"""Model clients answering `messages.create` in the Anthropic shape, over plain HTTP.

`AnthropicClient` speaks the Messages API; `OpenAIClient` speaks chat completions and answers in
the same shape, so the loop never sees a dialect. Both send the token as a bearer (a gateway)
or as `x-api-key` (Anthropic directly).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TIMEOUT_S = 120.0


class ModelError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"the model answered {status}: {body[:500]}")
        self.status = status
        self.body = body


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class Answer:
    content: list
    usage: dict
    model: str = ""
    stop_reason: str = ""


class _Messages:
    def __init__(self, owner: "AnthropicClient | OpenAIClient"):
        self.owner = owner

    def create(self, **kwargs: Any) -> Answer:
        return self.owner._create(**kwargs)

    def stream_create(self, *, on_text: Callable[[str], None], **kwargs: Any) -> Answer:
        """The same answer, its text handed to `on_text` piece by piece as it is written."""
        return self.owner._stream(on_text, **kwargs)


def _sse(lines) -> Any:
    """Server-sent events as `(event, data)` pairs, `data` parsed from JSON."""
    event, data = "", []
    for line in lines:
        if line == "":
            if data:
                raw = "\n".join(data)
                if raw != "[DONE]":
                    try:
                        yield event, json.loads(raw)
                    except ValueError:
                        pass
            event, data = "", []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data) != "[DONE]":
        try:
            yield event, json.loads("\n".join(data))
        except ValueError:
            pass


class AnthropicClient:
    def __init__(self, base_url: str, token: str, *, bearer: bool = True,
                 headers: Optional[dict] = None, timeout: float = DEFAULT_TIMEOUT_S):
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.headers = {"anthropic-version": ANTHROPIC_VERSION,
                        "content-type": "application/json", **(headers or {})}
        if bearer:
            self.headers["authorization"] = f"Bearer {token}"
        else:
            self.headers["x-api-key"] = token
        self.timeout = timeout
        self.messages = _Messages(self)

    def _create(self, **kwargs: Any) -> Answer:
        answer = httpx.post(self.url, json=kwargs, headers=self.headers, timeout=self.timeout)
        if answer.status_code >= 400:
            raise ModelError(answer.status_code, answer.text)
        data = answer.json()
        blocks = []
        for b in data.get("content") or []:
            if b.get("type") == "tool_use":
                blocks.append(Block("tool_use", id=b.get("id", ""), name=b.get("name", ""),
                                    input=dict(b.get("input") or {})))
            elif b.get("type") == "text":
                blocks.append(Block("text", text=b.get("text", "")))
        return Answer(blocks, dict(data.get("usage") or {}), str(data.get("model") or ""),
                      str(data.get("stop_reason") or ""))

    def _stream(self, on_text: Callable[[str], None], **kwargs: Any) -> Answer:
        blocks: dict[int, dict] = {}
        usage: dict = {}
        model, stop = "", ""
        with httpx.stream("POST", self.url, json={**kwargs, "stream": True},
                          headers=self.headers, timeout=self.timeout) as answer:
            if answer.status_code >= 400:
                answer.read()
                raise ModelError(answer.status_code, answer.text)
            for event, data in _sse(answer.iter_lines()):
                kind = data.get("type") or event
                if kind == "message_start":
                    message = data.get("message") or {}
                    model = str(message.get("model") or "")
                    usage.update(message.get("usage") or {})
                elif kind == "content_block_start":
                    block = dict(data.get("content_block") or {})
                    block["_json"] = ""
                    blocks[int(data.get("index", len(blocks)))] = block
                elif kind == "content_block_delta":
                    block = blocks.setdefault(int(data.get("index", 0)), {"type": "text"})
                    delta = data.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = str(delta.get("text") or "")
                        block["text"] = block.get("text", "") + text
                        if text:
                            on_text(text)
                    elif delta.get("type") == "input_json_delta":
                        block["_json"] = block.get("_json", "") + str(delta.get("partial_json")
                                                                      or "")
                elif kind == "message_delta":
                    usage.update(data.get("usage") or {})
                    stop = str((data.get("delta") or {}).get("stop_reason") or stop)
                elif kind == "error":
                    raise ModelError(500, json.dumps(data))
        out = []
        for index in sorted(blocks):
            b = blocks[index]
            if b.get("type") == "tool_use":
                try:
                    args = json.loads(b.get("_json") or "{}") if b.get("_json") \
                        else dict(b.get("input") or {})
                except ValueError:
                    args = {}
                out.append(Block("tool_use", id=b.get("id", ""), name=b.get("name", ""),
                                 input=args))
            elif b.get("type") == "text":
                out.append(Block("text", text=b.get("text", "")))
        return Answer(out, usage, model, stop)


class OpenAIClient:
    """Chat completions, answered in the Anthropic shape."""

    def __init__(self, base_url: str, token: str, *, headers: Optional[dict] = None,
                 timeout: float = DEFAULT_TIMEOUT_S):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"authorization": f"Bearer {token}",
                        "content-type": "application/json", **(headers or {})}
        self.timeout = timeout
        self.messages = _Messages(self)

    @staticmethod
    def _body(model: str, max_tokens: int, messages: list, system: Any,
              tools: Optional[list], tool_choice: Optional[dict], extra: dict) -> dict:
        body: dict = {"model": model, "max_completion_tokens": max_tokens,
                      "messages": _openai_messages(system, messages), **extra}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object"}}} for t in tools]
            choice = (tool_choice or {}).get("type", "auto")
            body["tool_choice"] = "required" if choice == "any" else "auto"
        return body

    def _stream(self, on_text: Callable[[str], None], *, model: str, max_tokens: int,
                messages: list, system: Any = None, tools: Optional[list] = None,
                tool_choice: Optional[dict] = None, **extra: Any) -> Answer:
        body = self._body(model, max_tokens, messages, system, tools, tool_choice, extra)
        body.update(stream=True, stream_options={"include_usage": True})
        text, calls, usage, named = "", {}, {}, model
        with httpx.stream("POST", self.url, json=body, headers=self.headers,
                          timeout=self.timeout) as answer:
            if answer.status_code >= 400:
                answer.read()
                raise ModelError(answer.status_code, answer.text)
            for _event, data in _sse(answer.iter_lines()):
                named = str(data.get("model") or named)
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        text += piece
                        on_text(piece)
                    for call in delta.get("tool_calls") or []:
                        held = calls.setdefault(int(call.get("index", 0)),
                                                {"id": "", "name": "", "args": ""})
                        held["id"] = call.get("id") or held["id"]
                        fn = call.get("function") or {}
                        held["name"] = fn.get("name") or held["name"]
                        held["args"] += fn.get("arguments") or ""
        blocks = [Block("text", text=text)] if text else []
        for index in sorted(calls):
            held = calls[index]
            try:
                args = json.loads(held["args"] or "{}")
            except ValueError:
                args = {}
            blocks.append(Block("tool_use", id=held["id"], name=held["name"], input=args))
        return Answer(blocks, {"input_tokens": usage.get("prompt_tokens", 0),
                               "output_tokens": usage.get("completion_tokens", 0)}, named)

    def _create(self, *, model: str, max_tokens: int, messages: list, system: Any = None,
                tools: Optional[list] = None, tool_choice: Optional[dict] = None,
                **extra: Any) -> Answer:
        body = self._body(model, max_tokens, messages, system, tools, tool_choice, extra)
        answer = httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
        if answer.status_code >= 400:
            raise ModelError(answer.status_code, answer.text)
        data = answer.json()
        message = ((data.get("choices") or [{}])[0]).get("message") or {}
        blocks = []
        if message.get("content"):
            blocks.append(Block("text", text=str(message["content"])))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            blocks.append(Block("tool_use", id=call.get("id", ""), name=fn.get("name", ""),
                                input=args))
        usage = data.get("usage") or {}
        return Answer(blocks, {"input_tokens": usage.get("prompt_tokens", 0),
                               "output_tokens": usage.get("completion_tokens", 0)},
                      str(data.get("model") or ""))


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text", "")) for b in content or []
                     if isinstance(b, dict) and b.get("type") == "text")


def _openai_messages(system: Any, messages: list) -> list:
    out = []
    if system:
        out.append({"role": "system", "content": _text_of(system)})
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input"))}}
                     for b in content if b.get("type") == "tool_use"]
            entry: dict = {"role": "assistant", "content": _text_of(content) or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        texts = [b for b in content if b.get("type") == "text"]
        for b in content:
            if b.get("type") == "tool_result":
                result = b.get("content")
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                            "content": result if isinstance(result, str) else _text_of(result)})
        if texts:
            out.append({"role": "user", "content": _text_of(texts)})
    return out
