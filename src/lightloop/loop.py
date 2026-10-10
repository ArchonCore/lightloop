"""The loop: a model calls tools until it answers, or until an ending tool accepts.

Each step is one `messages.create` on an Anthropic-shaped client (`client.py` gives both
dialects that shape). With `tool_choice="any"` the model must call a tool every step and the run
ends when an ending tool accepts: one decision after looking things up. With `"auto"` the run
ends when the model answers with no tool call: a conversation's turn. A tool's answer, or its
refusal, goes back as the tool's result.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

ANTHROPIC, OPENAI = "anthropic", "openai"
_EPHEMERAL = {"type": "ephemeral"}
_EPHEMERAL_1H = {"type": "ephemeral", "ttl": "1h"}


def _breakpoint(cache_ttl: str) -> dict:
    return dict(_EPHEMERAL_1H if cache_ttl == "1h" else _EPHEMERAL)


def shown(value: Any, limit: int = 4000) -> str:
    """A tool result or a message as text for the model and the log."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"


class Refused(Exception):
    """A tool that cannot do what it was asked; the reason goes back to the model."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[dict], Any]
    #: An ending tool that accepts ends the run with its result.
    ends: bool = False


@dataclass
class LoopResult:
    #: The ending tool that accepted; "" when the model answered or the step cap ended the run.
    ending: str = ""
    value: Any = None
    #: The model's last words when it answered with no tool call.
    text: str = ""
    #: `{"tool", "input", "result" | "refused"}` per tool call, in order.
    steps: list = field(default_factory=list)
    #: Each model call's usage, in order.
    usage: list = field(default_factory=list)
    #: The whole conversation after the run, to continue it.
    messages: list = field(default_factory=list)


def _definitions(tools: list[Tool], dialect: str, cache_ttl: str = "5m") -> list[dict]:
    out = [{"name": t.name, "description": t.description, "input_schema": t.schema}
           for t in tools]
    if dialect == ANTHROPIC and out:
        out[-1]["cache_control"] = _breakpoint(cache_ttl)
    return out


def _system(system: str, dialect: str, cache_ttl: str = "5m") -> Any:
    if dialect == ANTHROPIC:
        return [{"type": "text", "text": system, "cache_control": _breakpoint(cache_ttl)}]
    return system


def _marked(messages: list[dict], dialect: str, cache_ttl: str = "5m") -> list[dict]:
    """The conversation with its last block a cache breakpoint (Anthropic), so the next step
    reads everything before it from cache; the other dialect caches by prefix and is sent
    none."""
    if dialect != ANTHROPIC:
        return [{**m, "content": [{k: v for k, v in b.items() if k != "cache_control"}
                                  if isinstance(b, dict) else b for b in m["content"]]}
                if isinstance(m.get("content"), list) else m for m in messages]
    if not messages:
        return messages
    last = dict(messages[-1])
    content = last["content"]
    blocks = ([{"type": "text", "text": content}] if isinstance(content, str)
              else [dict(b) for b in content])
    blocks[-1]["cache_control"] = _breakpoint(cache_ttl)
    last["content"] = blocks
    return [*messages[:-1], last]


def _clean(messages: list[dict]) -> list[dict]:
    """The conversation without cache breakpoints, as it is kept between runs."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [{k: v for k, v in b.items() if k != "cache_control"}
                       if isinstance(b, dict) else b for b in content]
        out.append({**m, "content": content})
    return out


def _block(b: Any) -> dict:
    kind = getattr(b, "type", "") if not isinstance(b, dict) else b.get("type", "")
    get = (lambda k, d=None: b.get(k, d)) if isinstance(b, dict) \
        else (lambda k, d=None: getattr(b, k, d))
    if kind == "tool_use":
        return {"type": "tool_use", "id": get("id"), "name": get("name"),
                "input": dict(get("input") or {})}
    return {"type": "text", "text": str(get("text", "") or "")}


def run_loop(client, *, model: str, system: str, messages: list[dict], tools: list[Tool],
             dialect: str = ANTHROPIC, max_steps: int = 4, max_tokens: int = 512,
             tool_choice: str = "any", logger: Optional[logging.Logger] = None,
             cache_key: str = "", cache_ttl: str = "5m", reasoning_effort: str = "",
             on_step: Optional[Callable[[dict], None]] = None,
             before_call: Optional[Callable[[], list[dict]]] = None,
             stop: Optional[Callable[[], bool]] = None,
             on_text: Optional[Callable[[str], None]] = None) -> LoopResult:
    """One run. `tool_choice="any"`: tool calls until an ending tool accepts or `max_steps`
    model calls are spent. `"auto"`: until the model answers with no tool call. `on_step` hears
    each tool call and each answer as `{"kind", ...}`; `before_call` may add user messages
    before each model call after the first (a message that arrived during the run); `stop`
    ends the run before the next model call; `on_text` hears the model's text piece by piece
    as it is written, on a client that streams (`client.py`)."""
    by_name = {t.name: t for t in tools}
    definitions = _definitions(tools, dialect, cache_ttl)
    convo = [dict(m) for m in messages]
    result = LoopResult()
    if logger is not None:
        logger.info("=== run, model %s\n--- messages\n%s", model, shown(messages, 20000))
    extra: dict = {"prompt_cache_key": cache_key} if dialect == OPENAI and cache_key else {}
    if dialect == OPENAI and reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    for step in range(1, max_steps + 1):
        if stop is not None and stop():
            break
        if step > 1 and before_call is not None:
            for added in before_call() or []:
                if convo and convo[-1]["role"] == "user" and isinstance(convo[-1]["content"],
                                                                         list):
                    convo[-1]["content"].append({"type": "text", "text": added["content"]}
                                                if isinstance(added.get("content"), str)
                                                else added)
                else:
                    convo.append(added)
        kwargs = dict(model=model, max_tokens=max_tokens,
                      system=_system(system, dialect, cache_ttl),
                      messages=_marked(convo, dialect, cache_ttl), **extra)
        if definitions:
            kwargs.update(tools=definitions, tool_choice={"type": tool_choice})
        streams = on_text is not None and hasattr(client.messages, "stream_create")
        answer = (client.messages.stream_create(on_text=on_text, **kwargs) if streams
                  else client.messages.create(**kwargs))
        result.usage.append(getattr(answer, "usage", None))
        blocks = [b for b in (_block(x) for x in getattr(answer, "content", []) or [])
                  if b["type"] == "tool_use" or b["text"].strip()]
        blocks = blocks or [{"type": "text", "text": "(no answer)"}]
        uses = [b for b in blocks if b["type"] == "tool_use"]
        convo.append({"role": "assistant", "content": blocks})
        texts = [b["text"] for b in blocks if b["type"] == "text"]
        if on_step is not None and texts:
            on_step({"kind": "text", "text": "\n".join(texts), "final": not uses})
        if not uses and tool_choice == "auto":
            result.text = "\n".join(texts).strip()
            result.messages = _clean(convo)
            if logger is not None:
                logger.info("=== step %d answered\n%s", step, result.text)
            return result
        replies, ended = [], None
        for use in uses:
            tool = by_name.get(use["name"])
            entry = {"tool": use["name"], "input": use["input"]}
            if on_step is not None:
                on_step({"kind": "tool_call", "tool": use["name"], "input": use["input"]})
            try:
                if tool is None:
                    raise Refused(f"there is no tool {use['name']}; the tools are "
                                  f"{', '.join(by_name)}")
                out = tool.fn(use["input"])
            except Refused as refusal:
                entry["refused"] = refusal.reason
                replies.append({"type": "tool_result", "tool_use_id": use["id"],
                                "content": refusal.reason, "is_error": True})
                result.steps.append(entry)
                if on_step is not None:
                    on_step({"kind": "tool_result", "tool": use["name"],
                             "result": refusal.reason, "refused": True})
                continue
            entry["result"] = out
            result.steps.append(entry)
            if on_step is not None:
                on_step({"kind": "tool_result", "tool": use["name"], "result": out})
            if tool.ends:
                ended = ended or (tool.name, out)
                replies.append({"type": "tool_result", "tool_use_id": use["id"],
                                "content": "done"})
            else:
                replies.append({"type": "tool_result", "tool_use_id": use["id"],
                                "content": shown(out)})
        if logger is not None:
            logger.info("=== step %d\n--- model\n%s\n--- tools\n%s", step, shown(blocks),
                        shown(result.steps[-len(uses):] if uses else []))
        if ended is not None:
            result.ending, result.value = ended
            result.messages = _clean(convo)
            return result
        convo.append({"role": "user", "content": replies or [
            {"type": "text", "text": "Answer by calling one of the tools."}]})
    result.messages = _clean(convo)
    return result
