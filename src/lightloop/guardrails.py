"""Guardrails: cautions before the system prompt, hooks checked before each tool call.

A caution is a sentence the model reads first. A hook is a deterministic check on a tool call:
`tool` is a name or a glob, `field` names one input field (or none: the whole input) and
`matches` is a regular expression searched in it; a call it matches is refused with the rule.
They come from `LIGHTLOOP_GUARDRAILS_URL` (with `LIGHTLOOP_TOKEN`) when set, else from a JSON
file `{"cautions": [...], "hooks": [...]}` beside the app, else none.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import httpx

from .env import TOKEN_ENV
from .loop import Refused, Tool

GUARDRAILS_URL_ENV = "LIGHTLOOP_GUARDRAILS_URL"
GUARDRAILS_FILE = "guardrails.json"


@dataclass
class Hook:
    rule: str
    tool: str
    field: str = ""
    matches: str = ""

    def blocks(self, tool: str, args: dict) -> bool:
        if not fnmatch.fnmatchcase(tool, self.tool):
            return False
        if not self.matches:
            return True
        value = args.get(self.field) if self.field else args
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                                                              default=str)
        return re.search(self.matches, text or "") is not None


@dataclass
class Guardrails:
    cautions: list[str] = field(default_factory=list)
    hooks: list[Hook] = field(default_factory=list)

    @classmethod
    def parse(cls, data: dict) -> "Guardrails":
        cautions = [str(c.get("rule") if isinstance(c, dict) else c)
                    for c in data.get("cautions") or []]
        hooks = []
        for h in data.get("hooks") or []:
            shape = h.get("hook") if isinstance(h.get("hook"), dict) else h
            if shape.get("tool"):
                hooks.append(Hook(rule=str(h.get("rule") or shape.get("rule") or ""),
                                  tool=str(shape["tool"]), field=str(shape.get("field") or ""),
                                  matches=str(shape.get("matches") or "")))
        return cls([c for c in cautions if c.strip()], hooks)

    @classmethod
    def load(cls, *, url: str = "", token: str = "", path: Optional[str] = None) -> "Guardrails":
        """From `url` (or `LIGHTLOOP_GUARDRAILS_URL`), else from `path` (default
        `guardrails.json` in the working directory), else none."""
        url = url or os.environ.get(GUARDRAILS_URL_ENV, "").strip()
        token = token or os.environ.get(TOKEN_ENV, "").strip()
        if url:
            answer = httpx.get(url, headers={"authorization": f"Bearer {token}"}, timeout=30)
            answer.raise_for_status()
            return cls.parse(answer.json())
        file = Path(path or GUARDRAILS_FILE)
        if file.is_file():
            return cls.parse(json.loads(file.read_text(encoding="utf-8")))
        return cls()

    def system(self, system: str) -> str:
        """`system` with the cautions before it."""
        if not self.cautions:
            return system
        lines = "\n".join(f"- {c}" for c in self.cautions)
        return f"Rules you always keep:\n{lines}\n\n{system}"

    def check(self, tool: str, args: dict) -> None:
        for hook in self.hooks:
            if hook.blocks(tool, args):
                raise Refused(f"blocked by a guardrail: {hook.rule}")

    def tools(self, tools: list[Tool]) -> list[Tool]:
        """`tools`, each checked against the hooks before it runs."""
        def guarded(tool: Tool) -> Tool:
            def fn(args: dict, _tool=tool):
                self.check(_tool.name, args)
                return _tool.fn(args)
            return replace(tool, fn=fn)
        return [guarded(t) for t in tools] if self.hooks else list(tools)
