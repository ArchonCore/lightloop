"""An app's knowledge: the files of a folder, or of an endpoint serving them.

`LIGHTLOOP_KNOWLEDGE_URL` (with `LIGHTLOOP_TOKEN`) names an endpoint answering `GET <url>/` with
`{"files": [path, ...]}` and `GET <url>/<path>` with the file; without it the folder is read.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx

from .env import TOKEN_ENV
from .loop import Refused, Tool

KNOWLEDGE_URL_ENV = "LIGHTLOOP_KNOWLEDGE_URL"
KNOWLEDGE_FOLDER = "knowledge"
TEXT_SUFFIXES = (".md", ".txt", ".json", ".yaml", ".yml", ".csv")


class Knowledge:
    def __init__(self, folder: str = KNOWLEDGE_FOLDER, *, url: str = "", token: str = ""):
        self.folder = Path(folder)
        self.url = (url or os.environ.get(KNOWLEDGE_URL_ENV, "")).strip().rstrip("/")
        self.token = (token or os.environ.get(TOKEN_ENV, "")).strip()

    def _get(self, path: str) -> httpx.Response:
        answer = httpx.get(f"{self.url}/{path}", timeout=30,
                           headers={"authorization": f"Bearer {self.token}"})
        if answer.status_code == 404:
            raise Refused(f"no knowledge file {path!r}")
        answer.raise_for_status()
        return answer

    def index(self) -> list[str]:
        """Every file's path, relative to the folder."""
        if self.url:
            return [str(p) for p in self._get("").json().get("files") or []]
        if not self.folder.is_dir():
            return []
        return sorted(p.relative_to(self.folder).as_posix() for p in self.folder.rglob("*")
                      if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)

    def read(self, path: str) -> str:
        if self.url:
            return self._get(path.lstrip("/")).text
        root = self.folder.resolve()
        file = (root / path).resolve()
        if root not in file.parents or not file.is_file():
            raise Refused(f"no knowledge file {path!r}")
        return file.read_text(encoding="utf-8")

    def tool(self, name: str = "knowledge", *, about: Optional[str] = None) -> Tool:
        """One tool the model calls to list the files (no `path`) or read one."""
        def fn(args: dict):
            path = str(args.get("path") or "")
            return self.read(path) if path else {"files": self.index()}
        return Tool(name, about or "The app's knowledge: call with no path to list its files, "
                                   "with a path to read one.",
                    {"type": "object", "properties": {"path": {"type": "string"}}}, fn)
