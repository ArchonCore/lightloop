"""What a conversation taught: one more model call at its end, handed to a learning endpoint.

`LIGHTLOOP_LEARN_URL` (with `LIGHTLOOP_TOKEN`) names an endpoint answering `GET` with
`{"prompt": ...}` and taking `POST` `{conversation, caller, dump}`. Without it `learn`
does nothing. The call repeats the conversation's system prompt, tools and messages, so on a
provider that caches it reads them from cache; the tools are offered and never run.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Optional

import httpx

from .client import _Messages
from .env import TOKEN_ENV, Connection
from .loop import Refused, Tool, run_loop

LEARN_URL_ENV = "LIGHTLOOP_LEARN_URL"
#: The header saying a call is a conversation's lesson, not its work.
PURPOSE_HEADER = "x-lightloop-purpose"
NOTHING = "none"


def _marked(conn: Connection) -> object:
    client = copy.copy(conn.client)
    client.headers = {**client.headers, PURPOSE_HEADER: "learn"}
    client.messages = _Messages(client)
    return client


def learn(conn: Connection, *, system: str, messages: list[dict], tools: list[Tool],
          conversation: str, caller: str = "", url: str = "", token: str = "",
          max_tokens: int = 1500) -> Optional[dict]:
    """The conversation's dump, handed to the learning endpoint; None when there is no endpoint
    or nothing to tell, `{"kept": False, "error": ...}` when learning failed. It never raises:
    the conversation it follows is already answered."""
    url = (url or os.environ.get(LEARN_URL_ENV, "")).strip()
    if not url:
        return None
    try:
        return _learn(conn, system=system, messages=messages, tools=tools,
                      conversation=conversation, caller=caller, url=url, token=token,
                      max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 - learning never breaks a conversation
        return {"kept": False, "error": f"{type(exc).__name__}: {exc}"}


def _learn(conn: Connection, *, system: str, messages: list[dict], tools: list[Tool],
           conversation: str, caller: str, url: str, token: str,
           max_tokens: int) -> Optional[dict]:
    token = (token or os.environ.get(TOKEN_ENV, "")).strip()
    auth = {"authorization": f"Bearer {token}"}
    asked = httpx.get(url, headers=auth, timeout=30)
    asked.raise_for_status()
    prompt = str(asked.json().get("prompt") or "")
    if not prompt:
        return None

    def never(_args: dict) -> Any:
        raise Refused("the conversation is over; answer with text")

    offered = [Tool(t.name, t.description, t.schema, never) for t in tools]
    result = run_loop(_marked(conn), model=conn.model, dialect=conn.dialect, system=system,
                      messages=[*messages, {"role": "user", "content": prompt}], tools=offered,
                      tool_choice="auto", max_steps=1, max_tokens=max_tokens)
    dump = result.text.strip()
    if not dump or dump.lower() == NOTHING:
        return None
    sent = httpx.post(url, headers=auth, timeout=30,
                      json={"conversation": conversation, "caller": caller, "dump": dump})
    sent.raise_for_status()
    return sent.json()
