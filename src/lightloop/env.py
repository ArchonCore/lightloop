"""A client from the environment, the same code inside Archon and anywhere else.

Inside Archon a public app's copy is given `LIGHTLOOP_BASE_URL` (its node's model endpoint) and
`LIGHTLOOP_TOKEN`; the endpoint resolves a level to a model. Anywhere else, `LIGHTLOOP_BASE_URL`
names a provider (`https://api.anthropic.com` by default with `ANTHROPIC_API_KEY`) and
`LIGHTLOOP_MODEL` the model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import httpx

from .client import AnthropicClient, OpenAIClient
from .loop import ANTHROPIC

BASE_URL_ENV, TOKEN_ENV, MODEL_ENV = "LIGHTLOOP_BASE_URL", "LIGHTLOOP_TOKEN", "LIGHTLOOP_MODEL"
DIALECT_ENV = "LIGHTLOOP_DIALECT"
ANTHROPIC_API = "https://api.anthropic.com"
#: The header naming whom a call served, read by an Archon node.
CALLER_HEADER = "x-lightloop-caller"


@dataclass
class Connection:
    client: object
    model: str
    dialect: str


def connect(level: int = 0, *, caller: str = "", model: str = "") -> Connection:
    """A client, its model and its dialect. `level` asks Archon's node for a model of that level;
    `model` (or `LIGHTLOOP_MODEL`) names one directly; `caller` is who this call serves."""
    base = os.environ.get(BASE_URL_ENV, "").strip()
    token = os.environ.get(TOKEN_ENV, "").strip()
    headers = {CALLER_HEADER: caller} if caller else {}
    named = model or os.environ.get(MODEL_ENV, "").strip()
    if base and token and not named:
        answer = httpx.get(f"{base.rstrip('/')}/resolve", params={"level": level},
                           headers={"authorization": f"Bearer {token}"}, timeout=30)
        answer.raise_for_status()
        found = answer.json()
        url = base.rstrip("/") + str(found.get("base_path") or "")
        dialect = str(found.get("dialect") or ANTHROPIC)
        client = (AnthropicClient(url, token, headers=headers) if dialect == ANTHROPIC
                  else OpenAIClient(url, token, headers=headers))
        return Connection(client, str(found["model"]), dialect)
    dialect = os.environ.get(DIALECT_ENV, ANTHROPIC).strip() or ANTHROPIC
    if not named:
        raise RuntimeError(f"no model: set {MODEL_ENV}, or run where {BASE_URL_ENV} and "
                           f"{TOKEN_ENV} name an endpoint that resolves one")
    if dialect == ANTHROPIC:
        key = token or os.environ.get("ANTHROPIC_API_KEY", "")
        return Connection(AnthropicClient(base or ANTHROPIC_API, key, bearer=bool(token),
                                          headers=headers), named, dialect)
    key = token or os.environ.get("OPENAI_API_KEY", "")
    return Connection(OpenAIClient(base or "https://api.openai.com/v1", key, headers=headers),
                      named, dialect)


def env_or(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)
