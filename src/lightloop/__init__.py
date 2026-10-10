"""lightloop: a small model-and-tools loop, run in your own process."""
from .client import AnthropicClient, ModelError, OpenAIClient
from .env import CALLER_HEADER, Connection, connect
from .loop import ANTHROPIC, OPENAI, LoopResult, Refused, Tool, run_loop, shown

__all__ = ["ANTHROPIC", "OPENAI", "AnthropicClient", "OpenAIClient", "ModelError", "Connection",
           "connect", "CALLER_HEADER", "LoopResult", "Refused", "Tool", "run_loop", "shown"]
__version__ = "0.1.0"
