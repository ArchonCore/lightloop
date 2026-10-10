# lightloop

A small model-and-tools loop that runs inside your own process: no agent runtime to spawn,
no framework. One function, `run_loop`, calls a model with your tools until it answers, or
until an ending tool accepts.

```python
from lightloop import Tool, connect, run_loop

conn = connect(model="claude-haiku-5-5")          # ANTHROPIC_API_KEY from the environment

def order_status(args):
    return {"order": args["id"], "status": "shipped"}

tools = [Tool("order_status", "Look up an order by id.",
              {"type": "object", "properties": {"id": {"type": "string"}},
               "required": ["id"]}, order_status)]

result = run_loop(conn.client, model=conn.model, dialect=conn.dialect,
                  system="You answer customers about their orders.",
                  messages=[{"role": "user", "content": "Where is order 1042?"}],
                  tools=tools, tool_choice="auto", max_steps=6, max_tokens=800)
print(result.text)
```

- `tool_choice="auto"`: the run ends when the model answers with no tool call
  (`result.text`); `result.messages` continues the conversation.
- `tool_choice="any"`: the model must call a tool each step; the run ends when a tool marked
  `ends=True` accepts (`result.ending`, `result.value`). Raise `Refused(reason)` in a tool and
  the model reads the reason and decides again.
- Both the Anthropic Messages API and OpenAI chat completions, behind one shape
  (`AnthropicClient`, `OpenAIClient`).
- Prompt caching on Anthropic: the system prompt, the tools and the conversation's last block
  are cache breakpoints.

## Configuration

| variable | meaning |
|---|---|
| `LIGHTLOOP_MODEL` | the model, when you name one |
| `LIGHTLOOP_BASE_URL` | the provider's address (default `https://api.anthropic.com`) |
| `LIGHTLOOP_TOKEN` | a bearer token for that address; else `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| `LIGHTLOOP_DIALECT` | `anthropic` (default) or `openai` |

When `LIGHTLOOP_BASE_URL` and `LIGHTLOOP_TOKEN` name an endpoint that resolves models (an
Archon node gives its apps one), `connect(level=0)` asks it which model that level is, and
`connect(caller=...)` names whom the call serves.

## License

Apache-2.0.
