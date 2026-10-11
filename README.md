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
- `on_text=print_piece` streams the model's text as it is written, for a chat that shows the
  answer coming (`AnthropicClient` and `OpenAIClient` both stream).
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

## Knowledge, guardrails and learning

```python
from lightloop import Guardrails, Knowledge, learn

knowledge = Knowledge("knowledge")        # the app's folder, or LIGHTLOOP_KNOWLEDGE_URL
rails = Guardrails.load()                 # LIGHTLOOP_GUARDRAILS_URL, else guardrails.json
system = rails.system("You answer customers about their orders.")
tools = rails.tools([knowledge.tool(), *tools])

result = run_loop(conn.client, model=conn.model, dialect=conn.dialect, system=system,
                  messages=messages, tools=tools, tool_choice="auto", max_steps=6)
# when the conversation is over:
learn(conn, system=system, messages=result.messages, tools=tools, conversation="c-1042")
```

- `Knowledge` lists and reads the app's knowledge files, from its folder or from
  `LIGHTLOOP_KNOWLEDGE_URL`; `knowledge.tool()` gives the model one tool for both.
- `Guardrails` puts its cautions before the system prompt and refuses a tool call one of its
  hooks matches: `{"tool": "<name or glob>", "field": "<input field>", "matches": "<regex>"}`.
- `learn` asks the model once, at a conversation's end, what the conversation taught and hands
  it to `LIGHTLOOP_LEARN_URL`. With no such address it does nothing.

| variable | meaning |
|---|---|
| `LIGHTLOOP_KNOWLEDGE_URL` | an endpoint serving the knowledge files |
| `LIGHTLOOP_GUARDRAILS_URL` | an endpoint serving `{"cautions": [...], "hooks": [...]}` |
| `LIGHTLOOP_LEARN_URL` | an endpoint taking a conversation's lessons |

## License

Apache-2.0.
