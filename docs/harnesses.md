# Connecting harnesses and clients

ornith-serve speaks three wire formats on one port (default `http://127.0.0.1:8000`):

| Format | Base URL to enter | Notes |
|---|---|---|
| OpenAI | `http://127.0.0.1:8000/v1` | chat completions, tools, streaming, `reasoning_content` |
| Anthropic | `http://127.0.0.1:8000` | messages, tools, thinking blocks, `count_tokens` |
| Ollama | `http://127.0.0.1:8000` | `/api/chat`, `/api/generate`, `/api/tags` |

- **Model name:** `ornith-1.5-35b-a3b`. Any other name works too.
- **API key:** if the server runs without `--api-key`, any non-empty key works (some clients insist on one, so use `local`).
- **Context length:** if a client asks for it, use the server's `--ctx` (default 184320). The settings panel at
  `http://127.0.0.1:8000/` shows it under *Startup settings*.

Menus and field names change between client versions. What you always need is the base URL, the key and the model name
from above.

## OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(model="ornith-1.5-35b-a3b",
                                   messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

## Anthropic Python SDK

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8000", api_key="local")
r = client.messages.create(model="ornith-1.5-35b-a3b", max_tokens=1024,
                           messages=[{"role": "user", "content": "Hello!"}])
print(r.content[-1].text)
```

## Claude Code

Claude Code can talk to any server with the Anthropic Messages API. Set these before starting it:

```bash
set ANTHROPIC_BASE_URL=http://127.0.0.1:8000
set ANTHROPIC_AUTH_TOKEN=local
set ANTHROPIC_MODEL=ornith-1.5-35b-a3b
set ANTHROPIC_SMALL_FAST_MODEL=ornith-1.5-35b-a3b
claude
```

Agent harnesses send long system prompts. The first request reads them in (about 1 to 2 minutes for 20k tokens), and
later calls come from the prefix cache.

## Cline / Roo Code (VS Code)

- **API Provider:** OpenAI Compatible
- **Base URL:** `http://127.0.0.1:8000/v1`
- **API Key:** `local`
- **Model ID:** `ornith-1.5-35b-a3b`
- **Context window:** your `--ctx`

## Continue (VS Code / JetBrains)

`config.yaml`:

```yaml
models:
  - name: Ornith (GTX 1070)
    provider: openai
    model: ornith-1.5-35b-a3b
    apiBase: http://127.0.0.1:8000/v1
    apiKey: local
    roles: [chat, edit, apply]
```

## Open WebUI

There are two ways to connect, either works:
- **Ollama connection:** set the Ollama API URL to `http://127.0.0.1:8000`. The model then appears in the list.
- **OpenAI connection:** API base `http://127.0.0.1:8000/v1`, key `local`.

## aider

```bash
aider --model openai/ornith-1.5-35b-a3b --openai-api-base http://127.0.0.1:8000/v1 --openai-api-key local
```

## Hermes Agent and other agent frameworks

Choose a *custom OpenAI-compatible* provider and enter the base URL, key and model name from the top of this page. If
the framework can use Anthropic or Ollama providers, those work as well.

## curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
```

## Images

Send images the way your format usually does:

| Format | How to send an image |
|---|---|
| OpenAI | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` (http(s) URLs work too) |
| Anthropic | `{"type": "image", "source": {"type": "base64", ...}}` |
| Ollama | `"images": ["<base64>"]` |

The image encoder runs on the CPU, about 6 s for an 800×600 image, which becomes ~475 tokens. It loads on the first
image (about 15 s). Repeated images come from a cache.

## Tips for agent loops

- **Short tool rounds:** keep them short. Every new piece of prompt costs at least about 1.5 s, because each layer
  needs its experts once.
- **Keep the history stable:** keep earlier messages unchanged, without timestamps that change each call. The cache
  only reuses an identical prefix. `cached_tokens` in the usage block shows whether it works.
- **Speed:** `quality: turbo` (the default) is the fastest. Use `exact` when every bit of quality counts. You can set
  it per request with `"ornith": {"quality": "exact"}` or switch it in the panel.
