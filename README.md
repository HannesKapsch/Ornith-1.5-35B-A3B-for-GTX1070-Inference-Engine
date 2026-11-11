# Ornith-1.5-35B-A3B for GTX 1070: Inference Engine

**A 35-billion-parameter AI on a graphics card from 2016.**

I have a GTX 1070. It has 8 GB of VRAM, no tensor cores, and as far as most AI software is concerned it no longer
exists. I wanted a real local AI agent on it anyway: one that can read code, call tools, look at screenshots and keep up
a long conversation. Not a 7B toy.

This repository is the result. It runs [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4),
a 35B mixture-of-experts model, on that one card at about **25–30 tokens per second**. The model is 22 GB, almost three
times what fits into the card.

It is not a chat app. It is a local server that speaks the **OpenAI, Anthropic and Ollama APIs**, so the tools you already
use can simply point at it: Claude Code, Cline, Continue, Open WebUI, aider, your own scripts. Tool calls, thinking,
images and streaming all work.

> Built for the GTX 1070 class (Pascal, compute capability 6.1, 8 GB) and nothing else. Every kernel is tuned for this
> one card. On other GPUs it refuses to start unless you insist with `--force`.

## How does that fit?

The trick is that the model is big, but lazy. Every layer has 256 "experts", and for each token only 8 of them do any
work: about 3B of the 35B parameters. So the engine never tries to hold the whole model. It keeps the busy experts close
and fetches the rest when needed:

- **Three storage tiers.** The experts used most live in VRAM. Many more sit in pinned system RAM (up to 13.5 GB), which
  the GPU can read directly over PCIe. Everything else stays on the SSD. While one layer computes, a small predictor
  already guesses which experts the next layer will want.
- **Kernels that read 4-bit weights directly.** The weights are stored in NVFP4, 4 bits per value. Instead of unpacking
  them to 16 bit first, the kernels decode them in registers right inside the matrix multiplication. On a
  memory-bound card, reading fewer bytes simply means going faster. The 1070 has no FP4 hardware, so this is done in
  software, compiled at run time with CuPy.
- **Working around Windows.** On Pascal under Windows, CUDA streams barely run in parallel. So instead of hoping for
  overlap, every decode step is recorded as CUDA graphs and replayed.
- **A compact memory of the conversation.** The KV cache is int8 with fine-grained scales, plus an exact copy of the last
  2048 tokens. 30 of the 40 layers use linear attention with a fixed-size state, so long contexts stay affordable: up
  to 184k tokens.
- **Optional speed-ups.** In `fast` and `turbo` mode the engine skips experts that are not in VRAM and would barely
  change the result anyway. That costs a measured 0.6 % or 2 % in perplexity and gains 26 % or 42 % speed. `exact`
  mode is lossless.

## What you need

- Windows 10 or 11
- a GTX 1070 (or another 8 GB card of the same generation) with a current NVIDIA driver
- 32 GB of RAM. 16 GB works too, start it with `--ram-gb 4` then.
- about 51 GB of free disk space during the install and 22 GB afterwards. Please use an internal SSD. It runs from a USB
  hard disk too, but at 2–8 tokens/s.
- Python is not required. The installer brings its own if you have none.

## Install

```bash
git clone https://github.com/Kessler833/Ornith-1.5-35B-A3B-for-GTX1070-Inference-Engine.git
cd Ornith-1.5-35B-A3B-for-GTX1070-Inference-Engine
install.bat --dir C:\ornith
```

Then get a coffee, or two. The installer:
1. Checks your GPU, RAM and disk before it touches anything.
2. Sets up Python with exactly the package versions it was tested with. If you have no Python 3.10–3.12, it gets a
   private one via [uv](https://github.com/astral-sh/uv).
3. Downloads the model from Hugging Face (21.8 GB).
4. Repacks it into the format the engine streams from, then deletes the download.
5. Asks the model one question to make sure everything works.

It shows what it is doing (`download 12.3 / 21.8 GB, 38 MB/s, about 4 min left`). If anything stops it, a dropped
connection or a closed laptop lid, just run the same command again and it continues where it left off.

`install.bat status` shows what is installed where. `install.bat uninstall` shows what it would delete, asks you, and then
removes only what the installer created.

## Start

```bash
start-ornith.bat
```

After about 40 seconds it is ready. Open **http://127.0.0.1:8000/** in your browser: a small settings panel shows what
the engine is doing and lets you switch quality, RAM, thinking and more while it runs.

The very first answer after the install takes about a minute, because the CUDA kernels compile once. After that they
are cached.

## Connect your tools

Point any tool at one of these:

| If your tool speaks... | use this base URL |
|---|---|
| OpenAI | `http://127.0.0.1:8000/v1` |
| Anthropic | `http://127.0.0.1:8000` |
| Ollama | `http://127.0.0.1:8000` |

The model name is `ornith-1.5-35b-a3b`, though any name works. As the API key, anything goes (`local` is fine) unless you
set one with `--api-key`. Ready-made settings for Claude Code, Cline, Continue, Open WebUI and aider are in
**[docs/harnesses.md](docs/harnesses.md)**.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
print(client.chat.completions.create(model="ornith-1.5-35b-a3b",
      messages=[{"role": "user", "content": "Hello!"}]).choices[0].message.content)
```

## Why agents don't have to wait

Agent tools send the entire conversation again with every single request. With 50,000 tokens of history, reading that
in from scratch would take minutes, every time. So the server remembers:

- **It keeps exact snapshots of its own state** at the end of each message. When the next request arrives, it only reads
  the new part. Measured against a full re-read, the result is identical.
- **When you switch between conversations,** the previous one is parked in RAM, exactly as it was.
- **Long conversations are also saved compressed to disk.** Coming back to a 150k-token conversation, even after a
  restart, takes seconds instead of about 23 minutes. (This copy is slightly lossy, about 1–2 % perplexity.)

The `cached_tokens` field in the API response shows how much came from memory. One GPU means one request at a time. Any
others wait their turn.

## When something is off

| What you see | What helps |
|---|---|
| `cannot listen on ...` | something else is using port 8000: close it, or use `--port` |
| it refuses your GPU | it is only built for the GTX 1070 class. `--force` tries anyway, with no promises |
| only 2–8 tokens/s | the model is on a USB or hard disk. Install again to an internal SSD with `--weights-from <old folder>`, then uninstall the old one |
| slower in very long sessions | the conversation memory takes VRAM away from the experts. Start a fresh conversation for a new topic |
| `cached_tokens` is always 0 | your tool changes old messages, for example with timestamps. Only an unchanged history can be reused |
| every short reply waits ~1.5 s before starting | that is physics: each new piece of text needs every layer's experts once over PCIe |

## Reference

### Options

All options also work in `~/.ornith-serve/config.json` (see [config.example.json](config.example.json)). The command
line wins.

| Option | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8000` | where to listen. With `0.0.0.0` also set `--api-key` |
| `--api-key` | none | require a key (`Authorization: Bearer` or `x-api-key`); env `ORNITH_API_KEY` |
| `--quality` | `turbo` | `exact` (lossless), `fast` (−0.6 % ppl, +26 % speed), `turbo` (−2 % ppl, +42 % speed) |
| `--ram-gb` | `13.5` | pinned RAM for experts (0.5–13.5). More is faster |
| `--ctx` | `184320` | maximum context in tokens |
| `--think` | `on` | thinking default for requests that do not say |
| `--langs` | `auto` | output scripts: `auto` allows CJK and other scripts once they appear in the conversation. `all`, `latin` |
| `--temperature` / `--max-tokens` | `0.6` / `32768` | defaults for requests that do not say |
| `--max-image-pixels` | `1048576` | images are scaled to at most this (1 MP ≈ 1000 tokens) |
| `--free-gb` | `1.0` | VRAM left for Windows and other programs |
| `--cache` | `on` | conversation memory. `off` reads every prompt in full |
| `--park-gb` | `3.0` | RAM for parked conversations |
| `--snapshots` / `--snap-cap-gb` / `--snap-min` | `~/.ornith-serve/snapshots` / `20` / `8192` | compressed conversations on disk |
| `--max-queue` | `16` | requests that may wait. More get `429` |
| `--bench` | | measure your speed in every mode, print a report, exit |
| `--fake` | | developer mode without GPU or weights |

### Endpoints

| Format | Endpoints |
|---|---|
| OpenAI | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` |
| Anthropic | `POST /v1/messages`, `POST /v1/messages/count_tokens` |
| Ollama | `POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` |
| Own | `GET /` (settings panel), `GET /health`, `GET /ornith/status`, `POST /ornith/config` |

- **Thinking:**
  - OpenAI: comes back as `reasoning_content`. Controlled with `reasoning_effort`, OpenRouter's `reasoning` or
    `chat_template_kwargs.enable_thinking`.
  - Anthropic: comes back as `thinking` blocks, controlled with `thinking`.
  - Ollama: comes back as `thinking`, controlled with `think`.
- **Images:** base64 or URLs, in each format's usual way. They are encoded on the CPU (about 6 s per image) and cached.
- **JSON mode:** `response_format` and `format` become an instruction to the model. There is no constrained decoding.
- **Per request:** `"ornith": {"quality": "exact"}` overrides settings for a single request.
- **Not supported:** `n > 1`, logprobs, embeddings, forced `tool_choice`.

Errors come in each API's usual shape, with a message that says what went wrong:

| Code | Meaning |
|---|---|
| `400` | the request is wrong |
| `401` | key missing or wrong |
| `413` | body too large |
| `429` | queue full |
| `503` | still loading |
| `500` | the engine failed. It resets itself, and the next request starts clean |

## Benchmark

```bash
python ornith_serve.py --bench
```

It measures start-up, prompt reading and answer speed in every quality mode and prints a small JSON report. If you open
an issue about speed, please paste it there.

## Contributing

You can work on the server without the GPU or the 22 GB of weights:
- `python ornith_serve.py --fake` starts it with a stand-in engine.
- `python tests/test_serve.py` runs about 95 checks against that stand-in in about 20 seconds. It records every token,
  so it can prove that the conversation memory restores exactly the right state.
- With the model installed, `python tests/test_real.py` compares every memory path with a fresh read.

| Path | Contents |
|---|---|
| `ornith_serve.py` | the server: API formats, conversation memory, generation |
| `panel.html` | the settings panel |
| `install.py`, `install.bat` | installer, status and uninstaller |
| `engine/` | the inference engine: decoding (`engine3.py`), prompt reading (`prefill.py`), CUDA kernels (`kern*.py`), disk snapshots (`kvsnap.py`), image encoder (`vision.py`) |

## Standing on shoulders

None of the individual ideas here are new, and they deserve credit. What is new is putting them together until a 35B
model runs well on one old consumer card.

| Idea | Related work |
|---|---|
| fused low-bit kernels | llama.cpp, Marlin, ExLlama |
| streaming experts from RAM | Mixtral-offloading, MoE-Infinity, Fiddler |
| skipping experts that are not loaded | Mixture of Cache-Conditional Experts and related work on cache-aware routing |
| compressing the KV cache for storage | KVTC, CacheGen |

## Model and license

This is an unofficial project, not made by or affiliated with the authors of Ornith. It just makes their model run on
old hardware. The weights are not part of this repository. The installer downloads them from
[ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4), which has its own license
(MIT at the time of writing).

## License

[MIT](LICENSE) © Hannes Kapsch
