# Ornith-1.5-35B-A3B for GTX 1070: Inference Engine

**A 35B-parameter model on an 8 GB GTX 1070.**

This is a custom inference engine for [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4), a
35B mixture-of-experts model. It runs the model on a single GTX 1070 at about **25–30 tokens per second**, although the
weights (22 GB) are almost three times larger than the card's memory. The 1070 has no tensor cores and no FP4 support,
and most current inference software no longer targets it.

The engine runs as a local server with **OpenAI-, Anthropic- and Ollama-compatible APIs**. Existing tools connect to it
directly: Claude Code, Cline, Continue, Open WebUI, aider or your own scripts. Streaming, tool calls, thinking and image
input are supported.

> Built for the GTX 1070 class (Pascal, compute capability 6.1, 8 GB) and nothing else. Every kernel is tuned for this
> one card. On other GPUs it refuses to start unless you insist with `--force`.

## How it works

Each layer of the model has 256 experts, and each token uses only 8 of them: about 3B of the 35B parameters. The engine
therefore never holds the whole model in VRAM. It keeps frequently used experts close to the GPU and fetches the rest
on demand:

- **Three storage tiers.** The most used experts stay in VRAM, many more in pinned system RAM (up to 13.5 GB) that the
  GPU reads directly over PCIe, and the rest on the SSD. A predictor fetches the next layer's experts in advance.
- **Fused NVFP4 kernels.** The 4-bit weights are decoded in registers inside the matrix-vector kernels instead of being
  unpacked to fp16 first. Decoding is memory-bound, so reading fewer bytes makes it faster. The kernels are compiled at
  run time with CuPy/NVRTC.
- **CUDA graphs.** CUDA streams hardly overlap on Pascal under Windows (WDDM), so each decode step is captured as CUDA
  graphs and replayed.
- **Compact KV cache.** int8 with fine-grained scales, plus an exact fp16 window of the last 2048 tokens. 30 of the 40
  layers use linear attention with a fixed-size state, which keeps long contexts (up to 184k tokens) affordable.
- **Quality modes.** `fast` and `turbo` skip experts that are not in VRAM and carry little routing weight: −0.6 % or
  −2 % perplexity for +26 % or +42 % speed. `exact` is lossless.

## What you need

- Windows 10 or 11
- a GTX 1070 (or another 8 GB card of the same generation) with a current NVIDIA driver
- 32 GB of RAM. 16 GB works too, start it with `--ram-gb 4` then.
- about 51 GB of free disk space during the install and 22 GB afterwards, preferably on an internal SSD (from a USB hard
  disk it runs at only 2–8 tokens/s)
- Python is optional. The installer sets up its own if none is found.

## Install

```bash
git clone https://github.com/Kessler833/Ornith-1.5-35B-A3B-for-GTX1070-Inference-Engine.git
cd Ornith-1.5-35B-A3B-for-GTX1070-Inference-Engine
install.bat --dir C:\ornith
```

The installation takes 30–60 minutes, mostly for the download. The installer:
1. Checks GPU, RAM and disk space before it changes anything.
2. Sets up Python with the tested package versions. Without Python 3.10–3.12 it installs a private copy via
   [uv](https://github.com/astral-sh/uv).
3. Downloads the model from Hugging Face (21.8 GB).
4. Repacks it into the format the engine streams from and deletes the download.
5. Runs a short test query.

Progress is shown throughout (`download 12.3 / 21.8 GB, 38 MB/s, about 4 min left`). After an interruption, running the
same command again resumes where it stopped.

`install.bat status` shows what is installed where. `install.bat uninstall` shows what it would delete, asks you, and then
removes only what the installer created.

## Start

```bash
start-ornith.bat
```

The server is ready after about 40 seconds. **http://127.0.0.1:8000/** opens a settings panel with live status, where
quality, RAM, thinking and other settings can be changed while it runs.

The first answer after installation takes about a minute because the CUDA kernels are compiled once and then cached.

## Connect your tools

| API | Base URL |
|---|---|
| OpenAI | `http://127.0.0.1:8000/v1` |
| Anthropic | `http://127.0.0.1:8000` |
| Ollama | `http://127.0.0.1:8000` |

Model name: `ornith-1.5-35b-a3b` (other names are accepted). API key: any value, unless one is set with `--api-key`.
Configuration examples for Claude Code, Cline, Continue, Open WebUI and aider are in
**[docs/harnesses.md](docs/harnesses.md)**.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
print(client.chat.completions.create(model="ornith-1.5-35b-a3b",
      messages=[{"role": "user", "content": "Hello!"}]).choices[0].message.content)
```

## Prefix cache

Agent tools resend the full conversation with every request. Without a cache, a 50k-token history would be read in
again on each turn, which takes minutes. The server avoids this:

- **Checkpoints.** The engine state is saved at the end of each message, and the next request only processes the new
  part. The result is identical to a full re-read (measured).
- **Conversation switching.** When another conversation comes in, the previous one is kept as an exact copy in RAM.
- **Disk snapshots.** Long conversations are also stored compressed on disk. Resuming a 150k-token conversation, even
  after a restart, takes seconds instead of about 23 minutes. The compression is slightly lossy, about 1–2 %
  perplexity.

The `cached_tokens` field of the API response shows how much of the prompt came from the cache. Requests are processed
one at a time.

## Troubleshooting

| Symptom | Solution |
|---|---|
| `cannot listen on ...` | the port is in use: stop the other program or choose `--port` |
| the GPU is rejected | only GTX 1070-class cards are supported. `--force` starts anyway, without guarantees |
| only 2–8 tokens/s | the model is on a USB or hard disk. Reinstall to an internal SSD with `--weights-from <old folder>`, then uninstall the old copy |
| slower in very long sessions | the KV cache uses VRAM that the experts would otherwise use. Start a new conversation for new topics |
| `cached_tokens` is always 0 | the client modifies earlier messages, for example with timestamps. Only an unchanged prefix can be reused |
| about 1.5 s before every reply starts | expected: every new piece of prompt needs each layer's experts once over PCIe |

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

## Related work

The individual techniques build on existing work. This project combines them and tunes them for one older consumer GPU.

| Idea | Related work |
|---|---|
| fused low-bit kernels | llama.cpp, Marlin, ExLlama |
| streaming experts from RAM | Mixtral-offloading, MoE-Infinity, Fiddler |
| skipping experts that are not loaded | Mixture of Cache-Conditional Experts and related work on cache-aware routing |
| compressing the KV cache for storage | KVTC, CacheGen |

## Model and license

This is an unofficial project and is not affiliated with the authors of Ornith. The weights are not part of this repository. The installer downloads them from
[ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4), which has its own license
(MIT at the time of writing).

## License

[MIT](LICENSE) © Hannes Kapsch
