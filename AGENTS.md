# AGENTS.md: instructions for AI agents (Ornith-1.5-35B-A3B-for-GTX1070-Inference-Engine)

This repository runs **Ornith-1.5-35B-A3B** (a 35B mixture-of-experts LLM, NVFP4) on **one NVIDIA GTX 1070 (8 GB)** under
**Windows** and serves it over the OpenAI, Anthropic and Ollama APIs. Follow these steps exactly.

## 1. Check the machine first

Run these checks and stop with a clear message to the user if one fails:

| Check | Command | Pass |
|---|---|---|
| GPU | `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv` | GTX 1070-class: compute capability 6.1, at least 8 GB. Other GPUs are not supported (the installer has `--force`, use it only if the user insists) |
| RAM | none | 32 GB recommended, 16 GB minimum |
| Disk | none | the target folder needs about 51 GB free during the install and 22 GB afterwards. An internal SSD is strongly recommended, USB hard disks are very slow |
| OS | none | Windows 10 or 11 |

## 2. Install

Clone `https://github.com/Kessler833/Ornith-1.5-35B-A3B-for-GTX1070-Inference-Engine` and run from the repository folder:

```bash
install.bat --dir C:\ornith
```

- **Python:** `install.bat` uses an installed Python 3.10 to 3.12, or downloads a private one into `.tools`.
- **Duration:** 30 to 60 minutes, mostly the 21.8 GB download.
- **Resuming:** the installer can be resumed. If it stops (network, power), run the same command again.
- **Output:** progress lines like `download 12.3 / 21.8 GB, 38 MB/s, about 4 min left`.
- **Errors:** they start with `ERROR:` and say what to do. Do not work around a disk-space error by deleting the user's
  files. Ask the user.
- **Success:** the installer ends with `installation complete` after a smoke test that asks the model one question.

Options:
- `--python <python.exe>` reuses an existing environment.
- `--no-test` skips the smoke test.
- `--keep-download` keeps the Hugging Face files.

## 3. Start the server

```bash
start-ornith.bat
```

or: `<python> ornith_serve.py`, with `--help` for all options.

- **Loading:** about 40 s to 3 minutes. `GET http://127.0.0.1:8000/health` returns 503 while loading and 200 when ready.
- **Speed:** about 25 to 30 tokens/s.
- **Common options:**
  - `--port 8000`
  - `--quality exact|fast|turbo` (default turbo, about 2 % quality loss for +42 % speed)
  - `--ram-gb 13.5`: lower it on 16 GB machines, for example 4
  - `--ctx 184320`
  - `--api-key KEY`: required when you listen on the network with `--host 0.0.0.0`

## 4. Verify

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":10,\"reasoning_effort\":\"none\"}"
```

## 5. Connect the user's tools

- **Base URLs:**
  - OpenAI: `http://127.0.0.1:8000/v1`
  - Anthropic: `http://127.0.0.1:8000`
  - Ollama: `http://127.0.0.1:8000`
- **Model:** `ornith-1.5-35b-a3b`
- **Key:** `local`, unless `--api-key` is set.
- **Per-tool settings:** see `docs/harnesses.md`.
- **Settings panel:** `http://127.0.0.1:8000/` in a browser.

## 6. Uninstall

```bash
install.bat uninstall
```

It lists what will be deleted and asks first. It removes only what the installer created.

## Do not

- Do not open the port to the network without `--api-key`.
- Do not edit files under `engine/` to "fix" speed. The kernels are tuned for Pascal, and changes need the benchmarks in
  the repository.
- Do not commit or upload `install.json`, `start-ornith.bat`, `.tools/` or anything from `~/.ornith-serve`. They are
  machine-specific.
