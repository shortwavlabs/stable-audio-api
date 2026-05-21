# Stable Audio API

FastAPI server for Stability AI Stable Audio 3, using `uv` for Python environment and dependency management.

The Hugging Face models are gated. Before starting the server, accept the terms for each model you want to use and provide a token with access:

- [stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
- [stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
- [stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)

## Setup

```bash
uv sync
```

```bash
export HF_TOKEN=hf_your_token_here
export STABLE_AUDIO_DEFAULT_MODEL=small-sfx
export STABLE_AUDIO_DEVICE=cpu
uv run stable-audio-api --host 0.0.0.0 --port 8000
```

The server automatically loads a `.env` file from this project directory. You can also ask `uv` to load it explicitly:

```bash
uv run --env-file .env stable-audio-api --host 0.0.0.0 --port 8000
```

Use `STABLE_AUDIO_DEVICE=cuda` on a CUDA machine, or leave it unset to let `stable-audio-3` auto-detect `cuda`, `mps`, then `cpu`. The `medium` model requires CUDA with Flash Attention support in the upstream Stable Audio 3 package.

## Generate Audio

Choose a model per request with the `model` property. Valid values are `small-sfx`, `small-music`, and `medium`.

```bash
curl -X POST http://localhost:8000/v1/audio/generations \
  -H "Content-Type: application/json" \
  --output train.wav \
  -d '{
    "model": "small-sfx",
    "prompt": "chugging train coming into station with horn",
    "duration": 7,
    "steps": 8,
    "cfg_scale": 1.0,
    "seed": -1
  }'
```

The API also accepts full Hugging Face repo IDs as aliases, for example `"model": "stabilityai/stable-audio-3-medium"`.

## Endpoints

- `GET /health` returns available models, preloaded models, loaded models, and duration limits.
- `POST /v1/audio/generations` returns a `audio/wav` response.
- `POST /generate` is an alias for the generation endpoint.

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token for gated model access. |
| `STABLE_AUDIO_DEFAULT_MODEL` | `small-sfx` | Default model when a request omits `model`. |
| `STABLE_AUDIO_MODEL` | unset | Backward-compatible alias for `STABLE_AUDIO_DEFAULT_MODEL`. |
| `STABLE_AUDIO_PRELOAD_MODELS` | default model | Comma-separated models to load at startup. Set empty to lazy-load only. |
| `STABLE_AUDIO_DEVICE` | unset | Optional `cuda`, `mps`, or `cpu`. |
| `STABLE_AUDIO_MODEL_HALF` | `true` | Use fp16 on CUDA. Automatically disabled by the model on CPU/MPS. |
| `STABLE_AUDIO_MAX_DURATION` | `380` | API-wide duration cap. Small models still cap at 120s; medium caps at 380s. |
| `STABLE_AUDIO_MAX_STEPS` | `50` | API sampling step limit. |

The upstream Stable Audio 3 package pins PyTorch and torchaudio. This project mirrors its CUDA 12.6 `uv` source configuration for Linux x86_64; macOS uses the standard PyPI wheels.
