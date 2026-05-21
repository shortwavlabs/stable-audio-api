# Stable Audio API

FastAPI server for [stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx), using `uv` for Python environment and dependency management.

The Hugging Face model is gated. Before starting the server, accept the model terms in the browser and provide a token with access.

## Setup

```bash
uv sync
```

```bash
export HF_TOKEN=hf_your_token_here
export STABLE_AUDIO_MODEL=small-sfx
export STABLE_AUDIO_DEVICE=cpu
uv run stable-audio-api --host 0.0.0.0 --port 8000
```

The server automatically loads a `.env` file from this project directory. You can also ask `uv` to load it explicitly:

```bash
uv run --env-file .env stable-audio-api --host 0.0.0.0 --port 8000
```

Use `STABLE_AUDIO_DEVICE=cuda` on a CUDA machine, or leave it unset to let `stable-audio-3` auto-detect `cuda`, `mps`, then `cpu`.

## Generate Audio

```bash
curl -X POST http://localhost:8000/v1/audio/generations \
  -H "Content-Type: application/json" \
  --output train.wav \
  -d '{
    "prompt": "chugging train coming into station with horn",
    "duration": 7,
    "steps": 8,
    "cfg_scale": 1.0,
    "seed": -1
  }'
```

## Endpoints

- `GET /health` returns whether the model is loaded.
- `POST /v1/audio/generations` returns a `audio/wav` response.
- `POST /generate` is an alias for the generation endpoint.

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token for gated model access. |
| `STABLE_AUDIO_MODEL` | `small-sfx` | Stable Audio 3 model name. |
| `STABLE_AUDIO_DEVICE` | unset | Optional `cuda`, `mps`, or `cpu`. |
| `STABLE_AUDIO_MODEL_HALF` | `true` | Use fp16 on CUDA. Automatically disabled by the model on CPU/MPS. |
| `STABLE_AUDIO_MAX_DURATION` | `120` | API duration limit in seconds. |
| `STABLE_AUDIO_MAX_STEPS` | `50` | API sampling step limit. |

The upstream Stable Audio 3 package pins PyTorch and torchaudio. This project mirrors its CUDA 12.6 `uv` source configuration for Linux x86_64; macOS uses the standard PyPI wheels.
