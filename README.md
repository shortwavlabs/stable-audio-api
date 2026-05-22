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

For local development, the synchronous endpoint returns WAV bytes directly:

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
    "batch_size": 1,
    "seed": -1,
    "apg_scale": 1.0,
    "duration_padding_sec": 6.0,
    "sampler_kwargs": {}
  }'
```

The API also accepts full Hugging Face repo IDs as aliases, for example `"model": "stabilityai/stable-audio-3-medium"`.

When `batch_size` is greater than `1`, synchronous generation endpoints return `application/zip` containing one WAV per variation. A `batch_size` of `1` keeps returning a plain `audio/wav` response.

## Audio-to-Audio Variations

Use the multipart variation endpoint to edit or restyle an existing audio file. `init_noise_level` controls how strongly the source audio influences the output: lower values preserve more of the source, and `1.0` behaves like pure text-to-audio.

```bash
curl -X POST http://localhost:8000/v1/audio/variations \
  -F audio=@loop.wav \
  -F model=small-music \
  -F prompt="bossa nova bassline with warm upright bass" \
  -F duration=8 \
  -F init_noise_level=0.5 \
  --output variation.wav
```

The async version is `POST /jobs/variations`.

## Inpainting and Continuation

Use the multipart inpainting endpoint to regenerate a time range while preserving the rest of the uploaded audio.

```bash
curl -X POST http://localhost:8000/v1/audio/inpaint \
  -F audio=@loop.wav \
  -F model=small-music \
  -F prompt="punchy drum fill with tight snare rolls" \
  -F duration=16 \
  -F inpaint_start_seconds=4 \
  -F inpaint_end_seconds=8 \
  --output inpainted-loop.wav
```

Multiple regions can be passed as JSON lists:

```bash
curl -X POST http://localhost:8000/v1/audio/inpaint \
  -F audio=@loop.wav \
  -F model=small-music \
  -F prompt="cleaner percussion variation" \
  -F duration=16 \
  -F 'inpaint_start_seconds=[2,12]' \
  -F 'inpaint_end_seconds=[4,14]' \
  --output inpainted-loop.wav
```

For continuation, set `duration` longer than the source clip and start the inpaint region at the end of the source. The async version is `POST /jobs/inpaint`.

## Generate With Jobs

For cloud deployments, use the async job endpoints. They return quickly, generate audio in the background, write the artifact to local storage or S3/R2, and expose a download URL when complete.

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "small-sfx",
    "prompt": "short metallic impact with room reverb",
    "duration": 5,
    "steps": 8
  }'
```

Response:

```json
{
  "id": "68e48e7af36c4d829e3797a0b3e7687c",
  "status": "queued",
  "status_url": "http://localhost:8000/jobs/68e48e7af36c4d829e3797a0b3e7687c"
}
```

Poll status:

```bash
curl http://localhost:8000/jobs/68e48e7af36c4d829e3797a0b3e7687c
```

When `status` is `succeeded`, `download_url` points to the generated WAV or ZIP. Without object storage configured, job outputs are written under `outputs/` and served by the local API.

Job state is kept in memory. For multiple workers, restarts, or production serverless, use Redis/Postgres or another shared job store.

## Endpoints

- `GET /health` returns available models, preloaded models, loaded models, and duration limits.
- `POST /jobs` starts a background generation job and returns a job ID.
- `POST /jobs/variations` starts a background audio-to-audio variation job.
- `POST /jobs/inpaint` starts a background inpainting or continuation job.
- `GET /jobs/{id}` returns job status and a download URL when complete.
- `POST /v1/audio/generations` returns `audio/wav`, or `application/zip` when `batch_size > 1`.
- `POST /v1/audio/variations` accepts multipart audio input and returns generated audio.
- `POST /v1/audio/inpaint` accepts multipart audio input and returns inpainted audio.
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
| `STABLE_AUDIO_MAX_BATCH_SIZE` | `4` | Maximum number of variations per request. Batch outputs are returned as ZIP files. |
| `STABLE_AUDIO_MAX_UPLOAD_BYTES` | `104857600` | Maximum uploaded source-audio size for variation and inpainting endpoints. |
| `STABLE_AUDIO_OUTPUT_DIR` | `outputs` | Local output directory for job artifacts when S3/R2 is not configured. |
| `STABLE_AUDIO_STORAGE_BUCKET` | unset | S3/R2 bucket for job artifacts. Enables S3-compatible storage. |
| `STABLE_AUDIO_STORAGE_PREFIX` | `stable-audio/jobs` | Object key prefix for uploaded WAV files. |
| `STABLE_AUDIO_STORAGE_ENDPOINT_URL` | unset | S3-compatible endpoint URL, such as Cloudflare R2. |
| `STABLE_AUDIO_STORAGE_REGION` | `us-east-1` | S3 region. Use `auto` for Cloudflare R2 if desired. |
| `STABLE_AUDIO_STORAGE_PUBLIC_BASE_URL` | unset | Optional public/CDN base URL. If unset, the API generates presigned URLs. |
| `STABLE_AUDIO_PRESIGNED_URL_EXPIRES` | `3600` | Presigned download URL lifetime in seconds. |

The upstream Stable Audio 3 package pins PyTorch and torchaudio. This project mirrors its CUDA 12.6 `uv` source configuration for Linux x86_64; macOS uses the standard PyPI wheels.
