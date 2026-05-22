# GPU Cloud Deployment

This API can run on GPU cloud providers such as RunPod and Vast.ai. The main work is to package the server in Docker, expose port `8000`, provide the Hugging Face token as a secret, and avoid redownloading model weights every time the machine starts.

## Recommended First Deployment

Start with a long-running GPU instance:

- RunPod Pod
- Vast.ai instance

This is simpler than serverless for the first production pass because Stable Audio model startup can be slow, generation may run longer than HTTP proxy limits, and WAV responses can become large.

This API includes a job-based flow for cloud deployments:

- `POST /jobs` starts text-to-audio generation and returns a job ID.
- `POST /jobs/variations` starts audio-to-audio generation from uploaded source audio.
- `POST /jobs/inpaint` starts inpainting or continuation from uploaded source audio.
- A background worker writes the WAV or ZIP artifact to S3, R2, or provider storage.
- `GET /jobs/{id}` returns status and a download URL.

The original synchronous endpoints are still available for local development.

## Hardware

Use CUDA in cloud. Do not rely on CPU for production inference.

Suggested starting points:

- `small-sfx` and `small-music`: 16 GB GPU should be workable; 24 GB is a safer first target.
- `medium`: use CUDA with Flash Attention support. Start with 24 GB or larger, such as RTX 4090, A5000, L4, L40S, A40, A100, or similar.

Set:

```bash
STABLE_AUDIO_DEVICE=cuda
```

The upstream Stable Audio 3 package says `medium` requires CUDA with Flash Attention support.

## Environment Variables

Set secrets in the provider UI or account-level secret store, not in the Docker image.

```bash
HF_TOKEN=hf_your_token_here
STABLE_AUDIO_DEVICE=cuda
STABLE_AUDIO_DEFAULT_MODEL=small-sfx
STABLE_AUDIO_PRELOAD_MODELS=small-sfx
STABLE_AUDIO_MAX_DURATION=380
STABLE_AUDIO_MAX_STEPS=50
STABLE_AUDIO_MAX_BATCH_SIZE=4
STABLE_AUDIO_MAX_UPLOAD_BYTES=104857600
HF_HOME=/workspace/.cache/huggingface
STABLE_AUDIO_STORAGE_BUCKET=your-output-bucket
STABLE_AUDIO_STORAGE_PREFIX=stable-audio/jobs
STABLE_AUDIO_STORAGE_ENDPOINT_URL=https://account-id.r2.cloudflarestorage.com
STABLE_AUDIO_STORAGE_REGION=auto
```

Notes:

- `HF_TOKEN` must belong to an account that has accepted the gated model terms.
- Preload only the models you actually need. Preloading all three models increases startup time and VRAM pressure.
- `HF_HOME` should point to persistent storage so model files survive restarts.
- Leave `STABLE_AUDIO_STORAGE_BUCKET` unset for local filesystem job output in `outputs/`.
- For Cloudflare R2, set `STABLE_AUDIO_STORAGE_ENDPOINT_URL`, `STABLE_AUDIO_STORAGE_REGION=auto`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`.

## Storage and Model Cache

Avoid downloading weights from Hugging Face on every boot.

For RunPod Pods:

- Use a network volume or persistent pod storage.
- Mount it at `/workspace` if possible.
- Set `HF_HOME=/workspace/.cache/huggingface`.

For Vast.ai:

- Allocate enough instance disk before creating the instance. Disk size cannot always be changed afterward.
- Use account-level environment variables for sensitive values.
- Set `HF_HOME=/workspace/.cache/huggingface`.

Disk guidance:

- One small model cache needs several GB.
- All three models plus dependencies can grow quickly.
- Use at least 80 GB if you plan to test all models on the same machine.

Job output storage:

- Local fallback writes WAV or ZIP artifacts to `STABLE_AUDIO_OUTPUT_DIR`, default `outputs/`.
- S3/R2 storage is enabled when `STABLE_AUDIO_STORAGE_BUCKET` is set.
- If `STABLE_AUDIO_STORAGE_PUBLIC_BASE_URL` is set, `GET /jobs/{id}` returns public URLs from that base.
- Otherwise, `GET /jobs/{id}` returns presigned S3-compatible URLs.

## Docker

A CUDA-based image is the safest path. Use Python 3.10 or 3.11 for best compatibility with Flash Attention.

This repo includes a production-oriented [Dockerfile](../Dockerfile). Build it with:

```bash
docker build --platform linux/amd64 -t your-registry/stable-audio-api:latest .
docker push your-registry/stable-audio-api:latest
```

For the `medium` model, build with Flash Attention enabled:

```bash
docker build \
  --platform linux/amd64 \
  --build-arg INSTALL_FLASH_ATTN=1 \
  -t your-registry/stable-audio-api:latest .
```

The Dockerfile is based on this shape:

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV HF_HOME=/workspace/.cache/huggingface
ENV STABLE_AUDIO_DEVICE=cuda
ENV STABLE_AUDIO_DEFAULT_MODEL=small-sfx
ENV STABLE_AUDIO_PRELOAD_MODELS=small-sfx

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen

EXPOSE 8000

CMD ["stable-audio-api", "--host", "0.0.0.0", "--port", "8000"]
```

If you deploy to a platform that expects the image entrypoint to run directly, use the Dockerfile `CMD`. If the platform uses an on-start command, set it to:

```bash
stable-audio-api --host 0.0.0.0 --port 8000
```

## RunPod Pods

Use a custom container image and expose HTTP port `8000`.

Important settings:

- Image: `your-registry/stable-audio-api:latest`
- HTTP port: `8000`
- Volume mount: `/workspace`
- Env vars: set `HF_TOKEN`, `HF_HOME`, `STABLE_AUDIO_DEVICE`, and model defaults.

RunPod HTTP proxy URL format:

```text
https://[POD_ID]-8000.proxy.runpod.net
```

The API must bind to `0.0.0.0`; this repo already does that when started with:

```bash
stable-audio-api --host 0.0.0.0 --port 8000
```

RunPod HTTP proxy caveat:

- The HTTP proxy has a 100-second timeout.
- Long generations may fail through the proxy even if the server is still working.
- Use TCP exposure or change the API to a job/status flow for long audio.

## RunPod Serverless

RunPod Serverless has two relevant modes:

- Queue-based endpoints: better for long-running jobs, but the app must be wrapped in a handler.
- Load-balancing endpoints: can run FastAPI directly, but have request/processing/payload limits.

For this API, load-balancing serverless is the closer fit because it supports custom FastAPI routes. The job endpoints help avoid long client-held HTTP requests. However, RunPod documents these constraints:

- Processing timeout around minutes, not unlimited.
- Request/response payload limit around 30 MB.
- No built-in queue/backpressure for load-balancing endpoints.

If using RunPod Serverless:

- Store generated WAV files externally.
- Return URLs instead of raw WAV bytes.
- Consider active workers above zero if cold starts are unacceptable.
- Use model caching when possible, but note that RunPod cached model support is currently one model per endpoint.
- Replace the in-memory job store with Redis/Postgres if you run more than one worker.

## Vast.ai Instances

Vast.ai instances run Docker containers with exclusive GPU access. Create a template or rent an instance with a custom Docker image.

Template settings:

- Docker image: `your-registry/stable-audio-api:latest`
- Port: `8000`
- Launch mode: Docker entrypoint, or SSH/Jupyter with an on-start command
- Env vars: set `HF_TOKEN`, `HF_HOME`, `STABLE_AUDIO_DEVICE`, and model defaults
- Disk: allocate enough for dependencies and model cache

CLI-style shape:

```bash
vastai create instance <OFFER_ID> \
  --image your-registry/stable-audio-api:latest \
  --env '-p 8000:8000 -e STABLE_AUDIO_DEVICE=cuda -e STABLE_AUDIO_DEFAULT_MODEL=small-sfx -e STABLE_AUDIO_PRELOAD_MODELS=small-sfx -e HF_HOME=/workspace/.cache/huggingface' \
  --disk 80 \
  --onstart-cmd 'stable-audio-api --host 0.0.0.0 --port 8000' \
  --direct
```

Do not put `HF_TOKEN` directly in a public template. Use Vast account-level environment variables or a private launch configuration.

## Public API Security

Before exposing the endpoint publicly, add authentication. The current server will accept requests from anyone who can reach it.

Minimum recommended additions:

- `STABLE_AUDIO_API_KEY` environment variable.
- Require `Authorization: Bearer ...` on generation endpoints.
- Keep `/health` public or return only minimal status.
- Add rate limiting if the service will be internet-facing.

## Operational Notes

Start conservative:

- Preload one model.
- Test short durations first.
- Watch VRAM during model switching.
- Avoid loading all three models if they do not fit comfortably.

Good first cloud test:

```bash
curl -X POST https://YOUR_ENDPOINT/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "small-sfx",
    "prompt": "short metallic impact with room reverb",
    "duration": 5,
    "steps": 8
  }'
```

Then poll the returned `status_url` until `status` is `succeeded`, and download from `download_url`.

Then test `small-music`, and only test `medium` after verifying Flash Attention works on the chosen GPU image.

## References

- [RunPod Pods](https://docs.runpod.io/pods/overview)
- [RunPod port exposure](https://docs.runpod.io/pods/configuration/expose-ports)
- [RunPod Serverless overview](https://docs.runpod.io/serverless/overview)
- [RunPod load-balancing endpoints](https://docs.runpod.io/serverless/load-balancing/overview)
- [RunPod model caching](https://docs.runpod.io/serverless/endpoints/model-caching)
- [Vast.ai instances](https://docs.vast.ai/guides/instances/overview)
- [Vast.ai template creation](https://docs.vast.ai/guides/templates/creating-templates)
- [Vast.ai template settings](https://docs.vast.ai/guides/templates/template-settings)
