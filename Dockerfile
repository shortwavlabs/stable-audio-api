# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG INSTALL_FLASH_ATTN=0
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    HF_HOME=/workspace/.cache/huggingface \
    STABLE_AUDIO_DEVICE=cuda \
    STABLE_AUDIO_DEFAULT_MODEL=small-sfx \
    STABLE_AUDIO_PRELOAD_MODELS=small-sfx \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${PATH}"

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

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --python /usr/bin/python3

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --python /usr/bin/python3

# Required for the Stable Audio 3 medium model. Disabled by default because
# building flash-attn is GPU/CUDA/Python-version sensitive and can take time.
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        uv pip install --python "${VIRTUAL_ENV}/bin/python" ninja packaging \
        && MAX_JOBS="${MAX_JOBS}" uv pip install --python "${VIRTUAL_ENV}/bin/python" flash-attn --no-build-isolation; \
    fi

EXPOSE 8000

CMD ["stable-audio-api", "--host", "0.0.0.0", "--port", "8000"]
