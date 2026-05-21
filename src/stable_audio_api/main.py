from __future__ import annotations

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
import soundfile as sf
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=dotenv_path or None)
if os.getenv("HUGGING_FACE_HUB_TOKEN") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {value!r}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


MODEL_NAME = os.getenv("STABLE_AUDIO_MODEL", "small-sfx")
MODEL_DEVICE = os.getenv("STABLE_AUDIO_DEVICE") or None
MODEL_HALF = _env_bool("STABLE_AUDIO_MODEL_HALF", True)
MAX_DURATION_SECONDS = _env_float("STABLE_AUDIO_MAX_DURATION", 120.0)
MAX_STEPS = _env_int("STABLE_AUDIO_MAX_STEPS", 50)

Duration = Annotated[
    float,
    Field(
        gt=0,
        le=MAX_DURATION_SECONDS,
        description="Generated audio duration in seconds.",
    ),
]


class GenerateAudioRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt describing the sound effect.")
    negative_prompt: str | None = Field(
        default=None,
        description="Optional text prompt describing qualities to avoid.",
    )
    duration: Duration = 7.0
    steps: int = Field(default=8, ge=1, le=MAX_STEPS)
    cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    seed: int = Field(default=-1, description="-1 selects a random seed.")
    chunked_decode: bool | None = Field(
        default=None,
        description="Override the model default for chunked autoencoder decoding.",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        prompt = value.strip()
        if not prompt:
            raise ValueError("prompt must not be blank")
        return prompt


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str | None
    loaded: bool
    max_duration_seconds: float
    max_steps: int


class ModelRuntime:
    def __init__(self) -> None:
        self.model = None
        self.lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def sample_rate(self) -> int:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        return int(self.model.model.sample_rate)

    def load(self) -> None:
        from stable_audio_3 import StableAudioModel

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token is None:
            logger.warning(
                "HF_TOKEN is not set. Loading will fail unless the model is already cached."
            )

        logger.info("Loading Stable Audio model %s on device %s", MODEL_NAME, MODEL_DEVICE or "auto")
        self.model = StableAudioModel.from_pretrained(
            MODEL_NAME,
            device=MODEL_DEVICE,
            model_half=MODEL_HALF,
        )
        logger.info(
            "Loaded Stable Audio model %s at %s Hz",
            MODEL_NAME,
            self.sample_rate,
        )

    def generate(self, request: GenerateAudioRequest):
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        return self.model.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            duration=request.duration,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            batch_size=1,
            chunked_decode=request.chunked_decode,
        )


runtime = ModelRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await anyio.to_thread.run_sync(runtime.load)
    app.state.runtime = runtime
    yield


app = FastAPI(
    title="Stable Audio API",
    summary="Generate sound effects with Stability AI Stable Audio 3 Small SFX.",
    version="0.1.0",
    lifespan=lifespan,
)


def _audio_tensor_to_wav_bytes(audio, sample_rate: int) -> bytes:
    import torch

    if not isinstance(audio, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor audio, got {type(audio)!r}")

    waveform = audio.detach().to(torch.float32).cpu().clamp(-1, 1)
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Expected audio shape [channels, samples], got {tuple(waveform.shape)}")

    audio_np = waveform.transpose(0, 1).numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if runtime.loaded else "loading",
        model=MODEL_NAME,
        device=MODEL_DEVICE,
        loaded=runtime.loaded,
        max_duration_seconds=MAX_DURATION_SECONDS,
        max_steps=MAX_STEPS,
    )


@app.post(
    "/v1/audio/generations",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Generated WAV audio.",
        }
    },
)
async def generate_audio(request: GenerateAudioRequest) -> Response:
    async with runtime.lock:
        try:
            audio = await anyio.to_thread.run_sync(runtime.generate, request)
            wav_bytes = _audio_tensor_to_wav_bytes(audio, runtime.sample_rate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Audio generation failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = "stable-audio-3-small-sfx.wav"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Sample-Rate": str(runtime.sample_rate),
        },
    )


@app.post("/generate", include_in_schema=False)
async def generate_alias(request: GenerateAudioRequest) -> Response:
    return await generate_audio(request)
