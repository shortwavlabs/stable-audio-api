from __future__ import annotations

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

import anyio
import soundfile as sf
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, model_validator

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


SupportedModel = Literal["small-sfx", "small-music", "medium"]

SUPPORTED_MODELS: tuple[SupportedModel, ...] = ("small-sfx", "small-music", "medium")
MODEL_REPO_IDS: dict[SupportedModel, str] = {
    "small-sfx": "stabilityai/stable-audio-3-small-sfx",
    "small-music": "stabilityai/stable-audio-3-small-music",
    "medium": "stabilityai/stable-audio-3-medium",
}
MODEL_DURATION_LIMITS_SECONDS: dict[SupportedModel, float] = {
    "small-sfx": 120.0,
    "small-music": 120.0,
    "medium": 380.0,
}
MODEL_ALIASES: dict[str, SupportedModel] = {
    "small-sfx": "small-sfx",
    "stable-audio-3-small-sfx": "small-sfx",
    "stabilityai/stable-audio-3-small-sfx": "small-sfx",
    "small-music": "small-music",
    "stable-audio-3-small-music": "small-music",
    "stabilityai/stable-audio-3-small-music": "small-music",
    "medium": "medium",
    "stable-audio-3-medium": "medium",
    "stabilityai/stable-audio-3-medium": "medium",
}
DURATION_PADDING_SECONDS = 6.0


def _normalize_model_name(value: str) -> SupportedModel:
    model_name = MODEL_ALIASES.get(value.strip().lower())
    if model_name is None:
        valid_values = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model {value!r}. Use one of: {valid_values}.")
    return model_name


def _env_model_list(name: str, default: str) -> list[SupportedModel]:
    value = os.getenv(name, default)
    return [_normalize_model_name(item) for item in value.split(",") if item.strip()]


DEFAULT_MODEL_NAME = _normalize_model_name(
    os.getenv("STABLE_AUDIO_DEFAULT_MODEL", os.getenv("STABLE_AUDIO_MODEL", "small-sfx"))
)
PRELOAD_MODEL_NAMES = _env_model_list("STABLE_AUDIO_PRELOAD_MODELS", DEFAULT_MODEL_NAME)
MODEL_DEVICE = os.getenv("STABLE_AUDIO_DEVICE") or None
MODEL_HALF = _env_bool("STABLE_AUDIO_MODEL_HALF", True)
MAX_DURATION_SECONDS = _env_float(
    "STABLE_AUDIO_MAX_DURATION",
    max(MODEL_DURATION_LIMITS_SECONDS.values()),
)
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
    model: SupportedModel = Field(
        default=DEFAULT_MODEL_NAME,
        description="Stable Audio 3 model to use: small-sfx, small-music, or medium.",
    )
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

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: str) -> SupportedModel:
        if not isinstance(value, str):
            raise ValueError("model must be a string")
        return _normalize_model_name(value)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        prompt = value.strip()
        if not prompt:
            raise ValueError("prompt must not be blank")
        return prompt

    @model_validator(mode="after")
    def duration_must_fit_model(self) -> GenerateAudioRequest:
        model_limit = min(MAX_DURATION_SECONDS, MODEL_DURATION_LIMITS_SECONDS[self.model])
        if self.duration > model_limit:
            raise ValueError(
                f"duration must be <= {model_limit:g}s for model {self.model!r}"
            )
        return self


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str | None
    loaded: bool
    available_models: list[str]
    loaded_models: list[str]
    preload_models: list[str]
    model_duration_limits_seconds: dict[str, float]
    max_duration_seconds: float
    max_steps: int


@dataclass(frozen=True)
class GenerationResult:
    audio: object
    model_name: SupportedModel
    sample_rate: int


class ModelRuntime:
    def __init__(self) -> None:
        self.models: dict[SupportedModel, object] = {}
        self.lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return bool(self.models)

    @property
    def loaded_models(self) -> list[SupportedModel]:
        return sorted(self.models)

    def sample_rate(self, model_name: SupportedModel) -> int:
        model = self.models.get(model_name)
        if model is None:
            raise RuntimeError(f"Model {model_name!r} has not been loaded.")
        return int(model.model.sample_rate)

    def load_model(self, model_name: SupportedModel) -> object:
        if model_name in self.models:
            return self.models[model_name]

        from stable_audio_3 import StableAudioModel

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token is None:
            logger.warning(
                "HF_TOKEN is not set. Loading will fail unless the model is already cached."
            )

        logger.info(
            "Loading Stable Audio model %s (%s) on device %s",
            model_name,
            MODEL_REPO_IDS[model_name],
            MODEL_DEVICE or "auto",
        )
        model = StableAudioModel.from_pretrained(
            model_name,
            device=MODEL_DEVICE,
            model_half=MODEL_HALF,
        )
        self.models[model_name] = model
        logger.info(
            "Loaded Stable Audio model %s at %s Hz",
            model_name,
            self.sample_rate(model_name),
        )
        return model

    def load_preconfigured_models(self) -> None:
        for model_name in PRELOAD_MODEL_NAMES:
            self.load_model(model_name)

    def generate(self, request: GenerateAudioRequest) -> GenerationResult:
        model = self.load_model(request.model)
        sample_rate = self.sample_rate(request.model)
        max_model_seconds = MODEL_DURATION_LIMITS_SECONDS[request.model] + DURATION_PADDING_SECONDS
        audio = model.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            duration=request.duration,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            batch_size=1,
            sample_size=int(max_model_seconds * sample_rate),
            duration_padding_sec=DURATION_PADDING_SECONDS,
            chunked_decode=request.chunked_decode,
        )
        return GenerationResult(
            audio=audio,
            model_name=request.model,
            sample_rate=sample_rate,
        )


runtime = ModelRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await anyio.to_thread.run_sync(runtime.load_preconfigured_models)
    app.state.runtime = runtime
    yield


app = FastAPI(
    title="Stable Audio API",
    summary="Generate audio with Stability AI Stable Audio 3.",
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
    effective_duration_limits = {
        model_name: min(MAX_DURATION_SECONDS, model_limit)
        for model_name, model_limit in MODEL_DURATION_LIMITS_SECONDS.items()
    }
    return HealthResponse(
        status="ok" if runtime.loaded else "loading",
        model=DEFAULT_MODEL_NAME,
        device=MODEL_DEVICE,
        loaded=runtime.loaded,
        available_models=list(SUPPORTED_MODELS),
        loaded_models=runtime.loaded_models,
        preload_models=PRELOAD_MODEL_NAMES,
        model_duration_limits_seconds=effective_duration_limits,
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
            result = await anyio.to_thread.run_sync(runtime.generate, request)
            wav_bytes = _audio_tensor_to_wav_bytes(result.audio, result.sample_rate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Audio generation failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = f"stable-audio-3-{request.model}.wav"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Model": result.model_name,
            "X-Sample-Rate": str(result.sample_rate),
        },
    )


@app.post("/generate", include_in_schema=False)
async def generate_alias(request: GenerateAudioRequest) -> Response:
    return await generate_audio(request)
