from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import anyio
import soundfile as sf
from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

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
MAX_BATCH_SIZE = _env_int("STABLE_AUDIO_MAX_BATCH_SIZE", 4)
MAX_UPLOAD_BYTES = _env_int("STABLE_AUDIO_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
OUTPUT_DIR = Path(os.getenv("STABLE_AUDIO_OUTPUT_DIR", "outputs"))
STORAGE_BUCKET = (
    os.getenv("STABLE_AUDIO_STORAGE_BUCKET")
    or os.getenv("AWS_S3_BUCKET")
    or os.getenv("R2_BUCKET")
    or os.getenv("R2_BUCKET_NAME")
)
STORAGE_PREFIX = os.getenv("STABLE_AUDIO_STORAGE_PREFIX", "stable-audio/jobs").strip("/")
STORAGE_ENDPOINT_URL = (
    os.getenv("STABLE_AUDIO_STORAGE_ENDPOINT_URL")
    or os.getenv("AWS_ENDPOINT_URL_S3")
    or os.getenv("R2_ENDPOINT_URL")
)
STORAGE_REGION = (
    os.getenv("STABLE_AUDIO_STORAGE_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "us-east-1"
)
STORAGE_PUBLIC_BASE_URL = os.getenv("STABLE_AUDIO_STORAGE_PUBLIC_BASE_URL")
PRESIGNED_URL_EXPIRES = _env_int("STABLE_AUDIO_PRESIGNED_URL_EXPIRES", 3600)

Duration = Annotated[
    float,
    Field(
        gt=0,
        le=MAX_DURATION_SECONDS,
        description="Generated audio duration in seconds.",
    ),
]
SamplerKwargValue = str | int | float | bool | None
SAMPLER_KWARG_RESERVED_KEYS = {
    "prompt",
    "negative_prompt",
    "duration",
    "steps",
    "cfg_scale",
    "batch_size",
    "sample_size",
    "truncate_output_to_duration",
    "conditioning",
    "conditioning_tensors",
    "negative_conditioning",
    "negative_conditioning_tensors",
    "seed",
    "init_audio",
    "init_noise_level",
    "inpaint_audio",
    "inpaint_mask",
    "inpaint_mask_start_seconds",
    "inpaint_mask_end_seconds",
    "duration_padding_sec",
    "apg_scale",
    "dist_shift",
    "return_latents",
    "chunked_decode",
}


class GenerateAudioRequest(BaseModel):
    model: SupportedModel = Field(
        default=DEFAULT_MODEL_NAME,
        description="Stable Audio 3 model to use: small-sfx, small-music, or medium.",
    )
    prompt: str = Field(..., min_length=1, description="Text prompt describing the audio.")
    negative_prompt: str | None = Field(
        default=None,
        description="Optional text prompt describing qualities to avoid.",
    )
    duration: Duration = 7.0
    steps: int = Field(default=8, ge=1, le=MAX_STEPS)
    cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    batch_size: int = Field(
        default=1,
        ge=1,
        le=MAX_BATCH_SIZE,
        description="Number of variations to generate. Batch outputs are returned as a ZIP.",
    )
    seed: int = Field(default=-1, description="-1 selects a random seed.")
    chunked_decode: bool | None = Field(
        default=None,
        description="Override the model default for chunked autoencoder decoding.",
    )
    apg_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Adaptive Projected Guidance scale.",
    )
    duration_padding_sec: float = Field(
        default=DURATION_PADDING_SECONDS,
        ge=0.0,
        le=30.0,
        description="Extra seconds used by the upstream model when adapting generation length.",
    )
    sampler_kwargs: dict[str, SamplerKwargValue] = Field(
        default_factory=dict,
        description="Advanced sampler keyword arguments passed through to the upstream sampler.",
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

    @field_validator("sampler_kwargs")
    @classmethod
    def sampler_kwargs_must_be_safe_scalars(
        cls,
        value: dict[str, SamplerKwargValue] | None,
    ) -> dict[str, SamplerKwargValue]:
        if value is None:
            return {}

        for key, kwarg_value in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("sampler_kwargs keys must be non-empty strings")
            if key in SAMPLER_KWARG_RESERVED_KEYS:
                raise ValueError(f"sampler_kwargs cannot override explicit field {key!r}")
            if kwarg_value is not None and not isinstance(kwarg_value, (str, int, float, bool)):
                raise ValueError(
                    "sampler_kwargs values must be strings, numbers, booleans, or null"
                )
        return value

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
    storage_backend: str
    available_models: list[str]
    loaded_models: list[str]
    preload_models: list[str]
    model_duration_limits_seconds: dict[str, float]
    max_duration_seconds: float
    max_steps: int
    max_batch_size: int


JobStatus = Literal["queued", "running", "succeeded", "failed"]
StorageBackend = Literal["local", "s3"]
GenerationMode = Literal["text-to-audio", "audio-to-audio", "inpainting"]


class CreateJobResponse(BaseModel):
    id: str
    status: JobStatus
    status_url: str


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    mode: GenerationMode
    model: str
    duration: float
    steps: int
    output_count: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    download_url: str | None = None
    download_content_type: str | None = None
    error: str | None = None
    storage_backend: StorageBackend | None = None
    storage_key: str | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    audio: object
    model_name: SupportedModel
    sample_rate: int


@dataclass(frozen=True)
class AudioInput:
    sample_rate: int
    waveform: object


MaskSeconds = float | list[float]


@dataclass(frozen=True)
class RuntimeGenerationRequest:
    mode: GenerationMode
    controls: GenerateAudioRequest
    init_audio: AudioInput | None = None
    init_noise_level: float = 1.0
    inpaint_audio: AudioInput | None = None
    inpaint_mask_start_seconds: MaskSeconds | None = None
    inpaint_mask_end_seconds: MaskSeconds | None = None


@dataclass(frozen=True)
class AudioArtifact:
    content: bytes
    content_type: str
    filename: str
    output_count: int


@dataclass(frozen=True)
class StoredAudio:
    backend: StorageBackend
    key: str
    content_type: str
    filename: str
    output_count: int
    local_path: Path | None = None


@dataclass
class JobRecord:
    id: str
    request: RuntimeGenerationRequest
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stored_audio: StoredAudio | None = None
    sample_rate: int | None = None
    error: str | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _audio_input_tuple(audio_input: AudioInput | None) -> tuple[int, Any] | None:
    if audio_input is None:
        return None
    return audio_input.sample_rate, audio_input.waveform


class AudioStorage:
    def __init__(self) -> None:
        self.backend: StorageBackend = "s3" if STORAGE_BUCKET else "local"
        self._s3_client = None

    def save_artifact(
        self,
        job_id: str,
        artifact: AudioArtifact,
    ) -> StoredAudio:
        if self.backend == "s3":
            return self._save_s3(job_id, artifact)
        return self._save_local(job_id, artifact)

    def download_url(self, stored_audio: StoredAudio, request: Request) -> str:
        if stored_audio.backend == "s3":
            return self._s3_download_url(stored_audio.key)
        return str(request.url_for("download_job_audio", job_id=stored_audio.key))

    def _save_local(
        self,
        job_id: str,
        artifact: AudioArtifact,
    ) -> StoredAudio:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{job_id}-{artifact.filename}"
        path.write_bytes(artifact.content)
        return StoredAudio(
            backend="local",
            key=job_id,
            content_type=artifact.content_type,
            filename=path.name,
            output_count=artifact.output_count,
            local_path=path,
        )

    def _save_s3(
        self,
        job_id: str,
        artifact: AudioArtifact,
    ) -> StoredAudio:
        if STORAGE_BUCKET is None:
            raise RuntimeError("STABLE_AUDIO_STORAGE_BUCKET is required for S3 storage.")

        key_parts = [part for part in (STORAGE_PREFIX, f"{job_id}-{artifact.filename}") if part]
        key = "/".join(key_parts)
        self._s3().put_object(
            Bucket=STORAGE_BUCKET,
            Key=key,
            Body=artifact.content,
            ContentType=artifact.content_type,
        )
        return StoredAudio(
            backend="s3",
            key=key,
            content_type=artifact.content_type,
            filename=artifact.filename,
            output_count=artifact.output_count,
        )

    def _s3_download_url(self, key: str) -> str:
        if STORAGE_PUBLIC_BASE_URL:
            return f"{STORAGE_PUBLIC_BASE_URL.rstrip('/')}/{key}"

        if STORAGE_BUCKET is None:
            raise RuntimeError("STABLE_AUDIO_STORAGE_BUCKET is required for S3 storage.")

        return self._s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": STORAGE_BUCKET, "Key": key},
            ExpiresIn=PRESIGNED_URL_EXPIRES,
        )

    def _s3(self):
        if self._s3_client is not None:
            return self._s3_client

        import boto3
        from botocore.config import Config

        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")

        kwargs = {
            "service_name": "s3",
            "region_name": STORAGE_REGION,
            "endpoint_url": STORAGE_ENDPOINT_URL,
            "config": Config(signature_version="s3v4"),
        }
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key

        self._s3_client = boto3.client(**kwargs)
        return self._s3_client


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, generation_request: RuntimeGenerationRequest) -> JobRecord:
        job = JobRecord(
            id=uuid4().hex,
            request=generation_request,
            status="queued",
            created_at=_utc_now(),
        )
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_running(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "running"
            job.started_at = _utc_now()
            return job

    async def mark_succeeded(
        self,
        job_id: str,
        stored_audio: StoredAudio,
        sample_rate: int,
    ) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "succeeded"
            job.completed_at = _utc_now()
            job.stored_audio = stored_audio
            job.sample_rate = sample_rate

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.completed_at = _utc_now()
            job.error = error


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

    def generate(self, request: RuntimeGenerationRequest) -> GenerationResult:
        controls = request.controls
        model = self.load_model(controls.model)
        sample_rate = self.sample_rate(controls.model)
        max_model_seconds = (
            MODEL_DURATION_LIMITS_SECONDS[controls.model] + controls.duration_padding_sec
        )
        audio = model.generate(
            prompt=controls.prompt,
            negative_prompt=controls.negative_prompt,
            duration=controls.duration,
            steps=controls.steps,
            cfg_scale=controls.cfg_scale,
            seed=controls.seed,
            batch_size=controls.batch_size,
            sample_size=int(max_model_seconds * sample_rate),
            init_audio=_audio_input_tuple(request.init_audio),
            init_noise_level=request.init_noise_level,
            inpaint_audio=_audio_input_tuple(request.inpaint_audio),
            inpaint_mask_start_seconds=request.inpaint_mask_start_seconds,
            inpaint_mask_end_seconds=request.inpaint_mask_end_seconds,
            duration_padding_sec=controls.duration_padding_sec,
            apg_scale=controls.apg_scale,
            chunked_decode=controls.chunked_decode,
            **controls.sampler_kwargs,
        )
        return GenerationResult(
            audio=audio,
            model_name=controls.model,
            sample_rate=sample_rate,
        )


runtime = ModelRuntime()
jobs = JobStore()
audio_storage = AudioStorage()


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


def _audio_tensor_to_wav_outputs(audio, sample_rate: int) -> list[bytes]:
    import torch

    if not isinstance(audio, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor audio, got {type(audio)!r}")

    waveform = audio.detach().to(torch.float32).cpu().clamp(-1, 1)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 3:
        raise ValueError(
            f"Expected audio shape [batch, channels, samples], got {tuple(waveform.shape)}"
        )

    return [_audio_tensor_to_wav_bytes(batch_audio, sample_rate) for batch_audio in waveform]


def _generation_result_to_artifact(result: GenerationResult) -> AudioArtifact:
    wav_outputs = _audio_tensor_to_wav_outputs(result.audio, result.sample_rate)
    if len(wav_outputs) == 1:
        return AudioArtifact(
            content=wav_outputs[0],
            content_type="audio/wav",
            filename=f"stable-audio-3-{result.model_name}.wav",
            output_count=1,
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, wav_bytes in enumerate(wav_outputs, start=1):
            archive.writestr(
                f"stable-audio-3-{result.model_name}-{index:02d}.wav",
                wav_bytes,
            )

    return AudioArtifact(
        content=buffer.getvalue(),
        content_type="application/zip",
        filename=f"stable-audio-3-{result.model_name}-batch.zip",
        output_count=len(wav_outputs),
    )


def _job_response(job: JobRecord, request: Request) -> JobResponse:
    download_url = None
    download_content_type = None
    storage_backend = None
    storage_key = None

    if job.stored_audio is not None:
        download_url = audio_storage.download_url(job.stored_audio, request)
        download_content_type = job.stored_audio.content_type
        storage_backend = job.stored_audio.backend
        storage_key = job.stored_audio.key

    return JobResponse(
        id=job.id,
        status=job.status,
        mode=job.request.mode,
        model=job.request.controls.model,
        duration=job.request.controls.duration,
        steps=job.request.controls.steps,
        output_count=job.stored_audio.output_count if job.stored_audio else None,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        download_url=download_url,
        download_content_type=download_content_type,
        error=job.error,
        storage_backend=storage_backend,
        storage_key=storage_key,
        sample_rate=job.sample_rate,
    )


def _text_runtime_request(request: GenerateAudioRequest) -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest(mode="text-to-audio", controls=request)


def _artifact_response(artifact: AudioArtifact, result: GenerationResult) -> Response:
    return Response(
        content=artifact.content,
        media_type=artifact.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "X-Model": result.model_name,
            "X-Sample-Rate": str(result.sample_rate),
            "X-Output-Count": str(artifact.output_count),
        },
    )


async def _generate_artifact(
    runtime_request: RuntimeGenerationRequest,
) -> tuple[GenerationResult, AudioArtifact]:
    async with runtime.lock:
        try:
            result = await anyio.to_thread.run_sync(runtime.generate, runtime_request)
            artifact = _generation_result_to_artifact(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Audio generation failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result, artifact


async def _load_uploaded_audio(audio: UploadFile) -> AudioInput:
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded audio file exceeds {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode uploaded audio: {exc}",
        ) from exc

    return AudioInput(sample_rate=int(sample_rate), waveform=waveform)


def _parse_sampler_kwargs(value: str | None) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="sampler_kwargs must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="sampler_kwargs must be a JSON object.")
    return parsed


def _build_generation_request_from_form(
    *,
    model: str,
    prompt: str,
    negative_prompt: str | None,
    duration: float,
    steps: int,
    cfg_scale: float,
    batch_size: int,
    seed: int,
    chunked_decode: bool | None,
    apg_scale: float,
    duration_padding_sec: float,
    sampler_kwargs: str | None,
) -> GenerateAudioRequest:
    try:
        return GenerateAudioRequest(
            model=model,
            prompt=prompt,
            negative_prompt=(
                negative_prompt.strip() if negative_prompt and negative_prompt.strip() else None
            ),
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            seed=seed,
            chunked_decode=chunked_decode,
            apg_scale=apg_scale,
            duration_padding_sec=duration_padding_sec,
            sampler_kwargs=_parse_sampler_kwargs(sampler_kwargs),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _parse_mask_seconds(value: str, field_name: str) -> MaskSeconds:
    raw_value = value.strip()
    if not raw_value:
        raise HTTPException(status_code=400, detail=f"{field_name} must not be blank.")

    if raw_value.startswith("["):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} must be valid JSON.",
            ) from exc
        if not isinstance(parsed, list) or not parsed:
            raise HTTPException(status_code=400, detail=f"{field_name} must be a non-empty list.")
        try:
            return [float(item) for item in parsed]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} must contain numbers.",
            ) from exc

    if "," in raw_value:
        try:
            return [float(item.strip()) for item in raw_value.split(",") if item.strip()]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} must contain numbers.",
            ) from exc

    try:
        return float(raw_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a number.") from exc


def _validate_inpaint_mask_times(
    starts: MaskSeconds,
    ends: MaskSeconds,
    duration: float,
) -> None:
    starts_are_list = isinstance(starts, list)
    ends_are_list = isinstance(ends, list)
    if starts_are_list != ends_are_list:
        raise HTTPException(
            status_code=400,
            detail="inpaint start and end values must both be scalars or both be lists.",
        )

    start_values = starts if starts_are_list else [starts]
    end_values = ends if ends_are_list else [ends]
    if len(start_values) != len(end_values):
        raise HTTPException(
            status_code=400,
            detail="inpaint start and end lists must have the same length.",
        )

    for start, end in zip(start_values, end_values):
        if start < 0 or end < 0:
            raise HTTPException(status_code=400, detail="inpaint times must be non-negative.")
        if start >= end:
            raise HTTPException(
                status_code=400,
                detail="Each inpaint start time must be less than its end time.",
            )
        if end > duration:
            raise HTTPException(
                status_code=400,
                detail="Inpaint end times must be less than or equal to duration.",
            )


def _resolve_inpaint_mask_value(
    primary_value: str | None,
    alias_value: str | None,
    primary_name: str,
    alias_name: str,
) -> str:
    if primary_value is not None and alias_value is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Use either {primary_name} or {alias_name}, not both.",
        )
    value = primary_value if primary_value is not None else alias_value
    if value is None:
        raise HTTPException(status_code=400, detail=f"{primary_name} is required.")
    return value


async def _run_generation_job(job_id: str) -> None:
    job = await jobs.mark_running(job_id)
    if job is None:
        logger.error("Cannot run missing job %s", job_id)
        return

    try:
        async with runtime.lock:
            result = await anyio.to_thread.run_sync(runtime.generate, job.request)
            artifact = _generation_result_to_artifact(result)

        stored_audio = await anyio.to_thread.run_sync(
            audio_storage.save_artifact,
            job.id,
            artifact,
        )
        await jobs.mark_succeeded(job.id, stored_audio, result.sample_rate)
    except Exception as exc:
        logger.exception("Job %s failed.", job_id)
        await jobs.mark_failed(job_id, str(exc))


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
        storage_backend=audio_storage.backend,
        available_models=list(SUPPORTED_MODELS),
        loaded_models=runtime.loaded_models,
        preload_models=PRELOAD_MODEL_NAMES,
        model_duration_limits_seconds=effective_duration_limits,
        max_duration_seconds=MAX_DURATION_SECONDS,
        max_steps=MAX_STEPS,
        max_batch_size=MAX_BATCH_SIZE,
    )


@app.post("/jobs", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    generation_request: GenerateAudioRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> CreateJobResponse:
    job = await jobs.create(_text_runtime_request(generation_request))
    background_tasks.add_task(_run_generation_job, job.id)
    return CreateJobResponse(
        id=job.id,
        status=job.status,
        status_url=str(request.url_for("get_job", job_id=job.id)),
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, request: Request) -> JobResponse:
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job, request)


@app.get("/jobs/{job_id}/audio", include_in_schema=False)
async def download_job_audio(job_id: str) -> FileResponse:
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "succeeded" or job.stored_audio is None:
        raise HTTPException(status_code=409, detail=f"Job is {job.status}.")
    if job.stored_audio.backend != "local" or job.stored_audio.local_path is None:
        raise HTTPException(status_code=404, detail="Local audio file is not available.")
    if not job.stored_audio.local_path.exists():
        raise HTTPException(status_code=404, detail="Local audio file is missing.")

    return FileResponse(
        job.stored_audio.local_path,
        media_type=job.stored_audio.content_type,
        filename=job.stored_audio.filename,
    )


@app.post(
    "/v1/audio/generations",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/zip": {}},
            "description": "Generated WAV audio, or a ZIP of WAVs when batch_size > 1.",
        }
    },
)
async def generate_audio(request: GenerateAudioRequest) -> Response:
    result, artifact = await _generate_artifact(_text_runtime_request(request))
    return _artifact_response(artifact, result)


@app.post(
    "/v1/audio/variations",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/zip": {}},
            "description": "Generated audio variation from an uploaded source file.",
        }
    },
)
async def generate_audio_variation(
    audio: Annotated[UploadFile, File(description="Source audio file.")],
    prompt: Annotated[str, Form(min_length=1)],
    model: Annotated[str, Form()] = DEFAULT_MODEL_NAME,
    negative_prompt: Annotated[str | None, Form()] = None,
    duration: Annotated[float, Form(gt=0, le=MAX_DURATION_SECONDS)] = 7.0,
    steps: Annotated[int, Form(ge=1, le=MAX_STEPS)] = 8,
    cfg_scale: Annotated[float, Form(ge=0.0, le=20.0)] = 1.0,
    batch_size: Annotated[int, Form(ge=1, le=MAX_BATCH_SIZE)] = 1,
    seed: Annotated[int, Form()] = -1,
    chunked_decode: Annotated[bool | None, Form()] = None,
    apg_scale: Annotated[float, Form(ge=0.0, le=10.0)] = 1.0,
    duration_padding_sec: Annotated[float, Form(ge=0.0, le=30.0)] = DURATION_PADDING_SECONDS,
    sampler_kwargs: Annotated[str | None, Form()] = None,
    init_noise_level: Annotated[float, Form(ge=0.0, le=1.0)] = 0.9,
) -> Response:
    controls = _build_generation_request_from_form(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        seed=seed,
        chunked_decode=chunked_decode,
        apg_scale=apg_scale,
        duration_padding_sec=duration_padding_sec,
        sampler_kwargs=sampler_kwargs,
    )
    source_audio = await _load_uploaded_audio(audio)
    runtime_request = RuntimeGenerationRequest(
        mode="audio-to-audio",
        controls=controls,
        init_audio=source_audio,
        init_noise_level=init_noise_level,
    )
    result, artifact = await _generate_artifact(runtime_request)
    return _artifact_response(artifact, result)


@app.post(
    "/v1/audio/inpaint",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/zip": {}},
            "description": "Generated inpainted audio from an uploaded source file.",
        }
    },
)
async def generate_audio_inpaint(
    audio: Annotated[UploadFile, File(description="Source audio file.")],
    prompt: Annotated[str, Form(min_length=1)],
    model: Annotated[str, Form()] = DEFAULT_MODEL_NAME,
    negative_prompt: Annotated[str | None, Form()] = None,
    duration: Annotated[float, Form(gt=0, le=MAX_DURATION_SECONDS)] = 7.0,
    steps: Annotated[int, Form(ge=1, le=MAX_STEPS)] = 8,
    cfg_scale: Annotated[float, Form(ge=0.0, le=20.0)] = 1.0,
    batch_size: Annotated[int, Form(ge=1, le=MAX_BATCH_SIZE)] = 1,
    seed: Annotated[int, Form()] = -1,
    chunked_decode: Annotated[bool | None, Form()] = None,
    apg_scale: Annotated[float, Form(ge=0.0, le=10.0)] = 1.0,
    duration_padding_sec: Annotated[float, Form(ge=0.0, le=30.0)] = DURATION_PADDING_SECONDS,
    sampler_kwargs: Annotated[str | None, Form()] = None,
    inpaint_start_seconds: Annotated[str | None, Form()] = None,
    inpaint_end_seconds: Annotated[str | None, Form()] = None,
    inpaint_mask_start_seconds: Annotated[str | None, Form()] = None,
    inpaint_mask_end_seconds: Annotated[str | None, Form()] = None,
) -> Response:
    controls = _build_generation_request_from_form(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        seed=seed,
        chunked_decode=chunked_decode,
        apg_scale=apg_scale,
        duration_padding_sec=duration_padding_sec,
        sampler_kwargs=sampler_kwargs,
    )
    start_value = _resolve_inpaint_mask_value(
        inpaint_start_seconds,
        inpaint_mask_start_seconds,
        "inpaint_start_seconds",
        "inpaint_mask_start_seconds",
    )
    end_value = _resolve_inpaint_mask_value(
        inpaint_end_seconds,
        inpaint_mask_end_seconds,
        "inpaint_end_seconds",
        "inpaint_mask_end_seconds",
    )
    starts = _parse_mask_seconds(start_value, "inpaint_start_seconds")
    ends = _parse_mask_seconds(end_value, "inpaint_end_seconds")
    _validate_inpaint_mask_times(starts, ends, controls.duration)

    source_audio = await _load_uploaded_audio(audio)
    runtime_request = RuntimeGenerationRequest(
        mode="inpainting",
        controls=controls,
        inpaint_audio=source_audio,
        inpaint_mask_start_seconds=starts,
        inpaint_mask_end_seconds=ends,
    )
    result, artifact = await _generate_artifact(runtime_request)
    return _artifact_response(artifact, result)


@app.post(
    "/jobs/variations",
    response_model=CreateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_variation_job(
    background_tasks: BackgroundTasks,
    request: Request,
    audio: Annotated[UploadFile, File(description="Source audio file.")],
    prompt: Annotated[str, Form(min_length=1)],
    model: Annotated[str, Form()] = DEFAULT_MODEL_NAME,
    negative_prompt: Annotated[str | None, Form()] = None,
    duration: Annotated[float, Form(gt=0, le=MAX_DURATION_SECONDS)] = 7.0,
    steps: Annotated[int, Form(ge=1, le=MAX_STEPS)] = 8,
    cfg_scale: Annotated[float, Form(ge=0.0, le=20.0)] = 1.0,
    batch_size: Annotated[int, Form(ge=1, le=MAX_BATCH_SIZE)] = 1,
    seed: Annotated[int, Form()] = -1,
    chunked_decode: Annotated[bool | None, Form()] = None,
    apg_scale: Annotated[float, Form(ge=0.0, le=10.0)] = 1.0,
    duration_padding_sec: Annotated[float, Form(ge=0.0, le=30.0)] = DURATION_PADDING_SECONDS,
    sampler_kwargs: Annotated[str | None, Form()] = None,
    init_noise_level: Annotated[float, Form(ge=0.0, le=1.0)] = 0.9,
) -> CreateJobResponse:
    controls = _build_generation_request_from_form(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        seed=seed,
        chunked_decode=chunked_decode,
        apg_scale=apg_scale,
        duration_padding_sec=duration_padding_sec,
        sampler_kwargs=sampler_kwargs,
    )
    source_audio = await _load_uploaded_audio(audio)
    runtime_request = RuntimeGenerationRequest(
        mode="audio-to-audio",
        controls=controls,
        init_audio=source_audio,
        init_noise_level=init_noise_level,
    )
    job = await jobs.create(runtime_request)
    background_tasks.add_task(_run_generation_job, job.id)
    return CreateJobResponse(
        id=job.id,
        status=job.status,
        status_url=str(request.url_for("get_job", job_id=job.id)),
    )


@app.post("/jobs/inpaint", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_inpaint_job(
    background_tasks: BackgroundTasks,
    request: Request,
    audio: Annotated[UploadFile, File(description="Source audio file.")],
    prompt: Annotated[str, Form(min_length=1)],
    model: Annotated[str, Form()] = DEFAULT_MODEL_NAME,
    negative_prompt: Annotated[str | None, Form()] = None,
    duration: Annotated[float, Form(gt=0, le=MAX_DURATION_SECONDS)] = 7.0,
    steps: Annotated[int, Form(ge=1, le=MAX_STEPS)] = 8,
    cfg_scale: Annotated[float, Form(ge=0.0, le=20.0)] = 1.0,
    batch_size: Annotated[int, Form(ge=1, le=MAX_BATCH_SIZE)] = 1,
    seed: Annotated[int, Form()] = -1,
    chunked_decode: Annotated[bool | None, Form()] = None,
    apg_scale: Annotated[float, Form(ge=0.0, le=10.0)] = 1.0,
    duration_padding_sec: Annotated[float, Form(ge=0.0, le=30.0)] = DURATION_PADDING_SECONDS,
    sampler_kwargs: Annotated[str | None, Form()] = None,
    inpaint_start_seconds: Annotated[str | None, Form()] = None,
    inpaint_end_seconds: Annotated[str | None, Form()] = None,
    inpaint_mask_start_seconds: Annotated[str | None, Form()] = None,
    inpaint_mask_end_seconds: Annotated[str | None, Form()] = None,
) -> CreateJobResponse:
    controls = _build_generation_request_from_form(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        seed=seed,
        chunked_decode=chunked_decode,
        apg_scale=apg_scale,
        duration_padding_sec=duration_padding_sec,
        sampler_kwargs=sampler_kwargs,
    )
    start_value = _resolve_inpaint_mask_value(
        inpaint_start_seconds,
        inpaint_mask_start_seconds,
        "inpaint_start_seconds",
        "inpaint_mask_start_seconds",
    )
    end_value = _resolve_inpaint_mask_value(
        inpaint_end_seconds,
        inpaint_mask_end_seconds,
        "inpaint_end_seconds",
        "inpaint_mask_end_seconds",
    )
    starts = _parse_mask_seconds(start_value, "inpaint_start_seconds")
    ends = _parse_mask_seconds(end_value, "inpaint_end_seconds")
    _validate_inpaint_mask_times(starts, ends, controls.duration)

    source_audio = await _load_uploaded_audio(audio)
    runtime_request = RuntimeGenerationRequest(
        mode="inpainting",
        controls=controls,
        inpaint_audio=source_audio,
        inpaint_mask_start_seconds=starts,
        inpaint_mask_end_seconds=ends,
    )
    job = await jobs.create(runtime_request)
    background_tasks.add_task(_run_generation_job, job.id)
    return CreateJobResponse(
        id=job.id,
        status=job.status,
        status_url=str(request.url_for("get_job", job_id=job.id)),
    )


@app.post("/generate", include_in_schema=False)
async def generate_alias(request: GenerateAudioRequest) -> Response:
    return await generate_audio(request)
