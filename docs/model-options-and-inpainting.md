# Stable Audio 3 Model Options and Inpainting

This document summarizes the Stable Audio 3 model choices, the generation controls exposed by this API, the broader upstream `generate()` options, and how inpainting/continuation should be handled.

Sources:

- [Stable Audio 3 README](https://github.com/Stability-AI/stable-audio-3)
- [Stable Audio 3 Inference Methods](https://github.com/Stability-AI/stable-audio-3/blob/main/docs/workflows/inference.md)

## Model Choices

| API value | Hugging Face model | Best for | Hardware | Max length |
| --- | --- | --- | --- | --- |
| `small-sfx` | `stabilityai/stable-audio-3-small-sfx` | Sound effects, one-shots, foley, UI sounds, impacts | CPU or GPU | 120s |
| `small-music` | `stabilityai/stable-audio-3-small-music` | Loops, samples, music phrases | CPU or GPU | 120s |
| `medium` | `stabilityai/stable-audio-3-medium` | Higher quality music/audio and longer generations | CUDA GPU with Flash Attention | 380s |

For an app that generates audio samples for music, `small-music` is the main model to prioritize. `small-sfx` is useful for percussive hits, impacts, risers, transitions, foley, and texture layers. `medium` is useful when quality or longer output matters enough to justify GPU-only deployment.

The upstream repository also provides base checkpoints:

- `small-sfx-base`
- `small-music-base`
- `medium-base`

Those base models are not currently exposed by this API. The upstream docs note that `cfg_scale` and `negative_prompt` are primarily meaningful for the base checkpoints; the post-trained checkpoints still accept those parameters, but they may have little or no effect.

Stable Audio 3 Large is API-only and is not supported by the open-weight repository used by this server.

## Current API Request Options

The current JSON generation endpoints expose this request shape:

```json
{
  "model": "small-music",
  "prompt": "dusty boom bap drum loop, 90 BPM",
  "negative_prompt": "distorted, clipping",
  "duration": 8,
  "steps": 8,
  "cfg_scale": 1.0,
  "seed": -1,
  "chunked_decode": null
}
```

Available endpoints:

- `POST /v1/audio/generations` returns WAV bytes directly for local development.
- `POST /generate` is a local-development alias.
- `POST /jobs` starts a background text-to-audio job.
- `GET /jobs/{id}` returns job status and a download URL.

## Current API Fields

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | `small-sfx`, `small-music`, `medium`, or a full supported Hugging Face repo ID alias. |
| `prompt` | string | Required text description of the generated audio. |
| `negative_prompt` | string or null | Optional qualities to avoid. Mostly useful for base checkpoints upstream. |
| `duration` | number | Output length in seconds. Small models cap at 120s; medium caps at 380s. |
| `steps` | integer | Sampling steps. Upstream default is `8`; higher is not always better for post-trained models. |
| `cfg_scale` | number | Guidance scale. Mostly useful for base checkpoints upstream. |
| `seed` | integer | `-1` chooses a random seed. Use a fixed integer for repeatable output. |
| `chunked_decode` | boolean or null | Overrides chunked autoencoder decoding. `null` uses the model default. |

## Upstream Generation Options

The upstream `StableAudioModel.generate()` method supports more options than this API currently exposes:

| Option | Meaning |
| --- | --- |
| `prompt` | Text description. Can be a string or a list for batch customization. |
| `negative_prompt` | Things to avoid. Can be a string or a list for batch customization. |
| `duration` | Output length in seconds. Can be a number or a list for batch customization. |
| `steps` | Number of sampling steps. Default is `8`. |
| `cfg_scale` | Classifier-free guidance scale. |
| `batch_size` | Number of variations to generate in one call. Limited by VRAM. |
| `sample_size` | Output length in raw samples. The simple API usually uses `duration` instead. |
| `truncate_output_to_duration` | Whether to trim the decoded audio to the requested duration. |
| `seed` | Random seed, with `-1` meaning random. |
| `init_audio` | Source audio for audio-to-audio editing. |
| `init_noise_level` | How strongly the source audio is preserved in audio-to-audio editing. |
| `inpaint_audio` | Source audio for inpainting or continuation. |
| `inpaint_mask` | Custom tensor mask for advanced inpainting. |
| `inpaint_mask_start_seconds` | Start time for the region to regenerate. |
| `inpaint_mask_end_seconds` | End time for the region to regenerate. |
| `duration_padding_sec` | Extra seconds used when adapting variable-length generation. |
| `apg_scale` | Adaptive Projected Guidance scale. |
| `dist_shift` | Optional distribution-shift override for sampling. |
| `return_latents` | Return latents instead of decoded audio. |
| `chunked_decode` | Decode latents in overlapping chunks to reduce peak VRAM. |
| `conditioning` | Low-level prebuilt conditioning dictionaries. |
| `conditioning_tensors` | Low-level precomputed conditioning tensors. |
| `negative_conditioning` | Low-level negative conditioning dictionaries. |
| `negative_conditioning_tensors` | Low-level negative conditioning tensors. |
| `sampler_kwargs` | Additional sampler-specific options. |

The model object also supports LoRA methods:

| Method | Meaning |
| --- | --- |
| `load_lora([...])` | Load one or more LoRA checkpoints. |
| `set_lora_strength(strength)` | Adjust LoRA influence globally. |
| `set_lora_strength(strength, lora_index=0)` | Adjust one loaded LoRA by index. |

## Audio-to-Audio

Audio-to-audio editing uses an existing audio file as the starting point, then regenerates a variation based on a prompt.

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("small-music")

waveform, sr = torchaudio.load("loop.wav")

audio = model.generate(
    init_audio=(sr, waveform),
    init_noise_level=0.5,
    prompt="bossa nova bassline",
    duration=30,
)
```

`init_noise_level` controls how much the source matters:

| Value | Behavior |
| --- | --- |
| `0.1` | Close variation of the source audio. |
| `0.5` | Halfway blend between source and new generation. |
| `1.0` | Source has no effect; effectively pure text-to-audio generation. |

## Inpainting

Inpainting regenerates a specific region of an existing audio file while preserving everything outside that range. It is useful for fixing a bad section, replacing a sound, changing a fill, or extending a loop.

The upstream model call requires:

- `inpaint_audio`: the source audio as `(sample_rate, waveform)`.
- `inpaint_mask_start_seconds`: start of the region to regenerate.
- `inpaint_mask_end_seconds`: end of the region to regenerate.
- `prompt`: what the replaced region should become.
- `duration`: total target output duration.

Example:

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("small-music")

waveform, sr = torchaudio.load("loop.wav")

audio = model.generate(
    inpaint_audio=(sr, waveform),
    inpaint_mask_start_seconds=4.0,
    inpaint_mask_end_seconds=8.0,
    prompt="punchy drum fill with tight snare rolls",
    duration=16,
    steps=8,
    seed=1234,
)
```

The region from `4.0s` to `8.0s` is regenerated. Everything before `4.0s` and after `8.0s` is preserved.

## Multiple Inpaint Regions

The upstream model can regenerate multiple non-contiguous regions in one pass by passing lists for both time parameters:

```python
audio = model.generate(
    inpaint_audio=(sr, waveform),
    inpaint_mask_start_seconds=[2.0, 12.0],
    inpaint_mask_end_seconds=[4.0, 14.0],
    prompt="cleaner percussion variation",
    duration=16,
)
```

Both lists must have the same length. Each `(start, end)` pair defines one regenerated region.

## Continuation

Continuation is handled as inpainting beyond the end of the source clip.

For example, if the source clip is 8 seconds and the desired output is 16 seconds:

```python
audio = model.generate(
    inpaint_audio=(sr, waveform),
    inpaint_mask_start_seconds=8.0,
    inpaint_mask_end_seconds=16.0,
    prompt="continue as a dark synthwave loop",
    duration=16,
)
```

The original 0s to 8s region is preserved, and the model generates the 8s to 16s continuation.

## Recommended API Shape for Inpainting

The current API does not expose inpainting yet. Since inpainting requires input audio, the cleanest API shape is `multipart/form-data`.

Recommended local endpoint:

```http
POST /v1/audio/inpaint
```

Recommended async job endpoint:

```http
POST /jobs/inpaint
GET /jobs/{id}
```

Suggested form fields:

| Field | Type | Notes |
| --- | --- | --- |
| `audio` | file | Source WAV, FLAC, MP3, or other format supported by the audio loader. |
| `model` | string | Usually `small-music` or `small-sfx`. |
| `prompt` | string | Description for the regenerated region. |
| `duration` | number | Total target output duration. |
| `inpaint_start_seconds` | number or JSON list | Start time, or multiple start times. |
| `inpaint_end_seconds` | number or JSON list | End time, or multiple end times. |
| `steps` | integer | Sampling steps. Default `8`. |
| `seed` | integer | `-1` random, fixed integer for repeatable output. |
| `chunked_decode` | boolean or null | Optional decode override. |

Example multipart request:

```bash
curl -X POST http://localhost:8000/v1/audio/inpaint \
  -F model=small-music \
  -F prompt="punchy drum fill with tight snare rolls" \
  -F duration=16 \
  -F inpaint_start_seconds=4 \
  -F inpaint_end_seconds=8 \
  -F steps=8 \
  -F seed=1234 \
  -F audio=@loop.wav \
  --output inpainted-loop.wav
```

Example async request:

```bash
curl -X POST http://localhost:8000/jobs/inpaint \
  -F model=small-music \
  -F prompt="continue as a dusty house groove" \
  -F duration=16 \
  -F inpaint_start_seconds=8 \
  -F inpaint_end_seconds=16 \
  -F audio=@loop.wav
```

The async endpoint should follow the same storage pattern as `POST /jobs`: generate in the background, write the WAV to S3, R2, provider storage, or local output storage, then return the final `download_url` from `GET /jobs/{id}`.

## Implementation Notes for This API

To add inpainting support, extend the server with:

- A multipart parser using FastAPI `UploadFile` and `Form`.
- A helper that loads uploaded audio into `(sample_rate, waveform)`.
- A request model or form parser for scalar and list-based mask times.
- A synchronous local endpoint such as `POST /v1/audio/inpaint`.
- An async production endpoint such as `POST /jobs/inpaint`.
- Reuse of the existing model cache, WAV encoding, job store, and storage writer.

Validation should reject:

- Empty prompts.
- Unsupported models.
- Durations above the selected model limit.
- Mask start times greater than or equal to mask end times.
- Mismatched start/end list lengths.
- Negative mask times.
- Audio inputs that cannot be decoded.

