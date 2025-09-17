from typing import Optional
import io
import os
import torch
import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter()

# Config via env
MODEL_ID = os.getenv("SA_MODEL_ID", "stabilityai/stable-audio-open-1.0")
DEFAULT_SAMPLE_RATE = int(os.getenv("SA_SAMPLE_RATE", "44100"))
DEFAULT_STEPS = int(os.getenv("SA_STEPS", "100"))
DEFAULT_GUIDANCE = float(os.getenv("SA_GUIDANCE", "3.0"))
PREFERRED_DEVICE = os.getenv("SA_DEVICE", "auto")  # "auto" | "cuda" | "cpu"
DTYPE_ENV = os.getenv("SA_DTYPE", "float16")       # "float16" | "float32"

_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

_device = (
    "cuda" if (PREFERRED_DEVICE == "auto" and torch.cuda.is_available()) else
    ("cuda" if PREFERRED_DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
)
_dtype = torch.float16 if (_device == "cuda" and DTYPE_ENV.lower() == "float16") else torch.float32

_pipe = None
_pipe_sr = DEFAULT_SAMPLE_RATE

def _load_pipe():
    global _pipe, _pipe_sr
    if _pipe is not None:
        return _pipe
    try:
        # Use diffusers StableAudioOpenPipeline as the engine
        from diffusers import StableAudioOpenPipeline

        extra_kwargs = {}
        if _hf_token:
            # diffusers forwards this to huggingface_hub internally
            extra_kwargs["use_auth_token"] = _hf_token

        _pipe = StableAudioOpenPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=_dtype,
            use_safetensors=True,
            **extra_kwargs,
        )
        _pipe = _pipe.to(_device)
        # pull sample rate from pipeline/config if set; fallback to env default
        _pipe_sr = getattr(_pipe, "sample_rate", None) or getattr(getattr(_pipe, "config", None), "sample_rate", None) or DEFAULT_SAMPLE_RATE
        return _pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load Stable Audio Open pipeline: {e}")

def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    # audio: shape [samples] or [channels, samples], float32 [-1, 1]
    if audio.ndim == 1:
        audio = audio[None, ...]  # [1, samples]
    audio = np.transpose(audio)   # [samples, channels]
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf

@router.get("/stable-audio/ping")
def ping():
    return {"ok": True, "device": _device, "dtype": str(_dtype), "model": MODEL_ID, "sample_rate": _pipe_sr or DEFAULT_SAMPLE_RATE}

@router.get("/stable-audio/generate")
def generate(
    prompt: str = Query(..., description="Text prompt"),
    seconds: float = Query(8.0, ge=1.0, le=60.0, description="Duration in seconds"),
    seed: int = Query(0, description="Random seed"),
    steps: int = Query(DEFAULT_STEPS, ge=10, le=500, description="Diffusion steps"),
    guidance_scale: float = Query(DEFAULT_GUIDANCE, ge=0.0, le=30.0, description="Guidance scale"),
):
    try:
        pipe = _load_pipe()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        generator = torch.Generator(device=_device)
        if seed is not None:
            generator = generator.manual_seed(int(seed))

        # Run the pipeline. Returns AudioPipelineOutput with .audios: List[np.ndarray]
        out = pipe(
            prompt=prompt,
            audio_length_in_s=float(seconds),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        if not hasattr(out, "audios") or not out.audios:
            raise RuntimeError("Pipeline returned no audio.")
        audio = out.audios[0].astype(np.float32)
        sr = _pipe_sr or DEFAULT_SAMPLE_RATE

        buf = _to_wav_bytes(audio, sr)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")
