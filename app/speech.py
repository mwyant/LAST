# app/speech.py
import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger("speech")

def _nemo_synthesize(text: str, model_name: Optional[str] = None) -> Tuple[bytes, int]:
    """
    Use NeMo TTS if available. Returns (wav_bytes, sr)
    """
    try:
        # lazy import so module loads even if nemo not installed
        import nemo.collections.tts as nemo_tts  # may vary by nemo version
        import torch
        # pick a pretrained model name if not provided (example placeholder)
        if model_name is None:
            model_name = "nvidia/tacotron2pyt-tts"  # replace with an actual NeMo TTS model name if available
        # from_pretrained APIs vary; try commonsense approach
        tts = nemo_tts.models.TTSModel.from_pretrained(model_name)
        # to device if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tts.to(device)
        except Exception:
            logger.info("Could not move TTS model to device; continuing.")
        wav, sr = tts.convert_text_to_waveform(text)  # API varies; adjust to your NeMo version
        # WAV may be a numpy array; convert to bytes via soundfile
        import soundfile as sf
        bio = io.BytesIO()
        sf.write(bio, wav, sr, format="WAV")
        return bio.getvalue(), sr
    except Exception as e:
        logger.exception("NeMo TTS failed: %s", e)
        raise

def _coqui_synthesize(text: str, model_name: Optional[str] = None) -> Tuple[bytes, int]:
    """
    Use Coqui TTS (fallback). Returns (wav_bytes, sr)
    """
    try:
        from TTS.api import TTS
        # Use a small GPU-capable model or CPU model; let user override via model_name
        model_name = model_name or "tts_models/en/ljspeech/tacotron2-DDC"
        tts = TTS(model_name)
        # produce file in memory
        out_path = "/tmp/tts_out.wav"
        tts.tts_to_file(text=text, file_path=out_path)
        with open(out_path, "rb") as fh:
            data = fh.read()
        # read sample rate via soundfile
        import soundfile as sf
        _, sr = sf.read(out_path, always_2d=False)
        return data, sr
    except Exception:
        logger.exception("Coqui TTS failed")
        raise

def synthesize_text(text: str, model: Optional[str] = None) -> Tuple[bytes, int]:
    """
    Try NeMo first, then Coqui fallback. Returns (wav_bytes, sample_rate).
    """
    # try NeMo
    try:
        return _nemo_synthesize(text, model_name=model)
    except Exception:
        logger.info("NeMo synth failed or not available; trying Coqui TTS")
    # fallback
    return _coqui_synthesize(text, model_name=model)
