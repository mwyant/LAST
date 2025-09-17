# app/transcribe_functions.py
import subprocess
import glob
import os
import shutil
import time
import re
import datetime
from pathlib import Path
from typing import Callable, List, Tuple, Any, Optional

# ffmpeg will produce these chunk files
CHUNK_FILENAME_TEMPLATE = "chunk%03d.wav"
_TIME_PREFIX_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2}\.\d+|\d{2})")

def _shift_line_time_prefix(line: str, offset_seconds: float) -> str:
    m = _TIME_PREFIX_RE.match(line)
    if not m:
        return line
    hours = int(m.group(1)); mins = int(m.group(2)); secs = float(m.group(3))
    total = hours * 3600 + mins * 60 + secs + float(offset_seconds)
    dt = datetime.timedelta(seconds=total)
    total_seconds = int(dt.total_seconds())
    h = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = dt.total_seconds() - h * 3600 - mm * 60
    return f"{h:02d}:{mm:02d}:{ss:06.3f}" + line[m.end():]

def prepare_chunks_low_quality(src_path: str, out_dir: str, chunk_seconds: int = 300) -> List[str]:
    """
    Create low-quality (16k mono WAV) chunks from src_path using ffmpeg segmenting.
    Returns list of chunk file paths (in order).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, CHUNK_FILENAME_TEMPLATE)
    # -ar 16000 -ac 1 => 16k mono; codec default wav (pcm_s16le)
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        out_pattern
    ]
    subprocess.check_call(cmd)
    files = sorted(glob.glob(os.path.join(out_dir, "chunk*.wav")))
    return files

def _safe_transcribe_one(transcribe_fn: Callable[[Any, str], List[Any]], model_instance: Any, audio_path: str) -> Any:
    """
    Call the provided transcribe_fn in a defensive way.
    transcribe_fn is expected to have signature transcribe_fn(model_instance, audio_path) or transcribe_fn(audio_path)
    We try calling with model first, then without.
    """
    try:
        return transcribe_fn(model_instance, audio_path)
    except TypeError:
        try:
            return transcribe_fn(audio_path)
        except Exception:
            # propagate last exception
            return transcribe_fn(model_instance, audio_path)

def transcribe_and_aggregate(
    transcribe_fn: Callable[..., List[Any]],
    model_instance: Any,
    chunk_files: List[str],
    chunk_seconds: int = 300,
    progress_cb: Optional[Callable[[float, str], None]] = None
) -> Tuple[str, List[str]]:
    """
    transcribe_fn: a callable to run transcription (e.g., run_transcribe_model)
                  It can be either (model, path) or (path) — we try both styles.
    model_instance: the loaded ASR model instance (or None if transcribe_fn expects only path).
    chunk_files: ordered list of chunk wav file paths (16k mono).
    Returns (combined_plain_text, combined_lines_with_shifted_timestamps)
    progress_cb: optional callback(progress_percent_float, message_str)
    """
    combined_plain_parts: List[str] = []
    combined_lines: List[str] = []
    total = max(1, len(chunk_files))
    for i, chunk in enumerate(chunk_files):
        offset = i * chunk_seconds
        if progress_cb:
            progress_cb(55.0 + (i / total) * 30.0, f"transcribing chunk {i+1}/{total}")
        start = time.time()
        # call the provided transcribe function safely
        out = _safe_transcribe_one(transcribe_fn, model_instance, chunk)
        elapsed = time.time() - start
        # out may be a list or a single object — pick first if list
        transcript_obj = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
        # Convert transcript_obj to human readable text+lines.
        # We try to be compatible with the transcript_to_human_readable from main.py by
        # using a small portable renderer here.
        plain = None
        lines: List[str] = []
        try:
            # Try attribute lookup for common fields
            plain = getattr(transcript_obj, "text", None) or getattr(transcript_obj, "transcript", None) or str(transcript_obj)
        except Exception:
            plain = str(transcript_obj)
        # Attempt to extract timestamps (various shapes)
        ts_obj = getattr(transcript_obj, "timestamp", None) or getattr(transcript_obj, "timestamps", None) or getattr(transcript_obj, "word_timestamps", None) or getattr(transcript_obj, "segments", None)
        # Minimal conversion: if ts_obj is a list of dicts with start/text or list of (text,start,end)
        if isinstance(ts_obj, dict):
            # fallback to single block
            lines = [f"00:00:00 - SPKR1: {plain}"]
        elif isinstance(ts_obj, list):
            for itm in ts_obj:
                if isinstance(itm, dict):
                    start = itm.get("start") or itm.get("start_time") or 0.0
                    txt = itm.get("text") or itm.get("word") or str(itm)
                    t0 = datetime.timedelta(seconds=float(start or 0.0))
                    total_seconds = int(t0.total_seconds())
                    h = total_seconds // 3600
                    mm = (total_seconds % 3600) // 60
                    ss = t0.total_seconds() - h * 3600 - mm * 60
                    lines.append(f"{h:02d}:{mm:02d}:{ss:06.3f} - SPKR1: {txt}")
                elif isinstance(itm, (list, tuple)) and len(itm) >= 3:
                    txt, start = itm[0], itm[1]
                    t0 = datetime.timedelta(seconds=float(start or 0.0))
                    total_seconds = int(t0.total_seconds())
                    h = total_seconds // 3600
                    mm = (total_seconds % 3600) // 60
                    ss = t0.total_seconds() - h * 3600 - mm * 60
                    lines.append(f"{h:02d}:{mm:02d}:{ss:06.3f} - SPKR1: {txt}")
                else:
                    lines.append(f"00:00:00 - SPKR1: {plain}")
        else:
            lines = [f"00:00:00 - SPKR1: {plain}"]

        # shift timestamps on lines
        shifted = [_shift_line_time_prefix(ln, offset) for ln in lines]
        combined_plain_parts.append(plain)
        combined_lines.extend(shifted)

        if progress_cb:
            progress_cb(55.0 + ((i + 1) / total) * 30.0, f"finished chunk {i+1}/{total} ({elapsed:.1f}s)")

    full_plain = " ".join([p for p in combined_plain_parts if p])
    return full_plain, combined_lines
