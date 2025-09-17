# app/denoise_utils.py
import os
import subprocess
import tempfile
import glob
import shutil
import soundfile as sf
import numpy as np
import noisereduce as nr
import pyloudnorm as pyln

SR = 48000
CHANNELS = 2
SEG_CODEC = "pcm_s24le"  # 24-bit WAV

def ensure_wav_48k(in_path, out_path):
    """Transcode any input to 48k stereo 24-bit WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-vn", "-ar", str(SR), "-ac", str(CHANNELS),
        "-c:a", SEG_CODEC,
        out_path
    ]
    subprocess.check_call(cmd)

def split_into_chunks(in_wav, out_dir, seg_secs=300):
    """Split wav into fixed-length segments (seg000.wav...)."""
    out_pattern = os.path.join(out_dir, "seg%03d.wav")
    cmd = [
        "ffmpeg", "-y", "-i", in_wav,
        "-vn", "-ar", str(SR), "-ac", str(CHANNELS),
        "-c:a", SEG_CODEC,
        "-f", "segment",
        "-segment_time", str(seg_secs),
        "-reset_timestamps", "1",
        out_pattern
    ]
    subprocess.check_call(cmd)
    return sorted(glob.glob(os.path.join(out_dir, "seg*.wav")))

def concat_wavs(files, out_file):
    """Concatenate WAV files with ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for p in files:
            f.write("file '{}'\n".format(os.path.abspath(p)))
        listfile = f.name
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", out_file]
    subprocess.check_call(cmd)
    os.remove(listfile)

def cpu_denoise_and_normalize(in_wav, out_wav, target_lufs=-16.0):
    """CPU fallback denoise (noisereduce) + loudness normalize (pyloudnorm)."""
    data, sr = sf.read(in_wav, always_2d=True)
    processed = np.zeros_like(data)
    clip_dur = min(0.5, data.shape[0] / sr)
    clip_samples = int(clip_dur * sr)
    noise_clip = data[:clip_samples, :] if clip_samples >= 1 else None

    for ch in range(data.shape[1]):
        y = data[:, ch]
        if noise_clip is not None:
            noise = noise_clip[:, ch]
            reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise, prop_decrease=1.0)
        else:
            reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0)
        processed[:, ch] = reduced

    meter = pyln.Meter(sr)
    mono = np.mean(processed, axis=1)
    loudness = meter.integrated_loudness(mono)
    gain_db = target_lufs - loudness
    factor = 10.0 ** (gain_db / 20.0)
    processed = processed * factor
    processed = np.clip(processed, -0.9999, 0.9999)
    sf.write(out_wav, processed, sr, subtype='PCM_24')

def enhance_and_normalize(input_wav, output_wav, target_lufs=-16.0):
    """
    Enhance voice: EQ, dynamic normalization, and final LUFS normalization via ffmpeg.
    - input_wav: path to denoised/cleaned audio (can be mono or stereo)
    - output_wav: final output
    - target_lufs: target integrated loudness (default -16 LUFS)
    """
    filt = (
        # remove rumble, limit extreme highs, add low-mid warmth, presence boost, gentle de-harsh
        "highpass=f=80, "
        "lowpass=f=14000, "
        "equalizer=f=150:width_type=o:width=1:g=2.0, "
        "equalizer=f=3500:width_type=o:width=1:g=2.5, "
        "equalizer=f=8000:width_type=o:width=1:g=-1.5, "
        # perceptual dynamic normalization (helps intelligibility)
        "dynaudnorm=g=11:o=0.7, "
        # final LUFS normalization
        f"loudnorm=I={target_lufs}:LRA=7:TP=-1.5"
    )

    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-vn",
        "-af", filt,
        "-ar", str(SR), "-ac", str(CHANNELS),
        "-c:a", "pcm_s24le",
        output_wav
    ]
    subprocess.check_call(cmd)

def _denoise_then_enhance(src_wav, dst_wav, target_lufs=-16.0):
    """Helper: run cpu_denoise_and_normalize to temp file, then enhance_and_normalize to dst."""
    tmpf = None
    try:
        fd, tmpf = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cpu_denoise_and_normalize(src_wav, tmpf, target_lufs=target_lufs)
        enhance_and_normalize(tmpf, dst_wav, target_lufs=target_lufs)
    finally:
        if tmpf and os.path.exists(tmpf):
            try:
                os.remove(tmpf)
            except Exception:
                pass

def denoise_chunk_auto(input_wav, output_wav, target_lufs=-16.0):
    """
    Try GPU-demucs path if available (and torch.cuda), otherwise fallback to CPU noisereduce.
    Improved Demucs handling: prefer a vocal-like stem (by filename), else pick most 'speech-like'
    stem via a simple RMS/frame-energy heuristic, then normalize + enhance.
    """
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    demucs_path = shutil.which("demucs")
    if demucs_path and cuda_ok:
        tmpout = tempfile.mkdtemp(prefix="demucs_out_")
        try:
            # Model name can be changed; htdemucs is a decent general model.
            cmd = ["demucs", "-n", "htdemucs", "-o", tmpout, input_wav]
            subprocess.check_call(cmd)

            # Demucs produces files under tmpout/<model>/<basename>/*.wav
            found = sorted(glob.glob(os.path.join(tmpout, "*", "*", "*.wav")))
            if not found:
                # fallback to CPU denoise+enhance
                _denoise_then_enhance(input_wav, output_wav, target_lufs=target_lufs)
                return

            # Prefer filenames likely to contain vocals
            preferred_keywords = ("voc", "vocal", "voice", "lead", "singer")
            vocal_candidates = [p for p in found if any(k in os.path.basename(p).lower() for k in preferred_keywords)]

            chosen = None
            if vocal_candidates:
                # Pick candidate with highest RMS (tie-breaker)
                def rms(path):
                    y, sr = sf.read(path, always_2d=True)
                    mono = np.mean(y, axis=1)
                    return float(np.sqrt(np.mean(mono**2) + 1e-12))
                chosen = max(vocal_candidates, key=rms)
            else:
                # Energy / speech-like heuristic: count frames above threshold
                def speech_score(path, frame_ms=30, hop_ms=15):
                    y, sr = sf.read(path, always_2d=True)
                    mono = np.mean(y, axis=1)
                    frame_len = int(sr * frame_ms / 1000)
                    hop = int(sr * hop_ms / 1000)
                    if frame_len < 1:
                        return 0
                    rms_frames = []
                    for i in range(0, max(1, len(mono)-frame_len+1), max(1, hop)):
                        frame = mono[i:i+frame_len]
                        rms_frames.append(np.sqrt(np.mean(frame**2) + 1e-12))
                    if not rms_frames:
                        return 0
                    rms_arr = np.array(rms_frames)
                    # threshold relative to this file's max RMS
                    thresh = max(0.02 * np.max(rms_arr), 1e-6)
                    return float(np.sum(rms_arr > thresh))
                scores = [(p, speech_score(p)) for p in found]
                scores.sort(key=lambda x: x[1], reverse=True)
                chosen = scores[0][0] if scores else None

            if not chosen:
                _denoise_then_enhance(input_wav, output_wav, target_lufs=target_lufs)
                return

            # chosen is expected to be the denoised vocal stem. Normalize + enhance and write.
            _denoise_then_enhance(chosen, output_wav, target_lufs=target_lufs)
            return

        except subprocess.CalledProcessError:
            # demucs failed for some reason -> fallback
            _denoise_then_enhance(input_wav, output_wav, target_lufs=target_lufs)
            return
        finally:
            shutil.rmtree(tmpout, ignore_errors=True)
    else:
        # fallback CPU denoise + enhance
        _denoise_then_enhance(input_wav, output_wav, target_lufs=target_lufs)
