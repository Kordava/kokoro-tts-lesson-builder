from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import (
    SAMPLE_RATE,
    TRIM_ABSOLUTE_RMS,
    TRIM_FRAME_MS,
    TRIM_LEAD_GUARD_MS,
    TRIM_RELATIVE_RMS,
    TRIM_TAIL_GUARD_MS,
)


def mono_float32(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 2:
        value = value.mean(axis=1)
    return value.reshape(-1)


def trim_tts_padding(audio: np.ndarray) -> np.ndarray:
    """Trim long low-level TTS edge padding while preserving speech endings.

    The detector works on short RMS frames. Everything outside the detected
    audible region is treated as provider/technical edge padding; a small
    pre-roll and post-roll are retained to protect natural consonant attacks and
    releases. Contract pauses are inserted *after* this function and therefore
    remain independent of provider padding.
    """

    value = mono_float32(audio)
    if value.size == 0:
        raise ValueError("audio is empty")

    frame_size = max(1, int(round(SAMPLE_RATE * TRIM_FRAME_MS / 1000)))
    lead_guard = max(0, int(round(SAMPLE_RATE * TRIM_LEAD_GUARD_MS / 1000)))
    tail_guard = max(0, int(round(SAMPLE_RATE * TRIM_TAIL_GUARD_MS / 1000)))

    peak = float(np.max(np.abs(value)))
    if peak <= 0.0:
        return value.copy()

    frame_count = (len(value) + frame_size - 1) // frame_size
    padded = np.pad(value, (0, frame_count * frame_size - len(value)))
    frames = padded.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))

    # v0.4 used a very low threshold (~-70 dBFS in typical Kokoro output),
    # which classified Kokoro's low-level technical tail as active speech.
    # The v0.5 threshold is intentionally closer to audibility while still
    # leaving a guard after low-energy final consonants.
    threshold = max(TRIM_ABSOLUTE_RMS, peak * TRIM_RELATIVE_RMS)
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        # If thresholding finds nothing, preserve source rather than risk a cut.
        return value.copy()

    first_active_sample = int(active[0]) * frame_size
    last_active_end = min(len(value), (int(active[-1]) + 1) * frame_size)

    start = max(0, first_active_sample - lead_guard)
    end = min(len(value), last_active_end + tail_guard)

    if end <= start:
        return value.copy()
    return value[start:end].copy()


def silence(seconds: float) -> np.ndarray:
    if seconds <= 0:
        raise ValueError("silence duration must be positive")
    frames = int(round(seconds * SAMPLE_RATE))
    return np.zeros(frames, dtype=np.float32)


def read_mono_wav(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"Unexpected sample rate {rate}: {path}")
    return mono_float32(audio)


def atomic_write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.wav", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        sf.write(tmp, mono_float32(audio), SAMPLE_RATE, subtype="PCM_16")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
