from __future__ import annotations

import numpy as np

from local_kokoro_tts.audio import silence, trim_tts_padding
from local_kokoro_tts.config import (
    SAMPLE_RATE,
    TRIM_TAIL_GUARD_MS,
)


def test_exact_six_second_silence_samples() -> None:
    audio = silence(6.0)
    assert len(audio) == 6 * SAMPLE_RATE
    assert np.count_nonzero(audio) == 0


def test_trim_removes_systematic_kokoro_low_level_tail() -> None:
    # Reproduces the production failure seen in Episode 01: audible speech is
    # followed by ~0.9 s of low-level Kokoro technical tail rather than zeros.
    speech = np.full(int(0.50 * SAMPLE_RATE), 0.08, dtype=np.float32)
    technical_tail = np.full(int(0.90 * SAMPLE_RATE), 0.0008, dtype=np.float32)
    trimmed = trim_tts_padding(np.concatenate([speech, technical_tail]))

    retained_tail = len(trimmed) - len(speech)
    expected_guard = int(round(TRIM_TAIL_GUARD_MS / 1000 * SAMPLE_RATE))
    frame_tolerance = int(round(0.02 * SAMPLE_RATE))

    assert retained_tail >= 0
    assert retained_tail <= expected_guard + frame_tolerance
    assert len(trimmed) < len(speech) + int(0.20 * SAMPLE_RATE)


def test_trim_preserves_low_energy_final_consonant_before_padding() -> None:
    speech = np.full(int(0.40 * SAMPLE_RATE), 0.08, dtype=np.float32)
    final_consonant = np.full(int(0.12 * SAMPLE_RATE), 0.006, dtype=np.float32)
    technical_tail = np.full(int(0.90 * SAMPLE_RATE), 0.0008, dtype=np.float32)
    trimmed = trim_tts_padding(np.concatenate([speech, final_consonant, technical_tail]))

    assert len(trimmed) >= len(speech) + len(final_consonant)
    assert len(trimmed) <= len(speech) + len(final_consonant) + int(0.10 * SAMPLE_RATE)


def test_edge_padding_is_bounded_by_guards() -> None:
    lead = np.full(int(0.20 * SAMPLE_RATE), 0.0008, dtype=np.float32)
    speech = np.full(int(0.40 * SAMPLE_RATE), 0.08, dtype=np.float32)
    tail = np.full(int(0.15 * SAMPLE_RATE), 0.0008, dtype=np.float32)
    trimmed = trim_tts_padding(np.concatenate([lead, speech, tail]))

    # Technical edge padding is bounded even when it is shorter than the old
    # v0.4 250 ms heuristic. This is required for effective pauses to stay
    # within ±0.1 s of the contract.
    assert len(trimmed) <= len(speech) + int(0.11 * SAMPLE_RATE)
