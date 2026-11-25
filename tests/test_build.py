from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from local_kokoro_tts.audio import read_mono_wav
from local_kokoro_tts.config import SAMPLE_RATE, TRIM_TAIL_GUARD_MS
from local_kokoro_tts.episode import build_episode


class FakeEngine:
    def __init__(self, voice: str) -> None:
        self.voice = voice

    def synthesize(self, text: str, *, speed: float) -> np.ndarray:
        # Deterministic audible speech with no provider padding.
        return np.full(int(0.1 * SAMPLE_RATE), 0.05, dtype=np.float32)


class PaddedFakeEngine(FakeEngine):
    def synthesize(self, text: str, *, speed: float) -> np.ndarray:
        speech = np.full(int(0.20 * SAMPLE_RATE), 0.05, dtype=np.float32)
        tail = np.full(int(0.90 * SAMPLE_RATE), 0.0008, dtype=np.float32)
        return np.concatenate([speech, tail])


def test_build_manifest_and_pause_length(tmp_path: Path) -> None:
    script = tmp_path / "episode.txt"
    script.write_text(
        "[SECTION: ASK]\n\n"
        "[VOICE: COACH]\nQuestion?\n\n"
        "[PAUSE: ANSWER]\n\n"
        "[VOICE: MODEL]\nAnswer.\n\n"
        "[PAUSE: POST_MODEL]\n\n"
        "[VOICE: COACH]\nNext?\n",
        encoding="utf-8",
    )
    output = tmp_path / "episode.wav"
    manifest_path = tmp_path / "manifest.json"
    result = build_episode(
        script=script,
        output=output,
        manifest=manifest_path,
        cache_dir=tmp_path / "cache",
        engine_factory=FakeEngine,
    )

    assert output.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result
    assert result["manifest_schema_version"] == 4
    assert result["cache_schema_version"] >= 3
    assert result["renderer_fingerprint"]
    assert result["renderer_config"]["trim_tail_guard_ms"] == TRIM_TAIL_GUARD_MS
    assert result["script_sha256"]
    pauses = [item for item in result["items"] if item["type"] == "pause"]
    assert pauses[0]["pause_type"] == "ANSWER"
    assert pauses[1]["pause_type"] == "POST_MODEL"
    assert pauses[0]["frames"] == 10 * SAMPLE_RATE
    assert pauses[1]["frames"] == 6 * SAMPLE_RATE
    assert result["pause_durations"] == {
        "ANSWER": 10.0,
        "POST_MODEL": 6.0,
        "SECTION": 6.0,
        "FINAL_RETELLING": 60.0,
    }

    audio, rate = sf.read(output, dtype="float32")
    assert rate == SAMPLE_RATE
    expected_frames = 3 * int(0.1 * SAMPLE_RATE) + (10 + 6) * SAMPLE_RATE
    assert len(audio) == expected_frames


def test_builder_trims_provider_tail_before_contract_pause(tmp_path: Path) -> None:
    script = tmp_path / "episode.txt"
    script.write_text(
        "[SECTION: ASK]\n"
        "[VOICE: COACH]\nQuestion?\n"
        "[PAUSE: ANSWER]\n"
        "[VOICE: MODEL]\nAnswer.\n",
        encoding="utf-8",
    )
    output = tmp_path / "episode.wav"
    build_episode(
        script=script,
        output=output,
        cache_dir=tmp_path / "cache",
        engine_factory=PaddedFakeEngine,
    )

    audio = read_mono_wav(output)
    audible = np.flatnonzero(np.abs(audio) >= 0.01)
    # First speech is 200 ms; second speech starts after its retained technical
    # guard + exact 10 s contract pause. The silence measured from the last
    # clearly audible sample to the next clearly audible sample must stay
    # within the production tolerance target (10.0 ± 0.1 s).
    first_block_end = int(0.20 * SAMPLE_RATE) - 1
    second_start_candidates = audible[audible > first_block_end]
    second_start = int(second_start_candidates[0])
    measured_silence = (second_start - first_block_end - 1) / SAMPLE_RATE
    assert 10.0 <= measured_silence <= 10.1


def test_failed_build_does_not_replace_existing_output(tmp_path: Path) -> None:
    script = tmp_path / "episode.txt"
    script.write_text("[SECTION: A]\n[VOICE: MODEL]\nHello.\n", encoding="utf-8")
    output = tmp_path / "episode.wav"
    original = np.full(100, 0.02, dtype=np.float32)
    sf.write(output, original, SAMPLE_RATE, subtype="PCM_16")

    class BrokenEngine(FakeEngine):
        def synthesize(self, text: str, *, speed: float) -> np.ndarray:
            raise RuntimeError("boom")

    try:
        build_episode(
            script=script,
            output=output,
            cache_dir=tmp_path / "cache",
            no_cache=True,
            engine_factory=BrokenEngine,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("build should have failed")

    preserved = read_mono_wav(output)
    assert len(preserved) == len(original)


def test_pause_qa_reports_effective_boundary_within_tolerance(tmp_path: Path) -> None:
    from local_kokoro_tts.qa import analyze_pauses

    script = tmp_path / "episode.txt"
    script.write_text(
        "[SECTION: ASK]\n"
        "[VOICE: COACH]\nQuestion?\n"
        "[PAUSE: ANSWER]\n"
        "[VOICE: MODEL]\nAnswer.\n",
        encoding="utf-8",
    )
    output = tmp_path / "episode.wav"
    manifest_path = tmp_path / "manifest.json"
    build_episode(
        script=script,
        output=output,
        manifest=manifest_path,
        cache_dir=tmp_path / "cache",
        engine_factory=PaddedFakeEngine,
    )

    rows = analyze_pauses(output, manifest_path)
    assert len(rows) == 1
    assert rows[0]["pause_type"] == "ANSWER"
    assert rows[0]["contract_seconds"] == 10.0
    assert abs(float(rows[0]["delta_seconds"])) <= 0.1
