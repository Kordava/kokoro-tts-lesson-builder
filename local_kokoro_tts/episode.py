from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import __version__
from .audio import atomic_write_wav, read_mono_wav, silence, trim_tts_padding
from .config import (
    CACHE_SCHEMA_VERSION,
    COACH_SPEED,
    MODEL_SPEED,
    PAUSE_DURATIONS,
    SAMPLE_RATE,
    VOICE,
    render_config_fingerprint,
    render_config_payload,
)

VOICE_RE = re.compile(r"^\[VOICE:\s*(COACH|MODEL)\s*\]$")
PAUSE_RE = re.compile(r"^\[PAUSE:\s*([A-Z][A-Z0-9_]*)\s*\]$")
LEGACY_PAUSE_RE = re.compile(r"^\[PAUSE\s+([0-9]+(?:\.[0-9]+)?)s\]$")
SECTION_RE = re.compile(r"^\[SECTION:\s*([^\]]+)\]$")
TAG_RE = re.compile(r"^\[.*\]$")


@dataclass(frozen=True)
class Speech:
    role: str
    text: str
    section: str


@dataclass(frozen=True)
class Pause:
    pause_type: str
    seconds: float
    section: str


AudioItem = Speech | Pause


def parse_episode_script(path: Path) -> list[AudioItem]:
    """Parse the canonical episode format as a strict state machine."""

    for pause_type, seconds in PAUSE_DURATIONS.items():
        if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(
                f"Configured pause duration for {pause_type!r} must be a finite positive number"
            )

    items: list[AudioItem] = []
    current_section: str | None = None
    pending_role: str | None = None
    buffer: list[str] = []
    seen_sections: set[str] = set()
    section_speech_count = 0
    last_item_was_pause = False

    def fail(lineno: int, message: str) -> None:
        raise ValueError(f"Line {lineno}: {message}")

    def flush_speech(lineno: int) -> None:
        nonlocal buffer, pending_role, section_speech_count, last_item_was_pause
        if not buffer:
            return
        text = "\n".join(line.strip() for line in buffer if line.strip()).strip()
        buffer = []
        if not text:
            return
        if current_section is None:
            fail(lineno, f"spoken text outside [SECTION: ...]: {text[:80]!r}")
        if pending_role is None:
            fail(lineno, f"spoken text without explicit [VOICE: ...]: {text[:80]!r}")
        items.append(Speech(role=pending_role, text=text, section=current_section))
        pending_role = None
        section_speech_count += 1
        last_item_was_pause = False

    def ensure_no_pending_role(lineno: int, next_token: str) -> None:
        if pending_role is not None and not buffer:
            fail(lineno, f"[VOICE: {pending_role}] has no spoken block before {next_token}")

    def close_section(lineno: int) -> None:
        if current_section is not None and section_speech_count == 0:
            fail(lineno, f"section {current_section!r} is empty")

    lines = path.read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()

        if not line:
            flush_speech(lineno)
            continue
        if line.startswith("#"):
            # Comments do not consume a pending role. This allows a comment
            # between a VOICE tag and its spoken block without changing state.
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            flush_speech(lineno)
            ensure_no_pending_role(lineno, "[SECTION]")
            close_section(lineno)
            name = section_match.group(1).strip()
            if not name:
                fail(lineno, "section name must not be empty")
            if name in seen_sections:
                fail(lineno, f"duplicate section: {name!r}")
            seen_sections.add(name)
            current_section = name
            section_speech_count = 0
            last_item_was_pause = False
            continue

        voice_match = VOICE_RE.match(line)
        if voice_match:
            flush_speech(lineno)
            if current_section is None:
                fail(lineno, "[VOICE] outside [SECTION: ...]")
            ensure_no_pending_role(lineno, "another [VOICE]")
            pending_role = voice_match.group(1)
            continue

        pause_match = PAUSE_RE.match(line)
        if pause_match:
            flush_speech(lineno)
            if current_section is None:
                fail(lineno, "[PAUSE] outside [SECTION: ...]")
            ensure_no_pending_role(lineno, "[PAUSE]")
            if section_speech_count == 0:
                fail(lineno, "[PAUSE] cannot appear before the first spoken block in a section")
            if last_item_was_pause:
                fail(lineno, "consecutive [PAUSE] tags are not allowed")
            pause_type = pause_match.group(1)
            if pause_type not in PAUSE_DURATIONS:
                known = ", ".join(sorted(PAUSE_DURATIONS))
                fail(lineno, f"unknown pause type {pause_type!r}; expected one of: {known}")
            seconds = float(PAUSE_DURATIONS[pause_type])
            if seconds <= 0:
                fail(lineno, f"configured pause duration for {pause_type!r} must be positive")
            items.append(Pause(pause_type=pause_type, seconds=seconds, section=current_section))
            last_item_was_pause = True
            continue

        if LEGACY_PAUSE_RE.match(line):
            fail(
                lineno,
                "numeric pause tags are no longer supported; use a semantic tag such as "
                "[PAUSE: ANSWER], [PAUSE: POST_MODEL], [PAUSE: SECTION], or "
                "[PAUSE: FINAL_RETELLING]",
            )

        if TAG_RE.match(line):
            fail(lineno, f"unknown production tag: {line}")

        if current_section is None:
            fail(lineno, f"spoken text outside [SECTION: ...]: {line[:80]!r}")
        if pending_role is None:
            fail(lineno, f"spoken text without explicit [VOICE: ...]: {line[:80]!r}")
        buffer.append(raw)

    eof_line = len(lines) + 1
    flush_speech(eof_line)
    ensure_no_pending_role(eof_line, "end of file")
    close_section(eof_line)

    if not items:
        raise ValueError("Episode script produced no audio items")
    validate_episode_sequence(items)
    return items


def validate_episode_sequence(items: list[AudioItem]) -> None:
    for index, item in enumerate(items):
        if not isinstance(item, Pause):
            continue

        prev_item = items[index - 1] if index > 0 else None
        next_item = items[index + 1] if index + 1 < len(items) else None
        label = f"[PAUSE: {item.pause_type}]"

        if item.pause_type == "ANSWER":
            if not isinstance(prev_item, Speech) or prev_item.role != "COACH":
                raise ValueError(f"{label} must immediately follow a COACH speech unit")
            if not isinstance(next_item, Speech) or next_item.role != "MODEL":
                raise ValueError(f"{label} must immediately precede a MODEL answer")
            if prev_item.section != item.section or next_item.section != item.section:
                raise ValueError(f"{label} and its COACH/MODEL units must be in the same section")

        elif item.pause_type == "POST_MODEL":
            if not isinstance(prev_item, Speech) or prev_item.role != "MODEL":
                raise ValueError(f"{label} must immediately follow a MODEL speech unit")
            if not isinstance(next_item, Speech) or next_item.role != "COACH":
                raise ValueError(f"{label} must immediately precede the next COACH prompt")
            if prev_item.section != item.section or next_item.section != item.section:
                raise ValueError(f"{label} and its MODEL/COACH units must be in the same section")

        elif item.pause_type == "SECTION":
            if not isinstance(prev_item, Speech) or prev_item.section != item.section:
                raise ValueError(f"{label} must immediately follow speech in its section")
            if not isinstance(next_item, Speech) or next_item.section == item.section:
                raise ValueError(f"{label} must terminate a section and precede speech in the next section")

        elif item.pause_type == "FINAL_RETELLING":
            if not isinstance(prev_item, Speech) or prev_item.role != "COACH":
                raise ValueError(f"{label} must immediately follow a COACH retelling prompt")
            if not isinstance(next_item, Speech) or next_item.role != "COACH":
                raise ValueError(f"{label} must immediately precede the COACH model-introduction prompt")
            if prev_item.section != item.section or next_item.section != item.section:
                raise ValueError(f"{label} and its surrounding COACH units must be in the same section")

        else:
            raise ValueError(f"{label} has no sequence validation rule")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def cache_key(
    *,
    text: str,
    role: str,
    voice: str,
    speed: float,
    renderer_fingerprint: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "renderer_fingerprint": renderer_fingerprint or render_config_fingerprint(),
            "text": text,
            "role": role,
            "voice": voice,
            "speed": speed,
            "sample_rate": SAMPLE_RATE,
            "kokoro_version": _package_version("kokoro"),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_mp3(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed; install it with: sudo apt install ffmpeg")
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{mp3_path.name}.", suffix=".tmp.mp3", dir=mp3_path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-ac",
                "1",
                "-ar",
                "44100",
                "-b:a",
                "192k",
                str(tmp),
            ],
            check=True,
        )
        os.replace(tmp, mp3_path)
    finally:
        tmp.unlink(missing_ok=True)


def _default_engine_factory(voice: str):
    from .engine import KokoroEngine

    return KokoroEngine(voice=voice)


def build_episode(
    *,
    script: Path,
    output: Path,
    mp3: Path | None = None,
    manifest: Path | None = None,
    cache_dir: Path = Path(".tts-cache"),
    voice: str = VOICE,
    coach_speed: float = COACH_SPEED,
    model_speed: float = MODEL_SPEED,
    no_cache: bool = False,
    engine_factory: Callable[[str], object] | None = None,
) -> dict[str, object]:
    items = parse_episode_script(script)
    speeds = {"COACH": coach_speed, "MODEL": model_speed}
    renderer_fingerprint = render_config_fingerprint()

    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    speech_count = sum(isinstance(item, Speech) for item in items)
    pause_count = sum(isinstance(item, Pause) for item in items)
    section_order = list(dict.fromkeys(item.section for item in items))

    engine = None
    engine_factory = engine_factory or _default_engine_factory
    manifest_items: list[dict[str, object]] = []
    total_frames = 0
    speech_index = 0

    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp.wav", dir=output.parent)
    os.close(fd)
    tmp_output = Path(tmp_name)
    try:
        with sf.SoundFile(
            tmp_output,
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=1,
            subtype="PCM_16",
        ) as out:
            for item in items:
                if isinstance(item, Pause):
                    audio = silence(item.seconds)
                    out.write(audio)
                    total_frames += len(audio)
                    manifest_items.append(
                        {
                            "type": "pause",
                            "pause_type": item.pause_type,
                            "seconds": item.seconds,
                            "frames": len(audio),
                            "section": item.section,
                        }
                    )
                    continue

                speech_index += 1
                speed = speeds[item.role]
                key = cache_key(
                    text=item.text,
                    role=item.role,
                    voice=voice,
                    speed=speed,
                    renderer_fingerprint=renderer_fingerprint,
                )
                cache_file = cache_dir / f"{key}.wav"
                cache_hit = cache_file.exists() and not no_cache

                if cache_hit:
                    audio = read_mono_wav(cache_file)
                else:
                    if engine is None:
                        engine = engine_factory(voice)
                    print(f"[{speech_index}/{speech_count}] {item.role} {speed:.2f}: {item.text[:72]}")
                    raw_audio = engine.synthesize(item.text, speed=speed)
                    audio = trim_tts_padding(raw_audio)
                    atomic_write_wav(cache_file, audio)

                out.write(audio)
                total_frames += len(audio)
                manifest_items.append(
                    {
                        "type": "speech",
                        "role": item.role,
                        "section": item.section,
                        "speed": speed,
                        "text": item.text,
                        "cache_key": key,
                        "cache_hit": cache_hit,
                        "frames": len(audio),
                    }
                )

        os.replace(tmp_output, output)
    finally:
        tmp_output.unlink(missing_ok=True)

    duration = total_frames / SAMPLE_RATE
    build_manifest: dict[str, object] = {
        "manifest_schema_version": 4,
        "builder_version": __version__,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "renderer_fingerprint": renderer_fingerprint,
        "renderer_config": render_config_payload(),
        "script": str(script),
        "script_sha256": _sha256_file(script),
        "voice": voice,
        "coach_speed": coach_speed,
        "model_speed": model_speed,
        "pause_durations": dict(PAUSE_DURATIONS),
        "sample_rate": SAMPLE_RATE,
        "kokoro_version": _package_version("kokoro"),
        "numpy_version": _package_version("numpy"),
        "soundfile_version": _package_version("soundfile"),
        "duration_seconds": duration,
        "section_order": section_order,
        "speech_count": speech_count,
        "pause_count": pause_count,
        "items": manifest_items,
    }

    if mp3 is not None:
        _atomic_mp3(output, mp3)
    if manifest is not None:
        _atomic_write_json(manifest, build_manifest)
    return build_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-episode",
        description="Build an audio lesson from a tagged script.",
    )
    parser.add_argument("script", type=Path, help="Canonical episode .txt with SECTION/VOICE/semantic PAUSE tags")
    parser.add_argument("-o", "--output", type=Path, help="Full WAV output")
    parser.add_argument("--mp3", type=Path, help="Optional MP3 publication output (requires ffmpeg)")
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--coach-speed", type=float, default=COACH_SPEED)
    parser.add_argument("--model-speed", type=float, default=MODEL_SPEED)
    parser.add_argument("--cache-dir", type=Path, default=Path(".tts-cache"))
    parser.add_argument("--manifest", type=Path, help="Optional JSON build manifest")
    parser.add_argument("--no-cache", action="store_true", help="Regenerate speech units even if cached")
    parser.add_argument("--validate-only", action="store_true", help="Validate the episode format without generating audio")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, speed in (("coach", args.coach_speed), ("model", args.model_speed)):
        if not 0.5 <= speed <= 2.0:
            raise SystemExit(f"--{name}-speed must be between 0.5 and 2.0")

    items = parse_episode_script(args.script)
    speech_count = sum(isinstance(item, Speech) for item in items)
    pause_count = sum(isinstance(item, Pause) for item in items)
    section_count = len(dict.fromkeys(item.section for item in items))
    print(f"Validated: {section_count} sections, {speech_count} speech units, {pause_count} pauses")

    if args.validate_only:
        return
    if args.output is None:
        raise SystemExit("-o/--output is required unless --validate-only is used")

    print(f"Script: {args.script}")
    print(f"Voice: {args.voice}; COACH={args.coach_speed}; MODEL={args.model_speed}")
    result = build_episode(
        script=args.script,
        output=args.output,
        mp3=args.mp3,
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        voice=args.voice,
        coach_speed=args.coach_speed,
        model_speed=args.model_speed,
        no_cache=args.no_cache,
    )
    print(f"Saved WAV: {args.output} ({float(result['duration_seconds']) / 60:.1f} min)")
    if args.mp3:
        print(f"Saved MP3: {args.mp3}")
    if args.manifest:
        print(f"Saved manifest: {args.manifest}")


if __name__ == "__main__":
    main()
