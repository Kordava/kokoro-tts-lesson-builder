from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audio import read_mono_wav
from .config import SAMPLE_RATE, TRIM_ABSOLUTE_RMS, TRIM_FRAME_MS, TRIM_RELATIVE_RMS


def _active_bounds(audio: np.ndarray) -> tuple[int, int] | None:
    if audio.size == 0:
        return None
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return None

    frame_size = max(1, int(round(SAMPLE_RATE * TRIM_FRAME_MS / 1000)))
    frame_count = (len(audio) + frame_size - 1) // frame_size
    padded = np.pad(audio, (0, frame_count * frame_size - len(audio)))
    frames = padded.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    threshold = max(TRIM_ABSOLUTE_RMS, peak * TRIM_RELATIVE_RMS)
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return None
    start = int(active[0]) * frame_size
    end = min(len(audio), (int(active[-1]) + 1) * frame_size)
    return start, end


def analyze_pauses(wav: Path, manifest: Path) -> list[dict[str, float | str]]:
    audio = read_mono_wav(wav)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("manifest has no items list")

    spans: list[tuple[int, int, dict[str, object]]] = []
    cursor = 0
    for item in items:
        if not isinstance(item, dict) or "frames" not in item:
            raise ValueError("manifest item has no frame count")
        frames = int(item["frames"])
        spans.append((cursor, cursor + frames, item))
        cursor += frames

    if cursor != len(audio):
        raise ValueError(f"manifest/audio frame mismatch: manifest={cursor}, wav={len(audio)}")

    results: list[dict[str, float | str]] = []
    for index, (pause_start, pause_end, item) in enumerate(spans):
        if item.get("type") != "pause" or index == 0 or index + 1 >= len(spans):
            continue
        prev_start, prev_end, prev = spans[index - 1]
        next_start, next_end, nxt = spans[index + 1]
        if prev.get("type") != "speech" or nxt.get("type") != "speech":
            continue

        prev_bounds = _active_bounds(audio[prev_start:prev_end])
        next_bounds = _active_bounds(audio[next_start:next_end])
        if prev_bounds is None or next_bounds is None:
            continue

        prev_last_active = prev_start + prev_bounds[1]
        next_first_active = next_start + next_bounds[0]
        effective = (next_first_active - prev_last_active) / SAMPLE_RATE
        contract = float(item["seconds"])
        results.append(
            {
                "section": str(item.get("section", "")),
                "pause_type": str(item.get("pause_type", "")),
                "contract_seconds": contract,
                "effective_seconds": effective,
                "delta_seconds": effective - contract,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-qa-pauses",
        description="Measure effective audible-speech-to-audible-speech pauses from a WAV + build manifest.",
    )
    parser.add_argument("wav", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = analyze_pauses(args.wav, args.manifest)
    if not rows:
        raise SystemExit("No measurable speech-pause-speech boundaries found")

    deltas = [float(row["delta_seconds"]) for row in rows]
    effective = [float(row["effective_seconds"]) for row in rows]
    violations = [row for row in rows if abs(float(row["delta_seconds"])) > args.tolerance]

    print(
        f"Measured {len(rows)} pauses: "
        f"effective min={min(effective):.3f}s max={max(effective):.3f}s; "
        f"delta avg={sum(deltas)/len(deltas):+.3f}s "
        f"min={min(deltas):+.3f}s max={max(deltas):+.3f}s"
    )
    if violations:
        print(f"FAIL: {len(violations)} pause(s) exceed ±{args.tolerance:.3f}s")
        for row in violations[:20]:
            print(
                f"  {row['section']} [{row['pause_type']}]: contract={row['contract_seconds']:.3f}s "
                f"effective={row['effective_seconds']:.3f}s delta={row['delta_seconds']:+.3f}s"
            )
        raise SystemExit(2)
    print(f"PASS: all measurable pauses are within ±{args.tolerance:.3f}s")


if __name__ == "__main__":
    main()
