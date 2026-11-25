from __future__ import annotations

import argparse
from pathlib import Path

from .audio import atomic_write_wav, trim_tts_padding
from .config import PROFILE_SPEEDS, VOICE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts",
        description="Synthesize English speech from text with Kokoro.",
    )
    parser.add_argument(
        "profile",
        choices=["question", "model", "story", "custom"],
        help="Delivery profile. question=COACH speed; model/story=MODEL speed.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to synthesize.")
    source.add_argument("--file", type=Path, help="UTF-8 text file to synthesize.")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument(
        "--speed",
        type=float,
        help="Required only for custom profile; overrides a named profile if supplied.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8").strip()
    else:
        text = (args.text or "").strip()

    if not text:
        raise SystemExit("Text is empty")

    if args.speed is not None:
        speed = args.speed
    elif args.profile == "custom":
        raise SystemExit("custom profile requires --speed")
    else:
        speed = PROFILE_SPEEDS[args.profile]

    if not 0.5 <= speed <= 2.0:
        raise SystemExit("--speed must be between 0.5 and 2.0")

    # Lazy import: `tts --help` works without the Kokoro dependency installed.
    from .engine import KokoroEngine

    raw_audio = KokoroEngine(voice=args.voice).synthesize(text, speed=speed)
    audio = trim_tts_padding(raw_audio)
    atomic_write_wav(args.output, audio)
    print(f"Saved: {args.output}  voice={args.voice} speed={speed}")


if __name__ == "__main__":
    main()
