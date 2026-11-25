from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LANG_CODE, VOICE


def _extract_audio(result):
    audio = getattr(result, "audio", None)
    if audio is None and isinstance(result, (tuple, list)) and len(result) >= 3:
        audio = result[2]
    return audio


@dataclass
class KokoroEngine:
    voice: str = VOICE

    def __post_init__(self) -> None:
        # Keep the heavyweight dependency out of module import so --help and
        # --validate-only work without Kokoro installed.
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=LANG_CODE)

    def synthesize(self, text: str, *, speed: float) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for result in self.pipeline(
            text,
            voice=self.voice,
            speed=speed,
            split_pattern=r"\n+",
        ):
            audio = _extract_audio(result)
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach()
            if hasattr(audio, "cpu"):
                audio = audio.cpu()
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        return np.concatenate(chunks)
