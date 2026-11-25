from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-kokoro-tts")
except PackageNotFoundError:
    __version__ = "0.6.0"
