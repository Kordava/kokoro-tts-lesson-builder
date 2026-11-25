from __future__ import annotations

from local_kokoro_tts.episode import cache_key


def test_cache_key_changes_when_renderer_fingerprint_changes() -> None:
    kwargs = {
        "text": "Hello",
        "role": "MODEL",
        "voice": "af_heart",
        "speed": 1.0,
    }
    first = cache_key(**kwargs, renderer_fingerprint="renderer-a")
    second = cache_key(**kwargs, renderer_fingerprint="renderer-b")
    assert first != second


def test_cache_key_changes_by_delivery_settings() -> None:
    base = cache_key(text="Hello", role="MODEL", voice="af_heart", speed=1.0)
    assert base != cache_key(text="Hello!", role="MODEL", voice="af_heart", speed=1.0)
    assert base != cache_key(text="Hello", role="COACH", voice="af_heart", speed=1.0)
    assert base != cache_key(text="Hello", role="MODEL", voice="af_bella", speed=1.0)
    assert base != cache_key(text="Hello", role="MODEL", voice="af_heart", speed=0.95)
