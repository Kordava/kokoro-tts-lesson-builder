from __future__ import annotations

import hashlib
import json

VOICE = "af_heart"
LANG_CODE = "a"  # American English
COACH_SPEED = 0.9
MODEL_SPEED = 0.95
PROFILE_SPEEDS = {
    "question": COACH_SPEED,
    "model": MODEL_SPEED,
    "story": MODEL_SPEED,
}
SAMPLE_RATE = 24_000

# Semantic pause types. Episode scripts name the purpose of a pause; the
# generator owns the actual duration. Changing these values changes assembly
# timing without invalidating cached speech units.
PAUSE_DURATIONS = {
    "ANSWER": 10.0,
    "POST_MODEL": 6.0,
    "SECTION": 6.0,
    "FINAL_RETELLING": 60.0,
}

# Bump whenever cached audio semantics can change even if text/voice/speed do not.
CACHE_SCHEMA_VERSION = 3
RENDERER_SCHEMA_VERSION = "episode-renderer-v3"

# Kokoro can leave a long low-level technical tail. Detect edge activity on
# short RMS frames, trim only when the quiet edge is long enough to be provider
# padding, and retain a small natural guard around audible speech.
TRIM_FRAME_MS = 10
TRIM_LEAD_GUARD_MS = 10
TRIM_TAIL_GUARD_MS = 80
TRIM_ABSOLUTE_RMS = 0.0015
TRIM_RELATIVE_RMS = 0.01


def render_config_payload() -> dict[str, object]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "renderer_schema_version": RENDERER_SCHEMA_VERSION,
        "lang_code": LANG_CODE,
        "sample_rate": SAMPLE_RATE,
        "trim_frame_ms": TRIM_FRAME_MS,
        "trim_lead_guard_ms": TRIM_LEAD_GUARD_MS,
        "trim_tail_guard_ms": TRIM_TAIL_GUARD_MS,
        "trim_absolute_rms": TRIM_ABSOLUTE_RMS,
        "trim_relative_rms": TRIM_RELATIVE_RMS,
        # Pronunciation overrides are not implemented yet. Keeping this field
        # makes a future pronunciation layer an explicit cache-invalidating change.
        "pronunciation_config": None,
    }


def render_config_fingerprint() -> str:
    encoded = json.dumps(
        render_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
