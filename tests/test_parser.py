from __future__ import annotations

from pathlib import Path

import pytest

from local_kokoro_tts.episode import Pause, Speech, parse_episode_script


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "episode.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_train_station_example() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "train-station.txt"
    assert parse_episode_script(path) == [
        Speech("COACH", "Which platform does the train leave from?", "AT_THE_STATION"),
        Pause("ANSWER", 10.0, "AT_THE_STATION"),
        Speech("MODEL", "It leaves from platform four.", "AT_THE_STATION"),
    ]


@pytest.mark.parametrize(
    ("script", "message"),
    [
        (
            """
            [SECTION: A]
            [VOICE: MODEL]
            Hello.
            [SECTION: A]
            [VOICE: MODEL]
            Again.
            """,
            "duplicate section",
        ),
        (
            """
            [SECTION: A]
            [SECTION: B]
            [VOICE: MODEL]
            Hello.
            """,
            "is empty",
        ),
        (
            """
            [SECTION: A]
            [VOICE: COACH]
            """,
            "has no spoken block",
        ),
        (
            """
            [SECTION: A]
            [VOICE: COACH]
            [VOICE: MODEL]
            Hello.
            """,
            "has no spoken block",
        ),
        (
            """
            [SECTION: A]
            [VOICE: COACH]
            [PAUSE: ANSWER]
            """,
            "has no spoken block",
        ),
        (
            """
            [PAUSE: SECTION]
            [SECTION: A]
            [VOICE: MODEL]
            Hello.
            """,
            "outside [SECTION",
        ),
        (
            """
            [SECTION: A]
            [PAUSE: SECTION]
            [VOICE: MODEL]
            Hello.
            """,
            "before the first spoken block",
        ),
        (
            """
            [SECTION: A]
            [VOICE: MODEL]
            Hello.
            [PAUSE: SECTION]
            [PAUSE: SECTION]
            """,
            "consecutive [PAUSE]",
        ),
    ],
)
def test_invalid_states_are_rejected(tmp_path: Path, script: str, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[")):
        parse_episode_script(write(tmp_path, script))


def test_training_pause_contract(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [SECTION: ASK]
        [VOICE: COACH]
        Question?
        [PAUSE: ANSWER]
        [VOICE: MODEL]
        Answer.
        [PAUSE: POST_MODEL]
        [VOICE: COACH]
        Next question?
        """,
    )
    items = parse_episode_script(path)
    assert [type(item) for item in items] == [Speech, Pause, Speech, Pause, Speech]
    pauses = [item for item in items if isinstance(item, Pause)]
    assert [(item.pause_type, item.seconds) for item in pauses] == [
        ("ANSWER", 10.0),
        ("POST_MODEL", 6.0),
    ]


def test_legacy_numeric_pause_is_rejected_with_migration_hint(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [SECTION: ASK]
        [VOICE: COACH]
        Question?
        [PAUSE 6s]
        [VOICE: MODEL]
        Answer.
        """,
    )
    with pytest.raises(ValueError, match="numeric pause tags are no longer supported"):
        parse_episode_script(path)


def test_section_pause_must_really_cross_sections(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [SECTION: A]
        [VOICE: MODEL]
        One.
        [PAUSE: SECTION]
        [VOICE: COACH]
        Still A.
        """,
    )
    with pytest.raises(ValueError, match="must terminate a section"):
        parse_episode_script(path)


def test_final_retelling_pause_contract(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [SECTION: FINAL_RETELLING]
        [VOICE: COACH]
        Tell the whole story.
        [PAUSE: FINAL_RETELLING]
        [VOICE: COACH]
        Now listen to the model.
        [VOICE: MODEL]
        Model version.
        """,
    )
    items = parse_episode_script(path)
    pause = next(item for item in items if isinstance(item, Pause))
    assert pause.pause_type == "FINAL_RETELLING"
    assert pause.seconds == 60.0
