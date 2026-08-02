"""Offline tests for interactive demo question selection."""

from pytest import MonkeyPatch

from scripts.run_demo import SAMPLE_QUESTION, get_question


def test_get_question_uses_custom_input(monkeypatch: MonkeyPatch) -> None:
    """Return the user's question when they enter one."""
    monkeypatch.setattr("builtins.input", lambda _: "What are the earnings risks?")

    assert get_question() == "What are the earnings risks?"


def test_get_question_uses_sample_for_empty_input(monkeypatch: MonkeyPatch) -> None:
    """Return the displayed sample question when the user presses Enter."""
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert get_question() == SAMPLE_QUESTION
