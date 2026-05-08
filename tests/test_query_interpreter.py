"""Heuristic query interpreter (Phase 6)."""

from app.query_interpreter import interpret_transcript, InterpreterResult


def test_so_whats_one_plus_one():
    r = interpret_transcript('so what\'s one plus one', speaker="Interviewer", mode="coach")
    assert r["is_question"] is True
    assert r["should_coach"] is True
    assert r["interpreted_question"] == "What's one plus one?"
    assert r["reason"] == "question_detected"


def test_python_lowercase_question():
    r = interpret_transcript(
        "how would you make a python script lowercase",
        speaker="Interviewer",
        mode="coach",
    )
    assert r["is_question"] is True
    assert "Python" in (r["interpreted_question"] or "")
    assert (r["interpreted_question"] or "").endswith("?")


def test_tell_me_about_yourself():
    r = interpret_transcript("tell me about yourself", speaker="Interviewer", mode="coach")
    assert r["is_question"] is True
    assert r["should_coach"] is True
    assert r["interpreted_question"]
    assert "tell me about yourself" in (r["interpreted_question"] or "").lower()


def test_so_junk():
    r = interpret_transcript("so", speaker="Interviewer", mode="coach")
    assert r["is_question"] is False
    assert r["should_coach"] is False
    assert r["reason"] == "junk_fragment"
    assert r["confidence"] < 0.2


def test_me_turn_no_coach_flag():
    r = interpret_transcript(
        "I would solve it by adding the numbers.",
        speaker="Me",
        mode="coach",
    )
    assert r["should_coach"] is False
    assert r["reason"] == "coach_candidate_turn"
    assert r["interpreted_question"] is None


def test_notes_mode_no_coach():
    text = "Today we discussed project deadlines"
    r = interpret_transcript(text, speaker="Participant 3", mode="notes")
    assert r["should_coach"] is False
    assert r["reason"] == "notes_mode"
    assert r["cleaned_text"].lower() == text.lower()
    assert r["interpreted_question"] is None


def test_dataclass_export():
    row = InterpreterResult(
        interpreted_question="Hi?",
        cleaned_text="hi",
        is_question=True,
        confidence=0.9,
        should_coach=True,
        missing_context=False,
        reason="question_detected",
    )
    d = row.as_dict()
    assert d["interpreted_question"] == "Hi?"
    assert set(d.keys()) == {
        "interpreted_question",
        "cleaned_text",
        "is_question",
        "confidence",
        "should_coach",
        "missing_context",
        "reason",
    }
