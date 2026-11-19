from app.session_logger import ensure_logs_dir, log_session_event, read_session_log


def test_ensure_logs_dir_creates_directory(tmp_path, monkeypatch):
    log_dir = tmp_path / "sessions"
    monkeypatch.setenv("SESSION_LOG_DIR", str(log_dir))

    out = ensure_logs_dir()

    assert out == log_dir
    assert log_dir.exists()


def test_log_session_event_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))

    event = log_session_event(
        "abc",
        "coach_final",
        {
            "mode": "coach",
            "speaker": "Interviewer",
            "response_type": "coach_final",
            "route": "behavioral_story",
        },
    )

    assert event["session_id"] == "abc"
    assert event["event_type"] == "coach_final"
    assert (tmp_path / "abc.jsonl").exists()


def test_read_session_log_returns_events(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))
    log_session_event("abc", "notes_final", {"mode": "notes"})

    events = read_session_log("abc")

    assert len(events) == 1
    assert events[0]["event_type"] == "notes_final"
    assert "logged_at" in events[0]


def test_read_session_log_missing_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))

    assert read_session_log("missing") == []


def test_metadata_only_logging_is_default_and_does_not_persist_transcripts(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("SESSION_LOG_INCLUDE_CONTENT", raising=False)
    secret = "Sensitive interview answer with private personal details"
    event = log_session_event(
        "privacy",
        "coach_final",
        {
            "mode": "coach",
            "route": "behavioral_story",
            "raw_transcript": secret,
            "clean_transcript": secret,
            "interpreted_question": secret,
            "interpretation": {"question": secret},
            "retrieval_source": [secret],
            "filename": "confidential-document.pdf",
            "usage": {"total_tokens": 42},
        },
    )
    assert event["route"] == "behavioral_story"
    assert event["usage"] == {"total_tokens": 42}
    assert event["raw_transcript"] is None
    assert event["clean_transcript"] is None
    assert event["interpreted_question"] is None
    assert event["interpretation"] is None
    assert event["retrieval_source"] is None
    stored = (tmp_path / "privacy.jsonl").read_text(encoding="utf-8")
    assert secret not in stored
    assert "confidential-document.pdf" not in stored


def test_content_logging_requires_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SESSION_LOG_INCLUDE_CONTENT", "true")
    text = "Deliberately opted-in transcript"
    log_session_event("opt_in", "coach_final", {"raw_transcript": text})
    assert read_session_log("opt_in")[0]["raw_transcript"] == text


def test_unrecognized_opt_in_values_do_not_enable_content_logging(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SESSION_LOG_INCLUDE_CONTENT", "maybe")
    text = "Should not be persisted"
    log_session_event("unknown_opt", "coach_final", {"raw_transcript": text})
    assert text not in (tmp_path / "unknown_opt.jsonl").read_text(encoding="utf-8")
