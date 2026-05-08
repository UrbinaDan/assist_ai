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
