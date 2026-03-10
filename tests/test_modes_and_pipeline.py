"""Tests for mode switching, speculative vs final, notes, and speaker-change flush."""
from starlette.testclient import TestClient
from app.server import app
import app.agent as agent
from app.pipeline import append_delta, maybe_emit
from app.agent import AgentState, EndOfThought


def test_mode_switch_via_ws(monkeypatch):
    def fake_classify(text: str, **kwargs):
        return {"intent": "behavioral", "entities": {}, "confidence": 0.9}
    def fake_retrieve(query: str, k: int = 4, **kwargs):
        return [{"id": "f", "text": "x", "score": 0.9, "meta": {}}]
    monkeypatch.setattr(agent, "classify_question", fake_classify, raising=True)
    monkeypatch.setattr(agent, "retrieve_context", fake_retrieve, raising=True)

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        # Set notes mode
        ws.send_json({
            "session_id": "mode_test",
            "session_mode": "notes",
            "text_delta": "We decided to ship next week.",
            "final": True,
        })
        msg = ws.receive_json()
        assert msg.get("emit") is True
        data = msg.get("data", {})
        assert data.get("response_type") == "notes_final"
        assert "notes_final" in data
        notes = data["notes_final"].get("notes", {})
        assert "bullets" in notes
        assert "decisions" in notes or "bullets" in notes

        # Switch back to coach and send final
        ws.send_json({
            "session_id": "mode_test",
            "session_mode": "coach",
            "text_delta": "Tell me about a conflict you resolved.",
            "final": True,
        })
        msg2 = ws.receive_json()
        assert msg2.get("emit") is True
        assert msg2.get("data", {}).get("response_type") == "coach_final"


def test_speaker_change_flush(monkeypatch):
    def fake_classify(text: str, **kwargs):
        return {"intent": "behavioral", "entities": {}, "confidence": 0.9}
    def fake_retrieve(query: str, k: int = 4, **kwargs):
        return [{"id": "f", "text": "y", "score": 0.9, "meta": {}}]
    monkeypatch.setattr(agent, "classify_question", fake_classify, raising=True)
    monkeypatch.setattr(agent, "retrieve_context", fake_retrieve, raising=True)

    state = AgentState(session_id="s1")
    state.buffer_speaker = "Me"
    state.buffer_text = "I led the migration."
    state.last_token_ts = 0
    detector = EndOfThought(pause_ms=0, min_words=1, max_words=100)

    # Append delta from different speaker -> should trigger flush (pending_speaker_flush)
    append_delta(state, "And we shipped on time.", speaker="Interviewer")
    res = maybe_emit(state, final=False, detector=detector)
    assert res.emit is True
    assert res.kind == "final"
    assert res.reason == "speaker_change"
    assert state.buffer_text.strip() == "And we shipped on time."


def test_duplicate_final_dedupe(monkeypatch):
    state = AgentState(session_id="s2")
    state.buffer_text = "Same exact text."
    state.last_final_hash = __import__("hashlib").md5("Same exact text.".encode()).hexdigest()
    detector = EndOfThought()
    res = maybe_emit(state, final=True, detector=detector)
    assert res.emit is False
    assert res.reason == "duplicate_final"


def test_notes_extraction_heuristic():
    state = AgentState(session_id="s3")
    state.intent_history.append("behavioral")
    agent._update_notes(state, speaker="Alice", text="Action item: Bob will send the report by Friday.")
    assert any("action" in b.lower() or "bob" in b.lower() for b in state.notes.get("action_items", []))
    agent._update_notes(state, speaker="Bob", text="We decided to use API v2.")
    assert len(state.notes.get("decisions", [])) >= 1
    agent._update_notes(state, speaker="Alice", text="What about the timeline?")
    assert len(state.notes.get("open_questions", [])) >= 1
