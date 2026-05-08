from starlette.testclient import TestClient
from app.server import app
import app.agent as agent
from app.session_logger import read_session_log

def test_websocket_message_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))
    # --- stub OpenAI-dependent pieces so test is fast/offline ---
    def fake_route_question(text: str, **kwargs):
        return {"route": "behavioral_story", "intent": "behavioral", "confidence": 0.9, "rationale": ""}

    def fake_retrieve(query: str, k: int = 4, **kwargs):
        return [{"id": "fake::0", "text": "demo STAR note", "score": 0.92, "meta": {"source": "fake"}}]

    def fake_draft_background(route, text, ctx, prefs, *, state=None):
        return {
            "options": ["Answer angle A.", "Answer angle B."],
            "follow_up": "Can you quantify impact?",
            "bridge": "Link prior role to this question.",
            "ctx_ids": [c["id"] for c in ctx],
        }

    monkeypatch.setattr(agent, "route_question", fake_route_question, raising=True)
    monkeypatch.setattr(agent, "retrieve_context", fake_retrieve, raising=True)
    monkeypatch.setattr(agent, "_draft_background_answer", fake_draft_background, raising=True)

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        # Use text_delta (your server now tolerates both text_delta or text)
        ws.send_json({
            "session_id": "ws_test_1",
            "text_delta": "Can you walk me through a time you led a project end to end?",
            "final": True,
            "speaker": "Interviewer",
            "audio_source": "mic",
            "speaker_mode": "source",
            "client_sent_at": 123.0,
        })
        msg = ws.receive_json()
        assert msg.get("emit") is True, f"Expected emit=True, got {msg}"
        assert msg.get("kind") == "final"
        data = msg.get("data", {})
        assert data.get("response_type") == "coach_final"
        assert msg.get("kind") == "final"
        assert data.get("audio_source") == "mic"
        assert data.get("speaker_mode") == "source"
        assert "route" in data and data.get("route")
        assert "latency" in data and isinstance(data.get("latency"), dict)
        assert data["latency"].get("backend_sent_at") is not None
        assert data["latency"].get("frontend_sent_at") == 123.0
        assert "backend_total_ms" in data["latency"]
        assert "interpreter_ms" in data["latency"]
        assert "router_ms" in data["latency"]
        assert "retrieval_ms" in data["latency"]
        assert "llm_ms" in data["latency"]
        assert isinstance(data.get("interpretation"), dict)
        assert data["interpretation"].get("reason")
        cf = data.get("coach_final", {})
        assert "suggestions" in cf and isinstance(cf["suggestions"], list) and len(cf["suggestions"]) >= 1
        assert "follow_up" in cf and "bridge" in cf and "confidence" in cf and "context_ids" in cf
        assert "speaker" in data and "transcript" in data and "usage" in data

    events = read_session_log("ws_test_1")
    assert any(e.get("event_type") == "coach_final" for e in events)
