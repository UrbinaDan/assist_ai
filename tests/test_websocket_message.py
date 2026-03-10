from starlette.testclient import TestClient
from app.server import app
import app.agent as agent

def test_websocket_message_flow(monkeypatch):
    # --- stub OpenAI-dependent pieces so test is fast/offline ---
    def fake_classify(text: str, **kwargs):
        return {
            "intent": "behavioral",
            "entities": {"company": None, "role": None, "skills": [], "numbers": [], "dates": [], "times": []},
            "confidence": 0.9,
        }

    def fake_retrieve(query: str, k: int = 4, **kwargs):
        return [{"id": "fake::0", "text": "demo STAR note", "score": 0.92, "meta": {"source": "fake"}}]

    monkeypatch.setattr(agent, "classify_question", fake_classify, raising=True)
    monkeypatch.setattr(agent, "retrieve_context", fake_retrieve, raising=True)

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        # Use text_delta (your server now tolerates both text_delta or text)
        ws.send_json({
            "session_id": "ws_test_1",
            "text_delta": "Can you walk me through a time you led a project end to end?",
            "final": True
        })
        msg = ws.receive_json()
        assert msg.get("emit") is True, f"Expected emit=True, got {msg}"
        assert msg.get("kind") == "final"
        data = msg.get("data", {})
        assert data.get("response_type") == "coach_final"
        cf = data.get("coach_final", {})
        assert "suggestions" in cf and isinstance(cf["suggestions"], list) and len(cf["suggestions"]) >= 1
        assert "follow_up" in cf and "bridge" in cf and "confidence" in cf and "context_ids" in cf
        assert "speaker" in data and "transcript" in data and "usage" in data
