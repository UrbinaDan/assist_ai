from starlette.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_websocket_connection():
    with client.websocket_connect("/ws") as websocket:
        # Simulate minimal expected message from frontend
        websocket.send_json({
            "session_id": "test_session",
            "text_delta": "Hello?",
            "final": False
        })
        # Receive a response to confirm it's working, without forcing an LLM call.
        response = websocket.receive_json()
        assert response.get("emit") is False
        assert "reason" in response
