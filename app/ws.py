# app/ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .agent import AgentState, process_turn
from .schemas import DeltaIn, AgentResponse
import json

router = APIRouter()
sessions: dict[str, AgentState] = {}

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
                # tolerate text_delta from older clients/tests
                if "text" not in payload and "text_delta" in payload:
                    payload["text"] = payload.pop("text_delta")

                data = DeltaIn(**payload)
                state = sessions.setdefault(data.session_id, AgentState(session_id=data.session_id))
                state.buffer_text = data.text
                state.last_token_ts = data.ts or 0.0

                out = process_turn(state)
                if out:
                    await ws.send_json({"emit": True, "data": out})
                else:
                    await ws.send_json({"emit": False})
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
