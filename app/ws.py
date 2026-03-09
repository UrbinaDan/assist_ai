# app/ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import time
from .agent import AgentState, EndOfThought
from .schemas import DeltaIn
from .pipeline import append_delta, maybe_emit
import json

router = APIRouter()
sessions: dict[str, AgentState] = {}
DETECTOR = EndOfThought(pause_ms=900, stable_n=2, min_words=10, max_words=60)

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
                data = DeltaIn(**payload)
                state = sessions.setdefault(data.session_id, AgentState(session_id=data.session_id))
                speaker = data.speaker

                # Support both append-delta and replace semantics.
                if (data.mode or "append") == "replace":
                    state.buffer_text = (data.text or "").strip()
                    if speaker:
                        state.buffer_speaker = speaker
                    state.last_token_ts = data.ts if data.ts is not None else time.time()
                else:
                    delta = data.text_delta if data.text_delta is not None else (data.text or "")
                    append_delta(state, delta, ts=data.ts, speaker=speaker)

                res = maybe_emit(state, final=bool(data.final), detector=DETECTOR)
                if res.emit:
                    await ws.send_json({"emit": True, "data": res.data, "reason": res.reason})
                else:
                    await ws.send_json({"emit": False, "reason": res.reason})
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
