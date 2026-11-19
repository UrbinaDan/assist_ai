import json
import os
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent import EndOfThought
from app.pipeline import append_delta, maybe_emit
from app.schemas import DeltaIn
from app.session_store import get_or_create_session


router = APIRouter()

DETECTOR = EndOfThought(pause_ms=900, stable_n=2, min_words=10, max_words=60)
USE_STREAMING_COACH = os.getenv("USE_STREAMING_COACH", "false").lower() in ("1", "true", "yes")


def _latency_ms(latency: dict, start_key: str, end_key: str):
    start = latency.get(start_key)
    end = latency.get(end_key)
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return round(max(0.0, end - start) * 1000.0, 2)


def _finalize_latency(latency: dict | None) -> dict:
    out = dict(latency or {})
    out["backend_sent_at"] = time.time()
    out["backend_total_ms"] = _latency_ms(out, "backend_received_at", "backend_sent_at")
    out["interpreter_ms"] = _latency_ms(out, "interpreter_start_at", "interpreter_end_at")
    out["router_ms"] = _latency_ms(out, "router_start_at", "router_end_at")
    out["retrieval_ms"] = _latency_ms(out, "retrieval_start_at", "retrieval_end_at")
    out["llm_ms"] = _latency_ms(out, "llm_start_at", "llm_end_at")
    out["end_to_end_estimate_ms"] = _latency_ms(out, "frontend_sent_at", "backend_sent_at")
    return out


def _coach_stream_text(out: dict) -> str:
    coach_final = out.get("coach_final") if isinstance(out, dict) else None
    if not isinstance(coach_final, dict):
        return ""

    parts = []
    for idx, suggestion in enumerate(coach_final.get("suggestions") or [], start=1):
        if suggestion:
            parts.append(f"{idx}. {suggestion}")
    follow_up = coach_final.get("follow_up")
    if follow_up:
        parts.append(f"Follow-up: {follow_up}")
    bridge = coach_final.get("bridge")
    if bridge:
        parts.append(f"Bridge: {bridge}")
    return "\n".join(parts).strip()


def _chunk_stream_text(text: str, chunk_size: int = 80):
    text = text or ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk:
            yield chunk


def _should_stream_coach(out: dict) -> bool:
    return (
        USE_STREAMING_COACH
        and isinstance(out, dict)
        and out.get("kind") == "final"
        and out.get("response_type") == "coach_final"
    )


async def _send_coach_stream(ws: WebSocket, *, session_id: str, out: dict) -> None:
    if not _should_stream_coach(out):
        return
    for text_delta in _chunk_stream_text(_coach_stream_text(out)):
        await ws.send_json({
            "emit": True,
            "kind": "answer_delta",
            "data": {
                "text_delta": text_delta,
                "session_id": session_id,
                "response_type": "coach_stream",
            },
        })


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] connection accepted", flush=True)

    try:
        while True:
            raw = await ws.receive_text()
            backend_received_at = time.time()

            try:
                payload = json.loads(raw)
                data = DeltaIn(**payload)

                # Log only protocol metadata, never raw transcripts or speaker content.
                print(
                    f"[ws] received final={data.final} mode={data.session_mode!r} "
                    f"audio_source={data.audio_source!r}",
                    flush=True,
                )

                state = get_or_create_session(data.session_id)
                state.current_latency = {
                    "frontend_sent_at": data.client_sent_at,
                    "backend_received_at": backend_received_at,
                }
                speaker = data.speaker

                if data.audio_source is not None:
                    state.last_audio_source = data.audio_source
                if data.speaker_mode is not None:
                    state.last_speaker_mode = data.speaker_mode

                state.last_seen_at = time.time()

                if speaker:
                    state.roles.setdefault(speaker, speaker)

                if data.session_mode in ("coach", "notes"):
                    state.mode = data.session_mode

                # Support both append-delta and replace semantics
                if (data.mode or "append") == "replace":
                    print("[ws] handling replace mode", flush=True)

                    if speaker and state.buffer_text.strip() and state.buffer_speaker != speaker:
                        print("[ws] speaker changed during replace; forcing flush", flush=True)
                        _ = maybe_emit(state, final=True, detector=DETECTOR)

                    state.buffer_text = (data.text or "").strip()
                    if speaker:
                        state.buffer_speaker = speaker
                    state.last_token_ts = data.ts if data.ts is not None else time.time()

                else:
                    delta = data.text_delta if data.text_delta is not None else (data.text or "")
                    append_delta(state, delta, ts=data.ts, speaker=speaker)


                res = maybe_emit(state, final=bool(data.final), detector=DETECTOR)

                print(
                    f"[ws] emit={res.emit} kind={res.kind} reason={res.reason}",
                    flush=True,
                )

                if res.emit:
                    out = dict(res.data) if res.data else {}
                    out["latency"] = _finalize_latency(out.get("latency"))
                    out["created_at"] = getattr(state, "created_at", None)
                    out["last_seen_at"] = getattr(state, "last_seen_at", None)

                    if isinstance(out.get("usage"), dict):
                        out["usage"]["created_at"] = out["created_at"]
                        out["usage"]["last_seen_at"] = out["last_seen_at"]

                    print(f"[ws] sending emit payload keys={list(out.keys())}", flush=True)

                    await _send_coach_stream(ws, session_id=data.session_id, out=out)

                    await ws.send_json(
                        {
                            "emit": True,
                            "kind": res.kind,
                            "data": out,
                            "reason": res.reason,
                        }
                    )
                else:
                    await ws.send_json(
                        {
                            "emit": False,
                            "kind": res.kind,
                            "reason": res.reason,
                        }
                    )

            except Exception as e:
                # Avoid reflecting exception messages, which may contain private
                # transcript or provider details, into server logs or the browser.
                print(f"[ws] message handling failed ({type(e).__name__})", flush=True)
                await ws.send_json({"error": "Unable to process message"})

    except WebSocketDisconnect:
        print("[ws] connection closed", flush=True)
