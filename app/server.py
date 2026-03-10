from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time, sys
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import app.agent as agent               # import the module so we can inject into agent.RETRIEVER
from app.agent import AgentState, EndOfThought
from app.retriever import Retriever
from app.pipeline import append_delta, maybe_emit

app = FastAPI()
SESSIONS = {}
DETECTOR = EndOfThought(pause_ms=900, stable_n=2)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    if STATIC_DIR.exists():
        return FileResponse(str(STATIC_DIR / "index.html"))
    return {"ok": True, "hint": "Add app/static/index.html for a simple frontend."}

class TranscriptEvent(BaseModel):
    session_id: str
    text_delta: str
    final: bool = False
    speaker: str | None = None
    session_mode: str | None = None  # "coach" | "notes"


class SessionModeBody(BaseModel):
    mode: str  # "coach" | "notes"


class SessionSummary(BaseModel):
    session_id: str
    mode: str
    created_at: float
    last_seen_at: float
    roles: dict[str, str]


class UsageSummary(BaseModel):
    session_id: str
    mode: str
    created_at: float
    last_seen_at: float
    usage: dict


class NotesSummary(BaseModel):
    session_id: str
    mode: str
    notes: dict

@app.on_event("startup")
def _load_retriever():
    """Load FAISS index at startup and inject into agent.RETRIEVER."""
    try:
        agent.RETRIEVER = Retriever().load()        # inject the retriever into the agent module
        print("FAISS retriever loaded and injected into agent.", file=sys.stderr, flush=True)
    except Exception as e:
        agent.RETRIEVER = None
        print(f"FAISS retriever not available: {e}", file=sys.stderr, flush=True)

@app.post("/ingest")
def ingest(ev: TranscriptEvent):
    """Append delta; if final or end-of-thought -> process; otherwise no-op."""
    st = SESSIONS.get(ev.session_id)
    if not st:
        st = AgentState(session_id=ev.session_id)
        SESSIONS[ev.session_id] = st

    # Touch session activity
    st.last_seen_at = time.time()

    if ev.session_mode in ("coach", "notes"):
        st.mode = ev.session_mode

    append_delta(st, ev.text_delta, ts=time.time(), speaker=ev.speaker)
    res = maybe_emit(st, final=ev.final, detector=DETECTOR)
    if res.emit:
        out = dict(res.data) if res.data else {}
        out["created_at"] = getattr(st, "created_at", None)
        out["last_seen_at"] = getattr(st, "last_seen_at", None)
        if out.get("usage") and isinstance(out["usage"], dict):
            out["usage"]["created_at"] = out["created_at"]
            out["usage"]["last_seen_at"] = out["last_seen_at"]
        print(f"[ingest] emit sid={ev.session_id} kind={res.kind} reason={res.reason}", file=sys.stderr, flush=True)
        return {"emit": True, "kind": res.kind, "data": out, "reason": res.reason}
    return {"emit": False, "kind": res.kind, "reason": res.reason}


def _get_session_or_404(session_id: str) -> AgentState:
    st = SESSIONS.get(session_id)
    if not st:
        raise HTTPException(status_code=404, detail="Session not found")
    return st


@app.get("/session/{session_id}", response_model=SessionSummary)
def get_session(session_id: str):
    st = _get_session_or_404(session_id)
    return SessionSummary(
        session_id=session_id,
        mode=st.mode,
        created_at=st.created_at,
        last_seen_at=st.last_seen_at,
        roles=st.roles,
    )


@app.get("/session/{session_id}/mode", response_model=SessionModeBody)
def get_session_mode(session_id: str):
    st = _get_session_or_404(session_id)
    return SessionModeBody(mode=st.mode)


@app.post("/session/{session_id}/mode", response_model=SessionModeBody)
def set_session_mode(session_id: str, body: SessionModeBody):
    st = SESSIONS.get(session_id)
    if not st:
        st = AgentState(session_id=session_id)
        SESSIONS[session_id] = st
    if body.mode not in ("coach", "notes"):
        raise HTTPException(status_code=400, detail="mode must be 'coach' or 'notes'")
    st.mode = body.mode
    st.last_seen_at = time.time()
    return SessionModeBody(mode=st.mode)


@app.get("/session/{session_id}/usage", response_model=UsageSummary)
def get_session_usage(session_id: str):
    st = _get_session_or_404(session_id)
    return UsageSummary(
        session_id=session_id,
        mode=st.mode,
        created_at=st.created_at,
        last_seen_at=st.last_seen_at,
        usage=st.usage,
    )


@app.get("/session/{session_id}/notes", response_model=NotesSummary)
def get_session_notes(session_id: str):
    st = _get_session_or_404(session_id)
    return NotesSummary(
        session_id=session_id,
        mode=st.mode,
        notes=st.notes,
    )

@app.get("/health")
def health():
    status = "loaded" if (agent.RETRIEVER and agent.RETRIEVER.index is not None) else "missing"
    return {"retriever": status}

@app.get("/debug/retriever")
def debug_retriever():
    if not agent.RETRIEVER or not agent.RETRIEVER.index:
        return {"loaded": False}
    return {"loaded": True, "chunks": len(agent.RETRIEVER.meta)}

# Simple classifier probe to surface any OpenAI issues quickly
class ClassifyProbe(BaseModel):
    text: str

@app.post("/debug/classify")
def debug_classify(p: ClassifyProbe):
    try:
        result = agent.classify_question(p.text)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}



# End-to-end test 
# curl -X POST http://127.0.0.1:8000/ingest \
#   -H "Content-Type: application/json" \
#   -d '{"session_id":"t1","text_delta":"Can you walk me through a time you led a project end to end?","final":true}'


from app.ws import router as ws_router
app.include_router(ws_router)
