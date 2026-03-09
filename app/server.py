from fastapi import FastAPI
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

@app.on_event("startup")
def _load_retriever():
    """Load FAISS index at startup and inject into agent.RETRIEVER."""
    agent.RETRIEVER = Retriever().load()        # inject the retriever into the agent module
    print("FAISS retriever loaded and injected into agent.", file=sys.stderr, flush=True)

@app.post("/ingest")
def ingest(ev: TranscriptEvent):
    """Append delta; if final or end-of-thought -> process; otherwise no-op."""
    st = SESSIONS.get(ev.session_id)
    if not st:
        st = AgentState(session_id=ev.session_id)
        SESSIONS[ev.session_id] = st

    append_delta(st, ev.text_delta, ts=time.time(), speaker=ev.speaker)
    res = maybe_emit(st, final=ev.final, detector=DETECTOR)
    if res.emit:
        print(f"[ingest] emit sid={ev.session_id} reason={res.reason}", file=sys.stderr, flush=True)
        return {"emit": True, "data": res.data, "reason": res.reason}
    return {"emit": False, "reason": res.reason}

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
