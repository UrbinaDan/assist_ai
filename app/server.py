from fastapi import FastAPI
from pydantic import BaseModel
import time, hashlib, sys

import app.agent as agent               # import the module so we can inject into agent.RETRIEVER
from app.agent import AgentState, EndOfThought, process_turn
from app.retriever import Retriever

app = FastAPI()
SESSIONS = {}
DETECTOR = EndOfThought(pause_ms=900, stable_n=2)

def _h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

class TranscriptEvent(BaseModel):
    session_id: str
    text_delta: str
    final: bool = False

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

    # 1) Append incoming text
    delta = (ev.text_delta or "").strip()
    if delta:
        if st.buffer_text and not st.buffer_text.endswith(" "):
            st.buffer_text += " "
        st.buffer_text += delta
        st.last_token_ts = time.time()

    # 2) If FINAL -> always emit (dedupe repeated finals)
    if ev.final:
        buf = st.buffer_text.strip()
        h = _h(buf)
        if st.last_final_hash == h:
            print(f"[ingest] duplicate final ignored sid={ev.session_id}", file=sys.stderr, flush=True)
            return {"emit": False}
        st.last_final_hash = h

        out = process_turn(st)
        st.turns.append({"user": st.buffer_text, "assistant": out})
        st.buffer_text = ""
        print(f"[ingest] FINAL emit sid={ev.session_id}", file=sys.stderr, flush=True)
        return {"emit": True, "data": out}

    # 3) Otherwise gate on end-of-thought detector
    if DETECTOR.should_emit(st):
        out = process_turn(st)
        st.turns.append({"user": st.buffer_text, "assistant": out})
        st.buffer_text = ""
        print(f"[ingest] EOT emit sid={ev.session_id}", file=sys.stderr, flush=True)
        return {"emit": True, "data": out}

    # No emit yet
    return {"emit": False}

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
