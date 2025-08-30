from fastapi import FastAPI
from pydantic import BaseModel
import time

from app.agent import AgentState, EndOfThought, process_turn, RETRIEVER
from app.retriever import Retriever

app = FastAPI()
SESSIONS = {}
DETECTOR = EndOfThought(pause_ms=900, stable_n=2)

@app.on_event("startup")
def _load_retriever():
    global RETRIEVER
    RETRIEVER = Retriever().load()
    print("FAISS retriever loaded.")

class TranscriptEvent(BaseModel):
    session_id: str
    text_delta: str
    final: bool = False

@app.post("/ingest")
def ingest(ev: TranscriptEvent):
    st = SESSIONS.get(ev.session_id)
    if not st:
        st = AgentState(session_id=ev.session_id)
        SESSIONS[ev.session_id] = st

    # append incoming text
    if ev.text_delta is not None:
        delta = ev.text_delta.strip()
        if delta:
            if st.buffer_text and not st.buffer_text.endswith(" "):
                st.buffer_text += " "
            st.buffer_text += delta
            st.last_token_ts = time.time()

    # emit if finalized or end-of-thought
    if ev.final or DETECTOR.should_emit(st):
        out = process_turn(st)
        if out:
            st.turns.append({"user": st.buffer_text, "assistant": out})
            st.buffer_text = ""
            return {"emit": True, "data": out}
    return {"emit": False}
