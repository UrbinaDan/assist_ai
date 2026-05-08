# app/session_store.py
from app.agent import AgentState

SESSIONS: dict[str, AgentState] = {}


def get_or_create_session(session_id: str) -> AgentState:
    st = SESSIONS.get(session_id)
    if st is None:
        st = AgentState(session_id=session_id)
        SESSIONS[session_id] = st
    return st