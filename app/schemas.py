# app/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class DeltaIn(BaseModel):
    session_id: str
    text: Optional[str] = None
    text_delta: Optional[str] = None
    ts: Optional[float] = None
    final: Optional[bool] = False
    speaker: Optional[str] = None
    mode: Optional[str] = "append"  # "append" | "replace"
    session_mode: Optional[str] = None  # "coach" | "notes"


# -------- Explicit response payloads for frontend --------

class CoachSpeculativePayload(BaseModel):
    """Lightweight coach output: question type, outline, matched themes. No full answer."""
    question_type: str
    answer_outline: str
    matched_themes: List[str]


class CoachFinalPayload(BaseModel):
    """Full coach output: suggestions, follow-up, bridge, confidence."""
    suggestions: List[str]
    follow_up: str
    bridge: str
    confidence: float
    context_ids: List[str]


class NotesPayload(BaseModel):
    """Structured notes for display."""
    bullets: List[str] = []
    topics: List[Any] = []  # [{intent, tag}] or [str]
    action_items: List[str] = []
    decisions: List[str] = []
    follow_ups: List[str] = []
    summary_so_far: Optional[str] = None
    current_topic: Optional[str] = None
    open_questions: List[str] = []


class NotesFinalPayload(BaseModel):
    notes: NotesPayload


# Unified emit payload: one of three response types, plus common fields
class EmitPayload(BaseModel):
    kind: str  # "speculative" | "final"
    response_type: str  # "coach_speculative" | "coach_final" | "notes_final"
    speaker: str
    transcript: str
    usage: Dict[str, Any]

    coach_speculative: Optional[CoachSpeculativePayload] = None
    coach_final: Optional[CoachFinalPayload] = None
    notes_final: Optional[NotesFinalPayload] = None


class AgentResponse(BaseModel):
    emit: bool
    kind: str = "none"
    data: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
