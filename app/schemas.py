# Create this file so the WS route can parse incoming messages:
# app/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class DeltaIn(BaseModel):
    session_id: str
    # Clients may send either:
    # - text_delta (append semantics; preferred for realtime)
    # - text (legacy; treated as text_delta unless mode='replace')
    text: Optional[str] = None
    text_delta: Optional[str] = None
    ts: Optional[float] = None
    final: Optional[bool] = False
    speaker: Optional[str] = None
    mode: Optional[str] = "append"  # "append" | "replace"
    session_mode: Optional[str] = None  # "coach" | "notes"

class AgentResponse(BaseModel):
    emit: bool
    data: Dict[str, Any]
