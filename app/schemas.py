# Create this file so the WS route can parse incoming messages:
# app/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class DeltaIn(BaseModel):
    session_id: str
    text: str
    ts: Optional[float] = None
    final: Optional[bool] = False

class AgentResponse(BaseModel):
    emit: bool
    data: Dict[str, Any]
