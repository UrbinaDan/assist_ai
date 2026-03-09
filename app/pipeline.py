from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .agent import AgentState, process_turn, EndOfThought


def _h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


@dataclass
class IngestResult:
    emit: bool
    data: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


def append_delta(
    st: AgentState,
    text_delta: str,
    *,
    ts: Optional[float] = None,
    speaker: Optional[str] = None,
) -> None:
    delta = (text_delta or "").strip()
    if not delta and speaker is None:
        return

    # If speaker changes mid-buffer, we keep separation by starting a new buffer.
    if speaker is not None and getattr(st, "buffer_speaker", None) and st.buffer_text.strip():
        if st.buffer_speaker != speaker:
            st.pending_speaker_flush = True
            st.next_speaker = speaker
            return

    if speaker is not None:
        st.buffer_speaker = speaker

    if delta:
        if st.buffer_text and not st.buffer_text.endswith(" "):
            st.buffer_text += " "
        st.buffer_text += delta

    st.last_token_ts = ts if ts is not None else time.time()


def maybe_emit(
    st: AgentState,
    *,
    final: bool,
    detector: EndOfThought,
) -> IngestResult:
    # If we deferred because speaker changed, flush current buffer now.
    if getattr(st, "pending_speaker_flush", False) and st.buffer_text.strip():
        out = process_turn(st)
        st.turns.append(
            {"speaker": getattr(st, "buffer_speaker", None), "user": st.buffer_text, "assistant": out}
        )
        st.buffer_text = ""
        st.last_emit_ts = time.time()
        st.pending_speaker_flush = False
        st.buffer_speaker = getattr(st, "next_speaker", st.buffer_speaker)
        st.next_speaker = None
        return IngestResult(emit=True, data=out, reason="speaker_change")

    buf = st.buffer_text.strip()
    if not buf:
        return IngestResult(emit=False, reason="empty")

    # FINAL -> always emit (dedupe repeated finals)
    if final:
        h = _h(buf)
        if st.last_final_hash == h:
            return IngestResult(emit=False, reason="duplicate_final")
        st.last_final_hash = h

        out = process_turn(st)
        st.turns.append({"speaker": getattr(st, "buffer_speaker", None), "user": st.buffer_text, "assistant": out})
        st.buffer_text = ""
        st.last_emit_ts = time.time()
        return IngestResult(emit=True, data=out, reason="final")

    # Otherwise gate on end-of-thought detector
    if detector.should_emit(st):
        out = process_turn(st)
        st.turns.append({"speaker": getattr(st, "buffer_speaker", None), "user": st.buffer_text, "assistant": out})
        st.buffer_text = ""
        st.last_emit_ts = time.time()
        return IngestResult(emit=True, data=out, reason="eot")

    return IngestResult(emit=False, reason="no_emit")

