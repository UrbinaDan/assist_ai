# app/query_interpreter.py
"""Deterministic transcript interpretation for routing and coaching (Phase 6).

LLM-based refinement can be added later without changing the public dict shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# Single-token utterances treated as non-content (aligned with soft gate / Phase 7).
_JUNK_TOKENS = frozenset({
    "so",
    "ok",
    "okay",
    "um",
    "uh",
    "uhh",
    "hmm",
    "hm",
    "well",
    "like",
    "right",
    "yeah",
    "yep",
    "what's",
    "whats",
})

_STRONG_Q_PREFIX = re.compile(
    r"^(what\'?s|what\s+is|what\s+are|how\s+(do|does|did|would|can|could)|why\s+(do|does|did|would|is|are)|"
    r"can\s+you|could\s+you|would\s+you|tell\s+me\s+about|talk\s+to\s+me\s+about|walk\s+me\s+through|"
    r"describe|explain|solve|is\s+it\s+true)\b",
    re.IGNORECASE,
)

_INTERVIEW_PROMPT = re.compile(
    r"^(tell\s+me\s+about|walk\s+me\s+through|describe\s+a\s+time|give\s+me\s+an\s+example)\b",
    re.IGNORECASE,
)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _collapse_dup_words(s: str) -> str:
    words = s.split()
    out: list[str] = []
    for w in words:
        if out and out[-1].lower() == w.lower():
            continue
        out.append(w)
    return " ".join(out)


def _strip_leading_filler(s: str) -> str:
    patterns = [
        r"^(okay|ok|well|so|um|uh|uhh|hmm|hm|like)\s+",
        r"^(yeah|yep|right)\s*[,:]?\s+",
    ]
    out = s.strip()
    changed = True
    while changed:
        changed = False
        for p in patterns:
            nxt = re.sub(p, "", out, flags=re.IGNORECASE).strip()
            if nxt != out:
                out = nxt
                changed = True
    return out


def _minimal_clean(s: str) -> str:
    t = _normalize_ws(s)
    t = _collapse_dup_words(t)
    t = _strip_leading_filler(t)
    t = re.sub(r"\b(in in|to to)\b", lambda m: m.group(0).split()[0], t, flags=re.IGNORECASE)
    return _normalize_ws(t)


def _is_me_label(speaker: Optional[str]) -> bool:
    x = (speaker or "").strip().lower()
    return x in {"me", "candidate", "user", "self"}


def _fix_common_terms(s: str) -> str:
    out = re.sub(r"\bpython\b", "Python", s, flags=re.IGNORECASE)
    out = re.sub(r"\bjavascript\b", "JavaScript", out, flags=re.IGNORECASE)
    out = re.sub(r"\btypescript\b", "TypeScript", out, flags=re.IGNORECASE)
    return out


def _sentence_start_upper(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def _detect_question(t: str) -> bool:
    t = t.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    tl = t.lower()
    if _STRONG_Q_PREFIX.match(tl):
        return True
    if _INTERVIEW_PROMPT.match(tl):
        return True
    return False


def _junk_fragment(t: str) -> bool:
    u = t.strip().lower().rstrip(".!?")
    if not u:
        return True
    tokens = u.split()
    if len(tokens) == 1 and tokens[0] in _JUNK_TOKENS:
        return True
    return False


@dataclass
class InterpreterResult:
    interpreted_question: Optional[str]
    cleaned_text: str
    is_question: bool
    confidence: float
    should_coach: bool
    missing_context: bool
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_question_mark(p: str) -> str:
    p = p.strip()
    if not p:
        return p
    if p.endswith("?"):
        return p
    return p + "?"


def interpret_transcript(
    raw_text: str,
    cleaned_text: Optional[str] = None,
    speaker: Optional[str] = None,
    mode: str = "coach",
) -> Dict[str, Any]:
    """Normalize messy speech into routing/candidate question text (heuristics only).

    Returns a plain dict for JSON-friendly WebSocket payloads.

    ``cleaned_text`` is optional metadata from upstream normalization; the heuristic
    pipeline primarily uses ``raw_text`` so helpful prefixes like "tell me" are
    not stripped before question detection.
    """
    mode = (mode or "coach").strip().lower()
    src = raw_text.strip() if (raw_text or "").strip() else (cleaned_text or "").strip()
    base = _minimal_clean(src)
    base = _fix_common_terms(base)

    reason = "ok"
    interpreted_question: Optional[str] = None
    is_question = False
    confidence = 0.55
    missing_context = False
    should_coach = False

    if _junk_fragment(base):
        return InterpreterResult(
            interpreted_question=None,
            cleaned_text="",
            is_question=False,
            confidence=0.12,
            should_coach=False,
            missing_context=True,
            reason="junk_fragment",
        ).as_dict()

    wc = len(base.split())
    if wc < 4:
        missing_context = True
        confidence = min(confidence, 0.38)

    is_question = _detect_question(base)

    if mode == "notes":
        should_coach = False
        interpreted_question = None
        reason = "notes_mode"
        if is_question and wc >= 4:
            confidence = 0.72
        elif wc >= 8:
            confidence = 0.62
        else:
            confidence = max(confidence, 0.45)
        return InterpreterResult(
            interpreted_question=None,
            cleaned_text=base,
            is_question=is_question,
            confidence=confidence,
            should_coach=False,
            missing_context=missing_context,
            reason=reason,
        ).as_dict()

    # Coach mode
    if _is_me_label(speaker):
        interpreted_question = None
        should_coach = False
        confidence = 0.68 if wc >= 6 else 0.5
        return InterpreterResult(
            interpreted_question=None,
            cleaned_text=base,
            is_question=is_question,
            confidence=confidence,
            should_coach=False,
            missing_context=missing_context,
            reason="coach_candidate_turn",
        ).as_dict()

    # Interviewer (or unknown speaker): derive polished question when applicable
    should_coach = True
    polish = _sentence_start_upper(base)

    if is_question:
        polish = _ensure_question_mark(polish)
        interpreted_question = polish
        if polish.endswith("?"):
            confidence = 0.88
        elif _INTERVIEW_PROMPT.match(polish.lower()):
            confidence = 0.78
        else:
            confidence = 0.72
        reason = "question_detected"
    else:
        interpreted_question = None
        if wc >= 10:
            confidence = 0.58
            reason = "non_question_interviewer"
        else:
            confidence = 0.48
            reason = "ambiguous_interviewer"

    return InterpreterResult(
        interpreted_question=interpreted_question,
        cleaned_text=base,
        is_question=is_question,
        confidence=confidence,
        should_coach=should_coach,
        missing_context=missing_context,
        reason=reason,
    ).as_dict()
