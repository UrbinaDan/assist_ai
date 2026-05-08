from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import os
import json
import re
import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from app.retriever import Retriever
from app.query_interpreter import interpret_transcript
from app.memory_router import memory_filters_for_route
from app.session_logger import log_session_event

# =========================================================
# Models / config
# =========================================================
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4.1-mini")
ANSWER_MODEL_BG = os.getenv("ANSWER_MODEL_BG", "gpt-4o-mini")
ANSWER_MODEL_DIRECT = os.getenv("ANSWER_MODEL_DIRECT", "gpt-4o-mini")
ANSWER_MODEL_NOTES = os.getenv("ANSWER_MODEL_NOTES", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

USE_OAI_NOTES = os.getenv("USE_OAI_NOTES", "false").lower() in ("1", "true", "yes")

ROUTES = [
    "behavioral_story",
    "technical_project",
    "personal_background",
    "company_fit",
    "direct_knowledge",
    "coding_interview",
    "system_design",
    "small_talk",
    "unknown",
]

BACKGROUND_ROUTES = {"behavioral_story", "technical_project", "personal_background", "company_fit"}

# =========================================================
# OpenAI client
# =========================================================
_oai: Optional[OpenAI] = None

def _oai_client() -> OpenAI:
    global _oai
    if _oai is None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt")
        _oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _oai

# =========================================================
# Runtime state
# =========================================================
@dataclass
class AgentState:
    session_id: str
    buffer_text: str = ""
    buffer_speaker: str = "Speaker 1"
    mode: str = "coach"
    created_at: float = field(default_factory=lambda: time.time())
    last_seen_at: float = field(default_factory=lambda: time.time())
    last_emit_ts: float = 0.0
    last_token_ts: float = 0.0

    intent_history: List[str] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)

    prefs: Dict[str, Any] = field(default_factory=lambda: {
        "tone": "concise",
        "target_len": "2 sentences",
        "avoid": ["jargon", "overpromising"],
    })

    retrieval_cache: Dict[str, Any] = field(default_factory=dict)
    uploaded_contexts: List[Dict[str, Any]] = field(default_factory=list)
    session_context_chunks: List[Dict[str, Any]] = field(default_factory=list)
    current_latency: Dict[str, Any] = field(default_factory=dict)
    last_final_hash: str = ""
    pending_speaker_flush: bool = False
    next_speaker: Optional[str] = None
    next_text_delta: Optional[str] = None
    next_ts: Optional[float] = None

    usage: Dict[str, Any] = field(default_factory=lambda: {
        "by_model": {},
        "by_feature": {},
        "cost_usd_total": 0.0,
        "turn": {"by_model": {}, "cost_usd": 0.0},
    })

    notes: Dict[str, Any] = field(default_factory=lambda: {
        "bullets": [],
        "action_items": [],
        "decisions": [],
        "follow_ups": [],
        "topics": [],
        "summary_so_far": None,
        "current_topic": None,
        "open_questions": [],
    })

    roles: Dict[str, str] = field(default_factory=lambda: {})

    # Last client-reported audio / speaker UI mode (Phase X)
    last_audio_source: Optional[str] = None
    last_speaker_mode: Optional[str] = None

class EndOfThought:
    def __init__(self, pause_ms: int = 900, stable_n: int = 2, min_words: int = 10, max_words: int = 60):
        self.pause_ms = pause_ms
        self.stable_n = stable_n
        self.min_words = min_words
        self.max_words = max_words

    def strong_punct(self, text: str) -> bool:
        return text.strip().endswith(("?", ".", "!"))

    def should_emit(self, state: AgentState) -> bool:
        now = time.time()
        paused = (now - state.last_token_ts) * 1000 >= self.pause_ms
        punct = self.strong_punct(state.buffer_text)
        words = len(state.buffer_text.split())
        longish = words >= self.min_words
        too_long = words >= self.max_words
        return (paused and (punct or longish)) or too_long

# =========================================================
# Pricing / usage
# =========================================================
PRICES_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}

def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def _record_usage(
    state: AgentState,
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    feature: str = "core",
) -> None:
    by_model = state.usage.setdefault("by_model", {})
    row = by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    row["prompt_tokens"] += _safe_int(prompt_tokens)
    row["completion_tokens"] += _safe_int(completion_tokens)
    row["total_tokens"] += _safe_int(prompt_tokens) + _safe_int(completion_tokens)

    price = PRICES_PER_1M_TOKENS.get(model)
    if price:
        delta_cost = (
            (_safe_int(prompt_tokens) / 1_000_000.0) * price.get("input", 0.0)
            + (_safe_int(completion_tokens) / 1_000_000.0) * price.get("output", 0.0)
        )
        state.usage["cost_usd_total"] = round(float(state.usage.get("cost_usd_total", 0.0)) + delta_cost, 6)

        turn = state.usage.setdefault("turn", {"by_model": {}, "cost_usd": 0.0})
        turn["cost_usd"] = round(float(turn.get("cost_usd", 0.0)) + delta_cost, 6)

        t_by_model = turn.setdefault("by_model", {})
        trow = t_by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
        trow["prompt_tokens"] += _safe_int(prompt_tokens)
        trow["completion_tokens"] += _safe_int(completion_tokens)
        trow["total_tokens"] += _safe_int(prompt_tokens) + _safe_int(completion_tokens)
        trow["cost_usd"] = round(float(trow.get("cost_usd", 0.0)) + delta_cost, 6)

        by_feature = state.usage.setdefault("by_feature", {})
        f = by_feature.setdefault(feature, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
        f["prompt_tokens"] += _safe_int(prompt_tokens)
        f["completion_tokens"] += _safe_int(completion_tokens)
        f["total_tokens"] += _safe_int(prompt_tokens) + _safe_int(completion_tokens)
        f["cost_usd"] = round(float(f.get("cost_usd", 0.0)) + delta_cost, 6)

# =========================================================
# Retriever
# =========================================================
RETRIEVER: Optional[Retriever] = None

def embed_query(text: str, *, state: Optional["AgentState"] = None) -> np.ndarray:
    client = _oai_client()
    rsp = client.embeddings.create(model=EMBED_MODEL, input=text)
    u = getattr(rsp, "usage", None)
    if state is not None and u is not None:
        emb_prompt = _safe_int(getattr(u, "prompt_tokens", getattr(u, "total_tokens", 0)))
        _record_usage(
            state,
            model=EMBED_MODEL,
            prompt_tokens=emb_prompt,
            completion_tokens=0,
            feature="embed",
        )
    emb = rsp.data[0].embedding
    return np.array(emb, dtype="float32")

def retrieve_context(
    query: str,
    k: int = 6,
    *,
    state: Optional[AgentState] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    session_ctx = retrieve_uploaded_context(query, state=state, k=3)
    static_ctx: List[Dict[str, Any]] = []
    if RETRIEVER is not None:
        qv = embed_query(query, state=state)
        static_ctx = RETRIEVER.search(qv, k=k, filters=filters)

    if not session_ctx:
        return static_ctx

    seen = {c.get("id") for c in session_ctx}
    merged = list(session_ctx)
    for c in static_ctx:
        if c.get("id") not in seen:
            merged.append(c)
            seen.add(c.get("id"))
        if len(merged) >= k:
            break
    return merged[:k]


def _keyword_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9_+-]{2,}", (text or "").lower())
        if tok not in {"the", "and", "for", "with", "that", "this", "from", "you", "your", "are"}
    }


def retrieve_uploaded_context(
    query: str,
    *,
    state: Optional[AgentState],
    k: int = 3,
) -> List[Dict[str, Any]]:
    if state is None:
        return []
    chunks = getattr(state, "session_context_chunks", []) or []
    if not chunks:
        return []

    q_tokens = _keyword_tokens(query)
    if not q_tokens:
        return []

    scored: List[Dict[str, Any]] = []
    for chunk in chunks:
        c_tokens = _keyword_tokens(chunk.get("text", ""))
        if not c_tokens:
            continue
        overlap = q_tokens & c_tokens
        if not overlap:
            continue
        score = min(1.0, 0.72 + (len(overlap) / max(10, len(q_tokens))) * 0.28)
        meta = dict(chunk.get("meta") or {})
        meta.setdefault("source", chunk.get("source"))
        meta.setdefault("source_type", "uploaded_context")
        meta.setdefault("memory_type", "session")
        meta.setdefault("topic", chunk.get("topic"))
        meta.setdefault("route", chunk.get("route", "session_context"))
        meta.setdefault("priority", 1.0)
        scored.append({
            "id": chunk.get("id"),
            "text": chunk.get("text", ""),
            "score": round(score, 3),
            "meta": meta,
        })

    scored.sort(key=lambda row: (row["score"], row["meta"].get("priority", 0.0)), reverse=True)
    return scored[:k]

# =========================================================
# Text normalization / gating
# =========================================================
def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _collapse_duplicate_words(s: str) -> str:
    words = s.split()
    out = []
    for w in words:
        if out and out[-1].lower() == w.lower():
            continue
        out.append(w)
    return " ".join(out)

def _strip_leading_filler(s: str) -> str:
    filler_patterns = [
        r"^(hi|hello|hey)\s+\w+\s+",
        r"^(okay|ok|well|so|um|uh|hmm)\s+",
        r"^(that'?s good( then)?)\s+",
        r"^(sue|see|so)\s+\w+\s+",
        r"^(tell me)\s+",
    ]
    out = s
    changed = True
    while changed:
        changed = False
        for p in filler_patterns:
            new = re.sub(p, "", out, flags=re.IGNORECASE)
            if new != out:
                out = new.strip()
                changed = True
    return out

def clean_transcript_text(s: str) -> str:
    s = _normalize_text(s)
    s = _collapse_duplicate_words(s)
    s = _strip_leading_filler(s)
    s = re.sub(r"\b(in in|to to)\b", lambda m: m.group(0).split()[0], s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _looks_like_question(s: str) -> bool:
    s2 = s.lower().strip()
    if s.endswith("?"):
        return True
    return bool(re.match(
        r"^(talk to me about|tell me about|tell me why|why do you|what is|what are|how do you|how would you|can you|walk me through|solve|explain|describe)\b",
        s2
    ))

def _question_quality_gate(s: str) -> Dict[str, Any]:
    text = clean_transcript_text(s)
    words = text.split()

    if len(words) < 5:
        return {"ok": False, "clean_text": text, "reason": "too_short"}

    if len(words) > 60:
        return {"ok": False, "clean_text": text, "reason": "too_long"}

    garbage_patterns = [
        r"how old is .* that'?s good",
        r"sue \w+ tell me",
        r"keenan application",
        r"\b(back|pages)\b$",
    ]
    low = text.lower()
    for p in garbage_patterns:
        if re.search(p, low):
            return {"ok": False, "clean_text": text, "reason": "garbage_pattern"}

    abrupt_topic_signals = 0
    if "how old is" in low:
        abrupt_topic_signals += 1
    if "that's good then" in low:
        abrupt_topic_signals += 1
    if abrupt_topic_signals >= 1:
        return {"ok": False, "clean_text": text, "reason": "abrupt_topic"}

    if not _looks_like_question(text):
        return {"ok": False, "clean_text": text, "reason": "not_question_like"}

    return {"ok": True, "clean_text": text, "reason": "ok"}

def _normalize_speaker_label(s: str) -> str:
    return (s or "").strip().lower()

def _is_me_speaker(s: str) -> bool:
    return _normalize_speaker_label(s) in {"me", "candidate", "user", "self"}

# =========================================================
# Router
# =========================================================
def route_question(text: str, *, state: Optional[AgentState] = None) -> Dict[str, Any]:
    client = _oai_client()
    system_msg = (
        "You are a lightweight router for a realtime conversation assistant.\n"
        "Return ONLY JSON with keys: route, confidence, rationale.\n"
        f"route must be one of: {ROUTES}.\n\n"
        "Rules:\n"
        "- behavioral_story: examples about conflict, leadership, challenge, failure, teamwork.\n"
        "- personal_background: tell me about yourself, strengths, weaknesses, background, goals.\n"
        "- company_fit: why this company, why this role, why work here.\n"
        "- direct_knowledge: factual, math, conceptual, direct questions.\n"
        "- coding_interview: coding, algorithms, debugging, complexity, python/javascript/java, code-writing.\n"
        "- system_design: APIs, architecture, scalability, distributed systems.\n"
        "- small_talk: casual conversation.\n"
        "- unknown only if truly unclear.\n"
        "If the question does not require the user's background, prefer direct_knowledge/coding_interview/system_design."
    )

    try:
        rsp = client.chat.completions.create(
            model=ROUTER_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Utterance: {text}"}
            ],
            max_tokens=180,
            temperature=0.1,
        )
        u = getattr(rsp, "usage", None)
        if state is not None and u is not None:
            _record_usage(
                state,
                model=ROUTER_MODEL,
                prompt_tokens=_safe_int(getattr(u, "prompt_tokens", 0)),
                completion_tokens=_safe_int(getattr(u, "completion_tokens", 0)),
                feature="router",
            )

        raw = rsp.choices[0].message.content or "{}"
        data = json.loads(raw)
        route = data.get("route", "unknown")
        if route not in ROUTES:
            route = "unknown"
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        return {
            "route": route,
            "intent": route,
            "confidence": conf,
            "rationale": data.get("rationale", ""),
        }
    except Exception as e:
        print("[router] error:", e, flush=True)
        return {
            "route": "direct_knowledge",
            "intent": "direct_knowledge",
            "confidence": 0.45,
            "rationale": "router_error_fallback",
        }

# =========================================================
# Prompt builders
# =========================================================
def _join_ctx(ctx: List[Dict[str, Any]], limit: int = 5) -> str:
    return "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(ctx[:limit])) or "No context."

def _background_system_prompt(route: str) -> str:
    if route == "personal_background":
        return (
            "You are coaching a concise answer to 'Tell me about yourself' or a similar background question.\n"
            "Prefer this structure:\n"
            "1. mechanical engineering background\n"
            "2. transition into software engineering\n"
            "3. current hands-on work with APIs, Python, testing, automation, or AI projects\n"
            "4. why that makes the candidate strong for this role now\n"
            "Do not invent experiences. Use only the provided context.\n"
            "Return JSON with keys: options (2-3 strings), follow_up (string), bridge (string)."
        )
    if route == "company_fit":
        return (
            "You are coaching a concise answer to 'Why do you want to work with us?'\n"
            "Discourage vague lines like 'I am excited to work in this company.'\n"
            "Prefer this structure:\n"
            "1. what attracts the candidate about the company/team/role\n"
            "2. why the candidate's background fits\n"
            "3. why this is the right next step\n"
            "Do not invent details. Use only the provided context.\n"
            "Return JSON with keys: options (2-3 strings), follow_up (string), bridge (string)."
        )
    return (
        "You are coaching a behavioral interview answer.\n"
        "Use a natural STAR-style structure.\n"
        "Do not invent details. Use only the provided context.\n"
        "Return JSON with keys: options (2-3 strings), follow_up (string), bridge (string)."
    )

def _direct_system_prompt(route: str) -> str:
    if route == "coding_interview":
        return (
            "You are answering a coding interview question.\n"
            "Answer directly and simply.\n"
            "Do NOT use the candidate's background unless explicitly relevant.\n"
            "If code is needed, keep it short and practical.\n"
            "Return JSON with keys: options (1-2 strings), follow_up (string), bridge (string)."
        )
    if route == "system_design":
        return (
            "You are answering a system design interview question.\n"
            "Answer directly using: scope, components, flow, tradeoffs.\n"
            "Do NOT force the candidate's background.\n"
            "Return JSON with keys: options (1-2 strings), follow_up (string), bridge (string)."
        )
    if route == "small_talk":
        return (
            "You are answering light small-talk.\n"
            "Be natural and short.\n"
            "Return JSON with keys: options (1-2 strings), follow_up (string), bridge (string)."
        )
    return (
        "You are answering a direct factual or conceptual question.\n"
        "Answer directly first, then briefly explain.\n"
        "Do NOT force the candidate's background.\n"
        "Return JSON with keys: options (1-2 strings), follow_up (string), bridge (string)."
    )

def _draft_with_model(
    *,
    model: str,
    system_msg: str,
    user_msg: str,
    state: Optional[AgentState] = None,
    feature: str = "answer",
) -> Dict[str, Any]:
    client = _oai_client()
    try:
        rsp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=350,
            temperature=0.2,
        )
        u = getattr(rsp, "usage", None)
        if state is not None and u is not None:
            _record_usage(
                state,
                model=model,
                prompt_tokens=_safe_int(getattr(u, "prompt_tokens", 0)),
                completion_tokens=_safe_int(getattr(u, "completion_tokens", 0)),
                feature=feature,
            )
        raw = rsp.choices[0].message.content or "{}"
        data = json.loads(raw)
        return {
            "options": (data.get("options") or [])[:3],
            "follow_up": data.get("follow_up") or "",
            "bridge": data.get("bridge") or "",
        }
    except Exception as e:
        print("[answer-model] error:", e, flush=True)
        return {
            "options": ["I may need a little more context to answer that well."],
            "follow_up": "Do you want a concise answer or a more detailed one?",
            "bridge": "I can refine it once I have a little more context.",
        }

def _draft_background_answer(
    route: str,
    text: str,
    ctx: List[Dict[str, Any]],
    prefs: Dict[str, Any],
    *,
    state: Optional[AgentState] = None,
) -> Dict[str, Any]:
    system_msg = _background_system_prompt(route)
    user_msg = (
        f"Question: {text}\n\n"
        f"Candidate background:\n{_join_ctx(ctx)}\n\n"
        f"Prefs: {prefs}"
    )
    out = _draft_with_model(
        model=ANSWER_MODEL_BG,
        system_msg=system_msg,
        user_msg=user_msg,
        state=state,
        feature=f"{route}_bg_answer",
    )
    out["ctx_ids"] = [c["id"] for c in ctx]
    return out

def _draft_direct_answer(
    route: str,
    text: str,
    *,
    state: Optional[AgentState] = None,
) -> Dict[str, Any]:
    out = _draft_with_model(
        model=ANSWER_MODEL_DIRECT,
        system_msg=_direct_system_prompt(route),
        user_msg=f"Question: {text}",
        state=state,
        feature=f"{route}_direct_answer",
    )
    out["ctx_ids"] = []
    return out

# =========================================================
# Speculative builders
# =========================================================
def _build_speculative_background(route: str, ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    if route == "personal_background":
        outline = (
            "Structure: mechanical engineering background → transition into software engineering → "
            "current hands-on work with APIs/Python/testing/automation → why that makes you strong now."
        )
    elif route == "company_fit":
        outline = (
            "Structure: what attracts you about the role/company → why your background fits → "
            "why this is the right next step."
        )
    else:
        outline = "Use one concrete example with situation, action, and result."

    themes = [
        c.get("text", "")[:80].rstrip() + ("..." if len(c.get("text", "")) > 80 else "")
        for c in ctx[:3] if c.get("text")
    ]
    return {
        "question_type": route,
        "answer_outline": outline,
        "matched_themes": themes,
    }

def _build_speculative_direct(route: str) -> Dict[str, Any]:
    outlines = {
        "direct_knowledge": "Answer directly first, then briefly explain.",
        "coding_interview": "State the approach simply. If needed, give a short code example.",
        "system_design": "Clarify scope, then describe components, flow, and tradeoffs.",
        "small_talk": "Keep it natural and short.",
        "unknown": "Answer directly without forcing background.",
    }
    return {
        "question_type": route,
        "answer_outline": outlines.get(route, "Answer directly and clearly."),
        "matched_themes": [],
    }

# =========================================================
# Answer monitor
# =========================================================
def _build_answer_monitor(text: str) -> Dict[str, Any]:
    clean = clean_transcript_text(text)
    low = clean.lower()

    signals = []
    if len(clean.split()) > 45:
        signals.append("Your answer is getting long. Tighten it to the main point.")
    if re.search(r"\b(well|so|like|kind of|sort of)\b", low):
        signals.append("You are using filler. Try a cleaner opening.")
    if any(p in low for p in ["excited to work in this company", "i am excited", "impactful results"]):
        signals.append("This sounds generic. Add one specific reason or example.")
    if not re.search(r"\b(api|apis|python|testing|automation|latency|project|metrics?|result|built|implemented)\b", low):
        signals.append("You may be missing a concrete detail from your background.")
    if not re.search(r"\b(\d+|percent|ms|weeks|months)\b", low):
        signals.append("If possible, add one measurable detail or concrete outcome.")

    if not signals:
        signals.append("Your answer is on track. Keep it concise and specific.")

    return {
        "signals": signals[:3],
        "clean_answer": clean,
    }

# =========================================================
# Notes
# =========================================================
def _update_notes(state: AgentState, *, speaker: str, text: str) -> None:
    t = (text or "").strip()
    if not t:
        return

    bullet = f"{speaker}: {t}"
    state.notes["bullets"].append(bullet)
    state.notes["bullets"] = state.notes["bullets"][-60:]

    tl = t.lower()
    if any(p in tl for p in ["we decided", "decision:", "let's go with", "we will proceed with"]):
        state.notes["decisions"].append(bullet)
        state.notes["decisions"] = state.notes["decisions"][-20:]

    if any(p in tl for p in ["action item", "todo", "we need to", "we should", "i will", "i'll"]):
        state.notes["action_items"].append(bullet)
        state.notes["action_items"] = state.notes["action_items"][-30:]

    if "follow up" in tl or "follow-up" in tl:
        state.notes["follow_ups"].append(bullet)
        state.notes["follow_ups"] = state.notes["follow_ups"][-30:]

    topics = state.notes.setdefault("topics", [])
    intent = state.intent_history[-1] if state.intent_history else "unknown"
    coarse = "other"
    if any(k in tl for k in ["deadline", "date", "next week", "quarter"]):
        coarse = "timing"
    elif any(k in tl for k in ["metric", "kpi", "impact", "result", "%", "percent"]):
        coarse = "impact"
    elif any(k in tl for k in ["roadmap", "plan", "strategy"]):
        coarse = "planning"

    topics.append({"intent": intent, "tag": coarse})
    state.notes["topics"] = topics[-80:]
    state.notes["current_topic"] = coarse

    if "?" in t:
        state.notes.setdefault("open_questions", []).append(bullet)
        state.notes["open_questions"] = state.notes["open_questions"][-15:]

def _enhance_notes_with_llm(state: AgentState) -> None:
    if not USE_OAI_NOTES:
        return
    if not state.notes.get("bullets"):
        return

    client = _oai_client()
    bullets = state.notes.get("bullets", [])[-12:]

    prompt = (
        "You are a concise meeting notes assistant.\n"
        "Return JSON with keys: summary, current_topic, open_questions, action_items, decisions, follow_ups, topics.\n"
        "Be conservative."
    )

    try:
        rsp = client.chat.completions.create(
            model=ANSWER_MODEL_NOTES,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"bullets": bullets})}
            ],
            max_tokens=350,
            temperature=0.2,
        )
        raw = rsp.choices[0].message.content or "{}"
        data = json.loads(raw)

        u = getattr(rsp, "usage", None)
        if u is not None:
            _record_usage(
                state,
                model=ANSWER_MODEL_NOTES,
                prompt_tokens=_safe_int(getattr(u, "prompt_tokens", 0)),
                completion_tokens=_safe_int(getattr(u, "completion_tokens", 0)),
                feature="notes_llm",
            )

        if "summary" in data:
            state.notes["summary_so_far"] = data["summary"]
        if "current_topic" in data and data["current_topic"]:
            state.notes["current_topic"] = str(data["current_topic"])
        if "open_questions" in data:
            state.notes["open_questions"] = [str(q) for q in data.get("open_questions", [])][-15:]
        if "action_items" in data:
            state.notes["action_items"] = [str(x) for x in data.get("action_items", [])][-30:]
        if "decisions" in data:
            state.notes["decisions"] = [str(x) for x in data.get("decisions", [])][-20:]
        if "follow_ups" in data:
            state.notes["follow_ups"] = [str(x) for x in data.get("follow_ups", [])][-20:]
        if "topics" in data:
            cur = [t for t in state.notes.get("topics", []) if isinstance(t, dict)]
            for label in data.get("topics", []):
                cur.append({"intent": state.intent_history[-1] if state.intent_history else "unknown", "tag": str(label)})
            state.notes["topics"] = cur[-80:]

    except Exception as e:
        print("[notes-llm] error:", e, flush=True)

def _notes_payload(state: AgentState) -> Dict[str, Any]:
    n = state.notes
    return {
        "bullets": n.get("bullets", [])[-60:],
        "topics": n.get("topics", [])[-40:],
        "action_items": n.get("action_items", [])[-30:],
        "decisions": n.get("decisions", [])[-20:],
        "follow_ups": n.get("follow_ups", [])[-20:],
        "summary_so_far": n.get("summary_so_far"),
        "current_topic": n.get("current_topic"),
        "open_questions": n.get("open_questions", [])[-15:],
    }


def _interpretation_for_emit(interp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "confidence": interp["confidence"],
        "is_question": interp["is_question"],
        "reason": interp["reason"],
        "should_coach": interp["should_coach"],
        "missing_context": interp["missing_context"],
        "cleaned_text": interp["cleaned_text"],
    }


def _interpreted_question_for_emit(interp: Dict[str, Any]) -> Optional[str]:
    q = interp.get("interpreted_question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return None


def _latency_ms(latency: Dict[str, Any], start_key: str, end_key: str) -> Optional[float]:
    start = latency.get(start_key)
    end = latency.get(end_key)
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return round(max(0.0, end - start) * 1000.0, 2)


def _latency_with_metrics(latency: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(latency or {})
    out.setdefault("backend_sent_at", time.time())
    out["backend_total_ms"] = _latency_ms(out, "backend_received_at", "backend_sent_at")
    out["interpreter_ms"] = _latency_ms(out, "interpreter_start_at", "interpreter_end_at")
    out["router_ms"] = _latency_ms(out, "router_start_at", "router_end_at")
    out["retrieval_ms"] = _latency_ms(out, "retrieval_start_at", "retrieval_end_at")
    out["llm_ms"] = _latency_ms(out, "llm_start_at", "llm_end_at")
    out["end_to_end_estimate_ms"] = _latency_ms(out, "frontend_sent_at", "backend_sent_at")
    return out


# =========================================================
# Emit helper
# =========================================================
def _emit_payload(
    *,
    kind: str,
    response_type: str,
    speaker: str,
    transcript: str,
    usage: Dict[str, Any],
    coach_speculative: Optional[Dict[str, Any]] = None,
    coach_final: Optional[Dict[str, Any]] = None,
    notes_final: Optional[Dict[str, Any]] = None,
    answer_monitor: Optional[Dict[str, Any]] = None,
    skipped_reason: Optional[str] = None,
    audio_source: Optional[str] = None,
    speaker_mode: Optional[str] = None,
    route_info: Optional[Dict[str, Any]] = None,
    state: Optional["AgentState"] = None,
    interpreted_question: Optional[str] = None,
    interpretation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = {
        "kind": kind,
        "response_type": response_type,
        "speaker": speaker,
        "transcript": transcript,
        "usage": usage,
    }
    if coach_speculative is not None:
        out["coach_speculative"] = coach_speculative
    if coach_final is not None:
        out["coach_final"] = coach_final
    if notes_final is not None:
        out["notes_final"] = notes_final
    if answer_monitor is not None:
        out["answer_monitor"] = answer_monitor
    if skipped_reason is not None:
        out["skipped_reason"] = skipped_reason
    if audio_source is not None:
        out["audio_source"] = audio_source
    if speaker_mode is not None:
        out["speaker_mode"] = speaker_mode
    if route_info is not None:
        out["route"] = route_info.get("route")
        out["route_confidence"] = route_info.get("confidence")
        if route_info.get("memory_filters") is not None:
            out["memory_filters"] = route_info.get("memory_filters")
    if interpreted_question is not None:
        out["interpreted_question"] = interpreted_question
    if interpretation is not None:
        out["interpretation"] = interpretation
    if state is not None:
        rc = getattr(state, "retrieval_cache", None) or {}
        doc_ids = rc.get("doc_ids")
        if doc_ids:
            out["retrieval_source"] = doc_ids
    out["latency"] = _latency_with_metrics(getattr(state, "current_latency", None) if state is not None else None)
    return out


def _log_processed_turn(
    state: AgentState,
    out: Dict[str, Any],
    *,
    raw_text: str,
    clean_text: str,
    interpretation: Dict[str, Any],
) -> Dict[str, Any]:
    if out.get("kind") != "final":
        return out
    response_type = out.get("response_type")
    if response_type not in {"coach_final", "answer_monitor", "notes_final"}:
        return out

    try:
        log_session_event(
            state.session_id,
            response_type,
            {
                "mode": getattr(state, "mode", None),
                "speaker": out.get("speaker"),
                "audio_source": out.get("audio_source"),
                "speaker_mode": out.get("speaker_mode"),
                "raw_transcript": raw_text,
                "clean_transcript": clean_text or out.get("transcript"),
                "interpreted_question": out.get("interpreted_question"),
                "interpretation": interpretation,
                "route": out.get("route"),
                "route_confidence": out.get("route_confidence"),
                "memory_filters": out.get("memory_filters"),
                "retrieval_source": out.get("retrieval_source"),
                "response_type": response_type,
                "usage": out.get("usage"),
                "latency": out.get("latency"),
            },
        )
    except Exception as e:
        print("[session-log] error:", e, flush=True)
    return out

# =========================================================
# Main processor
# =========================================================
def process_turn(state: AgentState, *, kind: str = "final") -> Optional[Dict[str, Any]]:
    state.usage["turn"] = {"by_model": {}, "cost_usd": 0.0}

    mode = getattr(state, "mode", "coach") or "coach"
    speaker = getattr(state, "buffer_speaker", "Speaker 1")
    _asrc = getattr(state, "last_audio_source", None)
    _smode = getattr(state, "last_speaker_mode", None)
    raw_text = state.buffer_text.strip()
    clean_text = clean_transcript_text(raw_text)
    latency = dict(getattr(state, "current_latency", {}) or {})
    latency["interpreter_start_at"] = time.time()
    state.current_latency = latency
    interp = interpret_transcript(
        raw_text,
        cleaned_text=clean_text if clean_text else None,
        speaker=speaker,
        mode=mode,
    )
    latency["interpreter_end_at"] = time.time()
    state.current_latency = latency
    interp_emit = _interpretation_for_emit(interp)
    iq_emit = _interpreted_question_for_emit(interp)

    route_query = (interp.get("interpreted_question") or interp.get("cleaned_text") or "").strip()
    route_info = {"route": "unknown", "confidence": 0.5, "rationale": ""}
    if route_query:
        latency["router_start_at"] = time.time()
        state.current_latency = latency
        route_info = route_question(route_query, state=state)
        latency["router_end_at"] = time.time()
        state.current_latency = latency
        state.intent_history.append(route_info.get("route", "unknown"))
    else:
        state.intent_history.append("unknown")

    note_line = interp["cleaned_text"].strip() if interp["cleaned_text"].strip() else (clean_text or raw_text)
    _update_notes(state, speaker=speaker, text=note_line)
    if kind == "final" and USE_OAI_NOTES:
        latency["llm_start_at"] = time.time()
        state.current_latency = latency
        _enhance_notes_with_llm(state)
        latency["llm_end_at"] = time.time()
        state.current_latency = latency

    display_transcript = clean_text or raw_text

    def emit_logged(**kwargs: Any) -> Dict[str, Any]:
        out = _emit_payload(**kwargs)
        return _log_processed_turn(
            state,
            out,
            raw_text=raw_text,
            clean_text=clean_text,
            interpretation=interp_emit,
        )

    # Notes mode
    if mode == "notes":
        return emit_logged(
            kind=kind,
            response_type="notes_final",
            speaker=speaker,
            transcript=display_transcript,
            usage=state.usage,
            notes_final={"notes": _notes_payload(state)},
            audio_source=_asrc,
            speaker_mode=_smode,
            route_info=route_info,
            state=state,
            interpreted_question=iq_emit,
            interpretation=interp_emit,
        )

    # Coach mode, when candidate is speaking -> answer monitor, do not replace coaching
    if _is_me_speaker(speaker):
        monitor = _build_answer_monitor(clean_text or raw_text)
        return emit_logged(
            kind=kind,
            response_type="answer_monitor",
            speaker=speaker,
            transcript=display_transcript,
            usage=state.usage,
            answer_monitor=monitor,
            audio_source=_asrc,
            speaker_mode=_smode,
            route_info=route_info,
            state=state,
            interpreted_question=iq_emit,
            interpretation=interp_emit,
        )

    # Coach mode: interviewer path — soft junk gate only (interpretation handles fillers)
    coach_route_text = (interp.get("interpreted_question") or interp.get("cleaned_text") or "").strip()
    if interp.get("reason") == "junk_fragment" or not coach_route_text:
        return emit_logged(
            kind=kind,
            response_type="coach_skipped",
            speaker=speaker,
            transcript=display_transcript,
            usage=state.usage,
            skipped_reason="junk_fragment",
            audio_source=_asrc,
            speaker_mode=_smode,
            route_info=route_info,
            state=state,
            interpreted_question=iq_emit,
            interpretation=interp_emit,
        )

    if len(coach_route_text.split()) > 60:
        return emit_logged(
            kind=kind,
            response_type="coach_skipped",
            speaker=speaker,
            transcript=display_transcript,
            usage=state.usage,
            skipped_reason="too_long",
            audio_source=_asrc,
            speaker_mode=_smode,
            route_info=route_info,
            state=state,
            interpreted_question=iq_emit,
            interpretation=interp_emit,
        )

    route = route_info.get("route", "unknown")
    conf = route_info.get("confidence", 0.5)
    memory_filters = memory_filters_for_route(route)
    route_info = dict(route_info)
    route_info["memory_filters"] = memory_filters

    # Speculative
    if kind == "speculative":
        if route in BACKGROUND_ROUTES:
            latency["retrieval_start_at"] = time.time()
            state.current_latency = latency
            ctx = retrieve_context(coach_route_text, k=4, state=state, filters=memory_filters)
            latency["retrieval_end_at"] = time.time()
            state.current_latency = latency
            state.retrieval_cache = {"last_query": coach_route_text, "doc_ids": [c["id"] for c in ctx]}
            spec = _build_speculative_background(route, ctx)
        else:
            spec = _build_speculative_direct(route)
            state.retrieval_cache = {}

        return emit_logged(
            kind="speculative",
            response_type="coach_speculative",
            speaker=speaker,
            transcript=coach_route_text,
            usage=state.usage,
            coach_speculative=spec,
            audio_source=_asrc,
            speaker_mode=_smode,
            route_info=route_info,
            state=state,
            interpreted_question=iq_emit,
            interpretation=interp_emit,
        )

    # Final
    if route in BACKGROUND_ROUTES:
        latency["retrieval_start_at"] = time.time()
        state.current_latency = latency
        ctx = retrieve_context(coach_route_text, k=6, state=state, filters=memory_filters)
        latency["retrieval_end_at"] = time.time()
        state.current_latency = latency
        state.retrieval_cache = {"last_query": coach_route_text, "doc_ids": [c["id"] for c in ctx]}
        latency["llm_start_at"] = time.time()
        state.current_latency = latency
        draft = _draft_background_answer(route, coach_route_text, ctx, state.prefs, state=state)
        latency["llm_end_at"] = time.time()
        state.current_latency = latency
        score = round(0.65 * conf + 0.35 * (ctx[0]["score"] if ctx else 0.0), 2) if ctx else round(conf, 2)
    else:
        latency["llm_start_at"] = time.time()
        state.current_latency = latency
        draft = _draft_direct_answer(route, coach_route_text, state=state)
        latency["llm_end_at"] = time.time()
        state.current_latency = latency
        score = round(conf, 2)
        state.retrieval_cache = {}

    return emit_logged(
        kind="final",
        response_type="coach_final",
        speaker=speaker,
        transcript=coach_route_text,
        usage=state.usage,
        coach_final={
            "suggestions": draft.get("options", []),
            "follow_up": draft.get("follow_up", ""),
            "bridge": draft.get("bridge", ""),
            "confidence": score,
            "context_ids": draft.get("ctx_ids", []),
        },
        audio_source=_asrc,
        speaker_mode=_smode,
        route_info=route_info,
        state=state,
        interpreted_question=iq_emit,
        interpretation=interp_emit,
    )
