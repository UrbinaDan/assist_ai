from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time, os, json, re
import numpy as np
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
from app.retriever import Retriever

# -------- OpenAI client (reads OPENAI_API_KEY from env) --------
_oai: Optional[OpenAI] = None
def _oai_client() -> OpenAI:
    global _oai
    if _oai is None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt")
        _oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _oai

# -------- Embedding via OpenAI (keeps Codespace tiny) --------
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim

def embed_query(text: str, *, state: Optional["AgentState"] = None) -> np.ndarray:
    client = _oai_client()
    rsp = client.embeddings.create(model=EMBED_MODEL, input=text)
    u = getattr(rsp, "usage", None)
    if state is not None and u is not None:
        # Embeddings charge only input tokens; prefer prompt_tokens for consistency
        # with chat models, but fall back to total_tokens if needed.
        emb_prompt = _safe_int(
            getattr(u, "prompt_tokens", getattr(u, "total_tokens", 0))
        )
        _record_usage(
            state,
            model=EMBED_MODEL,
            prompt_tokens=emb_prompt,
            completion_tokens=0,
            feature="embed",
        )
    emb = rsp.data[0].embedding
    return np.array(emb, dtype="float32")

# -------- Minimal runtime state --------
@dataclass
class AgentState:
    session_id: str
    buffer_text: str = ""
    buffer_speaker: str = "Speaker 1"
    mode: str = "coach"  # "coach" | "notes" (or "hybrid" later)
    created_at: float = field(default_factory=lambda: time.time())
    last_seen_at: float = field(default_factory=lambda: time.time())
    last_emit_ts: float = 0.0
    last_token_ts: float = 0.0
    intent_history: List[str] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    prefs: Dict[str, Any] = field(default_factory=lambda: {
        "tone": "concise",
        "target_len": "2 sentences",
        "avoid": ["jargon", "overpromising"]
    })
    retrieval_cache: Dict[str, Any] = field(default_factory=dict)
    last_final_hash: str = ""
    pending_speaker_flush: bool = False
    next_speaker: Optional[str] = None
    next_text_delta: Optional[str] = None
    next_ts: Optional[float] = None

    usage: Dict[str, Any] = field(default_factory=lambda: {
        "by_model": {},  # model -> {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        "by_feature": {},  # feature -> same shape
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

    # Logical speaker roles per session, e.g. {"Me": "candidate", "Interviewer": "interviewer"}
    roles: Dict[str, str] = field(default_factory=lambda: {})

class EndOfThought:
    def __init__(self, pause_ms: int = 900, stable_n: int = 2, min_words: int = 10, max_words: int = 60):
        self.pause_ms = pause_ms
        self.stable_n = stable_n
        self.min_words = min_words
        self.max_words = max_words

    def intent_stable(self, intents: List[str]) -> bool:
        if len(intents) < self.stable_n:
            return False
        return len(set(intents[-self.stable_n:])) == 1

    def strong_punct(self, text: str) -> bool:
        return text.strip().endswith(("?", ".", "!"))

    def should_emit(self, state: AgentState) -> bool:
        now = time.time()
        paused = (now - state.last_token_ts) * 1000 >= self.pause_ms
        punct = self.strong_punct(state.buffer_text)
        words = len(state.buffer_text.split())
        longish = words >= self.min_words
        too_long = words >= self.max_words
        # Latency-sensitive: do not require an LLM classifier to decide turn boundaries.
        # Emit when there's a pause and the buffer looks like a meaningful unit.
        return (paused and (punct or longish)) or too_long

# -------- Classifier via OpenAI (Chat Completions JSON mode) --------
INTENTS = ["small_talk", "behavioral", "technical", "scheduling", "compensation", "unknown"]
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-4.1-nano")
DRAFTER_MODEL = os.getenv("DRAFTER_MODEL", "gpt-4o-mini")

PRICES_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    # Source: OpenAI pricing docs (Mar 2026). If you change models, add them here.
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
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
        row_delta_cost = (
            (_safe_int(prompt_tokens) / 1_000_000.0) * price.get("input", 0.0)
            + (_safe_int(completion_tokens) / 1_000_000.0) * price.get("output", 0.0)
        )
        state.usage["cost_usd_total"] = round(float(state.usage.get("cost_usd_total", 0.0)) + row_delta_cost, 6)

        turn = state.usage.setdefault("turn", {"by_model": {}, "cost_usd": 0.0})
        turn["cost_usd"] = round(float(turn.get("cost_usd", 0.0)) + row_delta_cost, 6)
        t_by_model = turn.setdefault("by_model", {})
        trow = t_by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
        trow["prompt_tokens"] += _safe_int(prompt_tokens)
        trow["completion_tokens"] += _safe_int(completion_tokens)
        trow["total_tokens"] += _safe_int(prompt_tokens) + _safe_int(completion_tokens)
        trow["cost_usd"] = round(float(trow.get("cost_usd", 0.0)) + row_delta_cost, 6)

        # Aggregate by feature as well (coach vs notes vs embed, etc.)
        by_feature = state.usage.setdefault("by_feature", {})
        f = by_feature.setdefault(feature, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
        f["prompt_tokens"] += _safe_int(prompt_tokens)
        f["completion_tokens"] += _safe_int(completion_tokens)
        f["total_tokens"] += _safe_int(prompt_tokens) + _safe_int(completion_tokens)
        f["cost_usd"] = round(float(f.get("cost_usd", 0.0)) + row_delta_cost, 6)

def classify_question(text: str, *, state: Optional[AgentState] = None) -> Dict[str, Any]:
    client = _oai_client()
    system_msg = (
        "You classify a single, short utterance from a live interview or conversation.\n"
        f"Return a compact JSON object with keys: intent (one of {INTENTS}), entities, confidence.\n"
        "entities must include: company (string|null), role (string|null), "
        "skills (string[]), numbers (string[]), dates (string[]), times (string[]).\n"
        "Be conservative: if unsure, intent='unknown' and confidence <= 0.6."
    )
    try:
        rsp = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Utterance: {text}"}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        u = getattr(rsp, "usage", None)
        if state is not None and u is not None:
            _record_usage(
                state,
                model=CLASSIFIER_MODEL,
                prompt_tokens=_safe_int(getattr(u, "prompt_tokens", 0)),
                completion_tokens=_safe_int(getattr(u, "completion_tokens", 0)),
                feature="classifier",
            )
        raw = rsp.choices[0].message.content or "{}"
        data = json.loads(raw)

        intent = data.get("intent", "unknown")
        if intent not in INTENTS:
            intent = "unknown"
        entities = data.get("entities") or {}
        entities.setdefault("company", None)
        entities.setdefault("role", None)
        for k in ("skills", "numbers", "dates", "times"):
            entities.setdefault(k, [])
        confidence = float(data.get("confidence", 0.5))
        return {
            "intent": intent,
            "entities": entities,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    except Exception as e:
        print("[classifier] error:", e, flush=True)
        return {
            "intent": "unknown",
            "entities": {"company": None, "role": None, "skills": [], "numbers": [], "dates": [], "times": []},
            "confidence": 0.4,
            "error": str(e),
        }

# -------- Retriever (injected at startup by server.py) --------
RETRIEVER: Optional[Retriever] = None

def retrieve_context(query: str, k: int = 4, *, state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
    if RETRIEVER is None: 
        return []
    qv = embed_query(query, state=state)
    return RETRIEVER.search(qv, k=k)

# -------- Drafting: OpenAI (optional) or local template --------
USE_OAI_DRAFTER = os.getenv("USE_OAI_DRAFTER", "false").lower() in ("1","true","yes")

def _draft_with_openai(
    text: str,
    ctx: List[Dict[str, Any]],
    prefs: Dict[str, Any],
    *,
    state: Optional[AgentState] = None,
) -> Dict[str, Any]:
    client = _oai_client()
    ctx_txt = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(ctx[:4])) or "No context."
    system_msg = (
        "You are a real-time conversation assistant and note taker.\n"
        "If the user asks for answers grounded in their background, ONLY use the provided context snippets.\n"
        "If context is missing or thin, do NOT guess details—ask 1-2 targeted questions instead.\n"
        "Return JSON with keys: options (2-3 strings), follow_up (string), bridge (string).\n"
        "First-person, concise, each option <= 2 sentences. If context is weak, say 'Context is limited'."
    )
    try:
        rsp = client.chat.completions.create(
            model=DRAFTER_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Utterance: {text}\n\nContext:\n{ctx_txt}\n\nPrefs: {prefs}"}
            ],
            max_tokens=300,
            temperature=0.3,
        )
        u = getattr(rsp, "usage", None)
        if state is not None and u is not None:
            _record_usage(
                state,
                model=DRAFTER_MODEL,
                prompt_tokens=_safe_int(getattr(u, "prompt_tokens", 0)),
                completion_tokens=_safe_int(getattr(u, "completion_tokens", 0)),
                feature=f"{getattr(state, 'mode', 'coach')}_drafter",
            )
        raw = rsp.choices[0].message.content or "{}"
        data = json.loads(raw)
        options = (data.get("options") or [])[:3]
        follow_up = data.get("follow_up") or "Would it help to go deeper on metrics or rollout?"
        bridge = data.get("bridge") or "Happy to share specifics."
        return {"options": options, "follow_up": follow_up, "bridge": bridge,
                "ctx_ids": [c["id"] for c in ctx]}
    except Exception as e:
        print("[drafter] error:", e, flush=True)
        return _draft_local(text, ctx, prefs)

def _draft_local(text: str, ctx: List[Dict[str, Any]], prefs: Dict[str, Any]) -> Dict[str, Any]:
    if not ctx:
        return {
            "options": [
                "Context is limited from your background docs—tell me which experience you want to use (project name + 1 line impact), and I’ll shape a strong answer.",
                "If you share 2–3 facts (your role, constraint, measurable result), I’ll draft a concise response and a follow-up question."
            ],
            "follow_up": "What role/company is this for, and which project should I anchor on?",
            "bridge": "Once I have those details, I can make it specific and credible.",
            "ctx_ids": []
        }
    snips = [c["text"] for c in ctx[:2]]
    base1 = "From a recent project, I led the effort to"
    base2 = "My approach is to"
    if snips:        base1 += f" — {snips[0][:160].rstrip()}..."
    if len(snips)>1: base2 += f" — {snips[1][:160].rstrip()}..."
    return {
        "options": [
            base1 + " I can apply the same steps here.",
            base2 + " and iterate quickly. That’s how we delivered impact."
        ],
        "follow_up": "Would it help to go deeper on metrics or the rollout plan?",
        "bridge": "Happy to share specifics if you want details.",
        "ctx_ids": [c["id"] for c in ctx]
    }

def draft_answer(
    text: str,
    ctx: List[Dict[str, Any]],
    prefs: Dict[str, Any],
    *,
    state: Optional[AgentState] = None,
    fast_only: bool = False,
) -> Dict[str, Any]:
    # For speculative emits, stay on the local path to avoid extra latency.
    if fast_only or not USE_OAI_DRAFTER:
        return _draft_local(text, ctx, prefs)
    return _draft_with_openai(text, ctx, prefs, state=state)

def refine_answer(draft: Dict[str, Any], ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    return draft

def style_adapter(draft: Dict[str, Any], prefs: Dict[str, Any]) -> Dict[str, Any]:
    def clamp(s: str) -> str:
        s = s.replace("utilize", "use")
        if len(s.split()) > 40:
            s = " ".join(s.split()[:40]) + "..."
        return s
    out = draft.copy()
    out["options"]   = [clamp(o) for o in draft["options"]]
    out["follow_up"] = clamp(draft["follow_up"])
    out["bridge"]    = clamp(draft["bridge"])
    return out

def confidence(ctx: List[Dict[str, Any]], cls_conf: float) -> float:
    top = ctx[0]["score"] if ctx else 0.0
    return round(0.5*cls_conf + 0.5*top, 2)


# -------- Lightweight speculative coach: no drafter, just classifier + retrieval summary --------
def _build_speculative_coach(cls: Dict[str, Any], ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Question type, one-line outline, and matched themes from retrieval. No full answer."""
    intent = cls.get("intent", "unknown")
    outline = "Use your background stories that match the question."
    if ctx:
        first = (ctx[0].get("text") or "")[:200].rstrip()
        if first:
            outline = first + ("..." if len((ctx[0].get("text") or "")) > 200 else "")
    themes = [c.get("text", "")[:80].rstrip() + ("..." if len(c.get("text", "")) > 80 else "") for c in ctx[:3] if c.get("text")]
    return {
        "question_type": intent,
        "answer_outline": outline,
        "matched_themes": themes,
    }


# -------- Framework-based draft for technical/conceptual (no retrieval-heavy personalization) --------
FRAMEWORK_INTENTS = ("technical", "unknown")


def _draft_framework(text: str, intent: str, *, state: Optional[AgentState] = None) -> Dict[str, Any]:
    """STAR / system design / conceptual outline. Used when retrieval is not the right fit."""
    # Lightweight: return a short framework reminder; no LLM call by default to keep latency low.
    if intent == "technical":
        return {
            "options": [
                "Structure: context, approach, trade-offs, result. Give one concrete example.",
                "If system design: clarify scope, then components, data flow, scale.",
            ],
            "follow_up": "Want to go deeper on one part or add metrics?",
            "bridge": "Happy to go into more detail.",
            "ctx_ids": [],
        }
    return {
        "options": [
            "Use a clear structure: situation, your role, action, result.",
            "Anchor on one example and one takeaway.",
        ],
        "follow_up": "Should we add a second example or a lesson learned?",
        "bridge": "I can expand on any part.",
        "ctx_ids": [],
    }


def process_turn(state: AgentState, *, kind: str = "final") -> Optional[Dict[str, Any]]:
    # Reset per-turn usage ledger
    state.usage["turn"] = {"by_model": {}, "cost_usd": 0.0}

    cls = classify_question(state.buffer_text, state=state)
    state.intent_history.append(cls["intent"])
    ctx = retrieve_context(state.buffer_text, k=4, state=state)
    state.retrieval_cache = {"last_query": state.buffer_text, "doc_ids": [c["id"] for c in ctx]}
    mode = getattr(state, "mode", "coach") or "coach"
    speaker = getattr(state, "buffer_speaker", "Speaker 1")

    _update_notes(state, speaker=speaker, text=state.buffer_text)
    if kind == "final":
        _enhance_notes_with_llm(state)

    # -------- Notes mode: explicit notes_final payload --------
    if mode == "notes":
        notes_obj = _notes_payload(state)
        return _emit_payload(
            kind=kind,
            response_type="notes_final",
            speaker=speaker,
            transcript=state.buffer_text,
            usage=state.usage,
            notes_final={"notes": notes_obj},
        )

    # -------- Coach mode --------
    if kind == "speculative":
        # Lightweight: no drafter; only classifier + retrieval summary.
        spec = _build_speculative_coach(cls, ctx)
        return _emit_payload(
            kind="speculative",
            response_type="coach_speculative",
            speaker=speaker,
            transcript=state.buffer_text,
            usage=state.usage,
            coach_speculative=spec,
        )

    # Final coach: route behavioral/background -> retrieval; technical/conceptual -> framework
    intent = cls.get("intent", "unknown")
    if intent in FRAMEWORK_INTENTS:
        draft = _draft_framework(state.buffer_text, intent, state=state)
    else:
        draft = draft_answer(
            state.buffer_text, ctx, state.prefs, state=state, fast_only=False,
        )
    final = style_adapter(refine_answer(draft, ctx), state.prefs)
    score = confidence(ctx, cls.get("confidence", 0.5))
    return _emit_payload(
        kind="final",
        response_type="coach_final",
        speaker=speaker,
        transcript=state.buffer_text,
        usage=state.usage,
        coach_final={
            "suggestions": final["options"],
            "follow_up": final["follow_up"],
            "bridge": final["bridge"],
            "confidence": score,
            "context_ids": final.get("ctx_ids", []),
        },
    )


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
    return out


def _update_notes(state: AgentState, *, speaker: str, text: str) -> None:
    t = (text or "").strip()
    if not t:
        return

    # Lightweight notes so we don't add extra model calls.
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

    # Extremely lightweight topic labeling: intent plus a coarse tag.
    topics = state.notes.setdefault("topics", [])
    intent = state.intent_history[-1] if state.intent_history else "unknown"
    coarse = "other"
    if any(k in tl for k in ["deadline", "date", "next week", "q", "quarter"]):
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


USE_OAI_NOTES = os.getenv("USE_OAI_NOTES", "false").lower() in ("1", "true", "yes")


def _enhance_notes_with_llm(state: AgentState) -> None:
    """Optional, LLM-based notes refinement. Only runs when USE_OAI_NOTES is true.

    It summarizes the last few bullets and refines action items/decisions/follow-ups/topics.
    """
    if not USE_OAI_NOTES:
        return
    if not state.notes.get("bullets"):
        return

    client = _oai_client()
    bullets = state.notes.get("bullets", [])[-12:]
    existing_actions = state.notes.get("action_items", [])[-12:]
    existing_decisions = state.notes.get("decisions", [])[-12:]
    prompt = (
        "You are a concise meeting notes assistant.\n"
        "Given recent bullets, action items, and decisions, return JSON with keys:\n"
        "summary (string, <= 3 sentences),\n"
        "current_topic (string, one short label),\n"
        "open_questions (array of strings, unresolved questions from the conversation),\n"
        "action_items (array of {text, owner?, due?}),\n"
        "decisions (array of strings),\n"
        "follow_ups (array of strings),\n"
        "topics (array of short labels).\n"
        "Be conservative and never hallucinate owners or dates; leave them null when unsure."
    )
    try:
        rsp = client.chat.completions.create(
            model=os.getenv("NOTES_MODEL", "gpt-4.1-nano"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    "bullets": bullets,
                    "action_items": existing_actions,
                    "decisions": existing_decisions,
                })}
            ],
            max_tokens=400,
            temperature=0.2,
        )
        raw = rsp.choices[0].message.content or "{}"
        data = json.loads(raw)
        if "summary" in data:
            state.notes["summary_so_far"] = data["summary"]
        if "action_items" in data:
            # Store plain strings for now for UI simplicity.
            items = []
            for it in data.get("action_items", []):
                if isinstance(it, str):
                    items.append(it)
                elif isinstance(it, dict) and it.get("text"):
                    owner = it.get("owner")
                    due = it.get("due")
                    extra = []
                    if owner:
                        extra.append(f"owner={owner}")
                    if due:
                        extra.append(f"due={due}")
                    suffix = f" ({', '.join(extra)})" if extra else ""
                    items.append(f"{it['text']}{suffix}")
            state.notes["action_items"] = items[-40:]
        if "decisions" in data:
            state.notes["decisions"] = [str(d) for d in data.get("decisions", [])][-40:]
        if "follow_ups" in data:
            state.notes["follow_ups"] = [str(f) for f in data.get("follow_ups", [])][-40:]
        if "topics" in data:
            cur = [t for t in state.notes.get("topics", []) if isinstance(t, dict)]
            for label in data.get("topics", []):
                cur.append({"intent": state.intent_history[-1] if state.intent_history else "unknown", "tag": str(label)})
            state.notes["topics"] = cur[-80:]
        if "current_topic" in data and data["current_topic"]:
            state.notes["current_topic"] = str(data["current_topic"])
        if "open_questions" in data:
            state.notes["open_questions"] = [str(q) for q in data.get("open_questions", [])][-15:]
    except Exception as e:
        print("[notes-llm] error:", e, flush=True)
