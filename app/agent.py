from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time, os, json, re
import numpy as np
from openai import OpenAI
from app.retriever import Retriever

# -------- OpenAI client (reads OPENAI_API_KEY from env) --------
_oai: Optional[OpenAI] = None
def _oai_client() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _oai

# -------- Embedding via OpenAI (keeps Codespace tiny) --------
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim

def embed_query(text: str) -> np.ndarray:
    client = _oai_client()
    emb = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    return np.array(emb, dtype="float32")

# -------- Minimal runtime state --------
@dataclass
class AgentState:
    session_id: str
    buffer_text: str = ""
    buffer_speaker: str = "Speaker 1"
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

    usage: Dict[str, Any] = field(default_factory=lambda: {
        "by_model": {},  # model -> {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        "cost_usd_total": 0.0,
        "last_turn": None,
    })

    notes: Dict[str, Any] = field(default_factory=lambda: {
        "bullets": [],
        "action_items": [],
        "decisions": [],
    })

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
) -> None:
    by_model = state.usage.setdefault("by_model", {})
    row = by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    row["prompt_tokens"] += _safe_int(prompt_tokens)
    row["completion_tokens"] += _safe_int(completion_tokens)
    row["total_tokens"] += _safe_int(prompt_tokens) + _safe_int(completion_tokens)

    price = PRICES_PER_1M_TOKENS.get(model)
    if price:
        state.usage["cost_usd_total"] = round(
            float(state.usage.get("cost_usd_total", 0.0))
            + (row_delta_cost := (
                (_safe_int(prompt_tokens) / 1_000_000.0) * price.get("input", 0.0)
                + (_safe_int(completion_tokens) / 1_000_000.0) * price.get("output", 0.0)
            )),
            6,
        )
        # keep the last delta cost available for per-turn display
        state.usage["last_turn"] = {"model": model, "cost_usd": round(row_delta_cost, 6)}

def classify_question(text: str) -> Dict[str, Any]:
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
        if u is not None:
            # prompt_tokens / completion_tokens are standard; fall back if missing
            _record_usage(
                AgentState(session_id="__tmp__") if False else state_placeholder,  # type: ignore[name-defined]
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

def _draft_with_openai(text: str, ctx: List[Dict[str, Any]], prefs: Dict[str, Any]) -> Dict[str, Any]:
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
                "I can walk through a recent end-to-end project covering goal, constraints, actions, and results.",
                "Do you prefer a technical performance example or a cross-team delivery story?"
            ],
            "follow_up": "Do you want the short version or a deeper STAR breakdown?",
            "bridge": "Happy to tailor the story to what’s most relevant.",
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

def draft_answer(text: str, ctx: List[Dict[str, Any]], prefs: Dict[str, Any]) -> Dict[str, Any]:
    return _draft_with_openai(text, ctx, prefs) if USE_OAI_DRAFTER else _draft_local(text, ctx, prefs)

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

def process_turn(state: AgentState) -> Optional[Dict[str, Any]]:
    # Reset per-turn usage delta
    state.usage["last_turn"] = None

    cls = classify_question(state.buffer_text)
    state.intent_history.append(cls["intent"])
    ctx = retrieve_context(state.buffer_text, k=4, state=state)
    state.retrieval_cache = {"last_query": state.buffer_text, "doc_ids": [c["id"] for c in ctx]}
    draft = draft_answer(state.buffer_text, ctx, state.prefs)
    final = style_adapter(refine_answer(draft, ctx), state.prefs)
    score = confidence(ctx, cls.get("confidence", 0.5))

    _update_notes(state, speaker=getattr(state, "buffer_speaker", "Speaker 1"), text=state.buffer_text)
    return {
        "speaker": getattr(state, "buffer_speaker", "Speaker 1"),
        "transcript": state.buffer_text,
        "suggestions": final["options"],
        "follow_up": final["follow_up"],
        "bridge": final["bridge"],
        "confidence": score,
        "context_ids": final["ctx_ids"],
        "intent": cls["intent"],
        "notes": state.notes,
        "usage": state.usage,
    }


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

    if any(p in tl for p in ["action item", "todo", "we need to", "we should", "follow up", "i will", "i'll"]):
        state.notes["action_items"].append(bullet)
        state.notes["action_items"] = state.notes["action_items"][-30:]
