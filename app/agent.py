# app/agent.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from app.retriever import Retriever

# NEW: OpenAI client
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
_oai: Optional[OpenAI] = None

def _get_oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _oai

# ---- Embedding model (local, free) ----
EMBED_MODEL = "all-MiniLM-L6-v2"
_st_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model

def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
    return vec

# ---- Agent state ----
@dataclass
class AgentState:
    session_id: str
    buffer_text: str = ""
    last_emit_ts: float = 0.0
    last_token_ts: float = 0.0
    intent_history: List[str] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    prefs: Dict[str, Any] = field(default_factory=lambda: {
        "tone": "concise",
        "target_len": "2 sentences",
        "avoid": ["overpromising", "jargon"]
    })
    # cache
    retrieval_cache: Dict[str, Any] = field(default_factory=dict)

# ---- END-OF-THOUGHT DETECTOR ----
class EndOfThought:
    def __init__(self, pause_ms: int = 900, stable_n: int = 2):
        self.pause_ms = pause_ms
        self.stable_n = stable_n

    def intent_stable(self, intents: List[str]) -> bool:
        if len(intents) < self.stable_n: 
            return False
        return len(set(intents[-self.stable_n:])) == 1

    def strong_punct(self, text: str) -> bool:
        text = text.strip()
        return text.endswith("?") or text.endswith(".") or text.endswith("!")

    def should_emit(self, state: AgentState) -> bool:
        now = time.time()
        paused = (now - state.last_token_ts) * 1000 >= self.pause_ms
        punct = self.strong_punct(state.buffer_text)
        stable = self.intent_stable(state.intent_history)
        longish = len(state.buffer_text.split()) >= 16  # fallback
        return (stable and (paused or punct)) or (stable and longish)

# ---- LLM classifier (OpenAI) ----
INTENTS = ["small_talk", "behavioral", "technical", "scheduling", "compensation", "unknown"]

def classify_question(text: str) -> Dict[str, Any]:
    """
    Uses an OpenAI model to classify live utterances into a fixed intent set
    and extract light slots. Returns strict JSON.
    """
    client = _get_oai()

    system_msg = (
        "You classify a single, short utterance from a live interview or conversation.\n"
        "Return a compact JSON object with keys: intent, entities, confidence.\n"
        f"Intent must be one of: {INTENTS}.\n"
        "entities must contain: company (string|null), role (string|null), "
        "skills (string[]), numbers (string[]), dates (string[]), times (string[]).\n"
        "confidence must be a number in [0,1].\n"
        "Be conservative: if unsure, use intent='unknown' and confidence <= 0.6."
    )

    user_msg = f"Utterance: {text}"

    try:
        # Use a fast/cheap reasoning-capable model; adjust if you have access to others
        rsp = client.responses.create(
            model="gpt-4o-mini",  # good balance of quality/latency/cost
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_output_tokens=200,
        )
        raw = rsp.output_text  # already JSON due to response_format
        data = json.loads(raw)

        # light validation / defaults
        intent = data.get("intent", "unknown")
        if intent not in INTENTS:
            intent = "unknown"
        entities = data.get("entities") or {}
        entities.setdefault("company", None)
        entities.setdefault("role", None)
        entities.setdefault("skills", [])
        entities.setdefault("numbers", [])
        entities.setdefault("dates", [])
        entities.setdefault("times", [])
        confidence = float(data.get("confidence", 0.5))

        return {
            "intent": intent,
            "entities": entities,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    except Exception as e:
        # Fallback if API fails: mark unknown but don't break the loop
        return {
            "intent": "unknown",
            "entities": {"company": None, "role": None, "skills": [], "numbers": [], "dates": [], "times": []},
            "confidence": 0.4,
            "error": str(e),
        }

# ---- Retriever (injected by server startup) ----
RETRIEVER: Optional[Retriever] = None

def retrieve_context(query: str, k: int = 4) -> List[Dict[str, Any]]:
    if RETRIEVER is None:
        return []
    qv = embed_query(query)
    return RETRIEVER.search(qv, k=k)

# ---- Draft + Refine + Style (template-based, local) ----
def draft_answer(text: str, ctx: List[Dict[str, Any]], prefs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask OpenAI to generate the suggestions, grounded on retrieved context.
    """
    client = _get_oai()
    context_txt = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(ctx[:4])) or "No context."

    system_msg = (
        "You are a real-time interview coach. Use ONLY the provided context snippets.\n"
        "Produce JSON with keys: options (string[] with 2-3 items), follow_up (string), bridge (string).\n"
        "Constraints: First-person voice, concise, specific; each option <= 2 sentences.\n"
        "If context is weak, say 'Context is limited' and ask for the missing detail."
    )
    user_msg = (
        f"User utterance: {text}\n\n"
        f"Context snippets:\n{context_txt}\n\n"
        f"Preferences: tone={prefs.get('tone','concise')}, target_len={prefs.get('target_len','2 sentences')}, avoid={prefs.get('avoid',[])}"
    )

    try:
        rsp = client.responses.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_output_tokens=300,
        )
        data = json.loads(rsp.output_text)
        options = data.get("options") or []
        follow_up = data.get("follow_up") or "Would you like more detail on metrics or rollout?"
        bridge = data.get("bridge") or "Happy to share specifics."

        return {
            "options": options[:3] if options else [
                "Based on a recent project, I led the effort to reduce latency; I can apply the same steps here.",
                "I’d start with a quick review of SLIs and traces, prototype two fixes, and A/B them for impact."
            ],
            "follow_up": follow_up,
            "bridge": bridge,
            "ctx_ids": [c["id"] for c in ctx]
        }
    except Exception:
        # fallback to the earlier local template if API fails
        snippets = [c["text"] for c in ctx[:2]]
        base1 = "From a recent project, I led the effort to"
        base2 = "My approach is to"
        if snippets:
            base1 += f" — {snippets[0][:160].rstrip()}..."
        if len(snippets) > 1:
            base2 += f" — {snippets[1][:160].rstrip()}..."
        return {
            "options": [
                base1 + " I can apply the same steps here.",
                base2 + " and iterate fast. That’s how we delivered impact."
            ],
            "follow_up": "Would it help to go deeper on the metrics or the rollout plan?",
            "bridge": "Happy to share specifics if you want details.",
            "ctx_ids": [c["id"] for c in ctx]
        }


def refine_answer(draft: Dict[str, Any], ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Minimal check to keep within retrieved context length
    return draft

def style_adapter(draft: Dict[str, Any], prefs: Dict[str, Any]) -> Dict[str, Any]:
    def clamp(s: str) -> str:
        s = s.replace("utilize", "use")
        # enforce length-ish
        if len(s.split()) > 40:
            s = " ".join(s.split()[:40]) + "..."
        return s
    out = draft.copy()
    out["options"] = [clamp(o) for o in draft["options"]]
    out["follow_up"] = clamp(draft["follow_up"])
    out["bridge"] = clamp(draft["bridge"])
    return out

def confidence(ctx: List[Dict[str, Any]], cls_conf: float) -> float:
    top = ctx[0]["score"] if ctx else 0.0  # cosine similarity [-1..1], usually [0..1]
    # Map to [0..1] and blend
    top01 = max(0.0, min(1.0, (top + 1.0) / 2.0))
    return round(0.5 * cls_conf + 0.5 * top01, 2)

def process_turn(state: AgentState) -> Optional[Dict[str, Any]]:
    cls = classify_question(state.buffer_text)
    state.intent_history.append(cls["intent"])

    ctx = retrieve_context(state.buffer_text, k=4)
    state.retrieval_cache = {"last_query": state.buffer_text, "doc_ids": [c["id"] for c in ctx]}

    draft = draft_answer(state.buffer_text, ctx, state.prefs)
    refined = refine_answer(draft, ctx)
    final = style_adapter(refined, state.prefs)

    score = confidence(ctx, cls.get("confidence", 0.5))
    return {
        "suggestions": final["options"],
        "follow_up": final["follow_up"],
        "bridge": final["bridge"],
        "confidence": score,
        "context_ids": final["ctx_ids"],
        "intent": cls["intent"]
    }

