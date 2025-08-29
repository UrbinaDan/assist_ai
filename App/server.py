# app/agent.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time

# ---- STATE ----
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

# ---- NODES (placeholders to wire your models) ----
def classify_question(text: str) -> Dict[str, Any]:
    # Call your LLM (cheap model) with a tiny prompt; return JSON
    # Example output:
    return {
        "intent": "behavioral",  # small_talk | behavioral | technical | scheduling | compensation | unknown
        "entities": {
            "company": None, "role": None,
            "skills": ["leadership", "python"],
            "numbers": [], "dates": [], "times": []
        },
        "confidence": 0.78
    }

def retrieve_context(query: str, k: int = 4) -> List[Dict[str, Any]]:
    # Wire FAISS/OpenAI File Search here; return list of {id, text, score, meta}
    return [
        {"id": "star-1", "text": "S: ... T: ... A: ... R: increased p95 latency by 40%.", "score": 0.71, "meta": {"tag":"STAR"}},
    ]

def draft_answer(text: str, ctx: List[Dict[str, Any]], prefs: Dict[str, Any]) -> Dict[str, Any]:
    # Call your LLM with system/few-shot:
    # Return 2–3 options + follow-up + bridge; keep them short.
    return {
        "options": [
            "I led a cross-team effort to remove a DB hotspot; p95 dropped by 40%. I can apply the same profiling process here.",
            "I’d start with a quick SLI/trace review, then prototype two fixes and A/B them. That’s how we cut auth errors last quarter."
        ],
        "follow_up": "Would it be more helpful to talk about scaling reads or write contention in your stack?",
        "bridge": "Happy to share the metrics if that’s useful.",
        "ctx_ids": [c["id"] for c in ctx]
    }

def refine_answer(draft: Dict[str, Any], ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Light guardrail: ensure claims appear in ctx; if not, soften or remove.
    return draft

def style_adapter(draft: Dict[str, Any], prefs: Dict[str, Any]) -> Dict[str, Any]:
    # Enforce tone/length/avoid lists (post-process strings).
    def clamp(s: str) -> str:
        return s.replace("utilize", "use")
    out = draft.copy()
    out["options"] = [clamp(o) for o in draft["options"]]
    out["bridge"] = clamp(draft["bridge"])
    out["follow_up"] = clamp(draft["follow_up"])
    return out

def confidence(ctx: List[Dict[str, Any]], cls_conf: float) -> float:
    # Blend classifier confidence with top retrieval score
    top = ctx[0]["score"] if ctx else 0.0
    return round(0.5*cls_conf + 0.5*top, 2)

# ---- GRAPH EXECUTION ----
def process_turn(state: AgentState) -> Optional[Dict[str, Any]]:
    # 1) Classify
    cls = classify_question(state.buffer_text)
    state.intent_history.append(cls["intent"])

    # 2) Retrieve
    ctx = retrieve_context(state.buffer_text, k=4)
    state.retrieval_cache = {"last_query": state.buffer_text, "doc_ids": [c["id"] for c in ctx]}

    # 3) Draft
    draft = draft_answer(state.buffer_text, ctx, state.prefs)

    # 4) Refine + Style
    refined = refine_answer(draft, ctx)
    final = style_adapter(refined, state.prefs)

    # 5) Emit package
    score = confidence(ctx, cls.get("confidence", 0.5))
    return {
        "suggestions": final["options"],
        "follow_up": final["follow_up"],
        "bridge": final["bridge"],
        "confidence": score,
        "context_ids": final["ctx_ids"],
        "intent": cls["intent"]
    }
