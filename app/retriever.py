import os, json
from typing import List, Dict, Any, Optional
import numpy as np
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore[assignment]

STORE_DIR = "store"

class Retriever:
    def __init__(self, store_dir: str = STORE_DIR):
        self.store_dir = store_dir
        self.index = None
        self.meta = []

    def load(self):
        if faiss is None:
            raise RuntimeError("faiss is not installed. Run: pip install -r requirements.txt")
        idx_path = os.path.join(self.store_dir, "index.faiss")
        meta_path = os.path.join(self.store_dir, "meta.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise RuntimeError("FAISS store not found. Run scripts/build_index.py first.")
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        return self

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Retriever not loaded.")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1).astype("float32")

        has_filters = bool(filters) and (
            bool(filters.get("memory_types"))
            or bool(filters.get("topics"))
            or float(filters.get("min_priority", 0.0) or 0.0) > 0.0
        )
        search_k = k
        if has_filters:
            search_k = min(len(self.meta), max(k * 4, 12))

        D, I = self.index.search(query_vec, search_k)
        # IndexFlatL2 returns L2 distance (smaller = better). Convert to similarity-ish [0..1]
        out: List[Dict[str, Any]] = []
        if len(D) and len(I):
            mind = float(np.min(D))
            maxd = float(np.max(D))
            den = (maxd - mind + 1e-9)
            for rank, idx in enumerate(I[0]):
                if idx == -1:
                    continue
                m = self.meta[idx]
                dist = float(D[0][rank])
                sim = 1.0 - ((dist - mind) / den)  # normalize & invert
                out.append({
                    "id": m["id"],
                    "text": m["text"],
                    "score": round(sim, 3),
                    "meta": {
                        "source": m.get("source"),
                        "source_path": m.get("source_path", m.get("source")),
                        "memory_type": m.get("memory_type", "legacy"),
                        "topic": m.get("topic"),
                        "route": m.get("route", "legacy"),
                        "priority": m.get("priority", 0.5),
                    }
                })

        if not has_filters:
            return out[:k]

        filtered = [row for row in out if _matches_filters(row.get("meta", {}), filters or {})]
        if not filtered:
            return out[:k]

        seen = {row["id"] for row in filtered}
        for row in out:
            if len(filtered) >= k:
                break
            if row["id"] not in seen:
                filtered.append(row)
                seen.add(row["id"])
        return filtered[:k]


def _matches_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    memory_types = filters.get("memory_types") or []
    if memory_types and meta.get("memory_type", "legacy") not in memory_types:
        return False

    topics = filters.get("topics") or []
    if topics and meta.get("topic") not in topics:
        return False

    min_priority = float(filters.get("min_priority", 0.0) or 0.0)
    try:
        priority = float(meta.get("priority", 0.0) or 0.0)
    except Exception:
        priority = 0.0
    if priority < min_priority:
        return False

    return True
