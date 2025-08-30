import os, json
from typing import List, Dict, Any
import numpy as np
import faiss

STORE_DIR = "store"

class Retriever:
    def __init__(self, store_dir: str = STORE_DIR):
        self.store_dir = store_dir
        self.index = None
        self.meta = []

    def load(self):
        idx_path = os.path.join(self.store_dir, "index.faiss")
        meta_path = os.path.join(self.store_dir, "meta.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise RuntimeError("FAISS store not found. Run scripts/build_index.py.")
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        return self

    def search(self, query_vec: np.ndarray, k: int = 4) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Retriever not loaded.")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1).astype("float32")
        D, I = self.index.search(query_vec, k)
        out: List[Dict[str, Any]] = []
        for rank, idx in enumerate(I[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            out.append({
                "id": m["id"],
                "text": m["text"],
                "score": round(float(D[0][rank]), 3),  # cosine similarity (higher is better)
                "meta": {"source": m.get("source")}
            })
        return out
