import os, glob, json
from typing import List
import numpy as np
import faiss
from openai import OpenAI

DATA_DIR = "data"
OUT_DIR = "store"
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim
CHUNK_CHARS = 800
OVERLAP = 150

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(txt: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> List[str]:
    txt = " ".join(txt.split())
    chunks, i = [], 0
    while i < len(txt):
        end = min(len(txt), i + size)
        chunks.append(txt[i:end])
        if end >= len(txt): break
        i = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
    if not files:
        raise SystemExit("Put some .txt/.md files in ./data first.")

    docs = []
    for fp in files:
        base = os.path.basename(fp)
        if not any(base.lower().endswith(ext) for ext in (".txt", ".md")):
            continue
        raw = read_text(fp)
        if not raw.strip():
            continue
        for idx, ch in enumerate(chunk_text(raw)):
            docs.append({"id": f"{base}::chunk{idx}", "text": ch, "source": base})

    if not docs:
        raise SystemExit("No usable text found to index.")

    print(f"Embedding {len(docs)} chunks...")
    vecs = embed_texts([d["text"] for d in docs])
    dim = vecs.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vecs)

    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(docs)} chunks → store/index.faiss (dim={dim})")

if __name__ == "__main__":
    main()
