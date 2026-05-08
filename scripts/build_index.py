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

MEMORY_PRIORITIES = {
    "semantic": 0.8,
    "episodic": 0.9,
    "procedural": 0.7,
    "preferences": 0.6,
    "legacy": 0.5,
}

ROUTE_BY_TYPED_PATH = {
    ("semantic", "projects"): "technical_project",
    ("semantic", "profile"): "personal_background",
    ("semantic", "technical_skills"): "technical_project",
    ("episodic", "interview_stories"): "behavioral_story",
    ("episodic", "debugging_stories"): "behavioral_story",
    ("episodic", "teamwork_stories"): "behavioral_story",
    ("procedural", "star_method"): "behavioral_story",
    ("procedural", "answering_rules"): "personal_background",
    ("procedural", "coding_interview_steps"): "coding_interview",
    ("preferences", "tone"): "preferences",
    ("preferences", "answer_length"): "preferences",
}

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

def metadata_for_path(path: str) -> dict:
    rel = os.path.relpath(path, DATA_DIR)
    parts = rel.split(os.sep)
    filename = os.path.basename(path)
    topic = os.path.splitext(filename)[0]

    memory_type = "legacy"
    if len(parts) > 1 and parts[0] in MEMORY_PRIORITIES:
        memory_type = parts[0]

    route = "legacy"
    if memory_type != "legacy":
        route = ROUTE_BY_TYPED_PATH.get((memory_type, topic), "unknown")

    return {
        "source": filename,
        "source_path": rel.replace(os.sep, "/"),
        "memory_type": memory_type,
        "topic": topic,
        "route": route,
        "priority": MEMORY_PRIORITIES.get(memory_type, 0.5),
    }

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.md"), recursive=True))
    if not files:
        raise SystemExit("Put some .md files in ./data first.")

    docs = []
    for fp in files:
        base = os.path.basename(fp)
        raw = read_text(fp)
        if not raw.strip():
            continue
        meta = metadata_for_path(fp)
        for idx, ch in enumerate(chunk_text(raw)):
            docs.append({
                "id": f"{meta['source_path']}::chunk{idx}",
                "text": ch,
                **meta,
            })

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
