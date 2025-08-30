import os, glob, json
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
OUT_DIR = "store"
EMBED_MODEL = "all-MiniLM-L6-v2"   # local, free
CHUNK_CHARS = 800                  # ~2–4 paragraphs per chunk
OVERLAP = 150

def read_text(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".pdf"):
        try:
            import pypdf
            with open(path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print(f"[WARN] Skipping PDF (need pypdf & extractable text): {path} ({e})")
            return ""
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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
    if not files:
        raise SystemExit("Put some .txt/.md (or simple PDFs) in ./data first.")

    # gather docs → chunks
    docs = []
    for fp in files:
        base = os.path.basename(fp)
        if not any(base.lower().endswith(ext) for ext in (".txt", ".md", ".pdf")):
            continue
        raw = read_text(fp)
        if not raw.strip():
            continue
        for idx, ch in enumerate(chunk_text(raw)):
            docs.append({"id": f"{base}::chunk{idx}", "text": ch, "source": base})

    if not docs:
        raise SystemExit("No usable text found to index.")

    print(f"Embedding {len(docs)} chunks...")
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode([d["text"] for d in docs],
                        batch_size=64,
                        convert_to_numpy=True,
                        normalize_embeddings=True).astype("float32")

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine via normalized vectors
    index.add(vecs)

    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(docs)} chunks → store/index.faiss (dim={dim})")

if __name__ == "__main__":
    main()
