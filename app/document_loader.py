from __future__ import annotations

from io import BytesIO
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]

SUPPORTED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf"}
CHUNK_CHARS = 800
CHUNK_OVERLAP = 150


def normalize_uploaded_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_upload(filename: str, content_bytes: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise ValueError(f"Unsupported context file type '{suffix or '(none)'}'. Supported types: {allowed}.")

    if suffix == ".pdf":
        return extract_text_from_pdf(content_bytes)

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = content_bytes.decode("utf-8", errors="replace")
    return normalize_uploaded_text(text)


def extract_text_from_pdf(content_bytes: bytes) -> str:
    if PdfReader is None:
        raise ValueError("PDF support requires the 'pypdf' package. Install dependencies from requirements.txt.")

    try:
        reader = PdfReader(BytesIO(content_bytes))
    except Exception as e:
        raise ValueError(f"Could not read PDF file: {e}") from e

    if getattr(reader, "is_encrypted", False):
        raise ValueError("Could not read PDF file: encrypted PDFs are not supported.")

    parts: List[str] = []
    try:
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
    except Exception as e:
        raise ValueError(f"Could not extract text from PDF file: {e}") from e

    text = normalize_uploaded_text("\n\n".join(parts))
    if not text:
        raise ValueError("Could not extract text from PDF file: no readable text found.")
    return text


def chunk_uploaded_context(
    *,
    filename: str,
    text: str,
    uploaded_at: float,
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    clean = normalize_uploaded_text(text)
    if not clean:
        return []

    topic = Path(filename).stem or "uploaded_context"
    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(clean):
        end = min(len(clean), i + size)
        chunk_text = clean[i:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"uploaded::{filename}::chunk{len(chunks)}",
                "text": chunk_text,
                "source": filename,
                "source_type": "uploaded_context",
                "memory_type": "session",
                "topic": topic,
                "route": "session_context",
                "priority": 1.0,
                "uploaded_at": uploaded_at,
                "meta": {
                    "source": filename,
                    "source_type": "uploaded_context",
                    "memory_type": "session",
                    "topic": topic,
                    "route": "session_context",
                    "priority": 1.0,
                    "uploaded_at": uploaded_at,
                },
            })
        if end >= len(clean):
            break
        i = max(0, end - overlap)
    return chunks
