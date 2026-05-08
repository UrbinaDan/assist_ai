from pathlib import Path
import re
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.agent as agent
from app.agent import EndOfThought
from app.document_loader import chunk_uploaded_context, extract_text_from_upload
from app.pipeline import append_delta, maybe_emit
from app.retriever import Retriever
from app.session_store import SESSIONS, get_or_create_session
from app.session_logger import log_session_event, read_session_log, session_log_path


app = FastAPI()

DETECTOR = EndOfThought(pause_ms=900, stable_n=2)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    if STATIC_DIR.exists():
        return FileResponse(str(STATIC_DIR / "index.html"))
    return {"ok": True, "hint": "Add app/static/index.html for a simple frontend."}


class TranscriptEvent(BaseModel):
    session_id: str
    text_delta: str
    final: bool = False
    speaker: str | None = None
    session_mode: str | None = None  # "coach" | "notes"


class SessionModeBody(BaseModel):
    mode: str  # "coach" | "notes"


class SessionSummary(BaseModel):
    session_id: str
    mode: str
    created_at: float
    last_seen_at: float
    roles: dict[str, str]


class UsageSummary(BaseModel):
    session_id: str
    mode: str
    created_at: float
    last_seen_at: float
    usage: dict


class NotesSummary(BaseModel):
    session_id: str
    mode: str
    notes: dict


class UploadContextSummary(BaseModel):
    filename: str
    chunk_count: int
    char_count: int


class SessionContextSummary(BaseModel):
    filename: str
    uploaded_at: float
    chunk_count: int
    source_type: str


@app.on_event("startup")
def _load_retriever():
    """Load FAISS index at startup and inject into agent.RETRIEVER."""
    try:
        agent.RETRIEVER = Retriever().load()
        print("FAISS retriever loaded and injected into agent.", file=sys.stderr, flush=True)
    except Exception as e:
        agent.RETRIEVER = None
        print(f"FAISS retriever not available: {e}", file=sys.stderr, flush=True)


def _get_session_or_404(session_id: str):
    st = SESSIONS.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return st


@app.post("/ingest")
def ingest(ev: TranscriptEvent):
    """
    Append transcript delta; if final or end-of-thought -> process.
    Otherwise return a no-emit response.
    """
    st = get_or_create_session(ev.session_id)

    st.last_seen_at = time.time()

    if ev.session_mode in ("coach", "notes"):
        st.mode = ev.session_mode

    if ev.speaker:
        st.roles.setdefault(ev.speaker, ev.speaker)

    append_delta(st, ev.text_delta, ts=time.time(), speaker=ev.speaker)
    res = maybe_emit(st, final=ev.final, detector=DETECTOR)

    if res.emit:
        out = dict(res.data) if res.data else {}
        out["created_at"] = getattr(st, "created_at", None)
        out["last_seen_at"] = getattr(st, "last_seen_at", None)

        if isinstance(out.get("usage"), dict):
            out["usage"]["created_at"] = out["created_at"]
            out["usage"]["last_seen_at"] = out["last_seen_at"]

        print(
            f"[ingest] emit sid={ev.session_id} kind={res.kind} reason={res.reason}",
            file=sys.stderr,
            flush=True,
        )
        return {"emit": True, "kind": res.kind, "data": out, "reason": res.reason}

    print(
        f"[ingest] no-emit sid={ev.session_id} kind={res.kind} reason={res.reason}",
        file=sys.stderr,
        flush=True,
    )
    return {"emit": False, "kind": res.kind, "reason": res.reason}


@app.get("/session/{session_id}", response_model=SessionSummary)
def get_session(session_id: str):
    st = get_or_create_session(session_id)
    return SessionSummary(
        session_id=session_id,
        mode=st.mode,
        created_at=st.created_at,
        last_seen_at=st.last_seen_at,
        roles=st.roles,
    )


@app.get("/session/{session_id}/mode", response_model=SessionModeBody)
def get_session_mode(session_id: str):
    st = get_or_create_session(session_id)
    return SessionModeBody(mode=st.mode)


@app.post("/session/{session_id}/mode", response_model=SessionModeBody)
def set_session_mode(session_id: str, body: SessionModeBody):
    if body.mode not in ("coach", "notes"):
        raise HTTPException(status_code=400, detail="mode must be 'coach' or 'notes'")

    st = get_or_create_session(session_id)
    st.mode = body.mode
    st.last_seen_at = time.time()

    return SessionModeBody(mode=st.mode)


@app.get("/session/{session_id}/usage", response_model=UsageSummary)
def get_session_usage(session_id: str):
    st = get_or_create_session(session_id)
    return UsageSummary(
        session_id=session_id,
        mode=st.mode,
        created_at=st.created_at,
        last_seen_at=st.last_seen_at,
        usage=st.usage,
    )


@app.get("/session/{session_id}/notes", response_model=NotesSummary)
def get_session_notes(session_id: str):
    st = get_or_create_session(session_id)
    return NotesSummary(
        session_id=session_id,
        mode=st.mode,
        notes=st.notes,
    )


@app.post("/session/{session_id}/upload-context", response_model=UploadContextSummary)
async def upload_session_context(session_id: str, request: Request):
    filename, content = await _extract_single_upload(request)
    try:
        text = extract_text_from_upload(filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    uploaded_at = time.time()
    chunks = chunk_uploaded_context(filename=filename, text=text, uploaded_at=uploaded_at)
    if not chunks:
        raise HTTPException(status_code=400, detail="Uploaded context file is empty after text extraction.")

    st = get_or_create_session(session_id)
    item = {
        "filename": filename,
        "uploaded_at": uploaded_at,
        "chunks": chunks,
        "source_type": "uploaded_context",
    }
    st.uploaded_contexts.append(item)
    st.session_context_chunks.extend(chunks)
    st.last_seen_at = uploaded_at

    log_session_event(
        session_id,
        "upload_context",
        {
            "mode": st.mode,
            "speaker": None,
            "audio_source": None,
            "speaker_mode": None,
            "raw_transcript": None,
            "clean_transcript": None,
            "interpreted_question": None,
            "interpretation": None,
            "route": None,
            "route_confidence": None,
            "memory_filters": None,
            "retrieval_source": [chunk["id"] for chunk in chunks],
            "response_type": "upload_context",
            "usage": st.usage,
            "latency": None,
            "filename": filename,
            "chunk_count": len(chunks),
            "char_count": len(text),
        },
    )

    return UploadContextSummary(
        filename=filename,
        chunk_count=len(chunks),
        char_count=len(text),
    )


@app.get("/session/{session_id}/upload-context")
def get_uploaded_contexts(session_id: str):
    st = get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "items": [
            SessionContextSummary(
                filename=item.get("filename", ""),
                uploaded_at=float(item.get("uploaded_at", 0.0)),
                chunk_count=len(item.get("chunks", [])),
                source_type=item.get("source_type", "uploaded_context"),
            ).model_dump()
            for item in getattr(st, "uploaded_contexts", [])
        ],
    }


async def _extract_single_upload(request: Request) -> tuple[str, bytes]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="Expected multipart/form-data upload with a file field.")

    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise HTTPException(status_code=400, detail="Missing multipart boundary.")

    boundary = ("--" + match.group(1)).encode("utf-8")
    for part in body.split(boundary):
        if b"filename=" not in part:
            continue
        header_blob, sep, file_blob = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        header_text = header_blob.decode("utf-8", errors="replace")
        filename_match = re.search(r'filename="([^"]+)"', header_text)
        if not filename_match:
            continue
        filename = Path(filename_match.group(1)).name
        content = file_blob
        if content.endswith(b"\r\n"):
            content = content[:-2]
        if content.endswith(b"--"):
            content = content[:-2]
        if content.endswith(b"\r\n"):
            content = content[:-2]
        return filename, content

    raise HTTPException(status_code=400, detail="No uploaded file found.")


@app.get("/session/{session_id}/log")
def get_session_log(session_id: str):
    return {
        "session_id": session_id,
        "events": read_session_log(session_id),
    }


@app.get("/session/{session_id}/log/download")
def download_session_log(session_id: str):
    path = session_log_path(session_id)
    if path.exists():
        return FileResponse(
            str(path),
            media_type="application/x-ndjson",
            filename=f"{session_id}.jsonl",
        )
    return PlainTextResponse("", media_type="application/x-ndjson")


@app.get("/health")
def health():
    status = "loaded" if (agent.RETRIEVER and agent.RETRIEVER.index is not None) else "missing"
    return {"retriever": status}


@app.get("/debug/retriever")
def debug_retriever():
    if not agent.RETRIEVER or not agent.RETRIEVER.index:
        return {"loaded": False}
    return {"loaded": True, "chunks": len(agent.RETRIEVER.meta)}


class ClassifyProbe(BaseModel):
    text: str


@app.post("/debug/classify")
def debug_classify(p: ClassifyProbe):
    try:
        result = agent.classify_question(p.text)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


from app.ws import router as ws_router

app.include_router(ws_router)
