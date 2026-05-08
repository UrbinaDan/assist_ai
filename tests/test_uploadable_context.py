import pytest
from starlette.testclient import TestClient

import app.agent as agent
import app.document_loader as document_loader
from app.agent import AgentState, retrieve_context
from app.document_loader import extract_text_from_upload
from app.server import app
from app.session_store import SESSIONS
from app.session_logger import read_session_log


def test_document_loader_supports_txt():
    out = extract_text_from_upload("resume.txt", b"Hello   world\r\n\r\n\r\nPython")
    assert out == "Hello world\n\nPython"


def test_document_loader_supports_md():
    out = extract_text_from_upload("notes.md", b"# Title\n\nCompany   notes")
    assert "# Title" in out
    assert "Company notes" in out


def test_document_loader_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported context file type"):
        extract_text_from_upload("resume.docx", b"docx")


def test_document_loader_supports_pdf_with_reader(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        is_encrypted = False
        pages = [FakePage("Resume   text"), FakePage("Backend role")]

        def __init__(self, stream):
            self.stream = stream

    monkeypatch.setattr(document_loader, "PdfReader", FakeReader)

    out = extract_text_from_upload("resume.pdf", b"%PDF fake")

    assert out == "Resume text\n\nBackend role"


def test_document_loader_pdf_missing_dependency_fails_gracefully(monkeypatch):
    monkeypatch.setattr(document_loader, "PdfReader", None)

    with pytest.raises(ValueError, match="PDF support requires"):
        extract_text_from_upload("resume.pdf", b"%PDF fake")


def test_document_loader_pdf_encrypted_fails_gracefully(monkeypatch):
    class FakeReader:
        is_encrypted = True
        pages = []

        def __init__(self, stream):
            self.stream = stream

    monkeypatch.setattr(document_loader, "PdfReader", FakeReader)

    with pytest.raises(ValueError, match="encrypted PDFs are not supported"):
        extract_text_from_upload("secret.pdf", b"%PDF fake")


def test_upload_context_endpoint_stores_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))
    SESSIONS.pop("upload_test", None)
    client = TestClient(app)
    res = client.post(
        "/session/upload_test/upload-context",
        files={"file": ("job.md", b"Python backend role with FastAPI and retrieval.", "text/markdown")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["filename"] == "job.md"
    assert body["chunk_count"] == 1
    assert body["char_count"] > 0

    st = SESSIONS["upload_test"]
    assert len(st.uploaded_contexts) == 1
    assert st.uploaded_contexts[0]["source_type"] == "uploaded_context"
    assert st.session_context_chunks[0]["memory_type"] == "session"

    events = read_session_log("upload_test")
    assert len(events) == 1
    assert events[0]["event_type"] == "upload_context"
    assert events[0]["response_type"] == "upload_context"


def test_retrieval_includes_uploaded_context_when_query_overlaps(monkeypatch):
    monkeypatch.setattr(agent, "RETRIEVER", None)
    state = AgentState(session_id="ctx_retrieval")
    state.session_context_chunks.append({
        "id": "uploaded::job.md::chunk0",
        "text": "The job description emphasizes FastAPI, Python, retrieval, and backend APIs.",
        "source": "job.md",
        "source_type": "uploaded_context",
        "memory_type": "session",
        "topic": "job",
        "route": "session_context",
        "priority": 1.0,
        "meta": {
            "source": "job.md",
            "source_type": "uploaded_context",
            "memory_type": "session",
            "topic": "job",
            "route": "session_context",
            "priority": 1.0,
        },
    })

    ctx = retrieve_context("How does my FastAPI backend experience fit this role?", state=state)
    assert ctx
    assert ctx[0]["id"] == "uploaded::job.md::chunk0"
    assert ctx[0]["meta"]["source_type"] == "uploaded_context"
