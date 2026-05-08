# Assist

Realtime conversation assistant: **Coach** mode (interview-style suggestions grounded in your materials) and **Notes** mode (bullets, action items, decisions, follow-ups). Same-device and text-stream first; audio capture and diarization are planned for a later phase.

## What this repo is

- **Backend**: Python (FastAPI), in-memory sessions, WebSocket at `/ws`, REST at `/ingest` and `/session/{id}/...`. FAISS retriever over your own docs; OpenAI for classifier, optional drafter, and optional notes refinement.
- **Frontend**: Single static app (HTML/JS/CSS) served by FastAPI at `/`. No Next.js or separate app repo; no PWA/CodeSpaces setup in this repo.

## Quick start

```bash
pip install -r requirements.txt
set OPENAI_API_KEY=your_key
python -m uvicorn app.server:app --reload --port 8000
```

Open `http://127.0.0.1:8000/`. Create a session, connect over WebSocket, choose **Coach** or **Notes**, and type (or use mic) to get live transcript, assistant output, notes, and cost.

## Project layout

```
assist_ai/
├── app/
│   ├── agent.py       # Classifier, retriever, drafter, notes, response shapes
│   ├── pipeline.py    # Turn detection, append_delta, maybe_emit
│   ├── retriever.py   # FAISS index load and search
│   ├── schemas.py     # DeltaIn, CoachSpeculativePayload, CoachFinalPayload, NotesPayload
│   ├── server.py      # FastAPI app, /ingest, /session/{id}, /session/{id}/mode|usage|notes
│   ├── ws.py          # WebSocket /ws
│   └── static/       # index.html, app.js, styles.css
├── data/              # Source texts for FAISS (e.g. resume, STAR notes)
├── scripts/
│   └── build_index.py # Build store/index.faiss and store/meta.json
├── store/             # index.faiss, meta.json (created by build_index)
├── tests/
├── requirements.txt
└── README.md
```

## Modes and response shapes

- **Coach**  
  - **Speculative** (end-of-thought, not final): `response_type: "coach_speculative"` with `question_type`, `answer_outline`, `matched_themes`. No full answer.  
  - **Final**: `response_type: "coach_final"` with `suggestions`, `follow_up`, `bridge`, `confidence`, `context_ids`.  
  - Behavioral/background questions use retrieval from your index; technical/conceptual use a lightweight framework path (no forced personalization).

- **Notes**  
  - **Final**: `response_type: "notes_final"` with `notes`: `bullets`, `topics`, `action_items`, `decisions`, `follow_ups`, `summary_so_far`, `current_topic`, `open_questions`.  
  - Set `USE_OAI_NOTES=true` to enable optional LLM refinement (summary, owner/due on action items).

## Memory architecture

Static background knowledge can be organized under typed memory folders:

- `data/semantic/`: stable facts, profile, projects, and skills.
- `data/episodic/`: stories and lived examples.
- `data/procedural/`: answer strategies, steps, and frameworks.
- `data/preferences/`: tone and style guidance.

Files directly under `data/` are still treated as legacy memory for backward compatibility. Rebuild the FAISS store after editing typed memory files:

```bash
python scripts/build_index.py
```

## Uploadable session context

Temporary context can be uploaded for the current in-memory session from the app UI or via `POST /session/{session_id}/upload-context`. Supported file types are `.txt`, `.md`, and readable `.pdf` files.

Useful examples include resumes, job descriptions, company notes, class notes, and meeting agendas. Uploaded context is chunked in memory, marked as `memory_type: "session"`, and considered before static FAISS memory during RAG retrieval. It is not persisted to disk.

PDF extraction uses `pypdf` and works best for text-based PDFs. Encrypted PDFs, scanned image-only PDFs, or unreadable files return a clear upload error instead of being stored.

## Session logging

Important session events are written locally as JSONL under `logs/sessions/{session_id}.jsonl`. Logs include final coach responses, answer-monitor events, notes updates, and uploaded context events.

Use `GET /session/{session_id}/log` to review parsed events, or `GET /session/{session_id}/log/download` to download the raw JSONL file. These logs are useful for reviewing interview practice, meeting notes, routing decisions, retrieval sources, and cost/debug metadata.

## Latency analytics

WebSocket messages include client and backend timestamps, and emitted payloads include optional latency fields for interpreter, router, retrieval, LLM, backend total, and end-to-end estimates. The debug panel shows a compact latency summary plus raw timestamps when available.

Latency fields are best-effort and backward compatible: older payloads can omit them, and unavailable steps are reported as `null`/`n/a`.

## Streaming coach responses

Coach streaming is optional and disabled by default with `USE_STREAMING_COACH=false`. When enabled, the WebSocket may send `kind: "answer_delta"` messages with `response_type: "coach_stream"` before the normal final `coach_final` payload.

The final payload contract does not change. Notes mode and answer-monitor responses are not streamed.

## Session and cost

- Sessions are in-memory; `created_at` and `last_seen_at` are tracked.  
- **GET /session/{id}/usage** returns usage and cost; **GET /session/{id}/notes** returns notes.  
- **POST /session/{id}/mode** sets `coach` or `notes`.  
- Cost is aggregated by model and by feature (embed, classifier, coach_drafter, notes_drafter).

## Building the FAISS index

```bash
python scripts/build_index.py
```

Uses `data/` (e.g. `resume.txt`, `star_latency.md`) and writes `store/index.faiss` and `store/meta.json`. If the store is missing, the server starts but retrieval returns empty.

## Tests

```bash
pytest tests/ -v
```

Covers: WebSocket connect, coach final message shape, mode switch (coach ↔ notes), speaker-change flush, duplicate final dedupe, notes heuristic extraction.

## Audio strategy (next phase)

- **V1 (current)**: Same-device, text-stream input (browser Speech Recognition or manual type). No raw audio upload.  
- **Later**: Same-device channel-separated capture (e.g. stereo → “Me” vs “Interviewer”), then optional second-device join (QR/token) and routing of that device’s stream into the same session.  
- **Later**: ASR + diarization (e.g. OpenAI or pyannote) producing `(speaker_id, start, end, text)` and mapping into the existing speaker/role model.

See `docs/AUDIO_STRATEGY.md` for a short roadmap.
