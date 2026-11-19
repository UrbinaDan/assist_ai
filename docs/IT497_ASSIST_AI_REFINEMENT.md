# IT 497 — assist_ai: Privacy-First AI Assistant Refinement

## Project selected and objective
I selected **assist_ai**, a real-time conversational AI assistant previously developed with FastAPI, Python, WebSockets, retrieval-augmented generation (RAG), and an optional OpenAI integration. Its **Coach** mode provides conversation suggestions grounded in user-provided documents; its **Notes** mode organizes action items, decisions, summaries, and follow-ups.

The refinement addresses a practical software engineering problem: **conversation content can be sensitive**. Before these changes, the WebSocket handler printed entire messages and transcript buffers in backend diagnostics; the browser logged outgoing WebSocket payloads and speech-recognition result objects. Session logs also stored raw and cleaned transcripts by default. These practices could expose interview answers, meeting discussions, or uploaded-context details to anyone able to read local logs or browser diagnostic output.

## Technical and ethical evaluation
- **Privacy and consent:** Users may reasonably expect a note-taking or coaching tool to process their words, but not to automatically persist full transcripts in diagnostic logs.
- **Data minimization:** Operational details such as route, model usage, latency, and event type support troubleshooting without requiring conversation content.
- **Transparency:** A developer should explain when full-content recording is enabled and obtain informed consent from all relevant participants.
- **Tradeoff:** Metadata-only logs make detailed debugging harder. Optional content logging remains possible, but should be used only deliberately with appropriate consent and secure storage.
- **Limits:** In-memory conversation turns, the current-session RAG context, responses in the browser, and any upstream AI API calls are **not** eliminated by this refinement. Local metadata logs remain present. The application is not made secure for multi-user internet hosting by these changes alone.

## Changes made on the IT 497 branch

### 1. Default to metadata-only disk logs
**File:** `app/session_logger.py`

`log_session_event()` now persists an explicit allowlist of operational fields rather than copying the entire event payload. Raw and cleaned transcripts, interpreted questions, interpretation details, uploaded filenames, and retrieved document identifiers are omitted from persisted events by default.

Operators can explicitly set `SESSION_LOG_INCLUDE_CONTENT=true` to opt back into full event payload logging **only in an environment where participants have provided appropriate consent**. This setting is not intended as a substitute for an in-app consent mechanism.

### 2. Protect server-side diagnostic output
**File:** `app/ws.py`

Removed debugging statements that printed raw WebSocket message bodies, incremental transcript text, and buffered conversation text. Retained limited protocol metadata. Generic WebSocket processing errors replace exception-detail responses to avoid inadvertently exposing private data or provider internals.

### 3. Protect browser console output
**File:** `app/static/app.js`

Removed logging of outgoing WebSocket objects and speech-recognition result events; warnings no longer include the full outgoing message. This reduces the chance that browser developer tools expose private conversation text.

### 4. Add regression tests
**File:** `tests/test_session_logger.py`

Added tests that verify:
- Default logging keeps useful metadata but does not persist transcript or retrieval text.
- Full-content logging requires an explicit recognized opt-in.
- Unrecognized values do not enable full-content logging.

## Development record and verification
The changes were implemented in a dedicated `it497-privacy-first-assistant` branch, leaving `main` unchanged.

**Verification status:** Source changes and test cases were committed. **A full automated test run has not been performed or verified from this environment.** Before a merge, run:

```bash
python -m pip install -r requirements.txt
pytest tests/test_session_logger.py -v
pytest tests/ -v
```

Then manually test Coach mode, Notes mode, WebSocket messages, and the session-log preview/download with `SESSION_LOG_INCLUDE_CONTENT` unset. The UI will still display basic event metadata; private transcript fields should not be written into newly created log files. Existing log files from earlier versions are **not automatically removed**, so they may still need deletion or protection.

## Important remaining work
The source review also identified a broader concern: session endpoints and WebSocket access are designed primarily for local use and are not a production-grade multi-user authorization system. Do **not** publicly expose this app as-is. Other future improvements include authenticated session ownership, deletion and retention controls, secure local file handling, user-facing informed consent, safe API key handling, and an explicit indicator whenever conversation text is sent to a third-party AI provider.

## Reproducible peer review
A peer can evaluate a sanitized project copy containing the relevant files and documentation. A GitHub repository link is not required to perform a code review. Never distribute `.env` files, API keys, real transcript logs, personal résumé files, or documents used as RAG input.

Suggested peer-review questions:
1. Does the default metadata allowlist actually prevent private transcript persistence?
2. Is the opt-in behavior appropriately documented?
3. Do the privacy changes preserve existing Coach and Notes behavior?
4. What user-consent and session-authorization changes would be necessary before deployment?
