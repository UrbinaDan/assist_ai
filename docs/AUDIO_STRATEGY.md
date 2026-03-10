# Audio strategy (next phase)

## Current (V1)

- **Input**: Text only. The frontend uses browser Speech Recognition (or manual typing) and sends transcript deltas over WebSocket. No raw audio is sent to the backend.
- **Speaker**: Logical label chosen by the user (Me, Interviewer, Participant 3). No acoustic diarization.

## Planned

1. **Same-device, text-first**
   - Keep improving turn detection and response quality for the existing text/WS flow.
   - Optional: WebRTC `getUserMedia` to capture stereo and map left/right channels to “Me” vs “Interviewer” when using an external stereo mic.

2. **Same-device, raw audio (optional)**
   - Send audio chunks over a dedicated channel (e.g. binary WebSocket or REST) to an ASR service.
   - Backend or a separate service runs Whisper (or similar) and returns segments; optionally run diarization (e.g. pyannote or API) to attach speaker IDs. Map speaker IDs to the session’s logical roles (Me, Interviewer, etc.).

3. **Second-device meeting flow**
   - “Join session” flow: generate a short-lived token or QR code; second device opens a join URL and connects to the same session (e.g. via WebSocket with token).
   - Second device streams its audio (or transcript) into the same session so the backend sees two (or more) streams. Combine with diarization when multiple people speak on one device.

4. **Diarization**
   - Prefer an API that returns segments with `(speaker_id, start, end, text)` (e.g. OpenAI transcribe-diarize or third-party). Backend maps `speaker_id` to the session’s role labels and merges into the existing transcript/notes pipeline.

## Out of scope for V1

- Multi-channel capture and diarization are not required for the first release. Focus on making the app excellent for same-device use and text-stream input.
