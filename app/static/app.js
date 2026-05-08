/* global webkitSpeechRecognition */

function randId() {
  return Math.random().toString(16).slice(2) + "-" + Date.now().toString(16);
}

/** @type {HTMLElement | null} */
const mainWorkspace = document.getElementById("mainWorkspace");

const els = {
  sessionId: document.getElementById("sessionId"),
  newSession: document.getElementById("newSession"),
  connect: document.getElementById("connect"),
  mic: document.getElementById("mic"),
  status: document.getElementById("status"),
  manual: document.getElementById("manual"),
  send: document.getElementById("send"),
  transcriptList: document.getElementById("transcriptList"),

  coachSpeculative: document.getElementById("coachSpeculative"),
  coachStream: document.getElementById("coachStream"),
  coachStreamText: document.getElementById("coachStreamText"),
  coachFinal: document.getElementById("coachFinal"),
  coachHistory: document.getElementById("coachHistory"),
  coachEmptyState: document.getElementById("coachEmptyState"),
  questionType: document.getElementById("questionType"),
  answerOutline: document.getElementById("answerOutline"),
  matchedThemes: document.getElementById("matchedThemes"),
  suggestions: document.getElementById("suggestions"),
  followUp: document.getElementById("followUp"),
  bridge: document.getElementById("bridge"),
  confidence: document.getElementById("confidence"),

  answerMonitor: document.getElementById("answerMonitor"),
  answerSignals: document.getElementById("answerSignals"),

  summarySoFar: document.getElementById("summarySoFar"),
  currentTopic: document.getElementById("currentTopic"),
  openQuestions: document.getElementById("openQuestions"),
  notesBullets: document.getElementById("notesBullets"),
  actionItems: document.getElementById("actionItems"),
  decisions: document.getElementById("decisions"),
  followUps: document.getElementById("followUps"),

  sessionCost: document.getElementById("sessionCost"),
  turnCost: document.getElementById("turnCost"),
  perMinEst: document.getElementById("perMinEst"),
  byFeature: document.getElementById("byFeature"),
  refreshUsage: document.getElementById("refreshUsage"),

  sessionCostNotes: document.getElementById("sessionCostNotes"),
  turnCostNotes: document.getElementById("turnCostNotes"),
  perMinEstNotes: document.getElementById("perMinEstNotes"),
  byFeatureNotes: document.getElementById("byFeatureNotes"),
  refreshUsageNotes: document.getElementById("refreshUsageNotes"),

  modeCoach: document.getElementById("modeCoach"),
  modeNotes: document.getElementById("modeNotes"),
  coachMiddlePanel: document.getElementById("coachMiddlePanel"),
  notesMiddlePanel: document.getElementById("notesMiddlePanel"),
  coachRightStack: document.getElementById("coachRightStack"),
  notesRightStack: document.getElementById("notesRightStack"),

  debugBody: document.getElementById("debugBody"),
  debugBodyNotes: document.getElementById("debugBodyNotes"),

  contextFile: document.getElementById("contextFile"),
  uploadContext: document.getElementById("uploadContext"),
  contextUploadStatus: document.getElementById("contextUploadStatus"),
  contextList: document.getElementById("contextList"),

  refreshSessionLog: document.getElementById("refreshSessionLog"),
  refreshSessionLogNotes: document.getElementById("refreshSessionLogNotes"),
  downloadSessionLog: document.getElementById("downloadSessionLog"),
  downloadSessionLogNotes: document.getElementById("downloadSessionLogNotes"),
  sessionLogCount: document.getElementById("sessionLogCount"),
  sessionLogCountNotes: document.getElementById("sessionLogCountNotes"),
  sessionLogPreview: document.getElementById("sessionLogPreview"),
  sessionLogPreviewNotes: document.getElementById("sessionLogPreviewNotes"),
};

const TRANSCRIPT_SCROLL_NEAR_BOTTOM_PX = 72;

/** Internal lifecycle labels for future streaming (Phase F). */
const TranscriptLifecycle = {
  LIVE_PREVIEW: "live_preview",
  TRANSCRIPT_FINAL: "transcript_final",
  COACH_RESPONSE: "coach_response",
  NOTES_UPDATE: "notes_update",
};

function getSpeaker() {
  const s = document.getElementById("speaker");
  return s ? s.value : "Me";
}

const LS_SPEAKER_MODE = "assist_speaker_mode";
const LS_NOTES_AUDIO = "assist_notes_audio";

/** @type {"manual"|"source"} */
let speakerMode = "manual";
/** @type {"mic"|"system"|"both"} */
let notesAudioSource = "both";

function loadAudioPrefs() {
  const sm = localStorage.getItem(LS_SPEAKER_MODE);
  if (sm === "manual" || sm === "source") speakerMode = sm;
  const na = localStorage.getItem(LS_NOTES_AUDIO);
  if (na === "mic" || na === "system" || na === "both") notesAudioSource = na;
}

function syncSpeakerModeRadios() {
  document.querySelectorAll('input[name="speakerMode"]').forEach((inp) => {
    inp.checked = inp.value === speakerMode;
  });
}

function syncNotesAudioRadios() {
  document.querySelectorAll('input[name="notesAudio"]').forEach((inp) => {
    inp.checked = inp.value === notesAudioSource;
  });
}

function syncSpeakerUi() {
  const manualRow = document.getElementById("manualSpeakerRow");
  const hint = document.getElementById("sourceSpeakerHint");
  const manualHelp = document.getElementById("manualModeHelp");
  const sourceHelp = document.getElementById("sourceModeHelp");
  if (manualRow) manualRow.hidden = speakerMode !== "manual";
  if (hint) hint.hidden = speakerMode !== "source";
  if (manualHelp) manualHelp.hidden = speakerMode !== "manual";
  if (sourceHelp) sourceHelp.hidden = speakerMode !== "source";
  syncSpeakerModeRadios();
}

function refreshMicAvailability() {
  const wsOpen = ws && ws.readyState === WebSocket.OPEN;
  const notesBlocksMic = currentMode === "notes" && notesAudioSource === "system";
  if (els.mic) {
    if (!wsOpen) els.mic.disabled = true;
    else els.mic.disabled = notesBlocksMic;
  }
}

/**
 * @param {"mic"|"system"} source
 */
function resolveSpeakerForSource(source) {
  if (speakerMode === "manual") {
    return getSpeaker();
  }
  if (speakerMode === "source") {
    if (source === "system") return "Interviewer";
    if (source === "mic") return "Me";
  }
  return getSpeaker();
}

function wsCommonFields() {
  return {
    session_id: els.sessionId.value,
    ts: Date.now() / 1000,
    client_sent_at: Date.now() / 1000,
    session_mode: currentMode,
    speaker_mode: speakerMode,
  };
}

els.sessionId.value = localStorage.getItem("assist_session_id") || randId();

let ws = null;
let recognition = null;
let micOn = false;
let userStoppedMic = false;
let micHardError = false;

let lastSentReplace = "";
let lastFinalizedMicText = "";
let lastLivePreviewText = "";

let micFinalizeTimer = null;

let currentMode = "coach";
let transcriptEntries = [];
let lastUsage = null;
let sessionStartAt = null;
let coachHistory = [];
let uploadedContextItems = [];
let sessionLogEvents = [];
let coachStreamText = "";

/** @type {Record<string, unknown>} */
let lastEmitSnapshot = {};

function setStatus(s) {
  if (els.status) els.status.textContent = s;
}

function setContextUploadStatus(s) {
  if (els.contextUploadStatus) els.contextUploadStatus.textContent = s || "";
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function sendMsg(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.warn("sendMsg skipped: websocket not open", { obj, wsState: ws?.readyState });
    return;
  }
  console.log("Sending WS message:", obj);
  ws.send(JSON.stringify(obj));
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtUsd(x) {
  const n = typeof x === "number" ? x : Number(x || 0);
  return `$${n.toFixed(4)}`;
}

function normalizeText(s) {
  return String(s || "")
    .replace(/\s+/g, " ")
    .trim();
}

function collapseDuplicateWords(s) {
  const words = String(s || "").split(/\s+/).filter(Boolean);
  const out = [];
  for (const w of words) {
    if (out.length && out[out.length - 1].toLowerCase() === w.toLowerCase()) continue;
    out.push(w);
  }
  return out.join(" ");
}

function stripLeadingFiller(s) {
  let out = String(s || "").trim();
  const patterns = [
    /^(hi|hello|hey)\s+\w+\s+/i,
    /^(okay|ok|well|so|um|uh|hmm)\s+/i,
    /^(that's good( then)?)\s+/i,
    /^(sue|see|so)\s+\w+\s+/i,
  ];

  let changed = true;
  while (changed) {
    changed = false;
    for (const p of patterns) {
      const next = out.replace(p, "").trim();
      if (next !== out) {
        out = next;
        changed = true;
      }
    }
  }
  return out;
}

function cleanTranscriptForSend(s) {
  let out = normalizeText(s);
  out = collapseDuplicateWords(out);
  out = stripLeadingFiller(out);
  out = out.replace(/\b(in in|to to)\b/gi, (m) => m.split(" ")[0]);
  out = out.replace(/\s+/g, " ").trim();
  return out;
}

function isQuestionLike(text) {
  const t = String(text || "").toLowerCase().trim();
  return /^(talk to me about|tell me about|tell me why|why do you|what is|what are|how do you|how would you|can you|walk me through|solve|explain|describe)\b/.test(t);
}

function isMeaningfulTranscript(text) {
  const raw = cleanTranscriptForSend(text);
  if (!raw) return false;
  const words = raw.split(/\s+/).filter(Boolean);
  if (words.length >= 8) return true;
  if (/[?.!]/.test(raw) && words.length >= 5) return true;
  if (isQuestionLike(raw) && words.length >= 5) return true;
  return false;
}

function isTooShortForFinal(text) {
  const t = cleanTranscriptForSend(text);
  const words = t ? t.split(/\s+/).filter(Boolean) : [];
  return words.length < 5;
}

function isSmallIncrement(newText, oldText) {
  const a = cleanTranscriptForSend(newText).toLowerCase();
  const b = cleanTranscriptForSend(oldText).toLowerCase();
  if (!a || !b) return false;
  if (a === b) return true;
  if (a.startsWith(b)) {
    const aWords = a.split(/\s+/).length;
    const bWords = b.split(/\s+/).length;
    return (aWords - bWords) <= 2;
  }
  return false;
}

/**
 * @param {HTMLElement | null} container
 */
function shouldAutoScroll(container) {
  if (!container) return true;
  const dist = container.scrollHeight - container.scrollTop - container.clientHeight;
  return dist <= TRANSCRIPT_SCROLL_NEAR_BOTTOM_PX;
}

/**
 * @param {boolean} [force]
 */
function scrollTranscriptToBottom(force) {
  const el = els.transcriptList;
  if (!el) return;
  if (force || shouldAutoScroll(el)) {
    el.scrollTop = el.scrollHeight;
  }
}

function removeLivePreview() {
  const livePreview = document.getElementById("liveTranscriptPreview");
  if (livePreview) livePreview.remove();
  lastLivePreviewText = "";
}

function updateLivePreview(text) {
  const full = cleanTranscriptForSend(text);
  if (!full || full === lastLivePreviewText) return;
  lastLivePreviewText = full;

  if (!els.transcriptList) return;

  removeLivePreview();

  const liveTs = Date.now() / 1000;
  const div = document.createElement("div");
  div.id = "liveTranscriptPreview";
  div.className = "transcriptEntry transcriptEntry--live";
  div.dataset.lifecycle = TranscriptLifecycle.LIVE_PREVIEW;
  div.innerHTML = `
    <span class="ts mono">${escapeHtml(
      new Date(liveTs * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    )}</span>
    <span class="kind live">live</span>
    <span class="speaker">${escapeHtml(resolveSpeakerForSource("mic"))}</span>:
    ${escapeHtml(full)}
  `;
  els.transcriptList.appendChild(div);
  scrollTranscriptToBottom(false);
}

function maybeSendMicFinal(text) {
  const clean = cleanTranscriptForSend(text);
  if (!clean) return false;
  if (!isMeaningfulTranscript(clean)) return false;
  if (clean === lastFinalizedMicText) return false;
  if (isSmallIncrement(clean, lastFinalizedMicText)) return false;

  lastFinalizedMicText = clean;

  const micSpeaker = resolveSpeakerForSource("mic");
  sendMsg({
    ...wsCommonFields(),
    mode: "replace",
    text: clean,
    final: true,
    speaker: micSpeaker,
    audio_source: "mic",
  });

  if (!alreadyHasRecentTranscript(clean, micSpeaker)) {
    appendTranscriptFromEvent(micSpeaker, Date.now() / 1000, "final", clean, TranscriptLifecycle.TRANSCRIPT_FINAL);
  }

  return true;
}

function scheduleMicFinalize(full) {
  if (micFinalizeTimer) clearTimeout(micFinalizeTimer);

  micFinalizeTimer = setTimeout(() => {
    const clean = cleanTranscriptForSend(full);
    if (!clean) return;
    maybeSendMicFinal(clean);
  }, 2500);
}

// -------- Transcript --------
function alreadyHasRecentTranscript(text, speaker) {
  const t = normalizeText(text).toLowerCase();
  if (!t) return true;
  const recent = transcriptEntries.slice(-16);
  for (let i = recent.length - 1; i >= 0; i--) {
    const e = recent[i];
    if (e.kind !== "final") continue;
    if (String(e.speaker) !== String(speaker)) continue;
    if (normalizeText(e.text).toLowerCase() === t) return true;
  }
  return false;
}

function transcriptRowModifier(kind) {
  if (kind === "live") return "transcriptEntry--live";
  if (kind === "speculative") return "transcriptEntry--speculative";
  return "transcriptEntry--final";
}

/**
 * @param {string} lifecycle
 */
function appendTranscriptFromEvent(speaker, ts, kind, text, lifecycle) {
  transcriptEntries.push({ speaker, ts, kind, text, lifecycle });
  const entry = transcriptEntries[transcriptEntries.length - 1];
  const timeStr = new Date(entry.ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const kindBadge =
    entry.kind === "final" ? "final" :
    entry.kind === "live" ? "live" :
    "speculative";

  const div = document.createElement("div");
  div.className = `transcriptEntry ${transcriptRowModifier(kindBadge)}`;
  if (lifecycle) div.dataset.lifecycle = lifecycle;
  div.innerHTML = `
    <span class="ts mono">${escapeHtml(timeStr)}</span>
    <span class="kind ${kindBadge}">${escapeHtml(kindBadge)}</span>
    <span class="speaker">${escapeHtml(entry.speaker)}</span>:
    ${escapeHtml(entry.text || "")}
  `;

  if (els.transcriptList) {
    els.transcriptList.appendChild(div);
    scrollTranscriptToBottom(false);
  }
}

function resetCoachUiToEmpty() {
  if (els.coachEmptyState) els.coachEmptyState.hidden = false;
  if (els.coachSpeculative) els.coachSpeculative.hidden = true;
  if (els.coachStream) els.coachStream.hidden = true;
  if (els.coachStreamText) els.coachStreamText.textContent = "";
  if (els.coachFinal) els.coachFinal.hidden = true;
  coachStreamText = "";
  if (els.questionType) els.questionType.textContent = "";
  if (els.answerOutline) els.answerOutline.textContent = "";
  if (els.matchedThemes) els.matchedThemes.innerHTML = "";
  if (els.suggestions) els.suggestions.innerHTML = "";
  if (els.followUp) els.followUp.textContent = "";
  if (els.bridge) els.bridge.textContent = "";
  if (els.confidence) els.confidence.textContent = "—";
}

// -------- Coach history --------
function addCoachHistoryCard(kind, speaker, transcript, payload) {
  if (!els.coachHistory || !payload) return;

  const card = {
    ts: Date.now() / 1000,
    kind,
    speaker,
    transcript,
    payload,
  };
  coachHistory.unshift(card);
  coachHistory = coachHistory.slice(0, 24);

  const items = coachHistory.map((item) => {
    const timeStr = new Date(item.ts * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    const suggestions = (item.payload.suggestions || [])
      .map((s) => `<li>${escapeHtml(s)}</li>`)
      .join("");

    const conf =
      typeof item.payload.confidence === "number" ? item.payload.confidence.toFixed(2) : "—";

    return `
      <div class="coachHistCard">
        <div class="histMeta">${escapeHtml(timeStr)}</div>
        <div class="histQ"><strong>Interviewer</strong>: ${escapeHtml(item.transcript || "")}</div>
        <ol class="list tightList">${suggestions}</ol>
        <div class="histConf">Confidence: ${escapeHtml(conf)}</div>
      </div>
    `;
  }).join("");

  els.coachHistory.innerHTML = items;
}

function renderCoachSpeculative(payload) {
  if (!payload) return;

  if (els.coachEmptyState) els.coachEmptyState.hidden = true;
  if (els.coachSpeculative) els.coachSpeculative.hidden = false;
  if (els.coachStream) els.coachStream.hidden = true;
  if (els.coachFinal) els.coachFinal.hidden = true;

  if (els.questionType) {
    els.questionType.textContent = "Question type: " + (payload.question_type || "—");
  }
  if (els.answerOutline) {
    els.answerOutline.textContent = payload.answer_outline || "";
  }
  if (els.matchedThemes) {
    const themes = payload.matched_themes || [];
    els.matchedThemes.innerHTML = themes.map((t) => `<li>${escapeHtml(t)}</li>`).join("");
  }
}

function renderCoachStreamDelta(payload) {
  if (!payload || payload.response_type !== "coach_stream") return;
  coachStreamText += payload.text_delta || "";

  if (els.coachEmptyState) els.coachEmptyState.hidden = true;
  if (els.coachSpeculative) els.coachSpeculative.hidden = true;
  if (els.coachFinal) els.coachFinal.hidden = true;
  if (els.coachStream) els.coachStream.hidden = false;
  if (els.coachStreamText) els.coachStreamText.textContent = coachStreamText;
}

function renderCoachFinal(payload) {
  if (!payload) return;

  if (els.coachEmptyState) els.coachEmptyState.hidden = true;
  if (els.coachSpeculative) els.coachSpeculative.hidden = true;
  if (els.coachStream) els.coachStream.hidden = true;
  if (els.coachStreamText) els.coachStreamText.textContent = "";
  coachStreamText = "";
  if (els.coachFinal) els.coachFinal.hidden = false;

  const sug = payload.suggestions || [];
  if (els.suggestions) {
    els.suggestions.innerHTML = sug.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  }

  if (els.followUp) els.followUp.textContent = payload.follow_up || "";
  if (els.bridge) els.bridge.textContent = payload.bridge || "";
  if (els.confidence) {
    els.confidence.textContent =
      typeof payload.confidence === "number" ? payload.confidence.toFixed(2) : "—";
  }
}

function renderAnswerMonitor(payload) {
  if (!payload || !els.answerSignals) return;
  const signals = payload.signals || [];
  els.answerSignals.innerHTML = signals.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
}

// -------- Notes panel --------
function renderNotes(notes) {
  if (!notes) return;

  if (els.summarySoFar) els.summarySoFar.textContent = notes.summary_so_far || "—";
  if (els.currentTopic) els.currentTopic.textContent = notes.current_topic || "—";

  const oq = notes.open_questions || [];
  if (els.openQuestions) {
    els.openQuestions.innerHTML = oq.length ? oq.map((q) => `<li>${escapeHtml(q)}</li>`).join("") : '<li class="muted">—</li>';
  }

  const bullets = (notes.bullets || []).slice(-25);
  if (els.notesBullets) {
    els.notesBullets.innerHTML = bullets.length
      ? bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join("")
      : '<li class="muted">Capture will appear here as you speak.</li>';
  }

  const actions = notes.action_items || [];
  if (els.actionItems) {
    els.actionItems.innerHTML = actions.length ? actions.map((a) => `<li>${escapeHtml(a)}</li>`).join("") : '<li class="muted">—</li>';
  }

  const dec = notes.decisions || [];
  if (els.decisions) {
    els.decisions.innerHTML = dec.length ? dec.map((d) => `<li>${escapeHtml(d)}</li>`).join("") : '<li class="muted">—</li>';
  }

  const fu = notes.follow_ups || [];
  if (els.followUps) {
    els.followUps.innerHTML = fu.length ? fu.map((f) => `<li>${escapeHtml(f)}</li>`).join("") : '<li class="muted">—</li>';
  }
}

// -------- Cost panel --------
function renderCost(data) {
  const usage = data?.usage || {};
  const turn = usage?.turn || {};

  const apply = () => {
    if (els.turnCost) els.turnCost.textContent = fmtUsd(turn?.cost_usd);
    if (els.sessionCost) els.sessionCost.textContent = fmtUsd(usage?.cost_usd_total);
    if (els.turnCostNotes) els.turnCostNotes.textContent = fmtUsd(turn?.cost_usd);
    if (els.sessionCostNotes) els.sessionCostNotes.textContent = fmtUsd(usage?.cost_usd_total);

    const byFeature = usage?.by_feature || {};
    const bf = JSON.stringify(byFeature, null, 2);
    if (els.byFeature) els.byFeature.textContent = bf;
    if (els.byFeatureNotes) els.byFeatureNotes.textContent = bf;

    const total = Number(usage?.cost_usd_total) || 0;
    const lastSeen = usage?.last_seen_at || Date.now() / 1000;
    const created = usage?.created_at || sessionStartAt || lastSeen;
    const minutes = Math.max(0.001, (lastSeen - created) / 60);
    const perMin = total / minutes;
    const pm = fmtUsd(perMin) + "/min";
    if (els.perMinEst) els.perMinEst.textContent = pm;
    if (els.perMinEstNotes) els.perMinEstNotes.textContent = pm;
  };

  apply();

  lastUsage = usage;

  if (!sessionStartAt && usage?.created_at) {
    sessionStartAt = usage.created_at;
  }
}

function renderContextList(items) {
  uploadedContextItems = Array.isArray(items) ? items : [];
  if (!els.contextList) return;
  if (!uploadedContextItems.length) {
    els.contextList.innerHTML = '<li class="muted">No uploaded context.</li>';
    return;
  }
  els.contextList.innerHTML = uploadedContextItems.map((item) => {
    const name = item.filename || "context";
    const count = Number(item.chunk_count || 0);
    return `<li><span>${escapeHtml(name)}</span> <span class="muted mono">${count} chunks</span></li>`;
  }).join("");
}

function updateSessionLogLinks() {
  const href = `/session/${encodeURIComponent(els.sessionId.value)}/log/download`;
  if (els.downloadSessionLog) els.downloadSessionLog.href = href;
  if (els.downloadSessionLogNotes) els.downloadSessionLogNotes.href = href;
}

function renderSessionLog(events) {
  sessionLogEvents = Array.isArray(events) ? events : [];
  const label = `${sessionLogEvents.length} events`;
  if (els.sessionLogCount) els.sessionLogCount.textContent = label;
  if (els.sessionLogCountNotes) els.sessionLogCountNotes.textContent = label;

  const recent = sessionLogEvents.slice(-5).reverse();
  const html = recent.length
    ? recent.map((event) => {
        const eventType = event.event_type || "event";
        const responseType = event.response_type || "";
        return `<li><span>${escapeHtml(eventType)}</span> <span class="muted mono">${escapeHtml(responseType)}</span></li>`;
      }).join("")
    : '<li class="muted">No logged events.</li>';

  if (els.sessionLogPreview) els.sessionLogPreview.innerHTML = html;
  if (els.sessionLogPreviewNotes) els.sessionLogPreviewNotes.innerHTML = html;
  updateSessionLogLinks();
}

async function refreshSessionLog() {
  updateSessionLogLinks();
  try {
    const res = await fetch(`/session/${encodeURIComponent(els.sessionId.value)}/log`);
    if (!res.ok) return;
    const data = await res.json();
    renderSessionLog(data.events || []);
  } catch (_) {}
}

async function refreshUploadedContexts() {
  try {
    const res = await fetch(`/session/${encodeURIComponent(els.sessionId.value)}/upload-context`);
    if (!res.ok) return;
    const data = await res.json();
    renderContextList(data.items || []);
  } catch (_) {}
}

async function uploadSelectedContext() {
  const file = els.contextFile?.files?.[0];
  if (!file) {
    setContextUploadStatus("Choose a .txt or .md file first.");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  setContextUploadStatus("Uploading...");

  try {
    const res = await fetch(`/session/${encodeURIComponent(els.sessionId.value)}/upload-context`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setContextUploadStatus(data.detail || "Upload failed.");
      return;
    }
    setContextUploadStatus(`${data.filename} uploaded (${data.chunk_count} chunks).`);
    if (els.contextFile) els.contextFile.value = "";
    await refreshUploadedContexts();
    await refreshSessionLog();
  } catch (_) {
    setContextUploadStatus("Upload failed.");
  }
}

function calcLatencyMs(lat, startKey, endKey) {
  const start = lat?.[startKey];
  const end = lat?.[endKey];
  if (typeof start !== "number" || typeof end !== "number") return null;
  return Math.max(0, (end - start) * 1000);
}

function fmtLatencyMs(value) {
  return typeof value === "number" ? `${value.toFixed(1)} ms` : "n/a";
}

function updateDebugPanel(data) {
  const lat = data?.latency && typeof data.latency === "object" ? data.latency : {};
  const renderedAt = lat.frontend_rendered_at || Date.now() / 1000;
  const latencyWithRender = { ...lat, frontend_rendered_at: renderedAt };
  if (typeof latencyWithRender.end_to_end_estimate_ms !== "number") {
    latencyWithRender.end_to_end_estimate_ms = calcLatencyMs(latencyWithRender, "frontend_sent_at", "frontend_rendered_at");
  }
  const interp = data?.interpretation && typeof data.interpretation === "object" ? data.interpretation : null;
  const lines = {
    ui_mode: currentMode,
    speaker_mode_ui: speakerMode,
    notes_audio: notesAudioSource,
    speaker_mode: data?.speaker_mode ?? "—",
    audio_source: data?.audio_source ?? "—",
    ws: ws ? (ws.readyState === WebSocket.OPEN ? "open" : String(ws.readyState)) : "null",
    interpreted_question: data?.interpreted_question ?? null,
    interpretation: interp,
    route: data?.route ?? "—",
    route_confidence: data?.route_confidence ?? "—",
    retrieval_source: data?.retrieval_source ?? "—",
    latency_summary: {
      backend_total: fmtLatencyMs(latencyWithRender.backend_total_ms),
      interpreter: fmtLatencyMs(latencyWithRender.interpreter_ms),
      router: fmtLatencyMs(latencyWithRender.router_ms),
      retrieval: fmtLatencyMs(latencyWithRender.retrieval_ms),
      llm: fmtLatencyMs(latencyWithRender.llm_ms),
      end_to_end_estimate: fmtLatencyMs(latencyWithRender.end_to_end_estimate_ms),
    },
    latency: latencyWithRender,
    system_audio: typeof getSystemAudioDebugInfo === 'function' ? getSystemAudioDebugInfo() : null,
  };
  const text = JSON.stringify(lines, null, 2);
  if (els.debugBody) els.debugBody.textContent = text;
  if (els.debugBodyNotes) els.debugBodyNotes.textContent = text;
}

// -------- Transcript / render pipeline (Phase F) --------

function ingestTranscriptFinalFromServer(data, kind) {
  const ts =
    typeof data?.usage?.last_seen_at === "number"
      ? data.usage.last_seen_at
      : Date.now() / 1000;
  if (data?.speaker != null && data?.transcript != null) {
    if (!alreadyHasRecentTranscript(data.transcript, data.speaker)) {
      const lineKind = kind || data.kind || "final";
      appendTranscriptFromEvent(
        data.speaker,
        ts,
        lineKind,
        data.transcript,
        TranscriptLifecycle.TRANSCRIPT_FINAL
      );
    }
  }
}

function ingestCoachResponse(data, kind) {
  const rt = data?.response_type;
  if (rt === "coach_speculative" && data.coach_speculative) {
    renderCoachSpeculative(data.coach_speculative);
    return;
  }
  if (rt === "coach_final" && data.coach_final) {
    renderCoachFinal(data.coach_final);
    addCoachHistoryCard(kind || "final", data.speaker, data.transcript, data.coach_final);
  }
}

function ingestNotesUpdate(data) {
  if (data?.response_type === "notes_final" && data.notes_final?.notes) {
    renderNotes(data.notes_final.notes);
  } else if (data?.notes_final?.notes) {
    renderNotes(data.notes_final.notes);
  } else if (data?.notes) {
    renderNotes(data.notes);
  }
}

function renderFromPayload(data, kind) {
  if (data && typeof data === "object") {
    if (!data.latency || typeof data.latency !== "object") data.latency = {};
    data.latency.frontend_rendered_at = Date.now() / 1000;
  }
  lastEmitSnapshot = { ...(data || {}), _ingest_kind: kind };
  ingestTranscriptFinalFromServer(data, kind);

  const rt = data?.response_type;

  if (rt === "coach_skipped") {
    renderCost(data);
    updateDebugPanel(data);
    return;
  }

  if (rt === "answer_monitor" && data.answer_monitor) {
    renderAnswerMonitor(data.answer_monitor);
    renderCost(data);
    updateDebugPanel(data);
    return;
  }

  if (rt === "coach_speculative" || rt === "coach_final") {
    ingestCoachResponse(data, kind);
  }

  if (rt === "notes_final" || data?.notes_final?.notes) {
    ingestNotesUpdate(data);
  }

  renderCost(data);
  updateDebugPanel(data);
}

// -------- Mode toggle --------
function updateModeButtons() {
  if (mainWorkspace) {
    mainWorkspace.dataset.mode = currentMode;
    mainWorkspace.classList.add("mode-transitioning");
    window.requestAnimationFrame(() => {
      setTimeout(() => mainWorkspace.classList.remove("mode-transitioning"), 220);
    });
  }

  if (els.modeCoach && els.modeNotes) {
    els.modeCoach.classList.toggle("active", currentMode === "coach");
    els.modeNotes.classList.toggle("active", currentMode === "notes");
  }

  const coachMid = els.coachMiddlePanel;
  const notesMid = els.notesMiddlePanel;
  const coachR = els.coachRightStack;
  const notesR = els.notesRightStack;

  if (coachMid) coachMid.hidden = currentMode !== "coach";
  if (notesMid) notesMid.hidden = currentMode !== "notes";
  if (coachR) coachR.hidden = currentMode !== "coach";
  if (notesR) notesR.hidden = currentMode !== "notes";

  const notesRow = document.getElementById("notesAudioRow");
  if (notesRow) notesRow.hidden = currentMode !== "notes";
  refreshMicAvailability();
  if (typeof updateSystemAudioEmphasize === 'function') updateSystemAudioEmphasize();
}

if (els.modeCoach) {
  els.modeCoach.addEventListener("click", () => {
    currentMode = "coach";
    updateModeButtons();
    fetch(`/session/${encodeURIComponent(els.sessionId.value)}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "coach" }),
    }).catch(() => {});
  });
}

if (els.modeNotes) {
  els.modeNotes.addEventListener("click", () => {
    currentMode = "notes";
    updateModeButtons();
    fetch(`/session/${encodeURIComponent(els.sessionId.value)}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "notes" }),
    }).catch(() => {});
  });
}

loadAudioPrefs();
syncNotesAudioRadios();
syncSpeakerUi();
updateModeButtons();

document.querySelectorAll('input[name="speakerMode"]').forEach((el) => {
  el.addEventListener("change", () => {
    if (!el.checked) return;
    speakerMode = el.value;
    if (speakerMode !== "manual" && speakerMode !== "source") return;
    localStorage.setItem(LS_SPEAKER_MODE, speakerMode);
    syncSpeakerUi();
  });
});

document.querySelectorAll('input[name="notesAudio"]').forEach((el) => {
  el.addEventListener("change", () => {
    if (!el.checked) return;
    notesAudioSource = el.value;
    if (!["mic", "system", "both"].includes(notesAudioSource)) return;
    localStorage.setItem(LS_NOTES_AUDIO, notesAudioSource);
    refreshMicAvailability();
    if (typeof updateSystemAudioEmphasize === 'function') updateSystemAudioEmphasize();
    if (micOn && currentMode === "notes" && notesAudioSource === "system") {
      stopMic();
    }
  });
});

let systemAudioStream = null;
const systemAudioBtn = document.getElementById("systemAudioBtn");

let sttConfig = { enabled: false, token: null };
let deepgramWs = null;
let deepgramRecorder = null;
let sttStatus = "disabled";

fetch("/api/stt-config")
  .then(res => res.json())
  .then(data => {
    if (data.enabled && data.token) {
      sttConfig = { enabled: true, token: data.token };
    }
  })
  .catch(err => console.error("Failed to fetch STT config:", err));


function setSystemAudioStatus(s) {
  const el = document.getElementById("systemAudioStatus");
  let display = s;
  if (sttConfig.enabled && !!systemAudioStream) {
    display += ` (STT: ${sttStatus})`;
  } else if (!sttConfig.enabled && !!systemAudioStream) {
    display += ` (System STT unavailable)`;
  }
  if (el) el.textContent = display;
}

function updateSttStatus(s) {
  sttStatus = s;
  if (systemAudioStream) {
    setSystemAudioStatus("active with audio track");
  }
}

function getSystemAudioDebugInfo() {
  const info = {
    active: !!systemAudioStream,
    stt_status: sttStatus,
    audioTrackCount: 0,
    videoTrackCount: 0,
    audioTracks: [],
    videoTracks: []
  };
  if (systemAudioStream) {
    const audio = systemAudioStream.getAudioTracks();
    const video = systemAudioStream.getVideoTracks();
    info.audioTrackCount = audio.length;
    info.videoTrackCount = video.length;
    info.audioTracks = audio.map(t => ({
      label: t.label,
      enabled: t.enabled,
      muted: t.muted,
      readyState: t.readyState
    }));
    info.videoTracks = video.map(t => ({
      label: t.label,
      enabled: t.enabled,
      muted: t.muted,
      readyState: t.readyState
    }));
  }
  return info;
}

function updateSystemAudioEmphasize() {
  const warningEl = document.getElementById("notesSystemWarning");
  if (!systemAudioBtn) return;
  
  const wantsSystem = currentMode === "notes" && (notesAudioSource === "system" || notesAudioSource === "both");
  if (wantsSystem && !systemAudioStream) {
    systemAudioBtn.classList.remove("btn-ghost");
    if (warningEl && notesAudioSource === "system") {
      warningEl.hidden = false;
    } else if (warningEl) {
      warningEl.hidden = true;
    }
  } else {
    systemAudioBtn.classList.add("btn-ghost");
    if (warningEl) warningEl.hidden = true;
  }
}

// Call initially
if (typeof updateSystemAudioEmphasize === 'function') {
  updateSystemAudioEmphasize();
}


function cleanupDeepgram() {
  if (deepgramRecorder) {
    try { deepgramRecorder.stop(); } catch(e) {}
    deepgramRecorder = null;
  }
  if (deepgramWs) {
    try { deepgramWs.close(); } catch(e) {}
    deepgramWs = null;
  }
  updateSttStatus(sttConfig.enabled ? "closed" : "disabled");
}

if (systemAudioBtn) {
  systemAudioBtn.addEventListener("click", async () => {
    if (systemAudioStream) {
      systemAudioStream.getTracks().forEach(t => t.stop());
      systemAudioStream = null;
      cleanupDeepgram();
      systemAudioBtn.textContent = "Start system audio";
      systemAudioBtn.classList.remove("active");
      setSystemAudioStatus("stopped");
      updateSystemAudioEmphasize();
      updateDebugPanel(lastEmitSnapshot);
      return;
    }

    try {
      setSystemAudioStatus("requesting permission");
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      const audioTracks = stream.getAudioTracks();
      
      if (audioTracks.length === 0) {
        stream.getTracks().forEach(t => t.stop());
        setSystemAudioStatus("stopped — no audio shared. Please check 'Share audio'.");
        return;
      }
      
      systemAudioStream = stream;

      if (sttConfig.enabled && sttConfig.token) {
        updateSttStatus("connecting");
        deepgramWs = new WebSocket("wss://api.deepgram.com/v1/listen?punctuate=true&model=nova-2", ["token", sttConfig.token]);
        
        deepgramWs.onopen = () => {
          updateSttStatus("connected");
          deepgramRecorder = new MediaRecorder(systemAudioStream, { mimeType: 'audio/webm' });
          deepgramRecorder.ondataavailable = (e) => {
            if (e.data.size > 0 && deepgramWs.readyState === WebSocket.OPEN) {
              deepgramWs.send(e.data);
            }
          };
          deepgramRecorder.start(250);
          updateSttStatus("transcribing");
          updateDebugPanel(lastEmitSnapshot);
        };

        deepgramWs.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            const isFinal = data.is_final;
            const transcript = data.channel?.alternatives?.[0]?.transcript;
            
            if (isFinal && transcript && transcript.trim()) {
              sendMsg({
                ...wsCommonFields(),
                mode: "replace",
                text_delta: transcript,
                final: true,
                speaker: resolveSpeakerForSource("system"),
                audio_source: "system"
              });
            }
          } catch(e) {
            console.error("Deepgram parsing error:", e);
          }
        };

        deepgramWs.onerror = () => {
          updateSttStatus("error");
          updateDebugPanel(lastEmitSnapshot);
        };
        
        deepgramWs.onclose = () => {
          updateSttStatus("closed");
          updateDebugPanel(lastEmitSnapshot);
        };
      } else {
        updateSttStatus("disabled");
      }

      setSystemAudioStatus("active with audio track");
      systemAudioBtn.textContent = "Stop system audio";
      systemAudioBtn.classList.add("active");
      updateSystemAudioEmphasize();
      updateDebugPanel(lastEmitSnapshot);
      
      stream.getTracks().forEach(track => {
        track.onended = () => {
          if (systemAudioStream) {
            systemAudioStream.getTracks().forEach(t => t.stop());
            systemAudioStream = null;
            cleanupDeepgram();
            systemAudioBtn.textContent = "Start system audio";
            systemAudioBtn.classList.remove("active");
            setSystemAudioStatus("stopped");
            updateSystemAudioEmphasize();
            updateDebugPanel(lastEmitSnapshot);
          }
        };
      });
    } catch (err) {
      console.error("Failed to get system audio:", err);
      setSystemAudioStatus("error");
      updateSystemAudioEmphasize();
      updateDebugPanel(lastEmitSnapshot);
    }
  });
}

// -------- New session --------
if (els.newSession) {
  els.newSession.addEventListener("click", () => {
    els.sessionId.value = randId();
    localStorage.setItem("assist_session_id", els.sessionId.value);

    if (systemAudioStream) {
      systemAudioStream.getTracks().forEach(t => t.stop());
      systemAudioStream = null;
      if (typeof cleanupDeepgram === 'function') cleanupDeepgram();
      if (systemAudioBtn) {
        systemAudioBtn.textContent = "Start system audio";
        systemAudioBtn.classList.remove("active");
      }
      setSystemAudioStatus("inactive");
      if (typeof updateSystemAudioEmphasize === 'function') updateSystemAudioEmphasize();
    }

    transcriptEntries = [];
    coachHistory = [];
    sessionStartAt = null;
    lastFinalizedMicText = "";
    lastSentReplace = "";
    removeLivePreview();

    if (els.transcriptList) els.transcriptList.innerHTML = "";
    if (els.coachHistory) els.coachHistory.innerHTML = '<div class="muted emptyCoachHist">No coach history yet.</div>';
    if (els.answerSignals) els.answerSignals.innerHTML = "";

    resetCoachUiToEmpty();

    renderNotes({});
    renderCost({});
    updateDebugPanel({});
    renderContextList([]);
    renderSessionLog([]);
    setContextUploadStatus("");
  });
}

// -------- Speech --------
function ensureSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;

  const r = new SR();
  r.continuous = false;
  r.interimResults = true;
  r.lang = "en-US";
  r.maxAlternatives = 1;
  return r;
}

function startMic() {
  if (micOn) return;

  if (currentMode === "notes" && notesAudioSource === "system") {
    setStatus("Notes (system only): use manual send until system capture is available");
    return;
  }

  recognition = ensureSpeech();
  if (!recognition) {
    alert("Speech recognition not supported. Use manual text.");
    return;
  }

  micOn = true;
  userStoppedMic = false;
  micHardError = false;
  lastSentReplace = "";
  lastFinalizedMicText = "";
  removeLivePreview();

  if (els.mic) els.mic.textContent = "Stop mic";

  recognition.onstart = () => {
    console.log("Speech recognition started");
    setStatus("listening...");
  };

  recognition.onerror = (e) => {
    console.log("Speech recognition error:", e);

    if (e?.error === "not-allowed" || e?.error === "service-not-allowed") {
      micHardError = true;
      micOn = false;
      if (els.mic) els.mic.textContent = "Mic";
      setStatus("mic permission denied");
      return;
    }

    if (e?.error === "audio-capture") {
      micHardError = true;
      micOn = false;
      if (els.mic) els.mic.textContent = "Mic";
      setStatus("mic not available");
      return;
    }

    if (e?.error === "network" || e?.error === "no-speech") {
      setStatus("mic listening...");
      return;
    }

    setStatus("mic error");
  };

  recognition.onend = () => {
    console.log("Speech recognition ended");

    if (micFinalizeTimer) {
      clearTimeout(micFinalizeTimer);
      micFinalizeTimer = null;
    }

    const full = cleanTranscriptForSend(lastSentReplace || "");
    if (full && !isTooShortForFinal(full)) {
      maybeSendMicFinal(full);
    }

    if (micOn && !userStoppedMic && !micHardError) {
      setStatus("listening...");
      try {
        recognition.start();
        return;
      } catch (_) {}
    }

    micOn = false;
    if (els.mic) els.mic.textContent = "Mic";
    setStatus(ws && ws.readyState === WebSocket.OPEN ? "connected" : "disconnected");
  };

  recognition.onresult = (ev) => {
    console.log("speech result", ev);

    let interim = "";
    let finalText = "";

    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const res = ev.results[i];
      const txt = res[0]?.transcript || "";
      if (res.isFinal) finalText += txt;
      else interim += txt;
    }

    const full = cleanTranscriptForSend((finalText + " " + interim).trim());
    if (!full) return;

    updateLivePreview(full);

    lastSentReplace = full;

    if (isMeaningfulTranscript(full)) {
      scheduleMicFinalize(full);
    }
  };

  try {
    recognition.start();
  } catch (_) {
    micOn = false;
    if (els.mic) els.mic.textContent = "Mic";
    setStatus("mic failed");
  }
}

function stopMic() {
  userStoppedMic = true;
  micOn = false;

  if (els.mic) els.mic.textContent = "Mic";

  if (micFinalizeTimer) {
    clearTimeout(micFinalizeTimer);
    micFinalizeTimer = null;
  }

  const full = cleanTranscriptForSend(lastSentReplace || "");
  if (full && !isTooShortForFinal(full)) {
    maybeSendMicFinal(full);
  }

  if (recognition) {
    try {
      recognition.stop();
    } catch (_) {}
  }

  setStatus(ws && ws.readyState === WebSocket.OPEN ? "connected" : "disconnected");
}

// -------- Connect --------
if (els.connect) {
  els.connect.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
      return;
    }

    ws = new WebSocket(wsUrl());
    setStatus("connecting…");

    ws.onopen = () => {
      console.log("WebSocket opened");
      setStatus("connected");
      refreshMicAvailability();
      if (els.send) els.send.disabled = false;
      els.connect.textContent = "Disconnect";
      localStorage.setItem("assist_session_id", els.sessionId.value);
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      setStatus("disconnected");
      if (els.mic) els.mic.disabled = true;
      if (els.send) els.send.disabled = true;
      els.connect.textContent = "Connect";
      if (micOn) stopMic();
      updateDebugPanel(lastEmitSnapshot);
    };

    ws.onerror = () => {
      setStatus("error");
      alert("Check your terminal for backend errors.");
    };

    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }

      if (msg?.error) {
        console.error("WS backend error:", msg.error);
        setStatus("error: " + msg.error);
        return;
      }

      if (msg?.emit && msg?.data) {
        if (msg.kind === "answer_delta" && msg.data.response_type === "coach_stream") {
          renderCoachStreamDelta(msg.data);
          return;
        }
        removeLivePreview();
        const kind = msg.kind || "final";
        renderFromPayload(msg.data, kind);
        refreshSessionLog();
      }
    };
  });
}

if (els.mic) {
  els.mic.addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!micOn) startMic();
    else stopMic();
  });
}

if (els.send) {
  els.send.addEventListener("click", () => {
    const txt = cleanTranscriptForSend(String(els.manual?.value || "").trim());
    if (!txt) return;

    sendMsg({
      ...wsCommonFields(),
      text_delta: txt,
      final: true,
      speaker: speakerMode === "manual" ? getSpeaker() : "Me",
      audio_source: "manual",
    });

    if (els.manual) els.manual.value = "";
  });
}

if (els.manual) {
  els.manual.addEventListener("keydown", (e) => {
    if (e.key === "Enter") els.send?.click();
  });
}

// -------- Cost refresh --------
async function refreshUsage() {
  try {
    const res = await fetch(`/session/${encodeURIComponent(els.sessionId.value)}/usage`);
    if (!res.ok) return;

    const data = await res.json();
    const usage = data.usage || {};
    usage.created_at = data.created_at ?? usage.created_at;
    usage.last_seen_at = data.last_seen_at ?? usage.last_seen_at;
    lastUsage = usage;

    if (usage.created_at) sessionStartAt = usage.created_at;
    renderCost({ usage });
  } catch (_) {}
}

if (els.refreshUsage) {
  els.refreshUsage.addEventListener("click", refreshUsage);
}
if (els.refreshUsageNotes) {
  els.refreshUsageNotes.addEventListener("click", refreshUsage);
}
if (els.uploadContext) {
  els.uploadContext.addEventListener("click", uploadSelectedContext);
}
if (els.refreshSessionLog) {
  els.refreshSessionLog.addEventListener("click", refreshSessionLog);
}
if (els.refreshSessionLogNotes) {
  els.refreshSessionLogNotes.addEventListener("click", refreshSessionLog);
}

// Initial debug
updateDebugPanel({});
renderContextList([]);
renderSessionLog([]);
if (els.coachHistory && !els.coachHistory.textContent?.trim()) {
  els.coachHistory.innerHTML = '<div class="muted emptyCoachHist">No coach history yet.</div>';
}
