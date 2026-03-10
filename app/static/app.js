/* global webkitSpeechRecognition */

function randId() {
  return Math.random().toString(16).slice(2) + "-" + Date.now().toString(16);
}

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
  coachFinal: document.getElementById("coachFinal"),
  questionType: document.getElementById("questionType"),
  answerOutline: document.getElementById("answerOutline"),
  matchedThemes: document.getElementById("matchedThemes"),
  suggestions: document.getElementById("suggestions"),
  followUp: document.getElementById("followUp"),
  bridge: document.getElementById("bridge"),
  confidence: document.getElementById("confidence"),
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
  modeCoach: document.getElementById("modeCoach"),
  modeNotes: document.getElementById("modeNotes"),
};

function getSpeaker() {
  const s = document.getElementById("speaker");
  return s ? s.value : "Me";
}

els.sessionId.value = localStorage.getItem("assist_session_id") || randId();

let ws = null;
let recognition = null;
let micOn = false;
let lastSentReplace = "";
let replaceTimer = null;
let currentMode = "coach";
let transcriptEntries = [];
let lastUsage = null;
let sessionStartAt = null;

function setStatus(s) {
  if (els.status) els.status.textContent = s;
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function sendMsg(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
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

// -------- 1. Transcript: speaker, timestamp, kind (speculative/final), raw text --------
function appendTranscript(speaker, ts, kind, text) {
  transcriptEntries.push({ speaker, ts, kind, text });
  const entry = transcriptEntries[transcriptEntries.length - 1];
  const timeStr = new Date(entry.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const kindBadge = entry.kind === "final" ? "final" : "speculative";
  const div = document.createElement("div");
  div.className = "transcriptEntry";
  div.innerHTML = `<span class="ts mono">${escapeHtml(timeStr)}</span> <span class="kind ${kindBadge}">${escapeHtml(kindBadge)}</span> <span class="speaker">${escapeHtml(entry.speaker)}</span>: ${escapeHtml(entry.text || "")}`;
  if (els.transcriptList) {
    els.transcriptList.appendChild(div);
    els.transcriptList.scrollTop = els.transcriptList.scrollHeight;
  }
}

// -------- 3. Coach panel: speculative vs final --------
function renderCoachSpeculative(payload) {
  if (!payload) return;
  if (els.questionType) els.questionType.textContent = "Question type: " + (payload.question_type || "—");
  if (els.answerOutline) els.answerOutline.textContent = payload.answer_outline || "";
  if (els.matchedThemes) {
    const themes = payload.matched_themes || [];
    els.matchedThemes.innerHTML = themes.map((t) => `<li>${escapeHtml(t)}</li>`).join("");
  }
  if (els.coachSpeculative) els.coachSpeculative.style.display = "block";
  if (els.coachFinal) els.coachFinal.style.display = "none";
}

function renderCoachFinal(payload) {
  if (!payload) return;
  const sug = payload.suggestions || [];
  if (els.suggestions) els.suggestions.innerHTML = sug.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  if (els.followUp) els.followUp.textContent = payload.follow_up || "";
  if (els.bridge) els.bridge.textContent = payload.bridge || "";
  if (els.confidence) els.confidence.textContent = typeof payload.confidence === "number" ? payload.confidence.toFixed(2) : "—";
  if (els.coachFinal) els.coachFinal.style.display = "block";
  if (els.coachSpeculative) els.coachSpeculative.style.display = "none";
}

// -------- 4. Notes panel: structured --------
function renderNotes(notes) {
  if (!notes) return;
  if (els.summarySoFar) els.summarySoFar.textContent = notes.summary_so_far || "—";
  if (els.currentTopic) els.currentTopic.textContent = notes.current_topic || "—";
  const oq = notes.open_questions || [];
  if (els.openQuestions) els.openQuestions.innerHTML = oq.map((q) => `<li>${escapeHtml(q)}</li>`).join("");
  const bullets = (notes.bullets || []).slice(-25);
  if (els.notesBullets) els.notesBullets.innerHTML = bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join("");
  const actions = notes.action_items || [];
  if (els.actionItems) els.actionItems.innerHTML = actions.map((a) => `<li>${escapeHtml(a)}</li>`).join("");
  const dec = notes.decisions || [];
  if (els.decisions) els.decisions.innerHTML = dec.map((d) => `<li>${escapeHtml(d)}</li>`).join("");
  const fu = notes.follow_ups || [];
  if (els.followUps) els.followUps.innerHTML = fu.map((f) => `<li>${escapeHtml(f)}</li>`).join("");
}

// -------- 5. Cost panel --------
function renderCost(data) {
  const usage = data?.usage || {};
  const turn = usage?.turn || {};
  if (els.turnCost) els.turnCost.textContent = fmtUsd(turn?.cost_usd);
  if (els.sessionCost) els.sessionCost.textContent = fmtUsd(usage?.cost_usd_total);
  const byFeature = usage?.by_feature || {};
  if (els.byFeature) els.byFeature.textContent = JSON.stringify(byFeature, null, 2);
  lastUsage = usage;
  if (!sessionStartAt && usage?.created_at) sessionStartAt = usage.created_at;
  const total = Number(usage?.cost_usd_total) || 0;
  const lastSeen = usage?.last_seen_at || Date.now() / 1000;
  const created = usage?.created_at || sessionStartAt || lastSeen;
  const minutes = Math.max(0.001, (lastSeen - created) / 60);
  const perMin = total / minutes;
  if (els.perMinEst) els.perMinEst.textContent = fmtUsd(perMin) + "/min";
}

function renderFromPayload(data, kind) {
  const rt = data?.response_type;
  const ts = typeof data?.usage?.last_seen_at === "number" ? data.usage.last_seen_at : Date.now() / 1000;

  if (data?.speaker != null && data?.transcript != null) {
    appendTranscript(data.speaker, ts, kind || data.kind || "final", data.transcript);
  }

  if (rt === "coach_speculative" && data.coach_speculative) {
    renderCoachSpeculative(data.coach_speculative);
  } else if (rt === "coach_final" && data.coach_final) {
    renderCoachFinal(data.coach_final);
  }

  if (rt === "notes_final" && data.notes_final?.notes) {
    renderNotes(data.notes_final.notes);
  } else if (data?.notes_final?.notes) {
    renderNotes(data.notes_final.notes);
  } else if (data?.notes) {
    renderNotes(data.notes);
  }

  renderCost(data);
}

// -------- Mode toggle --------
function updateModeButtons() {
  if (els.modeCoach && els.modeNotes) {
    els.modeCoach.classList.toggle("active", currentMode === "coach");
    els.modeNotes.classList.toggle("active", currentMode === "notes");
  }
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
updateModeButtons();

// -------- New session --------
if (els.newSession) {
  els.newSession.addEventListener("click", () => {
    els.sessionId.value = randId();
    localStorage.setItem("assist_session_id", els.sessionId.value);
    transcriptEntries = [];
    sessionStartAt = null;
    if (els.transcriptList) els.transcriptList.innerHTML = "";
    if (els.coachSpeculative) els.coachSpeculative.style.display = "block";
    if (els.coachFinal) els.coachFinal.style.display = "none";
    renderNotes({});
    renderCost({});
  });
}

// -------- Speech --------
function ensureSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const r = new SR();
  r.continuous = true;
  r.interimResults = true;
  r.lang = "en-US";
  return r;
}

function startMic() {
  if (micOn) return;
  recognition = ensureSpeech();
  if (!recognition) {
    alert("Speech recognition not supported. Use manual text.");
    return;
  }
  micOn = true;
  if (els.mic) els.mic.textContent = "Stop mic";
  lastSentReplace = "";
  recognition.onresult = (ev) => {
    let interim = "";
    let finalText = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const res = ev.results[i];
      const txt = res[0]?.transcript || "";
      if (res.isFinal) finalText += txt;
      else interim += txt;
    }
    const full = (finalText + " " + interim).trim().replace(/\s+/g, " ");
    if (replaceTimer) return;
    replaceTimer = setTimeout(() => {
      replaceTimer = null;
      if (full && full !== lastSentReplace) {
        lastSentReplace = full;
        sendMsg({
          session_id: els.sessionId.value,
          mode: "replace",
          text: full,
          final: false,
          ts: Date.now() / 1000,
          speaker: getSpeaker(),
          session_mode: currentMode,
        });
      }
    }, 250);
  };
  recognition.onend = () => {
    const full = String(lastSentReplace || "").trim();
    if (full) {
      sendMsg({
        session_id: els.sessionId.value,
        mode: "replace",
        text: full,
        final: true,
        ts: Date.now() / 1000,
        speaker: getSpeaker(),
        session_mode: currentMode,
      });
      lastSentReplace = "";
    }
    if (micOn) try { recognition.start(); } catch (_) {}
  };
  try { recognition.start(); } catch (_) { micOn = false; if (els.mic) els.mic.textContent = "Mic"; }
}

function stopMic() {
  micOn = false;
  if (els.mic) els.mic.textContent = "Mic";
  if (recognition) try { recognition.stop(); } catch (_) {}
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
      setStatus("connected");
      if (els.mic) els.mic.disabled = false;
      if (els.send) els.send.disabled = false;
      els.connect.textContent = "Disconnect";
      localStorage.setItem("assist_session_id", els.sessionId.value);
    };
    ws.onclose = () => {
      setStatus("disconnected");
      if (els.mic) els.mic.disabled = true;
      if (els.send) els.send.disabled = true;
      els.connect.textContent = "Connect";
      if (micOn) stopMic();
    };
    ws.onerror = () => setStatus("error");
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg?.error) return;
      if (msg?.emit && msg?.data) {
        const kind = msg.kind || "final";
        renderFromPayload(msg.data, kind);
      }
    };
  });
}

if (els.mic) els.mic.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!micOn) startMic();
  else stopMic();
});

if (els.send) {
  els.send.addEventListener("click", () => {
    const txt = String(els.manual?.value || "").trim();
    if (!txt) return;
    sendMsg({
      session_id: els.sessionId.value,
      text_delta: txt,
      final: true,
      ts: Date.now() / 1000,
      speaker: getSpeaker(),
      session_mode: currentMode,
    });
    if (els.manual) els.manual.value = "";
  });
}
if (els.manual) els.manual.addEventListener("keydown", (e) => { if (e.key === "Enter") els.send?.click(); });

// -------- Cost: refresh from API --------
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
if (els.refreshUsage) els.refreshUsage.addEventListener("click", refreshUsage);
