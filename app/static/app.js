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
  speaker: document.getElementById("speaker"),
  liveBuffer: document.getElementById("liveBuffer"),
  manual: document.getElementById("manual"),
  send: document.getElementById("send"),
  transcript: document.getElementById("transcript"),
  suggestions: document.getElementById("suggestions"),
  confidence: document.getElementById("confidence"),
  turnCost: document.getElementById("turnCost"),
  sessionCost: document.getElementById("sessionCost"),
  usage: document.getElementById("usage"),
  notes: document.getElementById("notes"),
  modeCoach: document.getElementById("modeCoach"),
  modeNotes: document.getElementById("modeNotes"),
  modeLabel: document.getElementById("modeLabel"),
  refreshUsage: document.getElementById("refreshUsage"),
  usageFull: document.getElementById("usageFull"),
};

els.sessionId.value = localStorage.getItem("assist_session_id") || randId();

els.newSession.addEventListener("click", () => {
  els.sessionId.value = randId();
  localStorage.setItem("assist_session_id", els.sessionId.value);
  els.transcript.textContent = "";
  els.liveBuffer.textContent = "";
  els.suggestions.innerHTML = "";
  els.notes.textContent = "";
  els.usage.textContent = "";
  els.confidence.textContent = "—";
  els.turnCost.textContent = "$0.000000";
  els.sessionCost.textContent = "$0.000000";
});

let ws = null;
let recognition = null;
let micOn = false;
let lastSentReplace = "";
let replaceTimer = null;
let currentMode = "coach";

function setStatus(s) {
  els.status.textContent = s;
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function sendMsg(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(obj));
}

function fmtUsd(x) {
  const n = typeof x === "number" ? x : Number(x || 0);
  return `$${n.toFixed(6)}`;
}

function renderAssistant(data) {
  const mode = data?.mode || currentMode;
  currentMode = mode;
  updateModeButtons();
  const sug = mode === "coach" && Array.isArray(data?.suggestions) ? data.suggestions : [];
  els.suggestions.innerHTML = sug.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  els.confidence.textContent =
    mode === "coach" && typeof data?.confidence === "number" ? data.confidence.toFixed(2) : "—";

  const usage = data?.usage || {};
  const turn = usage?.turn || {};
  els.turnCost.textContent = fmtUsd(turn?.cost_usd);
  els.sessionCost.textContent = fmtUsd(usage?.cost_usd_total);
  els.usage.textContent = JSON.stringify(turn?.by_model || {}, null, 2);

  const notes = data?.notes || {};
  const bullets = (notes?.bullets || []).slice(-18);
  const actions = (notes?.action_items || []).slice(-10);
  const decisions = (notes?.decisions || []).slice(-10);
  els.notes.textContent =
    `Bullets:\n- ${bullets.join("\n- ")}\n\nAction items:\n- ${actions.join("\n- ")}\n\nDecisions:\n- ${decisions.join("\n- ")}`.replace(/\n- $/g, "\n");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function appendTranscriptLine(speaker, text) {
  const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const line = `[${ts}] ${speaker || "Speaker"}: ${text || ""}`.trim();
  if (!line) return;
  els.transcript.textContent = (els.transcript.textContent ? els.transcript.textContent + "\n" : "") + line;
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

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
    alert("SpeechRecognition not supported in this browser. Use manual text input.");
    return;
  }

  micOn = true;
  els.mic.textContent = "Stop mic";
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
    els.liveBuffer.textContent = full;

    // Replace-mode streaming: backend turn detector will decide when to emit.
    // We send at most ~4 times/sec to keep WS traffic reasonable.
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
          speaker: els.speaker.value,
          session_mode: currentMode,
        });
      }
    }, 250);
  };

  recognition.onend = () => {
    // When recognition stops, flush as final if we have buffered text.
    const full = String(els.liveBuffer.textContent || "").trim();
    if (full) {
      sendMsg({
        session_id: els.sessionId.value,
        mode: "replace",
        text: full,
        final: true,
        ts: Date.now() / 1000,
        speaker: els.speaker.value,
        session_mode: currentMode,
      });
      els.liveBuffer.textContent = "";
      lastSentReplace = "";
    }
    if (micOn) {
      // Chrome may auto-stop; try to restart softly
      try { recognition.start(); } catch (_) {}
    }
  };

  recognition.onerror = () => {
    // Fall back to manual; keep UI alive
  };

  try {
    recognition.start();
  } catch (_) {
    micOn = false;
    els.mic.textContent = "Start mic";
  }
}

function stopMic() {
  micOn = false;
  els.mic.textContent = "Start mic";
  if (recognition) {
    try { recognition.stop(); } catch (_) {}
  }
}

els.connect.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
    return;
  }

  ws = new WebSocket(wsUrl());
  setStatus("connecting…");

  ws.onopen = () => {
    setStatus("connected");
    els.mic.disabled = false;
    els.send.disabled = false;
    localStorage.setItem("assist_session_id", els.sessionId.value);
    els.connect.textContent = "Disconnect";
  };

  ws.onclose = () => {
    setStatus("disconnected");
    els.mic.disabled = true;
    els.send.disabled = true;
    els.connect.textContent = "Connect";
    if (micOn) stopMic();
  };

  ws.onerror = () => setStatus("error");

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (_) { return; }
    if (msg?.error) return;
    if (msg?.emit && msg?.data) {
      const d = msg.data;
      appendTranscriptLine(d.speaker, d.transcript);
      renderAssistant(d);
      els.liveBuffer.textContent = "";
      lastSentReplace = "";
    }
  };
});

els.mic.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!micOn) startMic();
  else stopMic();
});

els.send.addEventListener("click", () => {
  const txt = String(els.manual.value || "").trim();
  if (!txt) return;
  sendMsg({
    session_id: els.sessionId.value,
    text_delta: txt,
    final: true,
    ts: Date.now() / 1000,
    speaker: els.speaker.value,
    session_mode: currentMode,
  });
  els.manual.value = "";
});

els.manual.addEventListener("keydown", (e) => {
  if (e.key === "Enter") els.send.click();
});

function updateModeButtons() {
  if (!els.modeCoach || !els.modeNotes) return;
  if (currentMode === "coach") {
    els.modeCoach.classList.add("active");
    els.modeNotes.classList.remove("active");
  } else {
    els.modeNotes.classList.add("active");
    els.modeCoach.classList.remove("active");
  }
  if (els.modeLabel) {
    els.modeLabel.textContent = `mode: ${currentMode}`;
  }
}

if (els.modeCoach && els.modeNotes) {
  els.modeCoach.addEventListener("click", () => {
    currentMode = "coach";
    updateModeButtons();
    // Optionally nudge backend mode immediately
    fetch(`/session/${encodeURIComponent(els.sessionId.value)}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "coach" }),
    }).catch(() => {});
  });
  els.modeNotes.addEventListener("click", () => {
    currentMode = "notes";
    updateModeButtons();
    fetch(`/session/${encodeURIComponent(els.sessionId.value)}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "notes" }),
    }).catch(() => {});
  });
  updateModeButtons();
}

async function refreshUsage() {
  try {
    const sid = els.sessionId.value;
    const res = await fetch(`/session/${encodeURIComponent(sid)}/usage`);
    if (!res.ok) return;
    const data = await res.json();
    els.usageFull.textContent = JSON.stringify(data, null, 2);
  } catch {
    // ignore
  }
}

if (els.refreshUsage) {
  els.refreshUsage.addEventListener("click", refreshUsage);
}

