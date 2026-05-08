import os

def patch_app_js():
    path = 'app/static/app.js'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. New Session must stop system audio
    target_new_session = """// -------- New session --------
if (els.newSession) {
  els.newSession.addEventListener("click", () => {
    els.sessionId.value = randId();
    localStorage.setItem("assist_session_id", els.sessionId.value);

    transcriptEntries = [];"""
    
    replace_new_session = """// -------- New session --------
if (els.newSession) {
  els.newSession.addEventListener("click", () => {
    els.sessionId.value = randId();
    localStorage.setItem("assist_session_id", els.sessionId.value);

    if (systemAudioStream) {
      systemAudioStream.getTracks().forEach(t => t.stop());
      systemAudioStream = null;
      if (systemAudioBtn) {
        systemAudioBtn.textContent = "Start system audio";
        systemAudioBtn.classList.remove("active");
      }
      setSystemAudioStatus("inactive");
      if (typeof updateSystemAudioEmphasize === 'function') updateSystemAudioEmphasize();
    }

    transcriptEntries = [];"""

    if target_new_session in content:
        content = content.replace(target_new_session, replace_new_session)
        print("Patched New Session logic")
    else:
        print("Failed to patch New Session logic")

    # 2 & 3. No-audio stream stops immediately & Attach onended to every track
    target_capture = """    try {
      setSystemAudioStatus("requesting permission");
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      const audioTracks = stream.getAudioTracks();
      
      if (audioTracks.length === 0) {
        setSystemAudioStatus("active but no audio track");
        systemAudioStream = stream;
      } else {
        setSystemAudioStatus("active with audio track");
        systemAudioStream = stream;
      }

      systemAudioBtn.textContent = "Stop system audio";
      systemAudioBtn.classList.add("active");
      updateSystemAudioEmphasize();
      updateDebugPanel(lastEmitSnapshot);
      
      const track = stream.getTracks()[0];
      if (track) {
        track.onended = () => {
          systemAudioStream = null;
          systemAudioBtn.textContent = "Start system audio";
          systemAudioBtn.classList.remove("active");
          setSystemAudioStatus("stopped");
          updateSystemAudioEmphasize();
          updateDebugPanel(lastEmitSnapshot);
        };
      }
    } catch (err) {"""

    replace_capture = """    try {
      setSystemAudioStatus("requesting permission");
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      const audioTracks = stream.getAudioTracks();
      
      if (audioTracks.length === 0) {
        stream.getTracks().forEach(t => t.stop());
        setSystemAudioStatus("stopped — no audio shared. Please check 'Share audio'.");
        return;
      }
      
      setSystemAudioStatus("active with audio track");
      systemAudioStream = stream;

      systemAudioBtn.textContent = "Stop system audio";
      systemAudioBtn.classList.add("active");
      updateSystemAudioEmphasize();
      updateDebugPanel(lastEmitSnapshot);
      
      stream.getTracks().forEach(track => {
        track.onended = () => {
          if (systemAudioStream) {
            systemAudioStream.getTracks().forEach(t => t.stop());
            systemAudioStream = null;
            systemAudioBtn.textContent = "Start system audio";
            systemAudioBtn.classList.remove("active");
            setSystemAudioStatus("stopped");
            updateSystemAudioEmphasize();
            updateDebugPanel(lastEmitSnapshot);
          }
        };
      });
    } catch (err) {"""

    if target_capture in content:
        content = content.replace(target_capture, replace_capture)
        print("Patched System Audio logic")
    else:
        print("Failed to patch System Audio logic")

    # 4. Remove refreshNotesSystemOptionStyle
    target_refresh_notes_style = """function refreshNotesSystemOptionStyle() {
  const seg = document.getElementById("notesSegSystem");
  if (seg) seg.classList.add("seg--future");
}

function refreshMicAvailability() {"""
    replace_refresh_notes_style = """function refreshMicAvailability() {"""
    
    if target_refresh_notes_style in content:
        content = content.replace(target_refresh_notes_style, replace_refresh_notes_style)
        print("Removed refreshNotesSystemOptionStyle definition")
    else:
        print("Failed to remove refreshNotesSystemOptionStyle definition")
        
    target_call_refresh = "refreshNotesSystemOptionStyle();\n"
    if target_call_refresh in content:
        content = content.replace(target_call_refresh, "")
        print("Removed refreshNotesSystemOptionStyle call")
    else:
        print("Failed to remove refreshNotesSystemOptionStyle call")

    # 5. Clean dead code (replaceTimer)
    target_replacetimer_decl = "let replaceTimer = null;\n"
    if target_replacetimer_decl in content:
        content = content.replace(target_replacetimer_decl, "")
        print("Removed replaceTimer declaration")
    else:
        print("Failed to remove replaceTimer declaration")
        
    target_replacetimer_usage = """  if (replaceTimer) {
    clearTimeout(replaceTimer);
    replaceTimer = null;
  }

"""
    if target_replacetimer_usage in content:
        content = content.replace(target_replacetimer_usage, "")
        print("Removed replaceTimer usage")
    else:
        print("Failed to remove replaceTimer usage")

    # 5. Clean dead code (appendTranscript)
    target_appendtranscript = """/** @deprecated name — use appendTranscriptFromEvent */
function appendTranscript(speaker, ts, kind, text) {
  appendTranscriptFromEvent(speaker, ts, kind, text, TranscriptLifecycle.TRANSCRIPT_FINAL);
}

function resetCoachUiToEmpty() {"""
    replace_appendtranscript = """function resetCoachUiToEmpty() {"""
    if target_appendtranscript in content:
        content = content.replace(target_appendtranscript, replace_appendtranscript)
        print("Removed appendTranscript")
    else:
        print("Failed to remove appendTranscript")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def patch_styles():
    path = 'app/static/styles.css'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    target_seg_future = """.seg--future {
  opacity: .72;
  border-style: dashed
}

"""
    if target_seg_future in content:
        content = content.replace(target_seg_future, "")
        print("Removed .seg--future")
    else:
        print("Failed to remove .seg--future")
        
    target_system_soon = """.systemSoon {
  opacity: .9;
  cursor: pointer
}

"""
    if target_system_soon in content:
        content = content.replace(target_system_soon, "")
        print("Removed .systemSoon")
    else:
        print("Failed to remove .systemSoon")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

patch_app_js()
patch_styles()
