/* Shooting Practice Video Analyzer — Frontend Controller */

const state = {
  firearms: [],
  session: null,
  signature: null,
  beepTimestamp: null,
  shotTimestamps: [],
  selectedMarker: null, // { type: 'beep' } or { type: 'shot', index: number }
  userModified: false,
  wavesurfer: null,
  regionsPlugin: null,
  isSyncingRegions: false,
  previewMode: "original", // "original" | "rendered"
  renderedPreviewUrl: null,
};

// DOM References
const els = {
  modeBadge: document.getElementById("modeBadge"),
  signatureBadge: document.getElementById("signatureBadge"),
  uploadForm: document.getElementById("uploadForm"),
  firearmPreset: document.getElementById("firearmPreset"),
  firearmInput: document.getElementById("firearmInput"),
  profileHint: document.getElementById("profileHint"),
  videoFileInput: document.getElementById("videoFileInput"),
  dropZone: document.getElementById("dropZone"),
  selectedFileName: document.getElementById("selectedFileName"),
  uploadBtn: document.getElementById("uploadBtn"),
  demoBtn: document.getElementById("demoBtn"),
  editorStatusText: document.getElementById("editorStatusText"),
  videoSourceToggle: document.getElementById("videoSourceToggle"),
  viewOriginalBtn: document.getElementById("viewOriginalBtn"),
  viewRenderedBtn: document.getElementById("viewRenderedBtn"),
  practiceVideo: document.getElementById("practiceVideo"),
  videoPlaceholder: document.getElementById("videoPlaceholder"),
  liveVideoHud: document.getElementById("liveVideoHud"),
  hudBeepBanner: document.getElementById("hudBeepBanner"),
  hudShotCard: document.getElementById("hudShotCard"),
  hudShotNumber: document.getElementById("hudShotNumber"),
  hudShotSplit: document.getElementById("hudShotSplit"),
  hudShotElapsed: document.getElementById("hudShotElapsed"),
  waveformContainer: document.getElementById("waveformContainer"),
  zoomSlider: document.getElementById("zoomSlider"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  stopBtn: document.getElementById("stopBtn"),
  speedSelect: document.getElementById("speedSelect"),
  currentTimeDisplay: document.getElementById("currentTimeDisplay"),
  durationDisplay: document.getElementById("durationDisplay"),
  stageTimeDisplay: document.getElementById("stageTimeDisplay"),
  addBeepBtn: document.getElementById("addBeepBtn"),
  addShotBtn: document.getElementById("addShotBtn"),
  deleteMarkerBtn: document.getElementById("deleteMarkerBtn"),
  kpiDrawTime: document.getElementById("kpiDrawTime"),
  kpiTotalTime: document.getElementById("kpiTotalTime"),
  kpiAvgSplit: document.getElementById("kpiAvgSplit"),
  kpiShotCount: document.getElementById("kpiShotCount"),
  splitsTableBody: document.getElementById("splitsTableBody"),
  sigGunName: document.getElementById("sigGunName"),
  sigVersion: document.getElementById("sigVersion"),
  sigBeepMeta: document.getElementById("sigBeepMeta"),
  sigShotMeta: document.getElementById("sigShotMeta"),
  saveProfileBtn: document.getElementById("saveProfileBtn"),
  processVideoBtn: document.getElementById("processVideoBtn"),
  downloadVideoLink: document.getElementById("downloadVideoLink"),
  statusToast: document.getElementById("statusToast"),
  openFirearmManagerHeaderBtn: document.getElementById("openFirearmManagerHeaderBtn"),
  openFirearmManagerInlineBtn: document.getElementById("openFirearmManagerInlineBtn"),
  openFirearmManagerPanelBtn: document.getElementById("openFirearmManagerPanelBtn"),
  firearmModal: document.getElementById("firearmModal"),
  closeFirearmModalBtn: document.getElementById("closeFirearmModalBtn"),
  firearmManageForm: document.getElementById("firearmManageForm"),
  editOriginalGunName: document.getElementById("editOriginalGunName"),
  manageGunNameInput: document.getElementById("manageGunNameInput"),
  manageBeepFreqInput: document.getElementById("manageBeepFreqInput"),
  manageBeepDurInput: document.getElementById("manageBeepDurInput"),
  manageShotDecayInput: document.getElementById("manageShotDecayInput"),
  saveFirearmModelBtn: document.getElementById("saveFirearmModelBtn"),
  cancelEditFirearmBtn: document.getElementById("cancelEditFirearmBtn"),
  firearmsManageTableBody: document.getElementById("firearmsManageTableBody"),
  restoreDefaultsBtn: document.getElementById("restoreDefaultsBtn"),
  modalFeedbackText: document.getElementById("modalFeedbackText"),
};

function showToast(message, isError = false) {
  els.statusToast.hidden = false;
  els.statusToast.textContent = message;
  els.statusToast.style.borderColor = isError
    ? "rgba(239, 68, 68, 0.55)"
    : "rgba(14, 165, 233, 0.45)";
  els.statusToast.style.color = isError ? "#fca5a5" : "#bae6fd";
}

function showModalFeedback(message, isError = false) {
  if (!els.modalFeedbackText) return;
  els.modalFeedbackText.textContent = message;
  els.modalFeedbackText.style.color = isError ? "#fca5a5" : "#34d399";
}

async function fetchFirearms() {
  try {
    const res = await fetch("/api/firearms");
    if (!res.ok) return;
    const data = await res.json();
    state.firearms = data.firearms || [];

    const currentTyped = els.firearmInput.value.trim();
    els.firearmPreset.innerHTML = "";

    state.firearms.forEach((sig) => {
      const opt = document.createElement("option");
      opt.value = sig.gun_name;
      opt.textContent =
        sig.signature_version > 0
          ? `${sig.gun_name} (v${sig.signature_version})`
          : `${sig.gun_name} (Cold Start)`;
      els.firearmPreset.appendChild(opt);
    });

    const customOpt = document.createElement("option");
    customOpt.value = "__custom__";
    customOpt.textContent = "+ Custom Firearm Name...";
    els.firearmPreset.appendChild(customOpt);

    // Match current input to dropdown option if present
    const matched = state.firearms.find(
      (f) => f.gun_name.toLowerCase() === currentTyped.toLowerCase()
    );
    if (matched) {
      els.firearmPreset.value = matched.gun_name;
      els.firearmInput.value = matched.gun_name;
    } else if (state.firearms.length > 0 && !currentTyped) {
      els.firearmPreset.value = state.firearms[0].gun_name;
      els.firearmInput.value = state.firearms[0].gun_name;
    } else {
      els.firearmPreset.value = "__custom__";
    }

    updateFirearmHint();
    renderFirearmManagerTable();
  } catch (err) {
    console.error("Failed to fetch firearms:", err);
  }
}

function findLocalSignature(gunName) {
  if (!gunName) return null;
  const key = gunName.trim().toLowerCase();
  return (
    state.firearms.find((f) => f.gun_name.trim().toLowerCase() === key) || null
  );
}

function updateFirearmHint() {
  const gun = els.firearmInput.value.trim();
  const sig = findLocalSignature(gun);
  if (!gun) {
    els.profileHint.textContent = "Please enter or select a firearm name.";
    return;
  }
  if (sig && sig.signature_version > 0) {
    els.profileHint.textContent = `✓ Existing Audio Signature found (v${sig.signature_version} • Beep ${Math.round(sig.beep_profile.frequency_hz)}Hz) — Automated Detection ready.`;
    els.profileHint.style.color = "#34d399";
  } else {
    els.profileHint.textContent = `ℹ No learned signature for "${gun}" yet — will start in Training Mode (Cold Start).`;
    els.profileHint.style.color = "#fbbf24";
  }
}

function resetFirearmModalForm() {
  els.editOriginalGunName.value = "";
  els.manageGunNameInput.value = "";
  els.manageBeepFreqInput.value = "3000";
  els.manageBeepDurInput.value = "220";
  els.manageShotDecayInput.value = "120";
  els.saveFirearmModelBtn.textContent = "+ Add Firearm";
  els.cancelEditFirearmBtn.hidden = true;
}

function startEditingFirearm(sig) {
  els.editOriginalGunName.value = sig.gun_name;
  els.manageGunNameInput.value = sig.gun_name;
  els.manageBeepFreqInput.value = Number(sig.beep_profile.frequency_hz).toFixed(2);
  els.manageBeepDurInput.value = String(sig.beep_profile.duration_ms);
  els.manageShotDecayInput.value = Number(sig.shot_profile.decay_rate).toFixed(1);
  els.saveFirearmModelBtn.textContent = "💾 Save Changes";
  els.cancelEditFirearmBtn.hidden = false;
  els.manageGunNameInput.focus();
  showModalFeedback(`Editing "${sig.gun_name}"...`);
}

function renderFirearmManagerTable() {
  if (!els.firearmsManageTableBody) return;
  els.firearmsManageTableBody.innerHTML = "";

  if (!state.firearms || state.firearms.length === 0) {
    els.firearmsManageTableBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6">No firearms configured. Add one above or click Restore Default Firearms.</td>
      </tr>`;
    return;
  }

  state.firearms.forEach((sig) => {
    const tr = document.createElement("tr");
    const isTrained = sig.signature_version > 0;
    const statusBadge = isTrained
      ? `<span class="status-badge badge-automated">Trained v${sig.signature_version}</span>`
      : `<span class="status-badge badge-training">Cold Start (v0)</span>`;
    const beepInfo = `${Number(sig.beep_profile.frequency_hz).toFixed(0)} Hz / ${sig.beep_profile.duration_ms} ms`;
    const hasNeural =
      Array.isArray(sig.shot_profile.neural_embedding) &&
      sig.shot_profile.neural_embedding.some((x) => Math.abs(x) > 1e-6);
    const yamnetBadge = hasNeural
      ? `YAMNet ${(Number(sig.shot_profile.yamnet_gunshot_score || 0) * 100).toFixed(0)}%`
      : "YAMNet Ready";
    const shotInfo = `${Number(sig.shot_profile.decay_rate).toFixed(1)} dB/s • ${yamnetBadge}`;
    const updatedDate = sig.updated_at
      ? new Date(sig.updated_at).toLocaleString()
      : "--";

    tr.innerHTML = `
      <td><strong>${sig.gun_name}</strong></td>
      <td>${statusBadge}</td>
      <td>${beepInfo}</td>
      <td>${shotInfo}</td>
      <td style="font-size:0.75rem;color:var(--text-secondary)">${updatedDate}</td>
      <td>
        <div class="action-btn-group">
          <button type="button" class="btn btn-xs btn-secondary" data-action="select">Use</button>
          <button type="button" class="btn btn-xs btn-cyan" data-action="edit">✎ Edit</button>
          <button type="button" class="btn btn-xs btn-amber" data-action="reset" ${!isTrained ? "disabled" : ""}>↺ Reset Signature</button>
          <button type="button" class="btn btn-xs btn-danger" data-action="delete">🗑 Remove</button>
        </div>
      </td>
    `;

    tr.querySelector('[data-action="select"]').addEventListener("click", () => {
      els.firearmInput.value = sig.gun_name;
      els.firearmPreset.value = sig.gun_name;
      updateFirearmHint();
      els.firearmModal.hidden = true;
      showToast(`Selected firearm "${sig.gun_name}".`);
    });

    tr.querySelector('[data-action="edit"]').addEventListener("click", () => {
      startEditingFirearm(sig);
    });

    tr.querySelector('[data-action="reset"]').addEventListener("click", async () => {
      try {
        const res = await fetch(
          `/api/firearms/${encodeURIComponent(sig.gun_name)}/reset`,
          { method: "POST" }
        );
        if (!res.ok) throw new Error("Failed to reset signature");
        await fetchFirearms();
        if (
          state.session &&
          state.session.gun_name.toLowerCase() === sig.gun_name.toLowerCase()
        ) {
          state.signature = null;
          renderSignatureUI();
        }
        showModalFeedback(`Reset acoustic signature for "${sig.gun_name}" to Cold Start.`);
      } catch (err) {
        showModalFeedback(err.message, true);
      }
    });

    tr.querySelector('[data-action="delete"]').addEventListener("click", async () => {
      try {
        const res = await fetch(
          `/api/firearms/${encodeURIComponent(sig.gun_name)}`,
          { method: "DELETE" }
        );
        if (!res.ok) throw new Error("Failed to remove firearm");
        if (els.editOriginalGunName.value === sig.gun_name) {
          resetFirearmModalForm();
        }
        await fetchFirearms();
        showModalFeedback(`Removed "${sig.gun_name}".`);
      } catch (err) {
        showModalFeedback(err.message, true);
      }
    });

    els.firearmsManageTableBody.appendChild(tr);
  });
}

// Compute client-side analytics (FR-4.1)
function computeClientAnalytics() {
  const beep =
    state.beepTimestamp !== null ? Number(state.beepTimestamp) : null;
  const sortedShots = [...state.shotTimestamps]
    .map((t) => Number(t))
    .sort((a, b) => a - b);

  const shots = sortedShots.map((t, idx) => {
    const elapsed = beep !== null ? Number((t - beep).toFixed(3)) : null;
    let split = null;
    if (idx === 0) {
      split = beep !== null ? Number((t - beep).toFixed(3)) : null;
    } else {
      split = Number((t - sortedShots[idx - 1]).toFixed(3));
    }
    return {
      shotNumber: idx + 1,
      timestamp: Number(t.toFixed(3)),
      splitTime: split,
      elapsedTime: elapsed,
    };
  });

  const drawTime =
    beep !== null && sortedShots.length > 0
      ? Number((sortedShots[0] - beep).toFixed(3))
      : null;
  const totalStageTime =
    beep !== null && sortedShots.length > 0
      ? Number((sortedShots[sortedShots.length - 1] - beep).toFixed(3))
      : null;

  let avgSplit = null;
  if (shots.length > 1) {
    const interShotSplits = shots
      .slice(1)
      .map((s) => s.splitTime)
      .filter((v) => v !== null);
    if (interShotSplits.length > 0) {
      const sum = interShotSplits.reduce((a, b) => a + b, 0);
      avgSplit = Number((sum / interShotSplits.length).toFixed(3));
    }
  } else if (shots.length === 1 && drawTime !== null) {
    avgSplit = drawTime;
  }

  return {
    beepTimestamp: beep,
    drawTime,
    totalStageTime,
    avgSplit,
    totalShots: shots.length,
    shots,
  };
}

function renderAnalyticsUI() {
  const metrics = computeClientAnalytics();

  els.kpiDrawTime.textContent =
    metrics.drawTime !== null ? `${metrics.drawTime.toFixed(2)}s` : "--";
  els.kpiTotalTime.textContent =
    metrics.totalStageTime !== null
      ? `${metrics.totalStageTime.toFixed(2)}s`
      : "--";
  els.kpiAvgSplit.textContent =
    metrics.avgSplit !== null ? `${metrics.avgSplit.toFixed(2)}s` : "--";
  els.kpiShotCount.textContent = `${metrics.totalShots} Shot${metrics.totalShots === 1 ? "" : "s"}`;

  // Populate Table
  els.splitsTableBody.innerHTML = "";

  if (metrics.beepTimestamp === null && metrics.shots.length === 0) {
    els.splitsTableBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="5">No markers placed yet. Press <kbd>B</kbd> for Beep and <kbd>S</kbd> for Shots.</td>
      </tr>`;
    return;
  }

  if (metrics.beepTimestamp !== null) {
    const tr = document.createElement("tr");
    if (state.selectedMarker && state.selectedMarker.type === "beep") {
      tr.classList.add("active-row");
    }
    tr.innerHTML = `
      <td><strong style="color:#fbbf24">⏱ Beep</strong></td>
      <td>${metrics.beepTimestamp.toFixed(3)}s</td>
      <td>--</td>
      <td>0.000s</td>
      <td><button type="button" class="btn btn-xs btn-danger" data-del-beep="true">✕</button></td>
    `;
    tr.addEventListener("click", (e) => {
      if (e.target.closest("[data-del-beep]")) {
        state.beepTimestamp = null;
        if (state.selectedMarker?.type === "beep") state.selectedMarker = null;
        state.userModified = true;
        syncWaveformMarkers();
        renderAnalyticsUI();
        return;
      }
      selectMarker({ type: "beep" }, true);
    });
    els.splitsTableBody.appendChild(tr);
  }

  metrics.shots.forEach((s, idx) => {
    const tr = document.createElement("tr");
    if (
      state.selectedMarker &&
      state.selectedMarker.type === "shot" &&
      state.selectedMarker.index === idx
    ) {
      tr.classList.add("active-row");
    }
    const splitStr =
      s.splitTime !== null
        ? s.shotNumber === 1
          ? `Draw ${s.splitTime.toFixed(3)}s`
          : `+${s.splitTime.toFixed(3)}s`
        : "--";
    const elapsedStr =
      s.elapsedTime !== null ? `${s.elapsedTime.toFixed(3)}s` : "--";

    tr.innerHTML = `
      <td><strong style="color:#67e8f9">Shot #${s.shotNumber}</strong></td>
      <td>${s.timestamp.toFixed(3)}s</td>
      <td>${splitStr}</td>
      <td>${elapsedStr}</td>
      <td><button type="button" class="btn btn-xs btn-danger" data-del-shot="${idx}">✕</button></td>
    `;
    tr.addEventListener("click", (e) => {
      if (e.target.closest("[data-del-shot]")) {
        state.shotTimestamps.splice(idx, 1);
        if (state.selectedMarker?.type === "shot") state.selectedMarker = null;
        state.userModified = true;
        syncWaveformMarkers();
        renderAnalyticsUI();
        return;
      }
      selectMarker({ type: "shot", index: idx }, true);
    });
    els.splitsTableBody.appendChild(tr);
  });
}

function renderSignatureUI() {
  if (!state.session) return;
  els.sigGunName.textContent = state.session.gun_name;

  if (state.signature) {
    const hasNeural =
      Array.isArray(state.signature.shot_profile.neural_embedding) &&
      state.signature.shot_profile.neural_embedding.some((x) => Math.abs(x) > 1e-6);
    els.sigVersion.textContent = `v${state.signature.signature_version} (${hasNeural ? "YAMNet + Slow/Fast" : "Trained"})`;
    els.sigBeepMeta.textContent = `${state.signature.beep_profile.frequency_hz.toFixed(0)} Hz / ${state.signature.beep_profile.duration_ms} ms`;
    const yamnetText = hasNeural
      ? ` • YAMNet ${(Number(state.signature.shot_profile.yamnet_gunshot_score || 0) * 100).toFixed(0)}%`
      : " • YAMNet Active";
    els.sigShotMeta.textContent = `${state.signature.shot_profile.decay_rate.toFixed(1)} dB/s${yamnetText}`;
    els.signatureBadge.textContent = `${state.session.gun_name} • Signature v${state.signature.signature_version}`;
    els.signatureBadge.className = "status-badge badge-automated";
  } else {
    els.sigVersion.textContent = "Cold Start (Unlearned)";
    els.sigBeepMeta.textContent = "Awaiting Training Confirmation";
    els.sigShotMeta.textContent = "Awaiting Training Confirmation";
    els.signatureBadge.textContent = `${state.session.gun_name} • Cold Start`;
    els.signatureBadge.className = "status-badge badge-training";
  }

  if (state.session.mode === "automated") {
    els.modeBadge.textContent = "Automated Detection Mode";
    els.modeBadge.className = "status-badge badge-automated";
  } else {
    els.modeBadge.textContent = "Training Mode (Cold Start)";
    els.modeBadge.className = "status-badge badge-training";
  }
}

function selectMarker(marker, seekToMarker = false) {
  state.selectedMarker = marker;
  els.deleteMarkerBtn.disabled = !marker;

  if (seekToMarker && state.wavesurfer && marker) {
    let targetTime = null;
    if (marker.type === "beep") {
      targetTime = state.beepTimestamp;
    } else if (
      marker.type === "shot" &&
      state.shotTimestamps[marker.index] !== undefined
    ) {
      targetTime = state.shotTimestamps[marker.index];
    }
    if (targetTime !== null) {
      els.practiceVideo.currentTime = targetTime;
    }
  }

  updateMarkerHighlightColors();
  renderAnalyticsUI();
}

function updateMarkerHighlightColors() {
  if (!state.regionsPlugin) return;
  const regions = state.regionsPlugin.getRegions();
  regions.forEach((reg) => {
    if (reg.id === "beep") {
      const isSel = state.selectedMarker?.type === "beep";
      reg.setOptions({
        color: isSel ? "rgba(244, 63, 94, 0.95)" : "rgba(245, 158, 11, 0.95)",
      });
    } else if (reg.id.startsWith("shot-")) {
      const idx = parseInt(reg.id.replace("shot-", ""), 10);
      const isSel =
        state.selectedMarker?.type === "shot" &&
        state.selectedMarker?.index === idx;
      reg.setOptions({
        color: isSel ? "rgba(244, 63, 94, 0.95)" : "rgba(6, 182, 212, 0.95)",
      });
    }
  });
}

function syncWaveformMarkers() {
  if (!state.regionsPlugin) return;
  state.isSyncingRegions = true;
  state.regionsPlugin.clearRegions();

  // Sort shots ascending
  state.shotTimestamps = [...state.shotTimestamps]
    .map((t) => Number(Number(t).toFixed(3)))
    .sort((a, b) => a - b);

  // Add Beep Marker
  if (state.beepTimestamp !== null) {
    const isSel = state.selectedMarker?.type === "beep";
    state.regionsPlugin.addRegion({
      id: "beep",
      start: state.beepTimestamp,
      content: "⏱ BEEP (0.00s)",
      color: isSel ? "rgba(244, 63, 94, 0.95)" : "rgba(245, 158, 11, 0.95)",
      drag: true,
      resize: false,
    });
  }

  // Add Shot Markers
  state.shotTimestamps.forEach((shotTime, idx) => {
    const isSel =
      state.selectedMarker?.type === "shot" &&
      state.selectedMarker?.index === idx;
    state.regionsPlugin.addRegion({
      id: `shot-${idx}`,
      start: shotTime,
      content: `S${idx + 1}`,
      color: isSel ? "rgba(244, 63, 94, 0.95)" : "rgba(6, 182, 212, 0.95)",
      drag: true,
      resize: false,
    });
  });

  state.isSyncingRegions = false;
}

function updateLiveVideoHud(currentTime) {
  if (!state.session || state.previewMode === "rendered") {
    els.liveVideoHud.hidden = true;
    return;
  }
  els.liveVideoHud.hidden = false;

  const metrics = computeClientAnalytics();

  // Timer Beep HUD indicator
  if (
    metrics.beepTimestamp !== null &&
    currentTime >= metrics.beepTimestamp &&
    currentTime <= metrics.beepTimestamp + 1.25
  ) {
    els.hudBeepBanner.hidden = false;
  } else {
    els.hudBeepBanner.hidden = true;
  }

  // Shot Pop-up HUD indicator (1.5s following each shot)
  let activeShot = null;
  for (const s of metrics.shots) {
    if (currentTime >= s.timestamp && currentTime <= s.timestamp + 1.5) {
      activeShot = s;
    }
  }

  if (activeShot) {
    els.hudShotCard.hidden = false;
    els.hudShotNumber.textContent = `SHOT #${activeShot.shotNumber}`;
    els.hudShotSplit.textContent =
      activeShot.splitTime !== null
        ? activeShot.shotNumber === 1
          ? `DRAW: ${activeShot.splitTime.toFixed(2)}s`
          : `SPLIT: +${activeShot.splitTime.toFixed(2)}s`
        : "SPLIT: --";
    els.hudShotElapsed.textContent =
      activeShot.elapsedTime !== null
        ? `TOTAL: ${activeShot.elapsedTime.toFixed(2)}s`
        : `TIME: ${activeShot.timestamp.toFixed(2)}s`;
  } else {
    els.hudShotCard.hidden = true;
  }

  // Stage time pill in toolbar
  if (metrics.beepTimestamp !== null) {
    const rel = currentTime - metrics.beepTimestamp;
    els.stageTimeDisplay.textContent = `Stage: ${rel >= 0 ? "+" : ""}${rel.toFixed(2)}s`;
  } else {
    els.stageTimeDisplay.textContent = "Stage: --";
  }
}

async function decodeAudioPeaks(audioUrl, targetBuckets = 3000) {
  const res = await fetch(audioUrl);
  const arrayBuf = await res.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioCtx();
  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuf);
    const rawData = audioBuffer.getChannelData(0);
    const total = rawData.length;
    const buckets = Math.min(targetBuckets, Math.max(500, total));
    const blockSize = Math.floor(total / buckets) || 1;
    const peaks = new Float32Array(buckets);

    for (let i = 0; i < buckets; i++) {
      const start = i * blockSize;
      let maxVal = 0;
      for (let j = 0; j < blockSize && start + j < total; j++) {
        const v = Math.abs(rawData[start + j]);
        if (v > maxVal) maxVal = v;
      }
      peaks[i] = maxVal;
    }
    return { peaks: Array.from(peaks), duration: audioBuffer.duration };
  } finally {
    if (audioCtx.state !== "closed") {
      audioCtx.close().catch(() => {});
    }
  }
}

async function initWaveSurfer(session) {
  if (state.wavesurfer) {
    state.wavesurfer.destroy();
    state.wavesurfer = null;
  }

  els.waveformContainer.innerHTML = "";
  els.zoomSlider.value = "0";
  els.videoPlaceholder.hidden = true;
  els.videoPlaceholder.style.display = "none";
  els.practiceVideo.src = session.video_url;
  els.practiceVideo.load();

  let decoded = null;
  try {
    decoded = await decodeAudioPeaks(session.audio_url);
  } catch (err) {
    console.warn("Audio peak decode fallback:", err);
  }

  const regions = WaveSurfer.Regions.create();
  const timeline = WaveSurfer.Timeline.create({
    height: 22,
    style: {
      fontSize: "11px",
      color: "#94a3b8",
    },
  });

  state.regionsPlugin = regions;

  const wsOptions = {
    container: els.waveformContainer,
    height: 118,
    waveColor: "#38bdf8",
    progressColor: "#0284c7",
    cursorColor: "#f8fafc",
    cursorWidth: 2,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    normalize: true,
    fillParent: true,
    minPxPerSec: 0,
    autoScroll: true,
    autoCenter: false,
    media: els.practiceVideo,
    url: session.video_url,
    plugins: [regions, timeline],
  };

  if (decoded && decoded.peaks.length > 0) {
    wsOptions.peaks = [decoded.peaks];
    wsOptions.duration = decoded.duration || session.duration_sec;
  }

  const ws = WaveSurfer.create(wsOptions);
  state.wavesurfer = ws;

  const onReady = () => {
    const dur = ws.getDuration() || session.duration_sec || 0;
    els.durationDisplay.textContent = `${dur.toFixed(3)}s`;
    els.playPauseBtn.disabled = false;
    els.stopBtn.disabled = false;
    els.speedSelect.disabled = false;
    els.zoomSlider.disabled = false;
    els.addBeepBtn.disabled = false;
    els.addShotBtn.disabled = false;
    els.saveProfileBtn.disabled = false;
    els.processVideoBtn.disabled = false;

    syncWaveformMarkers();
    renderAnalyticsUI();
  };

  ws.on("ready", onReady);

  ws.on("timeupdate", (currentTime) => {
    els.currentTimeDisplay.textContent = `${currentTime.toFixed(3)}s`;
    updateLiveVideoHud(currentTime);
  });

  ws.on("play", () => {
    els.playPauseBtn.textContent = "⏸ Pause";
  });

  ws.on("pause", () => {
    els.playPauseBtn.textContent = "▶ Play";
  });

  // Handle interactive marker dragging & clicking
  regions.on("region-clicked", (region, e) => {
    e.stopPropagation();
    if (region.id === "beep") {
      selectMarker({ type: "beep" }, true);
    } else if (region.id.startsWith("shot-")) {
      const idx = parseInt(region.id.replace("shot-", ""), 10);
      selectMarker({ type: "shot", index: idx }, true);
    }
  });

  regions.on("region-updated", (region) => {
    if (state.isSyncingRegions) return;
    const newTime = Number(region.start.toFixed(3));
    state.userModified = true;

    if (region.id === "beep") {
      state.beepTimestamp = newTime;
      state.selectedMarker = { type: "beep" };
      els.deleteMarkerBtn.disabled = false;
      updateMarkerHighlightColors();
      renderAnalyticsUI();
    } else if (region.id.startsWith("shot-")) {
      const idx = parseInt(region.id.replace("shot-", ""), 10);
      if (state.shotTimestamps[idx] !== undefined) {
        state.shotTimestamps[idx] = newTime;
        state.shotTimestamps.sort((a, b) => a - b);
        const newIdx = state.shotTimestamps.indexOf(newTime);
        state.selectedMarker = { type: "shot", index: Math.max(0, newIdx) };
        els.deleteMarkerBtn.disabled = false;
        requestAnimationFrame(() => {
          syncWaveformMarkers();
          renderAnalyticsUI();
        });
      }
    }
  });
}

function handleSessionLoaded(payload) {
  state.session = payload.session;
  state.signature = payload.signature;
  state.beepTimestamp = payload.session.beep_timestamp;
  state.shotTimestamps = payload.session.shot_timestamps || [];
  state.selectedMarker = null;
  state.userModified = false;
  state.previewMode = "original";
  state.renderedPreviewUrl = null;

  els.videoSourceToggle.hidden = true;
  els.downloadVideoLink.hidden = true;
  els.deleteMarkerBtn.disabled = true;

  if (state.session.mode === "automated") {
    els.editorStatusText.textContent = `Automated detection completed using "${state.session.gun_name}" profile (v${state.session.signature_version}). Review or drag markers to refine.`;
  } else {
    els.editorStatusText.textContent = `Training Mode (Cold Start) for "${state.session.gun_name}". Place Beep (B) and Shots (S) on the waveform, then confirm to build the profile.`;
  }

  renderSignatureUI();
  initWaveSurfer(state.session);
  fetchFirearms();
}

// Marker Actions (FR-3.3 & FR-3.4)
function setBeepAtPlayhead() {
  if (!state.wavesurfer || !state.session) return;
  const now = Number((els.practiceVideo.currentTime || 0).toFixed(3));
  state.beepTimestamp = now;
  state.userModified = true;
  state.selectedMarker = { type: "beep" };
  els.deleteMarkerBtn.disabled = false;
  syncWaveformMarkers();
  renderAnalyticsUI();
  showToast(`Timer Beep anchor set at ${now.toFixed(3)}s`);
}

function addShotAtPlayhead() {
  if (!state.wavesurfer || !state.session) return;
  const now = Number((els.practiceVideo.currentTime || 0).toFixed(3));
  state.shotTimestamps.push(now);
  state.shotTimestamps.sort((a, b) => a - b);
  const idx = state.shotTimestamps.indexOf(now);
  state.userModified = true;
  state.selectedMarker = { type: "shot", index: Math.max(0, idx) };
  els.deleteMarkerBtn.disabled = false;
  syncWaveformMarkers();
  renderAnalyticsUI();
  showToast(`Added Shot #${idx + 1} at ${now.toFixed(3)}s`);
}

function deleteSelectedMarker() {
  if (!state.selectedMarker) return;
  if (state.selectedMarker.type === "beep") {
    state.beepTimestamp = null;
    showToast("Removed Timer Beep marker.");
  } else if (state.selectedMarker.type === "shot") {
    const idx = state.selectedMarker.index;
    if (state.shotTimestamps[idx] !== undefined) {
      state.shotTimestamps.splice(idx, 1);
      showToast(`Removed Shot #${idx + 1} marker.`);
    }
  }
  state.selectedMarker = null;
  state.userModified = true;
  els.deleteMarkerBtn.disabled = true;
  syncWaveformMarkers();
  renderAnalyticsUI();
}

// API Calls for Profile Refinement & Video Rendering
async function saveAndRefineProfile() {
  if (!state.session) return;
  els.saveProfileBtn.disabled = true;
  els.saveProfileBtn.textContent = "Updating Acoustic Profile...";

  try {
    const body = {
      video_id: state.session.video_id,
      gun_name: els.firearmInput.value.trim() || state.session.gun_name,
      beep_timestamp: state.beepTimestamp,
      shot_timestamps: state.shotTimestamps,
      user_modified: state.userModified,
    };

    const res = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to save annotations");
    }

    const data = await res.json();
    state.session = data.session;
    state.signature = data.signature;
    state.userModified = false;

    renderSignatureUI();
    renderAnalyticsUI();
    await fetchFirearms();
    showToast(
      `✓ Profile "${state.signature.gun_name}" refined to v${state.signature.signature_version}!`
    );
  } catch (err) {
    showToast(err.message, true);
  } finally {
    els.saveProfileBtn.disabled = false;
    els.saveProfileBtn.textContent = "Confirm Markers & Refine Audio Profile";
  }
}

async function processAndRenderVideo() {
  if (!state.session) return;
  els.processVideoBtn.disabled = true;
  els.processVideoBtn.textContent = "⏳ Rendering OpenCV Video Overlays...";
  showToast("Rendering burnt-in performance overlays with OpenCV...");

  try {
    const body = {
      video_id: state.session.video_id,
      gun_name: els.firearmInput.value.trim() || state.session.gun_name,
      beep_timestamp: state.beepTimestamp,
      shot_timestamps: state.shotTimestamps,
      user_modified: state.userModified,
    };

    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Video rendering failed");
    }

    const data = await res.json();
    state.session = data.session;
    state.signature = data.signature;
    state.renderedPreviewUrl = `${data.preview_url}?t=${Date.now()}`;

    renderSignatureUI();
    renderAnalyticsUI();
    await fetchFirearms();

    els.downloadVideoLink.href = data.download_url;
    els.downloadVideoLink.hidden = false;
    els.videoSourceToggle.hidden = false;

    showToast("✓ Annotated video rendered! Click Download or switch to Rendered Preview.");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    els.processVideoBtn.disabled = false;
    els.processVideoBtn.textContent = "⚙ Process Video with Burnt-In Overlays";
  }
}

// Event Listeners Setup
function bindEvents() {
  els.firearmPreset.addEventListener("change", () => {
    if (els.firearmPreset.value === "__custom__") {
      els.firearmInput.value = "";
      els.firearmInput.focus();
    } else {
      els.firearmInput.value = els.firearmPreset.value;
    }
    updateFirearmHint();
  });

  els.firearmInput.addEventListener("input", updateFirearmHint);

  els.videoFileInput.addEventListener("change", () => {
    const file = els.videoFileInput.files[0];
    if (file) {
      els.selectedFileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    }
  });

  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("dragover");
  });

  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("dragover");
  });

  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      els.videoFileInput.files = e.dataTransfer.files;
      const file = e.dataTransfer.files[0];
      els.selectedFileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    }
  });

  // Upload Form Submit
  els.uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = els.videoFileInput.files[0];
    const gunName = els.firearmInput.value.trim();

    if (!gunName) {
      showToast("Please specify a firearm name (FR-1.2).", true);
      return;
    }
    if (!file) {
      showToast("Please select a video file (.mp4, .mov, .webm) or click 'Load Sample Drill Video'.", true);
      return;
    }

    els.uploadBtn.disabled = true;
    els.uploadBtn.textContent = "⏳ Extracting Audio & Analyzing...";

    try {
      const formData = new FormData();
      formData.append("video", file);
      formData.append("gun_name", gunName);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      handleSessionLoaded(data);
      showToast(`✓ Loaded "${file.name}" (${data.session.mode.toUpperCase()} mode).`);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      els.uploadBtn.disabled = false;
      els.uploadBtn.textContent = "Analyze Video & Extract PCM Audio";
    }
  });

  // Load Sample Drill Video Button
  els.demoBtn.addEventListener("click", async () => {
    const gunName = els.firearmInput.value.trim() || "CZ P-10C 9mm";
    els.demoBtn.disabled = true;
    els.demoBtn.textContent = "⏳ Generating Sample Drill...";

    try {
      const formData = new FormData();
      formData.append("gun_name", gunName);
      const res = await fetch("/api/demo-upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Demo creation failed");
      }
      const data = await res.json();
      handleSessionLoaded(data);
      showToast(
        data.session.mode === "automated"
          ? `✓ Sample drill loaded! Automated detection found Beep & ${data.session.shot_timestamps.length} shots.`
          : `✓ Sample drill loaded in Cold Start mode! Try placing Beep at 1.00s and Shots at 2.25s, 2.70s, 3.10s, 3.55s.`
      );
    } catch (err) {
      showToast(err.message, true);
    } finally {
      els.demoBtn.disabled = false;
      els.demoBtn.textContent = "Load Sample Drill Video";
    }
  });

  // Playback Controls (FR-3.2)
  els.playPauseBtn.addEventListener("click", () => {
    if (state.wavesurfer) {
      state.wavesurfer.playPause();
    }
  });

  els.stopBtn.addEventListener("click", () => {
    if (state.wavesurfer) {
      state.wavesurfer.stop();
      els.practiceVideo.currentTime = 0;
    }
  });

  els.speedSelect.addEventListener("change", () => {
    const rate = parseFloat(els.speedSelect.value) || 1.0;
    els.practiceVideo.playbackRate = rate;
    if (state.wavesurfer) {
      state.wavesurfer.setPlaybackRate(rate, true);
    }
  });

  els.zoomSlider.addEventListener("input", () => {
    if (state.wavesurfer) {
      state.wavesurfer.zoom(Number(els.zoomSlider.value));
    }
  });

  // Marker Toolbar Buttons
  els.addBeepBtn.addEventListener("click", setBeepAtPlayhead);
  els.addShotBtn.addEventListener("click", addShotAtPlayhead);
  els.deleteMarkerBtn.addEventListener("click", deleteSelectedMarker);

  // Profile & Render Buttons
  els.saveProfileBtn.addEventListener("click", saveAndRefineProfile);
  els.processVideoBtn.addEventListener("click", processAndRenderVideo);

  // Firearm Manager Modal Controls
  const openModal = () => {
    resetFirearmModalForm();
    showModalFeedback("");
    fetchFirearms();
    els.firearmModal.hidden = false;
  };

  if (els.openFirearmManagerHeaderBtn) {
    els.openFirearmManagerHeaderBtn.addEventListener("click", openModal);
  }
  if (els.openFirearmManagerInlineBtn) {
    els.openFirearmManagerInlineBtn.addEventListener("click", openModal);
  }
  if (els.openFirearmManagerPanelBtn) {
    els.openFirearmManagerPanelBtn.addEventListener("click", openModal);
  }
  if (els.closeFirearmModalBtn) {
    els.closeFirearmModalBtn.addEventListener("click", () => {
      els.firearmModal.hidden = true;
    });
  }
  if (els.firearmModal) {
    els.firearmModal.addEventListener("click", (e) => {
      if (e.target === els.firearmModal) {
        els.firearmModal.hidden = true;
      }
    });
  }
  if (els.cancelEditFirearmBtn) {
    els.cancelEditFirearmBtn.addEventListener("click", () => {
      resetFirearmModalForm();
      showModalFeedback("");
    });
  }

  if (els.firearmManageForm) {
    els.firearmManageForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const origName = els.editOriginalGunName.value.trim();
      const newName = els.manageGunNameInput.value.trim();
      const beepFreq = parseFloat(els.manageBeepFreqInput.value) || 3000.0;
      const beepDur = parseInt(els.manageBeepDurInput.value, 10) || 220;
      const shotDecay = parseFloat(els.manageShotDecayInput.value) || 120.0;

      if (!newName) {
        showModalFeedback("Firearm name is required.", true);
        return;
      }

      try {
        if (origName) {
          // Update existing firearm & signature parameters
          const res = await fetch(
            `/api/firearms/${encodeURIComponent(origName)}`,
            {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                new_gun_name: newName,
                beep_frequency_hz: beepFreq,
                beep_duration_ms: beepDur,
                shot_decay_rate: shotDecay,
              }),
            }
          );
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to update firearm");
          }
          const updated = await res.json();
          if (
            els.firearmInput.value.trim().toLowerCase() ===
            origName.toLowerCase()
          ) {
            els.firearmInput.value = updated.gun_name;
          }
          resetFirearmModalForm();
          await fetchFirearms();
          showModalFeedback(`✓ Updated "${updated.gun_name}".`);
        } else {
          // Create new firearm
          const res = await fetch("/api/firearms", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              gun_name: newName,
              beep_frequency_hz: beepFreq,
              beep_duration_ms: beepDur,
              shot_decay_rate: shotDecay,
            }),
          });
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to create firearm");
          }
          const created = await res.json();
          els.firearmInput.value = created.gun_name;
          resetFirearmModalForm();
          await fetchFirearms();
          showModalFeedback(`✓ Added firearm "${created.gun_name}".`);
        }
      } catch (err) {
        showModalFeedback(err.message, true);
      }
    });
  }

  if (els.restoreDefaultsBtn) {
    els.restoreDefaultsBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/firearms/restore-defaults", {
          method: "POST",
        });
        if (!res.ok) throw new Error("Failed to restore defaults");
        await fetchFirearms();
        showModalFeedback(
          "✓ Restored default firearms (CZ P-10C 9mm, T4 5.56x45mm, CBC 7022 .22LR)."
        );
      } catch (err) {
        showModalFeedback(err.message, true);
      }
    });
  }

  // Switch between Original Video and Rendered Overlay Preview
  els.viewOriginalBtn.addEventListener("click", () => {
    if (!state.session) return;
    state.previewMode = "original";
    els.viewOriginalBtn.classList.add("btn-active");
    els.viewRenderedBtn.classList.remove("btn-active");
    const cur = els.practiceVideo.currentTime;
    els.practiceVideo.src = state.session.video_url;
    els.practiceVideo.currentTime = cur;
  });

  els.viewRenderedBtn.addEventListener("click", () => {
    if (!state.renderedPreviewUrl) return;
    state.previewMode = "rendered";
    els.viewRenderedBtn.classList.add("btn-active");
    els.viewOriginalBtn.classList.remove("btn-active");
    els.liveVideoHud.hidden = true;
    const cur = els.practiceVideo.currentTime;
    els.practiceVideo.src = state.renderedPreviewUrl;
    els.practiceVideo.currentTime = cur;
  });

  // Global Keyboard Shortcuts (FR-3.4)
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") {
      return;
    }
    if (!state.session) return;

    if (e.key === "b" || e.key === "B") {
      e.preventDefault();
      setBeepAtPlayhead();
    } else if (e.key === "s" || e.key === "S") {
      e.preventDefault();
      addShotAtPlayhead();
    } else if (e.key === "Delete" || e.key === "Backspace") {
      if (state.selectedMarker) {
        e.preventDefault();
        deleteSelectedMarker();
      }
    } else if (e.code === "Space") {
      e.preventDefault();
      if (state.wavesurfer) {
        state.wavesurfer.playPause();
      }
    }
  });
}

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  fetchFirearms();
});
