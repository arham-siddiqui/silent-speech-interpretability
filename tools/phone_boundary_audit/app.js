const state = {
  items: [],
  filtered: [],
  current: null,
  audioBuffer: null,
  dragging: null,
};

const elements = {
  progress: document.querySelector("#progress"),
  statusFilter: document.querySelector("#status-filter"),
  qualityFilter: document.querySelector("#quality-filter"),
  list: document.querySelector("#item-list"),
  identifier: document.querySelector("#identifier"),
  transcript: document.querySelector("#transcript"),
  badge: document.querySelector("#quality-badge"),
  phoneError: document.querySelector("#phone-error"),
  confidence: document.querySelector("#confidence"),
  fallback: document.querySelector("#fallback"),
  duration: document.querySelector("#duration"),
  audio: document.querySelector("#audio"),
  canvas: document.querySelector("#timeline"),
  readout: document.querySelector("#hover-readout"),
  notes: document.querySelector("#notes"),
  saveState: document.querySelector("#save-state"),
  previous: document.querySelector("#previous"),
  next: document.querySelector("#next"),
  reset: document.querySelector("#reset"),
};

function reviewStatus(item) {
  return item.review?.status || "unreviewed";
}

function updateProgress() {
  const reviewed = state.items.filter((item) => reviewStatus(item) !== "unreviewed").length;
  elements.progress.textContent = `${reviewed} of ${state.items.length} reviewed`;
}

function applyFilters() {
  const previous = state.current;
  state.filtered = state.items.filter((item) => {
    const statusMatch = elements.statusFilter.value === "all" || reviewStatus(item) === elements.statusFilter.value;
    const qualityMatch = elements.qualityFilter.value === "all" || item.quality_bin === elements.qualityFilter.value;
    return statusMatch && qualityMatch;
  });
  if (!state.filtered.includes(state.current)) {
    state.current = state.filtered[0] || null;
  }
  if (state.current && state.current !== previous) {
    selectItem(state.current);
  } else {
    renderList();
    renderCurrent();
  }
}

function renderList() {
  elements.list.replaceChildren();
  for (const item of state.filtered) {
    const entry = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `item-button${item === state.current ? " active" : ""}`;
    const number = document.createElement("span");
    number.className = "item-number";
    number.textContent = String(item.order).padStart(2, "0");
    const copy = document.createElement("span");
    copy.className = "item-copy";
    const title = document.createElement("strong");
    title.textContent = `User ${item.user_id} · ${item.group_name}`;
    const subtitle = document.createElement("span");
    subtitle.textContent = item.transcript;
    copy.append(title, subtitle);
    const dot = document.createElement("span");
    dot.className = `status-dot ${reviewStatus(item)}`;
    button.append(number, copy, dot);
    button.addEventListener("click", () => selectItem(item));
    entry.append(button);
    elements.list.append(entry);
  }
}

async function selectItem(item) {
  const selectedKey = item.key;
  state.current = item;
  state.audioBuffer = null;
  elements.saveState.textContent = "";
  renderList();
  renderCurrent();
  try {
    const response = await fetch(item.audio_url);
    const bytes = await response.arrayBuffer();
    const context = new AudioContext();
    const decoded = await context.decodeAudioData(bytes);
    await context.close();
    if (state.current?.key !== selectedKey) return;
    state.audioBuffer = decoded;
    drawTimeline();
  } catch (error) {
    elements.saveState.textContent = `Audio decode failed: ${error.message}`;
    elements.saveState.className = "error";
  }
}

function renderCurrent() {
  const item = state.current;
  if (!item) {
    elements.identifier.textContent = "";
    elements.transcript.textContent = "No recordings match the current filters";
    elements.canvas.getContext("2d").clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    return;
  }
  elements.identifier.textContent = `USER ${item.user_id} · ${item.group_name.toUpperCase()} · ${reviewStatus(item).toUpperCase()}`;
  elements.transcript.textContent = item.transcript;
  elements.badge.textContent = item.quality_bin;
  elements.badge.className = `badge ${item.quality_bin}`;
  elements.phoneError.textContent = item.phone_error_rate.toFixed(3);
  elements.confidence.textContent = item.mean_token_probability.toFixed(3);
  elements.fallback.textContent = `${(item.fallback_word_fraction * 100).toFixed(0)}%`;
  elements.duration.textContent = `${item.duration.toFixed(2)} s`;
  elements.audio.src = item.audio_url;
  elements.notes.value = item.review?.notes || "";
  const index = state.filtered.indexOf(item);
  elements.previous.disabled = index <= 0;
  elements.next.disabled = index < 0 || index >= state.filtered.length - 1;
  drawTimeline();
}

function canvasMetrics() {
  const rect = elements.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (elements.canvas.width !== Math.round(rect.width * dpr) || elements.canvas.height !== Math.round(rect.height * dpr)) {
    elements.canvas.width = Math.round(rect.width * dpr);
    elements.canvas.height = Math.round(rect.height * dpr);
  }
  const context = elements.canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function timeToX(time, width) {
  return 12 + (time / state.current.duration) * (width - 24);
}

function xToTime(x, width) {
  return Math.max(0, Math.min(state.current.duration, ((x - 12) / (width - 24)) * state.current.duration));
}

function drawWaveform(context, width) {
  const top = 22;
  const height = 92;
  const center = top + height / 2;
  context.fillStyle = "#edf2f2";
  context.fillRect(12, top, width - 24, height);
  if (!state.audioBuffer) return;
  const data = state.audioBuffer.getChannelData(0);
  const columns = Math.max(1, Math.floor(width - 24));
  const step = Math.max(1, Math.floor(data.length / columns));
  context.strokeStyle = "#385a63";
  context.lineWidth = 1;
  context.beginPath();
  for (let x = 0; x < columns; x += 1) {
    let min = 1;
    let max = -1;
    const start = x * step;
    const end = Math.min(data.length, start + step);
    for (let index = start; index < end; index += 1) {
      min = Math.min(min, data[index]);
      max = Math.max(max, data[index]);
    }
    context.moveTo(12 + x, center + min * height * 0.45);
    context.lineTo(12 + x, center + max * height * 0.45);
  }
  context.stroke();
}

function drawTier(context, intervals, y, height, width, fill, labelKey) {
  context.font = "11px ui-sans-serif, system-ui, sans-serif";
  context.textBaseline = "middle";
  for (const interval of intervals) {
    const x1 = timeToX(interval.start, width);
    const x2 = timeToX(interval.end, width);
    context.fillStyle = fill;
    context.fillRect(x1, y, Math.max(1, x2 - x1), height);
    context.strokeStyle = "#ffffff";
    context.strokeRect(x1, y, Math.max(1, x2 - x1), height);
    context.save();
    context.beginPath();
    context.rect(x1 + 2, y, Math.max(0, x2 - x1 - 4), height);
    context.clip();
    context.fillStyle = "#18272e";
    context.fillText(String(interval[labelKey]), x1 + 5, y + height / 2);
    context.restore();
  }
}

function drawTimeline() {
  const { context, width, height } = canvasMetrics();
  context.clearRect(0, 0, width, height);
  if (!state.current) return;
  drawWaveform(context, width);
  context.fillStyle = "#5d6d74";
  context.font = "10px ui-sans-serif, system-ui, sans-serif";
  context.fillText("WORDS", 12, 139);
  drawTier(context, state.current.words, 148, 38, width, "#d5e4db", "text");
  context.fillText("PHONES", 12, 207);
  drawTier(context, state.current.phones, 216, 42, width, "#cfe2e6", "ipa");
  for (const phone of state.current.phones) {
    for (const boundary of [phone.start, phone.end]) {
      const x = timeToX(boundary, width);
      context.strokeStyle = "#0c7376";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x, 114);
      context.lineTo(x, 264);
      context.stroke();
    }
  }
  context.fillStyle = "#63737a";
  context.textAlign = "center";
  for (let second = 0; second <= state.current.duration; second += 1) {
    const x = timeToX(second, width);
    context.fillText(`${second}s`, x, height - 14);
  }
  context.textAlign = "left";
}

function nearestBoundary(time, width) {
  const pixelThreshold = 10;
  let best = null;
  state.current.phones.forEach((phone, index) => {
    ["start", "end"].forEach((edge) => {
      const distance = Math.abs(timeToX(phone[edge], width) - timeToX(time, width));
      if (distance <= pixelThreshold && (!best || distance < best.distance)) {
        best = { index, edge, distance };
      }
    });
  });
  return best;
}

function moveBoundary(target, time) {
  const phones = state.current.phones;
  const phone = phones[target.index];
  const minimum = 0.001;
  if (target.edge === "start") {
    const previous = phones[target.index - 1];
    const lower = previous ? previous.start + minimum : 0;
    const value = Math.max(lower, Math.min(phone.end - minimum, time));
    const shared = previous && Math.abs(previous.end - phone.start) < 1e-5;
    phone.start = value;
    if (shared) previous.end = value;
  } else {
    const next = phones[target.index + 1];
    const upper = next ? next.end - minimum : state.current.duration;
    const value = Math.max(phone.start + minimum, Math.min(upper, time));
    const shared = next && Math.abs(next.start - phone.end) < 1e-5;
    phone.end = value;
    if (shared) next.start = value;
  }
}

elements.canvas.addEventListener("pointerdown", (event) => {
  if (!state.current) return;
  const rect = elements.canvas.getBoundingClientRect();
  const time = xToTime(event.clientX - rect.left, rect.width);
  state.dragging = nearestBoundary(time, rect.width);
  if (state.dragging) elements.canvas.setPointerCapture(event.pointerId);
});

elements.canvas.addEventListener("pointermove", (event) => {
  if (!state.current) return;
  const rect = elements.canvas.getBoundingClientRect();
  const time = xToTime(event.clientX - rect.left, rect.width);
  const target = state.dragging || nearestBoundary(time, rect.width);
  if (state.dragging) {
    moveBoundary(state.dragging, time);
    drawTimeline();
  }
  elements.canvas.style.cursor = target ? "ew-resize" : "crosshair";
  elements.readout.textContent = target
    ? `${state.current.phones[target.index].ipa} ${target.edge}: ${time.toFixed(3)} s`
    : `Time: ${time.toFixed(3)} s`;
});

elements.canvas.addEventListener("pointerup", (event) => {
  state.dragging = null;
  if (elements.canvas.hasPointerCapture(event.pointerId)) elements.canvas.releasePointerCapture(event.pointerId);
});

async function saveDecision(status) {
  if (!state.current) return;
  elements.saveState.className = "";
  elements.saveState.textContent = "Saving";
  try {
    const response = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        key: state.current.key,
        status,
        notes: elements.notes.value,
        phones: state.current.phones,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Save failed");
    state.current.review = result.review;
    elements.saveState.textContent = "Saved";
    updateProgress();
    applyFilters();
  } catch (error) {
    elements.saveState.textContent = error.message;
    elements.saveState.className = "error";
  }
}

document.querySelectorAll("[data-status]").forEach((button) => {
  button.addEventListener("click", () => saveDecision(button.dataset.status));
});
elements.previous.addEventListener("click", () => selectItem(state.filtered[state.filtered.indexOf(state.current) - 1]));
elements.next.addEventListener("click", () => selectItem(state.filtered[state.filtered.indexOf(state.current) + 1]));
elements.reset.addEventListener("click", () => {
  if (!state.current) return;
  state.current.phones.forEach((phone) => {
    phone.start = phone.original_start;
    phone.end = phone.original_end;
  });
  drawTimeline();
  elements.saveState.textContent = "Boundaries reset";
});
elements.statusFilter.addEventListener("change", applyFilters);
elements.qualityFilter.addEventListener("change", applyFilters);
window.addEventListener("resize", drawTimeline);

fetch("/api/data")
  .then((response) => response.json())
  .then((data) => {
    state.items = data.items;
    state.current = state.items[0];
    updateProgress();
    applyFilters();
    selectItem(state.current);
  })
  .catch((error) => {
    elements.progress.textContent = "Audit set unavailable";
    elements.transcript.textContent = error.message;
  });
