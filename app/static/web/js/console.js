// JARVIS console. Everything here talks to the local API with the bearer token.
// Untrusted text (replies, web content, file names) is only ever set with textContent.
"use strict";

const TOKEN_KEY = "jarvis.token";
const SESSION_KEY = "jarvis.session";
const THEME_KEY = "jarvis.theme";
const PREFS_KEY = "jarvis.prefs";

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const storage = {
  get(key) { try { return localStorage.getItem(key) || ""; } catch { return ""; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* private mode */ } },
};

function newSessionId() {
  const rnd = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).replace(/-/g, "").slice(0, 10);
  return `console-${rnd}`;
}

const prefs = (() => { try { return JSON.parse(storage.get(PREFS_KEY) || "{}"); } catch { return {}; } })();
const state = {
  token: storage.get(TOKEN_KEY),
  sessionId: storage.get(SESSION_KEY) || newSessionId(),
  dryRun: false,
  serverDryRun: false,
  stopped: false,
  model: "",
  speakReplies: Boolean(prefs.speakReplies),
  rehearse: prefs.rehearse !== false,
  view: "talk",
  busy: false,
  pending: new Map(), // pending_id → data
  cards: new Map(), // pending_id → Set of card elements
  audit: [],
  riskFilter: "",
  voiceRunning: false,
  live: false,
};
storage.set(SESSION_KEY, state.sessionId);
const savePrefs = () => storage.set(PREFS_KEY, JSON.stringify({ speakReplies: state.speakReplies, rehearse: state.rehearse }));

// ── DOM helpers ──────────────────────────────────────────────────────────
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = String(text);
  return node;
}
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.appendChild(use);
  return svg;
}
function button(label, cls, onClick, iconName) {
  const b = el("button", `btn ${cls || ""}`.trim());
  b.type = "button";
  if (iconName) b.appendChild(icon(iconName));
  b.appendChild(document.createTextNode(label));
  if (onClick) b.addEventListener("click", onClick);
  return b;
}
function riskPill(risk) {
  const r = String(risk || "").toLowerCase();
  const known = ["low", "medium", "high"].includes(r);
  return el("span", `risk risk-${known ? r : "none"}`, known ? r[0].toUpperCase() + r.slice(1) : "No risk");
}
function setEmpty(listId, emptyId, empty) { $(emptyId).hidden = !empty; $(listId).hidden = empty; }
const plain = (s) => String(s ?? "").replace(/^[^\p{L}\p{N}“"']+/u, "").trim(); // drop leading emoji/symbols from titles

function fmtTime(ts) {
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}
function fmtWhen(ts) {
  const d = new Date(ts * 1000);
  const today = new Date();
  const tomorrow = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 1);
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (d.toDateString() === today.toDateString()) return `Today at ${time}`;
  if (d.toDateString() === tomorrow.toDateString()) return `Tomorrow at ${time}`;
  return `${d.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })} at ${time}`;
}
function fmtMs(v) {
  if (v == null || Number.isNaN(v)) return "—";
  if (v < 10) return `${v.toFixed(1)} ms`;
  if (v < 1000) return `${Math.round(v)} ms`;
  return `${(v / 1000).toFixed(v < 10000 ? 2 : 1)} s`;
}
function fmtDuration(s) {
  if (s == null) return "—";
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
  return `${Math.floor(s / 86400)} d ${Math.floor((s % 86400) / 3600)} h`;
}

// ── API ──────────────────────────────────────────────────────────────────
class AuthError extends Error {}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.token}`, ...(options.headers || {}) },
  });
  if (res.status === 401) {
    showGate("That token was rejected. Copy JARVIS_API_TOKEN from .env, without quotes.");
    throw new AuthError("unauthorized");
  }
  let body = null;
  try { body = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const detail = body && (typeof body.detail === "string" ? body.detail : body.detail?.[0]?.msg || body.message);
    throw new Error(detail || `JARVIS answered ${res.status}.`);
  }
  return body;
}
const post = (path, body) => api(path, { method: "POST", body: body == null ? undefined : JSON.stringify(body) });

// Server-Sent Events over fetch (EventSource cannot send the Authorization header).
async function readSSE(path, body, onEvent, signal) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.token}`, Accept: "text/event-stream" },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (res.status === 401) { showGate("That token was rejected."); throw new AuthError("unauthorized"); }
  if (!res.ok || !res.body) throw new Error(`JARVIS answered ${res.status}.`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let name = "";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      try {
        const ev = JSON.parse(data.join("\n"));
        if (name && !ev.type) ev.type = name;
        onEvent(ev);
      } catch { /* ignore a malformed frame */ }
    }
  }
}

// ── A small, safe Markdown renderer: builds DOM nodes, never parses HTML ───
function inline(parent, text) {
  const re = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    const tok = m[0];
    if (tok.startsWith("**")) parent.appendChild(el("strong", "", tok.slice(2, -2)));
    else if (tok.startsWith("`")) parent.appendChild(el("code", "", tok.slice(1, -1)));
    else {
      const [, label, href] = tok.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      const a = el("a", "", label);
      a.href = href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      parent.appendChild(a);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}
function renderMarkdown(container, text) {
  container.replaceChildren();
  const parts = String(text).split(/```[^\n]*\n?/);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const pre = el("pre");
      pre.appendChild(el("code", "", part.replace(/\n$/, "")));
      container.appendChild(pre);
      return;
    }
    for (const block of part.split(/\n{2,}/)) {
      const lines = block.split("\n").filter((l) => l.trim() !== "");
      if (!lines.length) continue;
      if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) {
        const ul = el("ul");
        for (const l of lines) inline(ul.appendChild(el("li")), l.replace(/^\s*[-*•]\s+/, ""));
        container.appendChild(ul);
      } else if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
        const ol = el("ol");
        for (const l of lines) inline(ol.appendChild(el("li")), l.replace(/^\s*\d+[.)]\s+/, ""));
        container.appendChild(ol);
      } else {
        const p = el("p");
        lines.forEach((l, n) => {
          if (n) p.appendChild(el("br"));
          const heading = l.match(/^#{1,4}\s+(.*)$/);
          if (heading) p.appendChild(el("strong", "", heading[1]));
          else inline(p, l);
        });
        container.appendChild(p);
      }
    }
  });
}

// ── The voice-reactive wordmark: JARVIS's name stretches with sound ───────
const presence = {
  setState(s) {
    const w = document.querySelector(".bar .wordmark");
    if (s) w.setAttribute("data-state", s); else w.removeAttribute("data-state");
  },
  setLevel(level) {
    // level 0..1 → width 85%..125%
    const stretch = `${Math.round(85 + Math.min(1, level) * 40)}%`;
    $("wordmark").style.setProperty("--stretch", stretch);
    $("voiceLine").style.setProperty("--stretch", stretch);
  },
  reset() {
    $("wordmark").style.removeProperty("--stretch");
    $("voiceLine").style.removeProperty("--stretch");
  },
};

// ── Approval cards (the same pending action can show in a turn and the rail) ──
function approvalCard(p) {
  const card = el("div", "approval");
  card.dataset.pending = p.pending_id;
  const top = el("div", "approval-top");
  top.appendChild(riskPill(p.risk));
  const meta = el("span", "approval-meta");
  meta.dataset.expires = String(Date.now() + (p.expires_in || 60) * 1000);
  const src = p.source && p.source !== "chat" ? `from ${p.source}` : "";
  meta.dataset.src = src;
  top.appendChild(meta);
  card.appendChild(top);
  card.appendChild(el("p", "approval-preview", p.preview || "An action is waiting for approval."));
  const actions = el("div", "approval-actions");
  const approve = button("Approve", "btn-primary btn-sm", () => decide(p.pending_id, true), "check");
  const cancel = button("Cancel", "btn-sm", () => decide(p.pending_id, false), "x");
  actions.append(approve, cancel);
  card.appendChild(actions);
  if (!state.cards.has(p.pending_id)) state.cards.set(p.pending_id, new Set());
  state.cards.get(p.pending_id).add(card);
  tickCard(card);
  return card;
}
function tickCard(card) {
  const meta = card.querySelector(".approval-meta");
  if (!meta || card.classList.contains("is-settled")) return;
  const left = Math.max(0, Math.round((Number(meta.dataset.expires) - Date.now()) / 1000));
  meta.textContent = [meta.dataset.src, left > 0 ? `${left} s left` : "expired"].filter(Boolean).join(", ");
  if (left <= 0) settleCards(card.dataset.pending, "This expired, so nothing was done.");
}
function settleCards(pendingId, message) {
  for (const card of state.cards.get(pendingId) || []) {
    if (card.classList.contains("is-settled")) continue;
    card.classList.add("is-settled");
    card.querySelector(".approval-actions")?.remove();
    const meta = card.querySelector(".approval-meta");
    if (meta) meta.textContent = "";
    card.appendChild(el("p", "approval-result", message));
  }
  state.cards.delete(pendingId);
}
async function decide(pendingId, approve) {
  for (const card of state.cards.get(pendingId) || []) card.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  try {
    if (approve) {
      const res = await post(`/automation/confirm/${encodeURIComponent(pendingId)}`);
      settleCards(pendingId, res.message || (res.success ? "Done." : "That didn't work."));
      if (!res.success) toastTurn(res.message);
    } else {
      await post(`/automation/cancel/${encodeURIComponent(pendingId)}`);
      settleCards(pendingId, "Cancelled. Nothing was done.");
    }
  } catch (err) {
    if (err instanceof AuthError) return;
    settleCards(pendingId, err.message);
  }
  state.pending.delete(pendingId);
  renderRail();
  refreshPending();
}
function toastTurn(message) {
  if (!message) return;
  const t = el("article", "turn");
  const j = el("div", "turn-jarvis");
  const a = el("div", "answer is-error");
  a.textContent = message;
  j.appendChild(a);
  t.appendChild(j);
  appendTurn(t);
}

async function refreshPending() {
  try {
    const { pending } = await api("/automation/pending");
    const ids = new Set(pending.map((p) => p.pending_id));
    for (const id of [...state.pending.keys()]) {
      if (!ids.has(id)) { state.pending.delete(id); settleCards(id, "No longer waiting."); }
    }
    for (const p of pending) state.pending.set(p.pending_id, p);
    renderRail();
  } catch { /* shown by the link state */ }
}
function renderRail() {
  const list = $("approvalList");
  const shown = new Set([...list.children].map((c) => c.dataset.pending));
  for (const child of [...list.children]) if (!state.pending.has(child.dataset.pending)) child.remove();
  for (const p of state.pending.values()) {
    if (shown.has(p.pending_id)) continue;
    const li = el("li");
    li.dataset.pending = p.pending_id;
    li.appendChild(approvalCard(p));
    list.prepend(li);
  }
  const n = state.pending.size;
  $("pendingCount").textContent = String(n);
  $("navPending").textContent = String(n);
  $("navPending").hidden = n === 0;
  $("approvalEmpty").hidden = n > 0;
  document.querySelector(".approvals").classList.toggle("is-empty", n === 0);
  document.title = n ? `(${n}) JARVIS console` : "JARVIS console";
}
setInterval(() => document.querySelectorAll(".approval:not(.is-settled)").forEach(tickCard), 1000);

// ── Talk ─────────────────────────────────────────────────────────────────
function appendTurn(node) {
  $("threadEmpty").hidden = true;
  $("thread").appendChild(node);
  $("thread").scrollTop = $("thread").scrollHeight;
}

function miniTrace(r) {
  const wrap = el("div", "mini-trace");
  const steps = [];
  const meta = r.metadata || {};
  if (r.route === "fast_path") steps.push(["Fast path", ""]);
  else if (r.route === "agent") steps.push([meta.model ? `${meta.provider}/${meta.model}` : "Language model", meta.fallback_used ? "is-wait" : ""]);
  else if (r.route === "confirmation") steps.push(["Your answer", ""]);
  else steps.push(["Couldn't answer", "is-stop"]);
  if (r.status === "needs_confirmation" && r.pending) {
    const risk = String(r.pending.risk || "high");
    steps.push([`${risk[0].toUpperCase()}${risk.slice(1)} risk, waiting for you`, risk === "high" ? "is-stop" : "is-wait"]);
  } else if (r.status === "dry_run") steps.push(["Dry run, nothing changed", "is-dim"]);
  else if (r.status === "error") steps.push(["Didn't work", "is-stop"]);
  else if (r.status === "blocked") steps.push(["Blocked by the stop switch", "is-stop"]);
  else if (r.status === "ok") steps.push(["Done", ""]);
  else steps.push([String(r.status || "Finished").replace(/_/g, " "), "is-dim"]);
  if (meta.total_latency_ms != null) steps.push([fmtMs(meta.total_latency_ms), "is-dim"]);
  steps.forEach(([label, cls], i) => {
    if (i) wrap.appendChild(el("span", "mini-sep"));
    wrap.appendChild(el("span", `mini-step ${cls}`.trim(), label));
  });
  for (const t of r.tools || []) if (t.name) wrap.appendChild(el("span", "tool-tag", t.name));
  return wrap;
}

function turnShell(query, origin) {
  const turn = el("article", "turn");
  if (origin) turn.appendChild(el("p", "turn-origin", origin));
  const you = el("div", "turn-you");
  you.appendChild(el("p", "you-text", query));
  turn.appendChild(you);
  const jarvis = el("div", "turn-jarvis");
  const answer = el("div", "answer is-waiting");
  jarvis.appendChild(answer);
  turn.appendChild(jarvis);
  return { turn, jarvis, answer };
}

function finishTurn(jarvis, answer, r, streamedText) {
  answer.classList.remove("is-waiting");
  const text = streamedText || r.response || "";
  renderMarkdown(answer, text);
  if (r.status === "error") answer.classList.add("is-error");
  jarvis.appendChild(miniTrace(r));
  if (r.status === "needs_confirmation" && r.pending) {
    state.pending.set(r.pending.pending_id, r.pending);
    jarvis.appendChild(approvalCard(r.pending));
    renderRail();
  }
}

let streamAbort = null;
async function send(text, { fromVoice = false } = {}) {
  text = text.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("sendBtn").disabled = true;
  stopSpeaking();
  const { turn, jarvis, answer } = turnShell(text);
  answer.textContent = "Thinking";
  answer.appendChild(el("span", "caret"));
  appendTurn(turn);
  presence.setState("thinking");

  let streamed = "";
  let result = null;
  let raf = 0;
  const paint = () => { raf = 0; renderMarkdown(answer, streamed); answer.appendChild(el("span", "caret")); $("thread").scrollTop = $("thread").scrollHeight; };
  streamAbort = new AbortController();
  try {
    await readSSE("/chat/stream", {
      query: text,
      session_id: state.sessionId,
      channel: fromVoice ? "voice" : "text",
      dry_run: state.dryRun ? true : null,
      model_override: state.model || null,
    }, (ev) => {
      if (ev.type === "text") {
        if (!streamed) answer.classList.remove("is-waiting");
        streamed += ev.delta || "";
        if (!raf) raf = requestAnimationFrame(paint);
      } else if (ev.type === "tool") {
        if (!streamed) { answer.textContent = `Using ${ev.name}`; answer.appendChild(el("span", "caret")); }
      } else if (ev.type === "status" && ev.message && !streamed) {
        answer.textContent = ev.message;
        answer.appendChild(el("span", "caret"));
      } else if (ev.type === "done") {
        result = ev.result;
      } else if (ev.type === "error") {
        throw new Error(ev.message || "The reply stopped.");
      }
    }, streamAbort.signal);
    if (raf) cancelAnimationFrame(raf);
    if (!result) throw new Error("The reply ended before JARVIS finished.");
    finishTurn(jarvis, answer, result, streamed);
    if (fromVoice || state.speakReplies) speak(result.response);
  } catch (err) {
    if (raf) cancelAnimationFrame(raf);
    if (!(err instanceof AuthError)) {
      answer.classList.remove("is-waiting");
      answer.classList.add("is-error");
      answer.textContent = err.name === "AbortError" ? "Stopped." : `${err.message} Check that JARVIS is still running, then try again.`;
    }
  } finally {
    state.busy = false;
    streamAbort = null;
    $("sendBtn").disabled = false;
    if (!speaking) presence.setState("");
    $("thread").scrollTop = $("thread").scrollHeight;
  }
}

// Turns that happened elsewhere (the PC microphone, schedules, other apps) appear in the thread too.
function externalTurn(ev) {
  const origin = ev.channel === "voice" ? "Said to the PC microphone" : ev.channel === "api" ? "From another app" : "From another window";
  const { turn, jarvis, answer } = turnShell(ev.query || "", origin);
  finishTurn(jarvis, answer, { response: ev.response, route: ev.route, status: ev.status, metadata: {} }, "");
  appendTurn(turn);
}

function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = `${Math.min(t.scrollHeight, 180)}px`;
}

// ── Voice in the browser (Web Speech API) with a live level meter ──────────
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
let listening = false;
let audio = null;

async function startLevel() {
  if (reduceMotion || !navigator.mediaDevices?.getUserMedia) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    let smooth = 0;
    const loop = () => {
      if (!audio) return;
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += ((v - 128) / 128) ** 2;
      const rms = Math.sqrt(sum / buf.length);
      smooth = smooth * 0.7 + Math.min(1, rms * 6) * 0.3;
      presence.setLevel(smooth);
      audio.raf = requestAnimationFrame(loop);
    };
    audio = { stream, ctx, raf: 0 };
    loop();
  } catch { /* no level meter: the recognizer still works */ }
}
function stopLevel() {
  if (!audio) return;
  cancelAnimationFrame(audio.raf);
  audio.stream.getTracks().forEach((t) => t.stop());
  audio.ctx.close().catch(() => {});
  audio = null;
  presence.reset();
}

function setMic(on) {
  listening = on;
  $("micBtn").setAttribute("aria-pressed", String(on));
  $("micBtn").setAttribute("aria-label", on ? "Stop listening" : "Speak");
  $("voiceLine").hidden = !on && !$("voiceLine").textContent;
  presence.setState(on ? "listening" : "");
}

function toggleMic() {
  if (!Recognition) {
    // No speech recognition in this browser: use the PC's own microphone loop instead.
    togglePcListening();
    return;
  }
  if (listening) { rec?.stop(); return; }
  stopSpeaking();
  rec = new Recognition();
  rec.lang = navigator.language || "en-US";
  rec.interimResults = true;
  rec.continuous = false;
  let finalText = "";
  const line = $("voiceLine");
  line.textContent = "Listening…";
  line.classList.add("is-interim");
  line.hidden = false;
  rec.onresult = (e) => {
    let interim = "";
    finalText = "";
    for (const r of e.results) {
      if (r.isFinal) finalText += r[0].transcript;
      else interim += r[0].transcript;
    }
    line.textContent = (finalText + interim).trim() || "Listening…";
    line.classList.toggle("is-interim", !finalText);
  };
  rec.onerror = (e) => {
    const why = { "not-allowed": "Microphone access was blocked. Allow it in the address bar, then try again.",
      "no-speech": "I didn't hear anything. Try again a little closer to the mic.",
      network: "Speech recognition needs a connection in this browser. Type instead, or use the PC microphone in System." }[e.error];
    line.textContent = why || `Speech recognition stopped (${e.error}).`;
    line.classList.add("is-interim");
  };
  rec.onend = () => {
    setMic(false);
    stopLevel();
    const text = finalText.trim();
    if (text) {
      line.textContent = "";
      line.hidden = true;
      send(text, { fromVoice: true });
    } else {
      setTimeout(() => { if (!listening) { line.hidden = true; line.textContent = ""; } }, 2600);
    }
  };
  try {
    rec.start();
    setMic(true);
    startLevel();
  } catch {
    setMic(false);
  }
}

let speaking = false;
function speak(text) {
  if (!("speechSynthesis" in window) || !text) return;
  const clean = String(text).replace(/```[\s\S]*?```/g, " (code shown on screen) ").replace(/[*_`#>]/g, "").slice(0, 1200);
  const u = new SpeechSynthesisUtterance(clean);
  u.lang = navigator.language || "en-US";
  u.rate = 1.04;
  u.onstart = () => { speaking = true; presence.setState("speaking"); };
  u.onboundary = () => { if (!reduceMotion) { presence.setLevel(0.9); setTimeout(() => presence.setLevel(0.35), 120); } };
  u.onend = u.onerror = () => { speaking = false; presence.reset(); if (!state.busy) presence.setState(""); };
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}
function stopSpeaking() {
  if ("speechSynthesis" in window && (speechSynthesis.speaking || speechSynthesis.pending)) speechSynthesis.cancel();
  speaking = false;
}

// ── Live activity (the event stream) ─────────────────────────────────────
const seen = new Set();
const OP_WORDS = {
  "system.power": "Power", "process.kill": "End process", "app.open": "Open app", "app.close": "Close app",
  "message.send": "Send message", "file.trash": "Move to Recycle Bin", "file.move": "Move file", "file.create": "Create file",
  "files.organize": "Organize folder", "computer.run_plan": "Computer task", "browser.agent": "Browser task",
  "shell.run_named": "Run command", "workflow.replay": "Replay workflow", "routine.run": "Routine", pending: "Approval",
};
function opLabel(op) { return OP_WORDS[op] || String(op || "action").replace(/[._]/g, " "); }
function targetText(t) {
  if (t == null) return "";
  if (typeof t === "string") return t;
  try { return Object.values(t).filter((v) => typeof v === "string" || typeof v === "number").join(", ").slice(0, 160); } catch { return ""; }
}

function feedItem(ev, isNew) {
  let kind = "info", title = "", sub = "";
  const d = ev.data || {};
  switch (ev.type) {
    case "turn":
      kind = "turn"; title = ev.query || ""; sub = `${ev.channel === "voice" ? "Voice" : ev.channel === "api" ? "App" : "Typed"}, ${String(ev.response || "").slice(0, 120)}`; break;
    case "gateway.audit":
      kind = String(d.risk || "info").toLowerCase(); title = `${opLabel(d.op)}${targetText(d.target) ? `: ${targetText(d.target)}` : ""}`;
      sub = `${DECISIONS[d.decision] || d.decision || ""}${d.success === false && d.decision === "executed" ? ", failed" : ""}`; break;
    case "gateway.pending_created":
      kind = "pending"; title = "Waiting for your approval"; sub = d.preview || ""; break;
    case "gateway.pending_resolved":
      title = "Approved and run"; break;
    case "gateway.pending_cancelled":
      title = "Approval cancelled"; break;
    case "health_alert":
      kind = "alert"; title = plain(ev.title) || "System health"; sub = ev.message || ""; break;
    default:
      title = plain(ev.title) || ev.type; sub = ev.message || "";
  }
  const li = el("li", `feed-item${isNew ? " is-new" : ""}`);
  li.dataset.kind = kind;
  li.appendChild(el("span", "feed-dot"));
  const body = el("p", "feed-text", title);
  if (sub) body.appendChild(el("small", "", sub));
  li.appendChild(body);
  li.appendChild(el("time", "feed-time", fmtTime(ev.ts)));
  return li;
}

function onEvent(ev, isNew) {
  const key = `${ev.type}|${ev.ts}|${ev.session_id || ""}|${ev.data?.id || ev.data?.pending_id || ""}`;
  if (seen.has(key)) return;
  seen.add(key);
  const feed = $("feed");
  feed.prepend(feedItem(ev, isNew));
  while (feed.children.length > 80) feed.lastChild.remove();
  $("feedEmpty").hidden = true;

  if (ev.type.startsWith("gateway.pending")) refreshPending();
  if (ev.type === "gateway.audit" && state.view === "log") loadLog();
  if (ev.type === "reminder" && state.view === "today") loadToday();
  if (isNew && ev.type === "turn" && ev.session_id !== state.sessionId && ev.session_id !== "site-preview") externalTurn(ev);
}

function setLink(s) {
  state.live = s === "live";
  $("linkState").dataset.state = s;
  $("linkText").textContent = { live: "Live", lost: "Reconnecting", connecting: "Connecting" }[s];
}
async function eventLoop() {
  for (;;) {
    const since = Date.now() / 1000;
    try {
      setLink("connecting");
      await readSSE("/events", null, (ev) => { setLink("live"); onEvent(ev, ev.ts >= since - 1); });
    } catch (err) {
      if (err instanceof AuthError) return;
    }
    setLink("lost");
    await new Promise((r) => setTimeout(r, 2500));
  }
}

// ── Today ────────────────────────────────────────────────────────────────
async function loadToday() {
  const [{ reminders }, { todos }] = await Promise.all([api("/reminders"), api("/todos")]);
  const rl = $("remList");
  rl.replaceChildren();
  for (const r of reminders) {
    const li = el("li", "row");
    const main = el("div", "row-main", r.text);
    main.appendChild(el("small", "", `${fmtWhen(r.due_ts)}${r.recurrence ? `, repeats ${r.recurrence}` : ""}`));
    li.appendChild(main);
    const acts = el("div", "row-actions");
    acts.appendChild(button("Cancel", "btn-sm btn-quiet", async () => { await api(`/reminders/${r.id}`, { method: "DELETE" }); loadToday(); }));
    li.appendChild(acts);
    rl.appendChild(li);
  }
  $("remCount").textContent = String(reminders.length);
  setEmpty("remList", "remEmpty", !reminders.length);

  const tl = $("todoList");
  tl.replaceChildren();
  for (const t of todos) {
    const li = el("li", "row");
    const check = el("button", "check");
    check.type = "button";
    check.setAttribute("aria-label", `Mark “${t.text}” done`);
    check.appendChild(icon("check"));
    check.addEventListener("click", async () => { check.disabled = true; await post(`/todos/${t.id}/done`); loadToday(); });
    li.appendChild(check);
    li.appendChild(el("div", "row-main", t.text));
    tl.appendChild(li);
  }
  $("todoCount").textContent = String(todos.length);
  setEmpty("todoList", "todoEmpty", !todos.length);
  await loadNotes();
}
async function loadNotes() {
  const q = $("noteSearch").value.trim();
  const { notes } = await api(`/notes${q ? `?search=${encodeURIComponent(q)}` : ""}`);
  const nl = $("noteList");
  nl.replaceChildren();
  for (const n of notes) {
    const li = el("li", "row");
    const main = el("div", "row-main", n.text);
    main.appendChild(el("small", "", fmtWhen(n.created_ts)));
    li.appendChild(main);
    nl.appendChild(li);
  }
  $("noteCount").textContent = String(notes.length);
  $("noteEmpty").textContent = q ? `No notes mention “${q}”.` : "No notes yet.";
  setEmpty("noteList", "noteEmpty", !notes.length);
}

// ── Routines and workflows ───────────────────────────────────────────────
function showRunOutput(lines, result) {
  const out = $("runOutput");
  out.replaceChildren();
  out.hidden = false;
  out.appendChild(document.createTextNode(lines.filter(Boolean).join("\n")));
  if (result && result.status === "needs_confirmation" && result.pending_id) {
    const p = { pending_id: result.pending_id, risk: result.risk, preview: result.preview || result.message, expires_in: 60 };
    state.pending.set(p.pending_id, p);
    out.appendChild(approvalCard(p));
    renderRail();
  }
}
async function loadRoutines() {
  const [{ routines }, wf] = await Promise.all([api("/routines"), api("/automation/workflows")]);
  const rl = $("routineList");
  rl.replaceChildren();
  for (const r of routines) {
    const li = el("li", "row");
    const main = el("div", "row-main", r.name.replace(/_/g, " "));
    if (r.description) main.appendChild(el("small", "", r.description));
    li.appendChild(main);
    const acts = el("div", "row-actions");
    const run = button(state.rehearse ? "Rehearse" : "Run", "btn-sm", async () => {
      run.disabled = true;
      try {
        const res = await post(`/routines/${encodeURIComponent(r.name)}/run`, { dry_run: state.rehearse, session_id: state.sessionId });
        showRunOutput([`${state.rehearse ? "Rehearsed" : "Ran"} “${r.name.replace(/_/g, " ")}”.`, ...(res.messages || [])], res.result);
      } catch (err) { showRunOutput([err.message]); }
      run.disabled = false;
    });
    acts.appendChild(run);
    li.appendChild(acts);
    rl.appendChild(li);
  }
  $("rtCount").textContent = String(routines.length);
  setEmpty("routineList", "routineEmpty", !routines.length);

  const wl = $("workflowList");
  wl.replaceChildren();
  for (const w of wf.workflows || []) {
    const li = el("li", "row");
    const main = el("div", "row-main", w.name);
    const bits = [`${w.steps} step${w.steps === 1 ? "" : "s"}`];
    if (w.run_count) bits.push(`run ${w.run_count} times, ${w.success_rate}% worked`);
    if (w.trigger_phrases?.length) bits.push(`say “${w.trigger_phrases[0]}”`);
    main.appendChild(el("small", "", bits.join(", ")));
    li.appendChild(main);
    const acts = el("div", "row-actions");
    const replay = button(state.rehearse ? "Rehearse" : "Replay", "btn-sm", async () => {
      replay.disabled = true;
      try {
        const res = await post("/automation/replay", { name: w.name, dry_run: state.rehearse, session_id: state.sessionId });
        showRunOutput([res.message], res);
      } catch (err) { showRunOutput([err.message]); }
      replay.disabled = false;
    });
    acts.appendChild(replay);
    li.appendChild(acts);
    wl.appendChild(li);
  }
  $("wfCount").textContent = String((wf.workflows || []).length);
  setEmpty("workflowList", "workflowEmpty", !(wf.workflows || []).length);
}

// ── Activity log ─────────────────────────────────────────────────────────
const DECISIONS = { executed: "Done", dry_run: "Dry run", needs_confirmation: "Asked you", confirmed: "Approved", cancelled: "Cancelled", blocked: "Blocked", rejected: "Refused", expired: "Expired", undone: "Undone" };
async function loadLog() {
  const { entries } = await api("/automation/audit?limit=300");
  state.audit = entries.reverse();
  renderLog();
}
function renderLog() {
  const body = $("logBody");
  body.replaceChildren();
  const rows = state.audit.filter((e) => !state.riskFilter || String(e.risk).toLowerCase() === state.riskFilter);
  for (const e of rows.slice(0, 200)) {
    const tr = el("tr");
    const when = new Date(e.ts);
    tr.appendChild(el("td", "t", Number.isNaN(when.getTime()) ? e.ts : `${when.toLocaleDateString([], { day: "numeric", month: "short" })} ${when.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })}`));
    tr.appendChild(el("td", "op", e.op));
    const msg = el("td", "msg", targetText(e.target) || "—");
    if (e.message) msg.appendChild(el("small", "", e.message));
    tr.appendChild(msg);
    const risk = el("td");
    risk.appendChild(riskPill(e.risk));
    tr.appendChild(risk);
    const decision = String(e.decision || "");
    const bad = e.success === false || ["blocked", "rejected"].includes(decision);
    tr.appendChild(el("td", `decision${bad ? " is-bad" : decision === "needs_confirmation" ? " is-wait" : ""}`,
      `${DECISIONS[decision] || decision}${e.success === false && decision === "executed" ? ", failed" : ""}`));
    tr.appendChild(el("td", "src", e.source || ""));
    body.appendChild(tr);
  }
  $("logEmpty").hidden = rows.length > 0;
  $("logEmpty").textContent = state.audit.length ? "No entries at this risk level." : "Nothing has run yet. Every action JARVIS takes or refuses is recorded here.";
  document.querySelector(".log-table").hidden = rows.length === 0;
}

// ── System ───────────────────────────────────────────────────────────────
const STAGES = {
  router: "Matching commands", "turn.fast_path": "Turn without a model", "turn.agent": "Turn with a model",
  "turn.confirmation": "Answering a yes or no", "turn.error": "Turn that failed", stt: "Speech to text", tts: "Text to speech",
};
function stageName(key) {
  if (STAGES[key]) return STAGES[key];
  if (key.startsWith("llm.")) return `Model ${key.slice(4)}`;
  if (key.startsWith("tool.")) return `Tool ${key.slice(5)}`;
  return key.replace(/[._]/g, " ");
}
function facts(dl, pairs) {
  dl.replaceChildren();
  for (const [k, v] of pairs) { dl.appendChild(el("dt", "", k)); dl.appendChild(el("dd", "", v)); }
}
async function loadSystem() {
  const [m, s, c] = await Promise.all([api("/metrics"), api("/status"), api("/config")]);
  const lat = m.latency || {};
  const turns = Object.entries(lat).filter(([k]) => k.startsWith("turn.")).reduce((n, [, v]) => n + v.count, 0);
  const fast = (lat["turn.fast_path"]?.count || 0) + (lat["turn.confirmation"]?.count || 0);
  $("kUptime").textContent = fmtDuration(m.uptime_s);
  $("kTurns").textContent = String(turns);
  $("kFast").textContent = turns ? `${Math.round((fast / turns) * 100)}%` : "0%";
  $("kTools").textContent = String(s.registered_tool_count);

  const box = $("latency");
  box.replaceChildren();
  const rows = Object.entries(lat).sort((a, b) => b[1].p50_ms - a[1].p50_ms).slice(0, 10);
  const max = Math.max(1, ...rows.map(([, v]) => v.p95_ms));
  for (const [key, v] of rows) {
    const row = el("div", "lat-row");
    const name = el("span", "lat-name", stageName(key));
    name.title = `${key}, ${v.count} samples`;
    row.appendChild(name);
    const track = el("span", "lat-track");
    const bar = el("span", "lat-bar");
    bar.style.width = `${Math.max(0.5, (v.p50_ms / max) * 100)}%`;
    const tick = el("span", "lat-p95");
    tick.style.left = `calc(${Math.min(100, (v.p95_ms / max) * 100)}% - 2px)`;
    track.append(bar, tick);
    row.appendChild(track);
    const val = el("span", "lat-val", fmtMs(v.p50_ms));
    val.appendChild(el("small", "", ` / ${fmtMs(v.p95_ms)}`));
    row.appendChild(val);
    box.appendChild(row);
  }
  $("latEmpty").hidden = rows.length > 0;
  box.hidden = rows.length === 0;

  facts($("modelFacts"), [
    ["Default", `${c.default_llm_provider}/${c.default_model}`],
    ["Quick tasks", c.fast_model],
    ["If it fails, then", (c.fallback_chain || []).join(", ") || "—"],
    ["Ready to use", (c.configured_models || []).join(", ") || "None. Add a key to .env or start Ollama."],
  ]);
  state.serverDryRun = Boolean(c.dry_run);
  facts($("safetyFacts"), [
    ["Dry run for everything", c.dry_run ? "On" : "Off"],
    ["Ask before medium risk", c.confirm_medium_risk ? "Yes" : "No, only high risk asks"],
    ["Private mode", c.privacy_mode === "local" ? "On: files and screen go to local models only" : "Off"],
    ["Emergency stop", state.stopped ? "Active" : "Clear"],
  ]);
  facts($("browserFacts"), [
    ["Speech recognition", Recognition ? "Available in this browser" : "Not available here; the mic button uses the PC microphone"],
    ["Conversation", state.sessionId],
    ["Theme", { dark: "Dark", light: "Light" }[storage.get(THEME_KEY)] || "Follows your system"],
  ]);
  $("sysFresh").textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
  await loadVoice(true);
}

let devicesLoaded = false;
async function loadVoice(withDevices) {
  try {
    const v = await api("/voice/status");
    state.voiceRunning = Boolean(v.is_running);
    $("pcVoice").hidden = !v.is_running;
    $("pcVoiceText").textContent = v.state === "speaking" ? "Speaking on this PC" : "Listening on this PC";
    const vs = $("voiceState");
    vs.textContent = v.is_running ? String(v.state || "listening").replace(/_/g, " ") : "Off";
    vs.classList.toggle("is-on", Boolean(v.is_running));
    $("pcListenBtn").textContent = v.is_running ? "Stop listening" : "Start listening";
    const top = Math.max(1, (v.energy_threshold || 1) * 3);
    $("meterFill").style.width = `${Math.min(100, ((v.current_energy || 0) / top) * 100)}%`;
    $("meterMark").style.left = `${Math.min(100, ((v.energy_threshold || 0) / top) * 100)}%`;
    $("voiceError").textContent = v.error || "";
    if (withDevices && !devicesLoaded) {
      const d = await api("/voice/devices");
      const sel = $("deviceSelect");
      for (const dev of d.devices || []) {
        const o = el("option", "", dev.name || `Microphone ${dev.index}`);
        o.value = String(dev.index);
        if (dev.index === d.selected_device_index) o.selected = true;
        sel.appendChild(o);
      }
      devicesLoaded = true;
    }
  } catch (err) {
    if (err instanceof AuthError) return;
    $("voiceState").textContent = "Unavailable";
    $("voiceError").textContent = "Voice isn't available on this PC. Install the voice extras: pip install -e \".[voice]\"";
  }
}
async function togglePcListening() {
  $("pcListenBtn").disabled = true;
  try {
    const res = await post(state.voiceRunning ? "/voice/stop" : "/voice/start");
    if (!state.voiceRunning && !res.success) $("voiceError").textContent = "The microphone didn't start. Check the device below and that no other app is using it.";
  } catch (err) { $("voiceError").textContent = err.message; }
  $("pcListenBtn").disabled = false;
  loadVoice(false);
}

// ── Safety controls ──────────────────────────────────────────────────────
async function loadStatus() {
  const s = await api("/automation/status");
  setStopped(Boolean(s.emergency_stopped));
  state.serverDryRun = Boolean(s.dry_run);
  const sw = $("dryRun");
  sw.disabled = state.serverDryRun;
  sw.setAttribute("aria-checked", String(state.serverDryRun || state.dryRun));
  sw.title = state.serverDryRun ? "Dry run is on for everything (DRY_RUN=true in .env)." : "Plan actions without doing them. Applies to what you send from this console.";
  for (const p of s.pending || []) state.pending.set(p.pending_id, p);
  renderRail();
}
function setStopped(stopped) {
  state.stopped = stopped;
  $("stoppedBanner").hidden = !stopped;
  $("stopBtn").disabled = stopped;
}

// ── Views ────────────────────────────────────────────────────────────────
const VIEWS = { talk: null, today: loadToday, routines: loadRoutines, log: loadLog, system: loadSystem };
function showView(name) {
  if (!(name in VIEWS)) name = "talk";
  state.view = name;
  for (const v of Object.keys(VIEWS)) $(`view-${v}`).hidden = v !== name;
  for (const a of document.querySelectorAll(".side-link[data-view]")) {
    if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  $("main").dataset.view = name;
  const load = VIEWS[name];
  if (load) load().catch((err) => { if (!(err instanceof AuthError)) console.warn(err); });
  if (name === "talk") setTimeout(() => $("input").focus({ preventScroll: true }), 0);
}

// ── Theme ────────────────────────────────────────────────────────────────
function applyTheme(t) {
  if (t === "dark" || t === "light") document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
  $("themeBtn").setAttribute("aria-label", `Theme: ${t === "dark" ? "dark" : t === "light" ? "light" : "follows your system"}`);
  $("themeBtn").title = $("themeBtn").getAttribute("aria-label");
}

// ── Connect / disconnect ─────────────────────────────────────────────────
let started = false;
function showGate(message) {
  $("app").hidden = true;
  $("gate").hidden = false;
  $("gateError").textContent = message || "";
  setTimeout(() => $("tokenInput").focus(), 0);
}
function disconnect() {
  storage.remove(TOKEN_KEY);
  state.token = "";
  location.reload();
}
async function start() {
  try {
    await loadStatus();
  } catch (err) {
    if (!(err instanceof AuthError)) showGate(`JARVIS isn't answering: ${err.message}`);
    return;
  }
  $("gate").hidden = true;
  $("app").hidden = false;
  if (started) return;
  started = true;
  showView(location.hash.slice(1) || "talk");
  eventLoop();
  setInterval(refreshPending, 5000);
  setInterval(() => { if (state.view === "system") loadVoice(false); }, 1000);
  setInterval(() => { if (state.view !== "system") loadVoice(false); }, 10000);
  setInterval(() => loadStatus().catch(() => {}), 15000);
  loadVoice(false);
  api("/config").then((c) => {
    for (const m of c.configured_models || []) {
      const o = el("option", "", m);
      o.value = m;
      $("modelSelect").appendChild(o);
    }
    $("modelSelect").options[0].textContent = `Default (${c.default_llm_provider}/${c.default_model})`;
  }).catch(() => {});
}

function wire() {
  $("gateForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const token = $("tokenInput").value.trim();
    if (!token) return;
    state.token = token;
    $("gateError").textContent = "Checking…";
    try {
      const res = await fetch("/automation/status", { headers: { Authorization: `Bearer ${token}` } });
      if (res.status === 401) { $("gateError").textContent = "That token was rejected. Copy JARVIS_API_TOKEN from .env, without quotes."; return; }
      if (!res.ok) { $("gateError").textContent = `JARVIS answered ${res.status}. Is it fully started?`; return; }
    } catch {
      $("gateError").textContent = "JARVIS isn't reachable. Start it with python jarvis.py, then try again.";
      return;
    }
    storage.set(TOKEN_KEY, token);
    $("tokenInput").value = "";
    $("gateError").textContent = "";
    start();
  });

  $("composer").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = $("input").value;
    $("input").value = "";
    autosize();
    send(text);
  });
  $("input").addEventListener("input", autosize);
  $("input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $("composer").requestSubmit(); }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { stopSpeaking(); if (listening) rec?.abort(); streamAbort?.abort(); }
  });
  for (const chip of document.querySelectorAll("[data-say]")) chip.addEventListener("click", () => send(chip.dataset.say));
  $("micBtn").addEventListener("click", toggleMic);
  if (!Recognition) $("micBtn").setAttribute("aria-label", "Listen with the PC microphone");

  const sw = (id, getter, setter) => {
    const b = $(id);
    b.setAttribute("aria-checked", String(getter()));
    b.addEventListener("click", () => { setter(!getter()); b.setAttribute("aria-checked", String(getter())); });
  };
  sw("speakReplies", () => state.speakReplies, (v) => { state.speakReplies = v; savePrefs(); if (!v) stopSpeaking(); });
  sw("dryRun", () => state.dryRun || state.serverDryRun, (v) => { if (!state.serverDryRun) state.dryRun = v; });
  sw("rehearse", () => state.rehearse, (v) => { state.rehearse = v; savePrefs(); loadRoutines().catch(() => {}); });

  $("modelSelect").addEventListener("change", () => { state.model = $("modelSelect").value; });
  $("newChatBtn").addEventListener("click", () => {
    state.sessionId = newSessionId();
    storage.set(SESSION_KEY, state.sessionId);
    for (const t of [...$("thread").querySelectorAll(".turn")]) t.remove();
    $("threadEmpty").hidden = false;
    $("input").focus();
  });

  $("stopBtn").addEventListener("click", async () => {
    streamAbort?.abort();
    stopSpeaking();
    try { await post("/automation/stop"); setStopped(true); } catch { /* link state shows it */ }
    refreshPending();
  });
  $("resumeBtn").addEventListener("click", async () => {
    try { await post("/automation/resume"); setStopped(false); } catch { /* link state shows it */ }
  });

  $("themeBtn").addEventListener("click", () => {
    const order = ["", "dark", "light"];
    const next = order[(order.indexOf(storage.get(THEME_KEY)) + 1) % order.length];
    if (next) storage.set(THEME_KEY, next); else storage.remove(THEME_KEY);
    applyTheme(next);
  });
  applyTheme(storage.get(THEME_KEY));

  $("disconnectBtn").addEventListener("click", disconnect);
  $("disconnectBtn2").addEventListener("click", disconnect);

  $("briefingBtn").addEventListener("click", async () => {
    const box = $("briefing");
    box.hidden = false;
    box.textContent = "Putting your briefing together…";
    try { box.textContent = (await api("/briefing")).briefing; } catch (err) { box.textContent = err.message; }
  });
  $("remForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    $("remError").textContent = "";
    try {
      await post("/reminders", { text: $("remText").value, when: $("remWhen").value });
      $("remText").value = "";
      $("remWhen").value = "";
    } catch (err) { $("remError").textContent = err.message; }
    loadToday().catch(() => {});
  });
  $("todoForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await post("/todos", { text: $("todoText").value }).catch(() => {});
    $("todoText").value = "";
    loadToday().catch(() => {});
  });
  $("noteForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await post("/notes", { text: $("noteText").value }).catch(() => {});
    $("noteText").value = "";
    loadNotes().catch(() => {});
  });
  let searchTimer = 0;
  $("noteSearch").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => loadNotes().catch(() => {}), 250); });

  for (const b of $("riskFilter").querySelectorAll("button")) {
    b.addEventListener("click", () => {
      state.riskFilter = b.dataset.risk;
      for (const o of $("riskFilter").querySelectorAll("button")) o.setAttribute("aria-pressed", String(o === b));
      renderLog();
    });
  }
  $("undoBtn").addEventListener("click", async () => {
    try {
      const res = await post(`/automation/undo?session_id=${encodeURIComponent(state.sessionId)}`);
      $("logNote").textContent = res.message || (res.success ? "Undone." : "There was nothing to undo.");
    } catch (err) { $("logNote").textContent = err.message; }
    loadLog().catch(() => {});
  });

  $("sysRefresh").addEventListener("click", () => loadSystem().catch(() => {}));
  $("pcListenBtn").addEventListener("click", togglePcListening);
  $("deviceSelect").addEventListener("change", async () => {
    const v = $("deviceSelect").value;
    if (v === "") return;
    try { await post("/voice/device/select", { device_index: Number(v) }); $("voiceError").textContent = ""; } catch (err) { $("voiceError").textContent = err.message; }
  });

  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
}

wire();
if (state.token) start(); else showGate("");
