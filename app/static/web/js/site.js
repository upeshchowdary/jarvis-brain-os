// Project site: presence check, the sentence trace, copy buttons.
// Example traces describe how JARVIS handles each sentence; your own sentence runs for real
// as a dry run (nothing changes) once this browser is connected to the console.
"use strict";

const TOKEN_KEY = "jarvis.token";
const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function readToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}

// ── Presence: is JARVIS running behind this page? (/health is the one public route) ──
async function checkPresence() {
  try {
    const res = await fetch("/health", { cache: "no-store" });
    if (!res.ok) return;
    const h = await res.json();
    $("presenceText").textContent = "Running on this PC";
    $("presence").title = `Version ${h.version}, model ${h.active_provider}/${h.active_model}`;
    $("presence").hidden = false;
  } catch { /* opened without the server: leave it hidden */ }
}

// ── The trace ──────────────────────────────────────────────────────────
const HEARD = { value: "Typed", sub: "or spoken after “Hey JARVIS”" };
const EXAMPLES = {
  volume: {
    q: "set volume to 30",
    understood: { value: "Fast path", sub: "a known command, no model needed" },
    checked: { value: "Low risk", sub: "runs straight away", state: "on" },
    done: { value: "Done", sub: "in a few milliseconds" },
    reply: "Volume set to 30%.",
  },
  restart: {
    q: "restart my pc",
    understood: { value: "Fast path", sub: "a known command, no model needed" },
    checked: { value: "High risk", sub: "waits for your yes", state: "stop" },
    done: { value: "Waiting", sub: "nothing happens until you answer", state: "wait" },
    reply: "This needs your confirmation (high risk): restart this PC in 30 seconds (you can say “cancel shutdown”). Say “yes” to confirm, or anything else to cancel.",
  },
  router: {
    q: "how do I reboot my router?",
    understood: { value: "Language model", sub: "a question, so read-only tools only" },
    checked: { value: "No action", sub: "questions can't act" },
    done: { value: "Answered", sub: "streamed as it's written" },
    reply: "It explains how: unplug the router, wait 30 seconds, plug it back in and give it two minutes. Nothing on your PC is touched.",
  },
  remind: {
    q: "remind me at 6 pm to submit the report",
    understood: { value: "Fast path", sub: "time understood without a model" },
    checked: { value: "Low risk", sub: "runs straight away", state: "on" },
    done: { value: "Done", sub: "a notification at 6:00 PM" },
    reply: "Okay, I'll remind you to submit the report today at 6:00 PM.",
  },
  whatsapp: {
    q: "tell Rahul on WhatsApp I'll be late",
    understood: { value: "Language model", sub: "picks the send-message tool" },
    checked: { value: "High risk", sub: "a preview, then your yes", state: "stop" },
    done: { value: "Waiting", sub: "checks the chat header before sending", state: "wait" },
    reply: "This needs your confirmation (high risk): send “I'll be late” to Rahul on WhatsApp. Say “yes” to confirm, or anything else to cancel.",
  },
};

let playId = 0;
const stages = () => Array.from(document.querySelectorAll("#lane .stage"));

function resetLane() {
  for (const li of stages()) {
    li.removeAttribute("data-state");
    li.querySelector("[data-v]").textContent = "—";
    li.querySelector("[data-s]").textContent = "";
  }
  $("replyText").textContent = "Working…";
  $("reply").classList.add("is-empty");
}

function fill(li, step) {
  li.querySelector("[data-v]").textContent = step.value;
  li.querySelector("[data-s]").textContent = step.sub || "";
  li.setAttribute("data-state", step.state || "on");
}

async function play(trace) {
  const id = ++playId;
  resetLane();
  const order = [HEARD, trace.understood, trace.checked, trace.done];
  const lis = stages();
  for (let i = 0; i < order.length; i++) {
    if (!reduceMotion) await new Promise((r) => setTimeout(r, i === 0 ? 120 : 380));
    if (id !== playId) return;
    fill(lis[i], order[i]);
  }
  $("replyText").textContent = trace.reply;
  $("reply").classList.remove("is-empty");
}

function pressChip(key) {
  for (const c of document.querySelectorAll(".chip")) c.setAttribute("aria-pressed", String(c.dataset.example === key));
}

function fromResult(r) {
  const meta = r.metadata || {};
  const timing = meta.timing_breakdown_ms || {};
  const ms = (v) => (v == null ? "" : v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`);
  let understood;
  if (r.route === "fast_path") understood = { value: "Fast path", sub: `matched in ${ms(timing.router_ms) || "under a millisecond"}` };
  else if (r.route === "agent") understood = { value: "Language model", sub: meta.model ? `${meta.provider}/${meta.model}` : "no model reachable" };
  else if (r.route === "confirmation") understood = { value: "Your answer", sub: "to an action that was waiting" };
  else understood = { value: "Not understood", sub: "try rephrasing it as a command", state: "stop" };

  let checked;
  if (r.status === "needs_confirmation" && r.pending) {
    const risk = String(r.pending.risk || "high");
    checked = { value: `${risk[0].toUpperCase()}${risk.slice(1)} risk`, sub: "waits for your yes", state: risk === "high" ? "stop" : "wait" };
  } else if (r.status === "dry_run") checked = { value: "Dry run", sub: "planned, nothing was changed" };
  else if ((r.tools || []).length) checked = { value: "Allowed", sub: (r.tools || []).map((t) => t.name).join(", ") };
  else checked = { value: "No action", sub: "nothing to run" };

  const label = { ok: "Done", needs_confirmation: "Waiting", dry_run: "Rehearsed", error: "Didn't work", blocked: "Blocked" }[r.status] || r.status;
  const bad = r.status === "error" || r.status === "blocked";
  const done = { value: label, sub: `${ms(meta.total_latency_ms)} in total`, state: bad ? "stop" : r.status === "needs_confirmation" ? "wait" : "on" };
  return { understood, checked, done, reply: r.response };
}

async function runOwn(text) {
  const token = readToken();
  const note = $("traceNote");
  if (!token) {
    note.classList.remove("is-live");
    note.textContent = "Connect the console once to try your own sentence. It runs as a dry run, so nothing changes.";
    return;
  }
  const id = ++playId;
  resetLane();
  fill(stages()[0], HEARD);
  try {
    const res = await fetch("/automation/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ command: text, dry_run: true, session_id: "site-preview" }),
    });
    if (res.status === 401) throw new Error("The saved token was rejected. Connect the console again.");
    if (!res.ok) throw new Error(`JARVIS answered ${res.status}.`);
    const r = await res.json();
    if (r.status === "needs_confirmation" && r.pending) {
      // The preview is all the visitor asked for; don't leave an action waiting.
      fetch(`/automation/cancel/${encodeURIComponent(r.pending.pending_id)}`, { method: "POST", headers: { Authorization: `Bearer ${token}` } }).catch(() => {});
    }
    if (id !== playId) return;
    note.classList.add("is-live");
    note.textContent = "Ran on your PC as a dry run. Nothing was changed.";
    await play(fromResult(r));
  } catch (err) {
    if (id !== playId) return;
    note.classList.remove("is-live");
    note.textContent = err.message || String(err);
    $("replyText").textContent = "Nothing ran.";
  }
}

function initTrace() {
  for (const chip of document.querySelectorAll(".chip")) {
    chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => {
      const ex = EXAMPLES[chip.dataset.example];
      $("traceInput").value = ex.q;
      pressChip(chip.dataset.example);
      $("traceNote").classList.remove("is-live");
      $("traceNote").textContent = "How JARVIS handles this sentence.";
      play(ex);
    });
  }
  $("traceForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = $("traceInput").value.trim();
    if (!text) return;
    const key = Object.keys(EXAMPLES).find((k) => EXAMPLES[k].q.toLowerCase() === text.toLowerCase());
    pressChip(key || "");
    if (key) {
      $("traceNote").classList.remove("is-live");
      $("traceNote").textContent = "How JARVIS handles this sentence.";
      play(EXAMPLES[key]);
    } else {
      runOwn(text);
    }
  });
  if (readToken()) $("traceNote").textContent = "Pick a sentence, or type your own: it runs on your PC as a dry run.";
  // The page's one authored moment: the headline opens, then the first sentence travels.
  setTimeout(() => { pressChip("volume"); play(EXAMPLES.volume); }, reduceMotion ? 0 : 900);
}

function initCopy() {
  for (const btn of document.querySelectorAll(".copy")) {
    btn.addEventListener("click", async () => {
      const code = btn.parentElement.querySelector("code").textContent;
      try {
        await navigator.clipboard.writeText(code);
        btn.textContent = "Copied";
        btn.classList.add("is-done");
      } catch {
        btn.textContent = "Select and copy";
      }
      setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("is-done"); }, 1600);
    });
  }
}

checkPresence();
initTrace();
initCopy();
