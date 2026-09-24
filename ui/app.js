"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = (x) => (x == null ? "–" : "$" + Number(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
const pct = (x) => (x == null ? "–" : Number(x).toFixed(2));
const human = (s) => String(s ?? "").replaceAll("_", " ");

const state = { cases: [], current: null, filter: "all", query: "", trace: null, network: null };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || r.statusText);
  }
  return r.json();
}

/* ---------------------------------------------------------------- shell */
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("fg-theme"); } catch { /* private mode */ }
  const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  document.documentElement.dataset.theme = saved || (prefersLight ? "light" : "dark");
  $("#themeBtn").onclick = () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("fg-theme", next); } catch { /* ignore */ }
    if (state.network) drawGraph();
  };
}

function showView(v) {
  $$(".nav button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("hidden", s.dataset.view !== v));
  if (v === "monitor") loadMonitor();
  if (v === "overview") loadOverview();
  window.scrollTo({ top: 0 });
}

function route() {
  const h = decodeURIComponent(location.hash.slice(1));
  if (h.startsWith("case/")) { showView("cases"); openCase(h.slice(5)); }
  else if (h === "monitor" || h === "cases") showView(h);
  else showView("overview");
}

async function boot() {
  initTheme();
  $$(".nav button").forEach((b) => (b.onclick = () => { location.hash = b.dataset.view; }));
  window.addEventListener("hashchange", route);
  $$("#filters button").forEach((b) => (b.onclick = () => {
    state.filter = b.dataset.f;
    $$("#filters button").forEach((x) => x.classList.toggle("on", x === b));
    renderQueue();
  }));
  $("#search").addEventListener("input", (e) => { state.query = e.target.value.toLowerCase(); renderQueue(); });
  $("#adhoc").addEventListener("submit", runAdhoc);
  await loadCases();
  route();
}

function renderHealth(h) {
  const llm = h.llm.includes("templates") ? `<span class="pill off"><i></i>LLM: templates</span>`
    : `<span class="pill" title="${esc(h.llm)}"><i></i>LLM: ${esc(h.llm.split(" -> ").map((x) => x.split(":")[1] || x).join(" → "))}</span>`;
  $("#sys").innerHTML = `<span class="pill"><i></i>TigerGraph · ${esc(h.graph_backend)}</span>${llm}`;
}

/* ---------------------------------------------------------------- overview */
async function loadOverview() {
  let o;
  try { o = await api("/api/overview"); } catch (e) { $("#kpis").innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  renderHealth(o.health);
  const v = o.verdicts;
  const kpi = (label, value, sub = "") => `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  $("#kpis").innerHTML = [
    kpi("Cases investigated", `${o.investigated}/${o.cases}`, "benchmark alerts"),
    kpi("Fraud", v.fraud, "confirmed or strongly suspected"),
    kpi("Legitimate", v.legitimate, "closed after evidence"),
    kpi("Uncertain", v.uncertain, "escalated to an analyst"),
    kpi("Exposure", money(o.exposure), "across fraud episodes"),
    kpi("SARs / approvals", `${o.sars} / ${o.pending_approvals}`, "reports · L1/L2 decisions pending"),
  ].join("");

  const total = Math.max(1, v.fraud + v.legitimate + v.uncertain);
  const seg = (n, color) => (n ? `<span style="width:${(n / total) * 100}%;background:${color}" title="${n}"></span>` : "");
  $("#verdictBar").innerHTML = `<div class="stack" role="img" aria-label="${v.fraud} fraud, ${v.legitimate} legitimate, ${v.uncertain} uncertain">
      ${seg(v.fraud, "var(--fraud)")}${seg(v.uncertain, "var(--warn)")}${seg(v.legitimate, "var(--legit)")}</div>
    <div class="legend-row"><span><i style="background:var(--fraud)"></i>Fraud ${v.fraud}</span>
      <span><i style="background:var(--warn)"></i>Uncertain ${v.uncertain}</span>
      <span><i style="background:var(--legit)"></i>Legitimate ${v.legitimate}</span></div>
    <p class="small">Half the benchmark is expected to be legitimate: the agent verifies before blocking (R1) and closes on confirmation (R3).</p>`;

  const pats = Object.entries(o.patterns).filter(([k]) => k !== "none").sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...pats.map((p) => p[1]));
  $("#patternBars").innerHTML = pats.map(([k, n]) => `<div class="hbar"><span>${esc(human(k))}</span>
      <div class="track"><div class="fill" style="width:${(n / max) * 100}%"></div></div><span class="n">${n}</span></div>`).join("")
    + `<p class="small">"Undocumented" = coordinated abuse that fits none of the five known typologies (R9).</p>`;

  const pending = [];
  for (const c of state.cases.filter((x) => x.pending)) {
    try {
      const ex = await api(`/api/cases/${c.case_id}/execution`);
      ex.filter((a) => a.route !== "auto" && a.state.startsWith("awaiting")).forEach((a) => pending.push({ ...a, case_id: c.case_id }));
    } catch { /* not investigated */ }
  }
  $("#approvalQueue").innerHTML = pending.length ? pending.map((a) => `<div class="aq">
      <button class="link" data-case="${esc(a.case_id)}">${esc(a.case_id)}</button>
      <div><b class="mono">${esc(a.action)}</b><div class="small">${esc(a.reason)}</div></div>
      <span class="chip r-${a.route}">${a.route}</span></div>`).join("") : `<p class="small">Nothing waiting. Auto actions have been executed by the agent.</p>`;
  $$("#approvalQueue .link").forEach((b) => (b.onclick = () => { location.hash = `case/${b.dataset.case}`; }));
}

async function runAdhoc(e) {
  e.preventDefault();
  const f = new FormData(e.target);
  const txn = String(f.get("txn") || "").trim();
  const msg = $("#adhocMsg");
  if (!/^\d{5,9}$/.test(txn)) { msg.textContent = "Enter a numeric transaction ID."; return; }
  const params = new URLSearchParams({ txn, trigger: f.get("trigger"), text: f.get("text") || "" });
  msg.textContent = "Investigating…";
  const res = await fetch(`/api/investigate?${params}`);
  if (!res.ok) { msg.textContent = (await res.json().catch(() => ({}))).detail || "Failed"; return; }
  await res.text();  // consume the stream; the trace is saved server-side
  msg.textContent = "";
  location.hash = `case/ADHOC-${txn}`;
}

/* ---------------------------------------------------------------- queue */
async function loadCases() {
  state.cases = await api("/api/cases");
  renderQueue();
}

function renderQueue() {
  const q = state.query;
  const list = state.cases.filter((c) => {
    const f = state.filter;
    if (f === "pending" ? !c.pending : f !== "all" && c.verdict !== f) return false;
    return !q || [c.case_id, c.card_id, c.pattern, c.trigger_type, c.verdict].join(" ").toLowerCase().includes(q);
  });
  $("#caseList").innerHTML = list.map((c) => `<li><button data-id="${esc(c.case_id)}" ${c.case_id === state.current ? 'aria-current="true"' : ""}>
      <div class="row1"><span class="cid">${esc(c.case_id)}</span>
        ${c.verdict ? `<span class="chip v-${c.verdict}">${c.verdict} ${pct(c.probability)}</span>` : `<span class="chip">new</span>`}</div>
      <div class="sub">${esc(human(c.trigger_type))} · ${esc(c.card_id)}${c.pattern && c.pattern !== "none" ? " · " + esc(human(c.pattern)) : ""}</div>
      <div class="sub">${c.exposure ? money(c.exposure) + " exposure" : "no exposure"}${c.sar ? " · SAR" : ""}${c.pending ? ` · ${c.pending} awaiting approval` : ""}</div>
    </button></li>`).join("") || `<li class="small" style="padding:14px">No cases match.</li>`;
  $$("#caseList button").forEach((b) => (b.onclick = () => { location.hash = `case/${b.dataset.id}`; }));
}

/* ---------------------------------------------------------------- case view */
function set(k, v, html = false) { $$(`[data-k="${k}"]`).forEach((el) => (html ? (el.innerHTML = v) : (el.textContent = v))); }

async function openCase(id) {
  state.current = id;
  state.trace = null;
  renderQueue();
  const meta = state.cases.find((c) => c.case_id === id) || { case_id: id, trigger_type: "analyst_request", trigger_text: "" };
  const root = $("#detail");
  root.innerHTML = "";
  root.appendChild($("#caseTpl").content.cloneNode(true));
  set("case_id", id);
  set("trigger", human(meta.trigger_type));
  set("trigger_text", meta.trigger_text || "");
  $$("#tabs button").forEach((b) => (b.onclick = () => tab(b.dataset.t)));
  $("#tabs").addEventListener("keydown", tabKeys);
  $("#runBtn").onclick = () => run(id);
  $("#runBtn").disabled = !/^HHG-/.test(id);
  $("#runBtn").title = /^HHG-/.test(id) ? "" : "Re-run from the Overview (ad-hoc) or scripts/monitor.py";
  renderStepper([]);
  try { render(await api(`/api/cases/${id}`)); }
  catch {
    $(".tab[data-t=summary]").innerHTML = `<div class="card">Not investigated yet. Press <b>Run investigation</b> to start the agent and watch each step.</div>`;
  }
}

function tabKeys(e) {
  const tabs = $$("#tabs button");
  const i = tabs.findIndex((t) => t.getAttribute("aria-selected") === "true");
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    tab(n.dataset.t); n.focus();
  }
}

function tab(t) {
  $$("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.t === t)));
  $$(".tab").forEach((d) => d.classList.toggle("hidden", d.dataset.t !== t));
  if (t === "graph") drawGraph();
}

const STEPS = [
  ["trigger", "Trigger", ["trigger"]], ["investigate", "Investigate", ["open_case", "tool"]],
  ["evidence", "Evidence", ["evidence"]], ["assess", "Assess", ["assess"]],
  ["more", "More evidence", ["request_evidence", "evidence_received"]], ["decide", "Decide", ["recommend_initial", "recommend_final"]],
  ["explain", "Explain", ["recommend_final"]], ["memory", "Memory", ["update_memory"]],
];

function renderStepper(timeline, live = false) {
  const seen = new Set(timeline.map((e) => e.step));
  const finished = seen.has("update_memory") || seen.has("done");
  let lastDone = -1;
  STEPS.forEach((s, i) => { if (s[2].some((x) => seen.has(x))) lastDone = i; });
  $("#stepper").innerHTML = STEPS.map(([key, label, steps], i) => {
    const hit = steps.some((x) => seen.has(x));
    let cls = hit ? "done" : "";
    let note = hit ? "" : "pending";
    if (key === "more" && finished && !hit) { cls = "skipped"; note = "not needed"; }
    if (live && !finished && i === lastDone + 1) cls = "live";
    if (hit) note = stepNote(key, timeline);
    return `<li class="${cls}"><b>${label}</b>${esc(note)}</li>`;
  }).join("");
}

function stepNote(key, tl) {
  const n = (s) => tl.filter((e) => e.step === s).length;
  const assess = tl.filter((e) => e.step === "assess").pop();
  if (key === "investigate") return `${n("tool")} graph calls`;
  if (key === "evidence") return `${n("evidence")} signals`;
  if (key === "assess") return assess ? `p = ${pct(assess.probability)}` : "";
  if (key === "more") return tl.some((e) => e.step === "evidence_received") ? "reply received" : "requested";
  if (key === "decide") return "policy engine";
  if (key === "explain") return "summary · SAR";
  if (key === "memory") return "written to graph";
  return "";
}

function render(d) {
  state.trace = d;
  const c = d.case;
  const sc = d.scoring || {};
  const p0 = sc.initial_probability ?? c.fraud_probability;
  const p1 = sc.final_probability ?? c.fraud_probability;
  set("status", human(c.status));
  set("trigger", human(d.trigger.type));
  set("trigger_text", d.trigger.text);
  set("meta", `card ${d.card_id} · flagged ${d.trigger.flagged_txn_id}${d.flag ? ` · ${money(d.flag.amt)} ${human(d.flag.channel)} · ${d.flag.ts}` : ""}`);
  set("verdict", `<span class="chip v-${c.verdict}">${c.verdict}</span>`, true);
  set("pattern", c.pattern === "none" ? "No fraud pattern" : human(c.pattern));
  set("exposure", money(c.exposure_usd));
  set("episode", c.affected_txn_ids.length ? `${c.affected_txn_ids.length} transaction(s) in the episode` : "nothing at risk");
  set("sar", d.sar.file ? "File SAR" : "No SAR");
  set("sarwhy", d.sar.file ? "Route L2 (fraud manager)" : "Case record only");
  renderStepper(d.timeline || []);
  renderProb(p0, p1, sc, d.evidence_requests.length > 0);

  const ev = d.evidence_requests;
  $(".tab[data-t=summary]").innerHTML = `<div class="stack-v">
    <div class="card"><h3>Case summary</h3><p class="summary-text">${esc(c.summary)}</p>
      <p class="ref">graph case ${esc(c.graph_case_id)} · written to TigerGraph: ${c.written_to_graph ? "yes" : "no"} ·
        ${d.tool_calls} tool calls · ${d.tokens} LLM tokens · ${d.latency_s}s · ${esc(d.llm || "")}</p></div>
    ${c.pattern_description ? `<div class="card"><h3>Undocumented pattern</h3><p>${esc(c.pattern_description)}</p></div>` : ""}
    <div class="two">
      <div class="card"><h3>Why more evidence${ev.length ? " was" : " was not"} requested</h3>
        ${ev.length ? ev.map((r) => `<p><span class="chip">${esc(human(r.type))}</span> after step ${r.asked_after_step}</p><p>${esc(r.assumed_response)}</p>`).join("")
          : `<p>${esc(d.stop_reason)}</p>`}</div>
      <div class="card"><h3>Why the investigation stopped</h3><p>${esc(d.stop_reason)}</p></div>
    </div>
    <div class="card"><h3>Episode</h3><p class="small">First suspicious: <span class="mono">${esc(c.first_suspicious_txn_id || "–")}</span></p>
      <div class="ids">${c.affected_txn_ids.map((x) => `<span>${esc(x)}</span>`).join("") || "<span>none</span>"}</div>
      ${c.connected_card_ids.length ? `<p class="small" style="margin-top:10px">Connected cards (${c.connected_card_ids.length})</p>
        <div class="ids">${c.connected_card_ids.map((x) => `<span>${esc(x)}</span>`).join("")}</div>` : ""}
      ${c.connected_device_profiles.length ? `<p class="small" style="margin-top:10px">Device profiles</p>
        <div class="ids">${c.connected_device_profiles.map((x) => `<span>${esc(x)}</span>`).join("")}</div>` : ""}</div>
  </div>`;

  renderEvidence(d);
  renderActions(d);
  renderSar(d);
  renderMemory(d);
  $("#timeline").innerHTML = "";
  (d.timeline || []).forEach(addEvent);
  if (!$(".tab[data-t=graph]").classList.contains("hidden")) drawGraph();
}

function renderProb(p0, p1, sc, asked) {
  const at = (p) => `${Math.max(0, Math.min(1, p)) * 100}%`;
  const left = Math.min(p0, p1), right = Math.max(p0, p1);
  $("#probTrack").innerHTML = `<div class="rail"></div>
    ${p0 !== p1 ? `<div class="arrow" style="left:${at(left)};width:${(right - left) * 100}%"></div>` : ""}
    <div class="mark init" style="left:${at(p0)}"></div>
    <div class="mark final" style="left:${at(p1)}"></div>
    <span class="tick" style="left:0%">0</span><span class="tick" style="left:15%">0.15</span>
    <span class="tick" style="left:85%">0.85</span><span class="tick" style="left:100%">1</span>`;
  $("#probReadout")?.remove();
  $("#probTrack").insertAdjacentHTML("afterend", `<div class="prob-readout" id="probReadout">
    <span><i style="background:var(--text-2)"></i>before evidence <b>${pct(p0)}</b></span>
    <span><i style="background:var(--text)"></i>after evidence <b>${pct(p1)}</b></span>
    <span>band 0.15–0.85 = gather more evidence</span></div>`);
  $("#probTrack").setAttribute("role", "img");
  $("#probTrack").setAttribute("aria-label", `Fraud probability ${pct(p0)} before evidence, ${pct(p1)} after`);
  const inBand = p0 > 0.15 && p0 < 0.85;
  const fams = sc.independent_families ?? "–";
  let why;
  if (!asked) why = "Enough independent evidence to act without asking anyone (policy §6).";
  else if (inBand) why = "Started inside the 0.15–0.85 band, so the agent gathered more evidence before acting (policy §6, R1).";
  else why = "Outside the band, but resting on too few independent evidence families, so the agent verified before acting (R1).";
  set("uncertainty", `${fams} independent evidence ${fams === 1 ? "family" : "families"}. ${why}`);
}

function renderEvidence(d) {
  const sc = d.scoring || { families: {}, signals: [] };
  const fams = Object.entries(sc.families || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const max = Math.max(1, ...fams.map((f) => Math.abs(f[1])));
  const chart = fams.length ? `<div class="card"><h3>What moved the probability (log-odds by evidence family)</h3>
      <div class="contrib" role="img" aria-label="Evidence contributions by family">
      ${fams.map(([k, v]) => `<span class="fam">${esc(k)}</span><div class="lane" data-tip="${esc(k)}: ${v > 0 ? "+" : ""}${v.toFixed(2)} log-odds ${v > 0 ? "toward fraud" : "toward legitimate"}">
          <span class="axis"></span><span class="bar ${v > 0 ? "pos" : "neg"}" style="width:${(Math.abs(v) / max) * 42}%"></span>
          <span class="val" style="${v > 0 ? `left:calc(50% + ${(Math.abs(v) / max) * 42}% + 6px)` : `right:calc(50% + ${(Math.abs(v) / max) * 42}% + 6px)`}">${v > 0 ? "+" : ""}${v.toFixed(1)}</span></div>`).join("")}
      </div>
      <div class="contrib-scale"><span>← toward legitimate</span><span>toward fraud →</span></div>
      <div class="legend-row"><span><i style="background:var(--legit)"></i>Supports legitimate</span><span><i style="background:var(--fraud)"></i>Supports fraud</span>
        <span class="small">Each family is capped so correlated signals count once.</span></div></div>` : "";
  const weights = Object.fromEntries((sc.signals || []).map((s) => [s.claim, s.weight]));
  const sources = [...new Set(d.case.evidence.map((e) => e.source))];
  $(".tab[data-t=evidence]").innerHTML = `<div class="stack-v">${chart}
    <div class="card"><h3>Evidence (${d.case.evidence.length})</h3>
      <div class="chips" id="srcFilter" style="margin-bottom:6px">${["all", ...sources].map((s, i) => `<button data-s="${s}" class="${i ? "" : "on"}">${s}</button>`).join("")}</div>
      <div id="evList">${d.case.evidence.map((e) => {
        const w = weights[e.claim];
        return `<div class="ev" data-src="${esc(e.source)}"><div class="src">${esc(e.source)}</div>
          <div>${esc(e.claim)}<div class="ref">${esc(e.ref)}</div><div class="ids">${e.entity_ids.map((x) => `<span>${esc(x)}</span>`).join("")}</div></div>
          <div class="w ${w > 0 ? "pos" : w < 0 ? "neg" : ""}">${w ? (w > 0 ? "+" : "") + w.toFixed(1) : ""}</div></div>`;
      }).join("")}</div></div></div>`;
  $$("#srcFilter button").forEach((b) => (b.onclick = () => {
    $$("#srcFilter button").forEach((x) => x.classList.toggle("on", x === b));
    $$("#evList .ev").forEach((r) => r.classList.toggle("hidden", b.dataset.s !== "all" && r.dataset.src !== b.dataset.s));
  }));
  $$(".lane[data-tip]").forEach(tooltip);
}

function tooltip(el) {
  const t = $("#tooltip");
  el.addEventListener("mousemove", (e) => { t.hidden = false; t.textContent = el.dataset.tip; t.style.left = `${e.clientX + 12}px`; t.style.top = `${e.clientY + 12}px`; });
  el.addEventListener("mouseleave", () => { t.hidden = true; });
}

async function renderActions(d) {
  let exec = [];
  if (/^HHG-/.test(d.case_id)) { try { exec = await api(`/api/cases/${d.case_id}/execution`); } catch { /* ignore */ } }
  const st = Object.fromEntries(exec.map((e) => [e.action, e.state]));
  const init = new Set(d.next_best_actions.initial.map((a) => a.action));
  const fin = new Set(d.next_best_actions.final.map((a) => a.action));
  const row = (a, final) => {
    const tag = final ? (!init.has(a.action) ? '<span class="tag add">added</span>' : "") : (!fin.has(a.action) ? '<span class="tag drop">dropped</span>' : "");
    const s = st[a.action] || "";
    const approve = final && a.route !== "auto" && s.startsWith("awaiting") ? `<div class="approve" data-a="${a.action}">
        <select aria-label="Approver role"><option value="L1">as L1 team lead</option><option value="L2">as L2 fraud manager</option></select>
        <button class="ok">Approve</button><button class="no">Reject</button></div>` : "";
    return `<tr><td>${a.action}${tag}</td><td><span class="chip r-${a.route}">${a.route}</span></td>
      <td>${esc(a.reason)}${final && s ? `<div class="state ${s.startsWith("awaiting") ? "" : "ok"}">${esc(s)}</div>` : ""}${approve}</td></tr>`;
  };
  $(".tab[data-t=actions]").innerHTML = `<div class="stack-v">
    <div class="nba">
      <div class="card"><h3>1 · Before evidence (initial)</h3><table class="actions">${d.next_best_actions.initial.map((a) => row(a, false)).join("")}</table></div>
      <div class="card changed"><h3>2 · What changed</h3><p>${esc(d.next_best_actions.what_changed === "nothing"
        ? "Nothing: the evidence was already decisive (policy §6), so no extra evidence was requested and the recommendation stands."
        : d.next_best_actions.what_changed)}</p></div>
      <div class="card"><h3>3 · After evidence (final)</h3><table class="actions">${d.next_best_actions.final.map((a) => row(a, true)).join("")}</table></div>
    </div>
    <div class="card"><h3>Permissions</h3><p class="small"><span class="chip r-auto">auto</span> executed by the agent (simulated API) ·
      <span class="chip r-L1">L1</span> team lead approval · <span class="chip r-L2">L2</span> fraud manager approval.
      The API refuses an approval from the wrong role and records every decision in the case in TigerGraph.</p></div></div>`;
  $$(".approve").forEach((box) => {
    const send = async (decision) => {
      box.querySelectorAll("button").forEach((b) => (b.disabled = true));
      try {
        await api(`/api/cases/${d.case_id}/approve`, { method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ action: box.dataset.a, decision, role: $("select", box).value }) });
        await loadCases();
        renderActions(d);
      } catch (e) {
        box.querySelectorAll("button").forEach((b) => (b.disabled = false));
        box.insertAdjacentHTML("afterend", `<div class="err">${esc(e.message)}</div>`);
      }
    };
    $(".ok", box).onclick = () => send("approve");
    $(".no", box).onclick = () => send("reject");
  });
}

function renderSar(d) {
  const s = d.sar;
  $(".tab[data-t=sar]").innerHTML = s.file ? `<article class="sar-doc">
      <header><div><h2>Suspicious Activity Report</h2><p class="small">Draft for regulator filing · awaiting L2 approval</p></div>
        <span class="chip r-L2">FILE_REPORT · L2</span></header>
      <dl><dt>Case</dt><dd class="mono">${esc(d.case_id)} / ${esc(d.case.graph_case_id)}</dd>
        <dt>Total amount</dt><dd>${money(s.total_amount_usd)}</dd>
        <dt>Activity dates</dt><dd>${esc(s.activity_dates.join(" → "))}</dd>
        <dt>Pattern</dt><dd>${esc(human(d.case.pattern))}</dd>
        <dt>Subjects</dt><dd><div class="ids">${s.subjects.map((x) => `<span>${esc(x)}</span>`).join("")}</div></dd></dl>
      <h3 class="small" style="text-transform:uppercase;letter-spacing:.06em">Narrative</h3>
      <p class="sar-text">${esc(s.narrative)}</p>
      <p class="small">Why file: ${esc(s.reason)}</p></article>`
    : `<div class="card"><h3>No report required</h3><p>${esc(s.reason)}</p>
       <p class="small">Policy 3a: a SAR needs confirmed or strongly suspected fraud and one of: exposure over $1,000, a shared device /
       region / other customer's fraud, or a coordinated or undocumented pattern. Most cases are "case only".</p></div>`;
}

function renderMemory(d) {
  const mem = d.memory || d.case.similar_prior_cases.map((x) => ({ case_id: x }));
  const hist = d.card_history || [];
  $(".tab[data-t=memory]").innerHTML = `<div class="stack-v">
    <div class="card"><h3>Prior cases retrieved (TigerVector similarity + shared-device links)</h3>
      ${mem.map((m) => `<div class="mem"><span class="mono">${esc(m.case_id)}</span>
        <span>${m.outcome ? `<span class="chip ${m.outcome === "confirmed_fraud" ? "v-fraud" : "v-legitimate"}">${esc(human(m.outcome))}</span>` : ""}</span>
        <span class="notes"><b>${esc(human(m.pattern || ""))}</b>${m.relation ? ` · ${esc(m.relation)}` : ""} — ${esc(m.notes || "")}</span>
        <span class="mono small">${m.similarity != null ? `sim ${pct(m.similarity)}` : ""}</span></div>`).join("") || "<p class='small'>none</p>"}</div>
    <div class="card"><h3>History on this card and customer</h3>
      ${hist.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Case</th><th>Relation</th><th>Outcome</th><th>Pattern</th><th>Opened</th></tr></thead><tbody>
        ${hist.map((h) => `<tr><td class="mono">${esc(h.case_id)}</td><td>${esc(human(h.relation))}</td><td>${esc(human(h.outcome))}</td><td>${esc(human(h.pattern))}</td><td class="mono">${esc(h.opened_at)}</td></tr>`).join("")}
        </tbody></table></div>` : "<p class='small'>No earlier cases.</p>"}
      <p class="small">This investigation was written back as <span class="mono">${esc(d.case.graph_case_id)}</span>, linked to its card, transactions,
        devices, pattern and the prior cases above, so the next case on the same card, device or pattern retrieves it.</p></div></div>`;
}

function addEvent(e) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${esc(e.at || "")}</span><span class="s s-${esc(e.step)}">${esc(human(e.step))}</span>
    <span>${esc(e.detail)}</span><span class="ms">${e.ms != null ? `${Math.round(e.ms)} ms` : ""}</span>`;
  $("#timeline")?.appendChild(li);
}

function run(id) {
  const btn = $("#runBtn");
  btn.disabled = true; btn.textContent = "Investigating…";
  tab("timeline");
  $("#timeline").innerHTML = "";
  const live = [];
  renderStepper(live, true);
  const es = new EventSource(`/api/cases/${encodeURIComponent(id)}/investigate`);
  es.onmessage = async (m) => {
    const e = JSON.parse(m.data);
    live.push(e); addEvent(e); renderStepper(live, true);
    if (e.step === "done" || e.step === "error") {
      es.close();
      btn.disabled = false; btn.textContent = "Run again";
      if (e.step === "done") { await loadCases(); render(await api(`/api/cases/${id}`)); tab("timeline"); }
    }
  };
  es.onerror = () => { es.close(); btn.disabled = false; btn.textContent = "Run investigation"; addEvent({ step: "error", detail: "connection lost" }); };
}

/* ---------------------------------------------------------------- graph */
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

async function drawGraph() {
  const d = state.trace;
  if (!d || typeof vis === "undefined") return;
  let g;
  try { g = await api(`/api/cases/${d.case_id}/graph`); } catch { return; }
  const colors = { customer: cssVar("--muted"), card: cssVar("--accent"), card_linked: cssVar("--l2"), txn: cssVar("--l1"),
    txn_flag: cssVar("--fraud"), device: cssVar("--warn"), closed_case: cssVar("--good"), case: cssVar("--text") };
  const labels = { customer: "Customer", card: "Card", card_linked: "Linked card", txn: "Transaction", txn_flag: "Flagged txn",
    device: "Device profile", closed_case: "Prior case", case: "This case" };
  const nodes = new vis.DataSet(g.nodes.map((n) => ({ ...n, color: { background: colors[n.group], border: colors[n.group] },
    font: { color: cssVar("--text"), size: 11, face: "JetBrains Mono" }, shape: n.group === "case" ? "box" : "dot",
    size: n.group.startsWith("txn") ? 9 : 13, title: n.title })));
  const edges = new vis.DataSet(g.edges.map((e) => ({ ...e, arrows: "to", color: cssVar("--line"),
    font: { color: cssVar("--muted"), size: 9, strokeWidth: 0 } })));
  state.network?.destroy();
  state.network = new vis.Network($("#graph"), { nodes, edges }, { physics: { stabilization: { iterations: 200 }, barnesHut: { springLength: 120 } },
    interaction: { hover: true, keyboard: true } });
  state.network.once("stabilizationIterationsDone", () => state.network.fit({ animation: false }));
  $("#legend").innerHTML = Object.entries(labels).map(([k, v]) => `<span><i style="background:${colors[k]}"></i>${v}</span>`).join("");
}

/* ---------------------------------------------------------------- monitor */
async function loadMonitor() {
  let rows = [];
  try { rows = await api("/api/monitor"); } catch { /* none */ }
  const by = (k) => rows.filter((r) => r.kind === k).length;
  const kpi = (l, v, s = "") => `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`;
  $("#monKpis").innerHTML = [kpi("Alerts investigated", rows.length, "outside the case pack"),
    kpi("Structuring bursts", by("structuring"), "band_bursts (GSQL)"), kpi("Device-ring cards", by("device_ring"), "ring_components (WCC)"),
    kpi("Fraud verdicts", rows.filter((r) => r.verdict === "fraud").length), kpi("SARs", rows.filter((r) => r.sar).length),
    kpi("Exposure", money(rows.reduce((s, r) => s + (r.exposure_usd || 0), 0)))].join("");
  $("#monTable").innerHTML = rows.length ? `<thead><tr><th>Alert</th><th>Detector</th><th>Card</th><th>Verdict</th><th>Pattern</th><th class="num">Exposure</th><th>SAR</th><th>Detail</th></tr></thead>
    <tbody>${rows.map((r) => `<tr class="click" tabindex="0" data-case="${esc(r.case_id)}"><td class="mono">${esc(r.case_id)}</td><td>${esc(human(r.kind))}</td><td class="mono">${esc(r.card_id)}</td>
      <td><span class="chip v-${r.verdict}">${r.verdict} ${pct(r.probability)}</span></td><td>${esc(human(r.pattern))}</td>
      <td class="num">${money(r.exposure_usd)}</td><td>${r.sar ? "yes" : "no"}</td><td class="small wrap">${esc(r.detail)}</td></tr>`).join("")}</tbody>`
    : "<tbody><tr><td>No monitoring run yet.</td></tr></tbody>";
  $$("#monTable tr.click").forEach((tr) => {
    const go = () => { location.hash = `case/${tr.dataset.case}`; };
    tr.onclick = go;
    tr.onkeydown = (e) => { if (e.key === "Enter") go(); };
  });
}

boot().catch((e) => { document.body.insertAdjacentHTML("beforeend", `<p class="err" style="padding:20px">${esc(e.message)}</p>`); });
