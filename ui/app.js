const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const money = (x) => (x == null ? "–" : "$" + Number(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));

let CASES = [];
let current = null;
let filter = "all";

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

async function boot() {
  try {
    const h = await api("/api/health");
    $("#status").textContent = `graph: ${h.graph_backend} · llm: ${h.llm}`;
  } catch { $("#status").textContent = "API offline"; }
  $$("#filters button").forEach((b) => (b.onclick = () => {
    filter = b.dataset.f;
    $$("#filters button").forEach((x) => x.classList.toggle("on", x === b));
    renderQueue();
  }));
  await loadQueue();
}

async function loadQueue() {
  CASES = await api("/api/cases");
  renderQueue();
}

function renderQueue() {
  const ul = $("#caseList");
  ul.innerHTML = "";
  CASES.filter((c) => filter === "all" || c.verdict === filter || (filter === "pending" && c.pending > 0))
    .forEach((c) => {
      const li = document.createElement("li");
      li.className = c.case_id === current ? "sel" : "";
      li.innerHTML = `
        <div class="row1"><span class="cid">${c.case_id}</span>
          ${c.verdict ? `<span class="chip v-${c.verdict}">${c.verdict} ${c.probability?.toFixed(2)}</span>` : `<span class="chip">new</span>`}</div>
        <div class="sub">${esc(c.trigger_type.replace("_", " "))} · ${c.card_id} · ${c.pattern && c.pattern !== "none" ? c.pattern.replaceAll("_", " ") : "–"}</div>
        <div class="sub">${c.exposure ? money(c.exposure) + " exposure" : ""}${c.sar ? " · SAR" : ""}${c.pending ? ` · ${c.pending} awaiting approval` : ""}</div>`;
      li.onclick = () => open(c.case_id);
      ul.appendChild(li);
    });
}

async function open(id) {
  current = id;
  renderQueue();
  const meta = CASES.find((c) => c.case_id === id);
  const root = $("#detail");
  root.innerHTML = "";
  root.appendChild($("#detailTpl").content.cloneNode(true));
  set("case_id", id);
  set("trigger_type", meta.trigger_type.replace("_", " "));
  set("trigger_text", meta.trigger_text);
  $$("#tabs button").forEach((b) => (b.onclick = () => tab(b.dataset.t)));
  $("#runBtn").onclick = () => run(id);
  try { render(await api(`/api/cases/${id}`)); }
  catch { $(".tab[data-t=summary]").innerHTML = `<div class="card">Not investigated yet. Press <b>Run investigation</b> to start the agent.</div>`; }
}

function set(k, v, html = false) {
  $$(`[data-k="${k}"]`).forEach((el) => (html ? (el.innerHTML = v) : (el.textContent = v)));
}

function tab(t) {
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  $$(".tab").forEach((d) => d.classList.toggle("hidden", d.dataset.t !== t));
  if (t === "graph") drawGraph();
}

function probs(d) {
  const a = d.timeline?.filter((e) => e.step === "assess").map((e) => e.probability) || [];
  return [a.length ? a[a.length - 1] : d.case.fraud_probability, d.case.fraud_probability];
}

function render(d) {
  const c = d.case;
  const [p0, p1] = probs(d);
  set("verdict", `<span class="chip v-${c.verdict}">${c.verdict}</span>`, true);
  set("p0", p0.toFixed(2)); set("p1", p1.toFixed(2));
  $("[data-k=bar0]").style.left = `${p0 * 100}%`;
  $("[data-k=bar1]").style.left = `${p1 * 100}%`;
  set("pattern", c.pattern.replaceAll("_", " "));
  set("exposure", money(c.exposure_usd));
  set("status", c.status.replace("_", " "));
  set("sar", d.sar.file ? "File (L2)" : "Not required");

  $(".tab[data-t=summary]").innerHTML = `
    <div class="card"><h3>Case summary</h3><p>${esc(c.summary)}</p>
      <p class="ref">graph case ${esc(c.graph_case_id)} · written to TigerGraph: ${c.written_to_graph ? "yes" : "no"} ·
      ${d.tool_calls} tool calls · ${d.tokens} LLM tokens · ${d.latency_s}s</p></div>
    <div class="grid2">
      <div class="card"><h3>Episode</h3>
        <p>First suspicious: <span class="ref">${esc(c.first_suspicious_txn_id || "–")}</span></p>
        <div class="ids">${c.affected_txn_ids.map((x) => `<span>${esc(x)}</span>`).join("") || "<span>none</span>"}</div></div>
      <div class="card"><h3>Stop reason</h3><p>${esc(d.stop_reason)}</p></div>
    </div>
    ${c.pattern_description ? `<div class="card"><h3>Undocumented pattern</h3><p>${esc(c.pattern_description)}</p></div>` : ""}
    ${d.evidence_requests.length ? `<div class="card"><h3>Evidence requested</h3>${d.evidence_requests.map((r) =>
      `<p><span class="chip">${r.type}</span> after step ${r.asked_after_step}: ${esc(r.assumed_response)}</p>`).join("")}</div>` : ""}`;

  $("#timeline").innerHTML = "";
  (d.timeline || []).forEach(addEvent);

  $(".tab[data-t=evidence]").innerHTML = `<div class="card">${c.evidence.map((e) => `
    <div class="ev"><div class="src">${e.source}</div><div>${esc(e.claim)}
      <div class="ref">${esc(e.ref)}</div>
      <div class="ids">${e.entity_ids.map((x) => `<span>${esc(x)}</span>`).join("")}</div></div></div>`).join("")}</div>`;

  renderActions(d);

  $(".tab[data-t=sar]").innerHTML = d.sar.file ? `
    <div class="card"><h3>Suspicious activity report · awaiting L2 approval to file</h3>
      <p class="sar-text">${esc(d.sar.narrative)}</p>
      <p class="ref">Total ${money(d.sar.total_amount_usd)} · ${d.sar.activity_dates.join(" → ")}</p>
      <div class="ids">${d.sar.subjects.map((x) => `<span>${esc(x)}</span>`).join("")}</div></div>
    <div class="card"><h3>Why file</h3><p>${esc(d.sar.reason)}</p></div>` :
    `<div class="card"><h3>No report</h3><p>${esc(d.sar.reason)}</p></div>`;

  $(".tab[data-t=memory]").innerHTML = `<div class="card"><h3>Prior cases retrieved (graph + TigerVector)</h3>
    ${c.similar_prior_cases.map((x) => `<div class="memo"><span class="cid">${esc(x)}</span></div>`).join("") || "none"}
    <p class="ref">This investigation was written back as ${esc(c.graph_case_id)} so future cases on the same card, device or pattern retrieve it.</p></div>`;
  window.__case = d;
}

async function renderActions(d) {
  let exec = [];
  try { exec = await api(`/api/cases/${d.case_id}/execution`); } catch {}
  const st = Object.fromEntries(exec.map((e) => [e.action, e.state]));
  const row = (a, final) => `<tr><td>${a.action}</td><td><span class="chip r-${a.route}">${a.route}</span></td>
    <td>${esc(a.reason)}${final ? `<div class="state">${esc(st[a.action] || "")}</div>` : ""}</td>
    <td>${final && a.route !== "auto" ? `<div class="approve" data-a="${a.action}" data-r="${a.route}">
      <button class="ok">Approve as ${a.route}</button><button class="no">Reject</button></div>` : ""}</td></tr>`;
  $(".tab[data-t=actions]").innerHTML = `
    <div class="card"><h3>Before evidence</h3><table class="actions">${d.next_best_actions.initial.map((a) => row(a, false)).join("")}</table></div>
    <div class="card changed"><h3>What changed</h3><p>${esc(d.next_best_actions.what_changed)}</p></div>
    <div class="card"><h3>After evidence (final)</h3><table class="actions">${d.next_best_actions.final.map((a) => row(a, true)).join("")}</table></div>`;
  $$(".approve").forEach((box) => {
    const send = async (decision) => {
      try {
        await api(`/api/cases/${d.case_id}/approve`, { method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ action: box.dataset.a, decision, role: box.dataset.r }) });
        renderActions(d);
      } catch (e) { alertInline(box, e.message); }
    };
    $(".ok", box).onclick = () => send("approve");
    $(".no", box).onclick = () => send("reject");
  });
}

function alertInline(el, msg) { el.insertAdjacentHTML("afterend", `<div class="state" style="color:var(--fraud)">${esc(msg)}</div>`); }

function addEvent(e) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${esc(e.at || "")}</span><span class="s s-${e.step}">${esc(e.step)}</span><span>${esc(e.detail)}</span>`;
  $("#timeline")?.appendChild(li);
}

function run(id) {
  const btn = $("#runBtn");
  btn.disabled = true; btn.textContent = "Investigating…";
  tab("timeline");
  $("#timeline").innerHTML = "";
  const es = new EventSource(`/api/cases/${id}/investigate`);
  es.onmessage = async (m) => {
    const e = JSON.parse(m.data);
    addEvent(e);
    if (e.step === "done" || e.step === "error") {
      es.close();
      btn.disabled = false; btn.textContent = "▶ Run again";
      if (e.step === "done") { await loadQueue(); render(await api(`/api/cases/${id}`)); tab("timeline"); }
    }
  };
  es.onerror = () => { es.close(); btn.disabled = false; btn.textContent = "▶ Run investigation"; };
}

const COLORS = { customer: "#8b98a8", card: "#f28c28", card_linked: "#c58af9", txn: "#6aa9ff", txn_flag: "#f25f5c",
  device: "#f2c14e", closed_case: "#3ecf8e", case: "#e6edf3" };

async function drawGraph() {
  const d = window.__case;
  if (!d) return;
  const g = await api(`/api/cases/${d.case_id}/graph`);
  const nodes = new vis.DataSet(g.nodes.map((n) => ({ ...n, color: { background: COLORS[n.group], border: COLORS[n.group] },
    font: { color: "#e6edf3", size: 11, face: "JetBrains Mono" }, shape: n.group === "case" ? "box" : "dot", size: n.group.startsWith("txn") ? 9 : 13 })));
  const edges = new vis.DataSet(g.edges.map((e) => ({ ...e, arrows: "to", color: "#3a4655", font: { color: "#8b98a8", size: 9, strokeWidth: 0 } })));
  new vis.Network($("#graph"), { nodes, edges }, { physics: { stabilization: true, barnesHut: { springLength: 120 } }, interaction: { hover: true } });
  $("#legend").innerHTML = Object.entries(COLORS).map(([k, v]) => `<span><i style="background:${v}"></i>${k.replace("_", " ")}</span>`).join("");
}

boot();
