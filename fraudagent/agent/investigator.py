"""The investigation agent.

A small explicit state machine, one node per step of the investigation flow:

    TRIGGER -> OPEN_CASE -> INVESTIGATE (planner picks graph tools until nothing useful is left)
            -> ASSESS -> [REQUEST_EVIDENCE -> REASSESS] -> DECIDE -> EXPLAIN -> UPDATE_MEMORY

Every node appends to the case timeline (who did what, why, with which evidence), which is the
decision record written to the graph and shown in the UI. The LLM is used where it helps
(choosing the next tool, synthesising the summary, writing the SAR narrative); graph analysis,
scoring and policy stay deterministic so the outcome is reproducible and auditable.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fraudagent.agent import detectors as D
from fraudagent.agent import narrative as N
from fraudagent.agent.llm import LLM
from fraudagent.agent.policy import (
    STOP_HIGH,
    STOP_LOW,
    ActionPlan,
    PolicyFacts,
    final_plan,
    initial_plan,
    sar_required,
    should_stop,
)
from fraudagent.agent.simulator import CustomerSimulator
from fraudagent.graph.tools import TOOL_DOCS, GraphTools
from fraudagent.memory.case_store import embedder

TS = "%Y-%m-%d %H:%M:%S"


@dataclass
class Trigger:
    case_id: str
    opened_at: str
    trigger_type: str
    trigger_text: str
    flagged_txn_id: str
    card_id: str
    customer_id: str
    risk_score: float | None = None


@dataclass
class State:
    trigger: Trigger
    flag: dict = field(default_factory=dict)
    window: list[dict] = field(default_factory=list)
    wide_window: list[dict] = field(default_factory=list)
    baseline: dict = field(default_factory=dict)
    neighbors: dict | None = None
    ring: dict | None = None
    device_cases: list[dict] = field(default_factory=list)
    region: list[dict] | None = None
    recurring: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    similar: list[dict] = field(default_factory=list)
    docs: list[dict] = field(default_factory=list)
    done_tools: set[str] = field(default_factory=set)
    signals: list[D.Signal] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)


class Investigator:
    def __init__(self, tools: GraphTools, llm: LLM | None = None,
                 on_event: Callable[[dict], None] | None = None) -> None:
        self.tools = tools
        self.llm = llm or LLM()
        self.on_event = on_event or (lambda e: None)
        self.sim = CustomerSimulator()

    # ------------------------------------------------------------------------------------
    def log(self, st: State, step: str, detail: str, **data) -> None:
        ev = {"step": step, "at": datetime.now().strftime("%H:%M:%S"), "detail": detail, **data}
        st.timeline.append(ev)
        self.on_event(ev)

    def tool(self, st: State, name: str, **args):
        out = self.tools.call(name, **args)
        st.done_tools.add(name)
        last = self.tools.log[-1]
        self.log(st, "tool", f"{name}({', '.join(f'{k}={v}' for k, v in last.args.items())}) -> {last.summary}",
                 tool=name, ms=round(last.ms, 1))
        return out

    # ------------------------------------------------------------------------------------
    def run(self, trig: Trigger, persist: bool = True) -> dict:
        t0 = time.perf_counter()
        self.tools.log.clear()
        tokens0 = self.llm.tokens
        # per-investigation scratch state: never let one case's findings leak into the next
        self._episode, self._pattern, self._findings = [], "none", D.Findings()
        st = State(trig)
        self.log(st, "trigger", f"{trig.trigger_type}: {trig.trigger_text}")

        # OPEN_CASE / first look
        st.flag = self.tool(st, "txn_detail", txn_id=trig.flagged_txn_id)
        if not st.flag:
            raise ValueError(f"flagged transaction {trig.flagged_txn_id} not found")
        st.flag.setdefault("card", trig.card_id)
        self.log(st, "open_case", f"Case opened for {trig.card_id}; flagged ${st.flag['amt']:.2f} "
                 f"{st.flag['channel']} txn {st.flag['id']} at {st.flag['ts']}")

        # INVESTIGATE: planner loop
        self.investigate(st)

        # ASSESS
        facts, findings, prob, indep = self.assess(st, "initial")
        prob0 = prob
        decided = should_stop(prob, indep) and not (trig.trigger_type == "customer_report" and prob < STOP_HIGH)
        evidence_type = None if decided else self.choose_evidence(st, facts)
        if decided and trig.trigger_type == "customer_report":
            facts.customer_response = "denied"  # the report is the cardholder's denial (R2)
        plan0 = initial_plan(facts, evidence_type)
        self.log(st, "recommend_initial", f"Initial next best action: {', '.join(i.action for i in plan0.items)}",
                 actions=plan0.sorted())

        evidence_requests: list[dict] = []
        response = ""
        if evidence_type:
            step_no = len([e for e in st.timeline if e["step"] == "tool"])
            reason = (f"probability {prob:.2f} sits between {STOP_LOW} and {STOP_HIGH}" if STOP_LOW < prob < STOP_HIGH
                      else f"only {indep} independent signal(s)")
            self.log(st, "request_evidence", f"Requested {evidence_type} after step {step_no}: {reason} (policy 6, R1)")
            response, assumed, prob = self.sim.respond(evidence_type, trig.trigger_type, prob, facts, st.flag)
            evidence_requests.append({"type": evidence_type, "asked_after_step": step_no, "assumed_response": assumed})
            st.signals.append(D.Signal("evidence_response", assumed, 0.0, "customer", source="customer",
                                       ref="evidence_request:1", entity_ids=[st.flag["id"]]))
            self.log(st, "evidence_received", f"Simulated response ({response}): {assumed}. Probability "
                     f"{prob0:.2f} -> {prob:.2f}")

        # DECIDE (a customer report is itself a denial of the transaction: R2)
        if not response and trig.trigger_type == "customer_report":
            response = "denied"
        facts = self.facts(st, findings, prob, indep, response)
        plan1 = final_plan(facts) if evidence_type else plan0
        if not evidence_type:
            plan1 = initial_plan(facts, None)
        self.log(st, "recommend_final", f"Final next best action: {', '.join(i.action for i in plan1.items)}",
                 actions=plan1.sorted())

        # EXPLAIN + package
        case = self.package(st, facts, findings, prob0, prob, indep, plan0, plan1, evidence_requests, response)
        # UPDATE_MEMORY
        case["embedding"] = embedder().encode_one(self.profile_text(st, findings, facts))
        written = self.tools.write_case(case) if persist else False
        case["case"]["written_to_graph"] = bool(written)
        self.log(st, "update_memory", ("Case written to TigerGraph as " + case["graph_case_id"]) if written
                 else "Case stored in local memory (graph write unavailable)")
        case["tool_calls"] = len(self.tools.log)
        case["tokens"] = self.llm.tokens - tokens0
        case["latency_s"] = round(time.perf_counter() - t0, 2)
        case["timeline"] = st.timeline
        _, _, fams = D.score([x for x in st.signals if x.name != "evidence_response"])
        case["scoring"] = {
            "initial_probability": prob0, "final_probability": prob, "independent_families": indep,
            "families": {k: round(v, 2) for k, v in fams.items() if v},
            "signals": [{"name": x.name, "family": x.family, "weight": x.weight, "claim": x.claim}
                        for x in st.signals if x.weight],
            "stop_band": [STOP_LOW, STOP_HIGH],
        }
        case["memory"] = [{"case_id": c["case_id"], "outcome": c.get("outcome"), "pattern": c.get("pattern"),
                           "exposure": c.get("exposure"), "similarity": round(1 - (c.get("distance") or 1), 3),
                           "notes": (c.get("notes") or "")[:280]} for c in st.similar[:8]]
        case["memory"] += [{"case_id": c["case_id"], "outcome": c.get("outcome"), "pattern": c.get("pattern"),
                            "relation": "shared device", "notes": (c.get("notes") or "")[:280]}
                           for c in st.device_cases[:4]]
        case["card_history"] = [{"case_id": c["case_id"], "relation": c["relation"], "outcome": c["outcome"],
                                 "pattern": c["pattern"], "opened_at": c["opened_at"]} for c in st.history[:12]]
        case["flag"] = st.flag
        case["llm"] = self.llm.label
        case["graph_backend"] = self.tools.backend
        return case

    # ------------------------------------------------------------------------------------
    def plan_next(self, st: State) -> str | None:
        """Heuristic planner: the next most informative tool given what is known so far."""
        f = st.flag
        order = ["card_window", "card_baseline", "card_case_history"]
        if f.get("device"):
            order += ["device_neighbors", "device_case_links"]
        if f.get("channel") == "in_person" and f.get("region"):
            order += ["region_cluster"]
        if st.trigger.trigger_type == "customer_report":
            order += ["recurring_matches"]
        order += ["similar_closed_cases", "search_docs"]
        if st.neighbors and len(st.neighbors.get("cards", [])) >= 5 and not D.is_generic_device(
                f.get("device", ""), int(st.neighbors.get("all_time_cards", 0))):
            order.insert(order.index("similar_closed_cases"), "device_ring")
        for name in order:
            if name not in st.done_tools:
                return name
        return None

    def llm_plan(self, st: State, candidate: str) -> str:
        """Let the LLM re-order the remaining tools (it may only choose among allowed ones)."""
        if not self.llm.enabled:
            return candidate
        remaining = [t for t in TOOL_DOCS if t not in st.done_tools and t != "txn_detail"]
        if st.flag.get("channel") != "in_person":
            remaining = [t for t in remaining if t != "region_cluster"]
        if not st.flag.get("device"):
            remaining = [t for t in remaining if t not in ("device_neighbors", "device_case_links", "device_ring")]
        if len(remaining) <= 1:
            return candidate
        sig = "; ".join(f"{s.name}({s.weight:+.1f})" for s in st.signals[-6:]) or "none yet"
        ans = self.llm.complete_json(
            "You are a fraud investigator choosing the next graph tool. Reply as JSON {\"tool\": name, \"why\": reason}.",
            f"Alert: {st.trigger.trigger_text}\nFlagged: {st.flag}\nSignals so far: {sig}\n"
            f"Tools available: {{{', '.join(f'{t}: {TOOL_DOCS[t]}' for t in remaining)}}}\n"
            f"Default next tool: {candidate}. Pick the most informative one.", max_tokens=120)
        if ans and ans.get("tool") in remaining:
            verb = f"chose {ans['tool']} over {candidate}" if ans["tool"] != candidate else f"confirmed {candidate}"
            self.log(st, "plan", f"LLM planner {verb}: {str(ans.get('why', ''))[:180]}")
            return ans["tool"]
        return candidate

    def investigate(self, st: State) -> None:
        budget = 12
        while budget:
            budget -= 1
            name = self.plan_next(st)
            if name is None:
                break
            if len(st.done_tools) == 4:  # one LLM planning decision per case, once the basics are known
                name = self.llm_plan(st, name)
            self.execute(st, name)
            self.update_signals(st)
            p, indep, _ = D.score(st.signals)
            if len(st.done_tools) >= 6 and should_stop(p, indep) and "similar_closed_cases" in st.done_tools:
                self.log(st, "stop_gathering", f"Probability {p:.2f} with {indep} independent signals: enough to act (policy 6)")
                break

    def execute(self, st: State, name: str) -> None:
        f, trig = st.flag, st.trigger
        if name == "card_window":
            st.window = self.tool(st, "card_window", card=trig.card_id, center_ts=f["ts"], hours_before=48, hours_after=48)
        elif name == "card_baseline":
            before = (datetime.strptime(f["ts"], TS) - timedelta(hours=48)).strftime(TS)
            st.baseline = self.tool(st, "card_baseline", card=trig.card_id, before_ts=before, lookback_days=150)
        elif name == "card_case_history":
            st.history = self.tool(st, "card_case_history", card=trig.card_id)
        elif name == "device_neighbors":
            st.neighbors = self.tool(st, "device_neighbors", device=f["device"], center_ts=f["ts"], days=30)
        elif name == "device_case_links":
            generic = D.is_generic_device(f["device"], int((st.neighbors or {}).get("all_time_cards", 0)))
            st.device_cases = [] if generic else self.tool(st, "device_case_links", device=f["device"])
            if generic:
                st.done_tools.add(name)
                self.log(st, "skip", f"device_case_links skipped: '{f['device']}' is a generic profile shared by "
                         f"{(st.neighbors or {}).get('all_time_cards', 0)} cards")
        elif name == "device_ring":
            st.ring = self.tool(st, "device_ring", seed=trig.card_id, center_ts=f["ts"], days=30, hops=2)
        elif name == "region_cluster":
            st.region = self.tool(st, "region_cluster", region=f["region"], center_ts=f["ts"], hours=72)
        elif name == "recurring_matches":
            st.recurring = self.tool(st, "recurring_matches", card=trig.card_id, before_ts=f["ts"], amt=f["amt"],
                                     product=f["product"], tolerance=0.02)
        elif name == "similar_closed_cases":
            st.similar = self.tool(st, "similar_closed_cases", qv=embedder().encode_one(self.query_text(st)), k=8)
        elif name == "search_docs":
            q = " ".join(s.claim for s in st.signals if abs(s.weight) >= 0.5)[:600] or st.trigger.trigger_text
            st.docs = self.tool(st, "search_docs", qv=embedder().encode_one(q), k=5)
        else:
            st.done_tools.add(name)

    # ------------------------------------------------------------------------------------
    def update_signals(self, st: State) -> None:
        f, trig = st.flag, st.trigger
        sig: list[D.Signal] = []
        for s in (D.trigger_signal(trig.trigger_type, f, trig.trigger_text), D.risk_score_signal(f, trig.trigger_type),
                  D.case_model_signal(f)):
            if s:
                sig.append(s)
        self._findings = D.Findings()
        if st.window:
            s, ids, cleared = D.card_testing(f, st.window)
            if s:
                sig.append(s)
                self._findings.card_testing, self._findings.large_purchase_cleared = True, cleared
                self._findings.affected_txn_ids = ids
            s, ids = D.structuring(f, st.window)
            if s:
                sig.append(s)
                self._findings.structuring = True
                self._findings.affected_txn_ids = ids
        if st.baseline:
            sig += D.amount_signals(f, st.baseline)
            sig += D.device_signals(f, st.baseline, st.neighbors)
            rs, rf = D.region_signals(f, st.baseline, st.region)
            sig += rs
            if rf.region_cluster:
                self._findings.region_cluster = True
                self._findings.connected_card_ids = rf.connected_card_ids
                self._findings.shared_element = rf.shared_element
        if st.neighbors:
            s, rf = D.device_ring(f, st.neighbors, st.ring, st.device_cases)
            if s:
                # a handset that belongs to a ring is not reassuring just because it was seen on this card before
                sig = [x for x in sig if x.name != "known_device"]
                sig.append(s)
                self._findings.device_ring = True
                self._findings.connected_card_ids = rf.connected_card_ids
                self._findings.connected_device_profiles = rf.connected_device_profiles
                self._findings.shared_element = rf.shared_element
        if st.recurring:
            s, rec = D.recurring_signal(f, st.recurring, trig.trigger_type)
            if s:
                sig.append(s)
                self._findings.recurring = rec
        if st.history or st.similar:
            sig += D.memory_signals(f, st.history, st.similar)
        if st.docs:
            top = st.docs[0]
            sig.append(D.Signal("policy_context", f"Retrieved guidance: {top['section']} ({top['source']})", 0.0,
                                "document", source="document", ref=f"doc:{top['id']}",
                                entity_ids=[d["id"] for d in st.docs[:3]]))
        new = {s.name for s in sig} - {s.name for s in st.signals}
        for s in sig:
            if s.name in new and s.weight:
                self.log(st, "evidence", s.claim, weight=s.weight, family=s.family)
        st.signals = sig

    # ------------------------------------------------------------------------------------
    def episode(self, st: State, findings: D.Findings, prob: float) -> list[dict]:
        """Transactions that belong to the same fraud episode as the flagged one."""
        f = st.flag
        by_id = {t["id"]: t for t in st.window}
        if findings.affected_txn_ids:
            return [by_id[i] for i in findings.affected_txn_ids if i in by_id] or [f]
        ep = [f]
        t0 = datetime.strptime(f["ts"], TS)
        if findings.device_ring and st.neighbors:
            own = [u for u in st.neighbors.get("uses", []) if u["card"] == st.trigger.card_id]
            ids = {f["id"]} | {u["txn"] for u in own}
            rows = {t["id"]: t for t in st.window}
            for u in own:
                rows.setdefault(u["txn"], {"id": u["txn"], "ts": u["ts"], "amt": u["amt"], "channel": "online",
                                           "product": u["product"], "device": f.get("device"),
                                           "dev_status": u.get("dev_status", ""), "region": u.get("region", "")})
            return sorted((rows[i] for i in ids if i in rows), key=lambda x: x["ts"])
        dev = f.get("device") or ""
        generic = D.is_generic_device(dev, int((st.neighbors or {}).get("all_time_cards", 0)))
        for t in st.window:
            if t["id"] == f["id"]:
                continue
            dt = abs(datetime.strptime(t["ts"], TS) - t0)
            if findings.device_ring and dev and t.get("device") == dev:
                ep.append(t)
            elif dev and not generic and t.get("device") == dev and dt <= timedelta(hours=48) and t.get("cm_score", 0) >= 0.1:
                ep.append(t)
            elif (f["channel"] == "in_person" and t["channel"] == "in_person" and t.get("region") == f.get("region")
                  and dt <= timedelta(hours=24) and (st.baseline.get("regions") or {}).get(f.get("region"), 0) == 0):
                ep.append(t)
            elif (t["channel"] == f["channel"] and dt <= timedelta(hours=6) and f["amt"] > 0
                  and abs(t["amt"] - f["amt"]) <= 0.01 * f["amt"] + 0.1):
                ep.append(t)  # near-identical repeat charge in the same burst
            elif t["channel"] == f["channel"] and dt <= timedelta(hours=24) and t.get("cm_score", 0) >= 0.5 and prob >= 0.5:
                ep.append(t)
        return sorted(ep, key=lambda x: x["ts"])[:25]

    def assess(self, st: State, label: str):
        self.update_signals(st)
        prob, indep, fams = D.score(st.signals)
        findings = self._findings
        self.log(st, "assess", f"{label} assessment: fraud probability {prob:.2f}, {indep} independent signal families "
                 f"({', '.join(f'{k} {v:+.1f}' for k, v in fams.items() if v)})", probability=prob)
        facts = self.facts(st, findings, prob, indep, "")
        return facts, findings, prob, indep

    def facts(self, st: State, findings: D.Findings, prob: float, indep: int, response: str) -> PolicyFacts:
        verdict = "fraud" if prob >= 0.7 else "legitimate" if prob <= 0.3 else "uncertain"
        ep = self.episode(st, findings, prob) if verdict != "legitimate" else []
        pattern = D.classify_pattern(st.flag, st.signals, ep) if verdict != "legitimate" else "none"
        exposure = round(sum(abs(t["amt"]) for t in ep), 2)
        # R8 "evidence conflicts": strong evidence families pulling in opposite directions
        _, _, fams = D.score([x for x in st.signals if x.name != "evidence_response"])
        conflicts = max(fams.values(), default=0) >= 0.8 and min(fams.values(), default=0) <= -0.8
        self._episode, self._pattern = ep, pattern
        return PolicyFacts(
            trigger_type=st.trigger.trigger_type, probability=prob, independent_signals=indep, verdict=verdict,
            pattern=pattern, exposure=exposure, shared_origin=findings.device_ring or findings.region_cluster,
            coordinated=findings.device_ring or findings.region_cluster or findings.structuring,
            card_testing=findings.card_testing, large_purchase_cleared=findings.large_purchase_cleared,
            recurring_dispute=findings.recurring and prob < 0.5, evidence_conflicts=conflicts,
            customer_response=response, has_connected_cards=bool(findings.connected_card_ids),
        )

    def choose_evidence(self, st: State, facts: PolicyFacts) -> str:
        f = st.flag
        if st.trigger.trigger_type == "customer_report":
            return "customer_validation"
        if f.get("channel") == "online" and (f.get("dev_status") == "New" or facts.card_testing):
            return "step_up_auth"
        return "customer_validation"

    # ------------------------------------------------------------------------------------
    def query_text(self, st: State) -> str:
        return self.profile_text(st, self._findings, None)

    def profile_text(self, st: State, findings: D.Findings, facts: PolicyFacts | None) -> str:
        from fraudagent.memory.embed import DIM  # noqa: F401
        f = st.flag
        rows = getattr(self, "_episode", None) or [f]
        tok = [f"channel_{f['channel']}", f"product_{f['product']}"]
        from fraudagent.graph.profiles import amount_bucket
        tok += sorted({amount_bucket(t["amt"]) for t in rows})
        tok.append("single_txn" if len(rows) == 1 else "few_txns" if len(rows) <= 4 else "many_txns")
        if f.get("dev_status") == "New":
            tok.append("device_new")
        if f.get("proxy_type"):
            tok.append("proxy_" + f["proxy_type"].split(":")[-1].lower())
        if findings.structuring:
            tok.append("sub_threshold_cluster")
        if findings.card_testing:
            tok.append("micro_auth_run")
        # structure only: the closed-case notes are templated, so free-text claims would match on wording
        # ("reported they did not make") rather than on what actually happened
        hints = {"new_region": "billing region the cardholder had no history in card-present",
                 "new_device": "device not previously seen on this account",
                 "sub_threshold_structuring": "just under $500 authorization threshold",
                 "shared_device_ring": "same device profile other cardholders anonymous proxy",
                 "card_testing": "very small online authorizations followed by a larger purchase"}
        tok += [hints[s.name] for s in st.signals if s.name in hints and s.weight > 0]
        if f["channel"] == "in_person":
            tok.append("card-present")
        text = " ".join(tok)
        if facts:
            text += f" pattern {facts.pattern} verdict {facts.verdict}"
        return text

    def package(self, st, facts, findings, prob0, prob, indep, plan0: ActionPlan, plan1: ActionPlan,
                evidence_requests, response) -> dict:
        return N.build_case(self.llm, st, facts, findings, getattr(self, "_episode", []), prob0, prob, indep,
                            plan0, plan1, evidence_requests, response, sar_required(facts))
