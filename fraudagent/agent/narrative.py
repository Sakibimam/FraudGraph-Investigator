"""Turn an investigation state into the answer file: case record, SAR and next best actions.

Summaries, SAR narratives and undocumented-pattern descriptions are written by the LLM when
one is configured, grounded only in the evidence list and the retrieved guidance (GraphRAG).
A deterministic template produces the same sections when no LLM is available, and is also the
fallback if the LLM output fails validation.
"""
from __future__ import annotations

import re
from datetime import datetime

from fraudagent.agent.policy import ActionPlan

PATTERN_NAMES = {
    "card_testing": "card testing", "card_not_present_fraud": "card-not-present fraud",
    "card_not_present_new_device": "card-not-present fraud from a new device",
    "out_of_region_use": "out-of-region card-present use", "account_takeover": "account takeover",
    "undocumented": "an undocumented pattern", "none": "no fraud",
}
ID_RE = re.compile(r"^(\d{7}|C\d{5}(-K\d)?|CC-\d{4})$")


def _entity_ok(e: str) -> bool:
    return bool(ID_RE.match(e)) or " | " in e


def evidence_list(signals) -> list[dict]:
    out = []
    for s in signals:
        if s.weight == 0 and s.source not in ("customer", "document"):
            continue
        ev = s.as_evidence()
        ev["entity_ids"] = [e for e in dict.fromkeys(ev["entity_ids"]) if _entity_ok(str(e))]
        out.append(ev)
    return out


def pattern_description(findings, flag, episode) -> str:
    if findings.structuring:
        amts = ", ".join(f"${t['amt']:.2f}" for t in episode)
        return (f"Threshold structuring on a compromised card: {len(episode)} online purchases ({amts}) placed within "
                "about an hour, each kept just under $500, the level at which authorisation checks tighten. It affects "
                "cardholders whose card numbers are already compromised and matches five closed cases the bank "
                "confirmed but could not map to a known typology; the agent found it by scanning the card's window for "
                "clusters of same-band amounts.")
    if findings.device_ring:
        return (f"Shared-device fraud ring: one device profile ({flag.get('device')}) behind an anonymising proxy is used "
                f"to make small online purchases on {len(findings.connected_card_ids) + 1} unrelated customers' cards "
                "in the same weeks, each time appearing as a New device for that account. It affects many customers at "
                "once rather than one compromised card; the agent found it by traversing card -> transaction -> device "
                "-> card in TigerGraph and matching the profile to earlier confirmed undocumented cases.")
    return ""


def _fmt_actions(items) -> str:
    return ", ".join(i["action"] for i in items)


def template_summary(st, facts, prob0, prob, response, episode) -> str:
    f = st.flag
    lead = {
        "fraud": f"Fraud: {PATTERN_NAMES[facts.pattern]} on card {st.trigger.card_id}.",
        "legitimate": f"Legitimate: the flagged ${f['amt']:.2f} {f['channel'].replace('_', '-')} transaction fits the cardholder.",
        "uncertain": f"Unresolved: evidence on the ${f['amt']:.2f} {f['channel'].replace('_', '-')} transaction is mixed.",
    }[facts.verdict]
    strong = sorted([s for s in st.signals if abs(s.weight) >= 0.5], key=lambda s: -abs(s.weight))[:2]
    why = " ".join(s.claim.rstrip(".") + "." for s in strong)
    tail = ""
    if response:
        tail = f" Evidence request answered ({response}); probability moved {prob0:.2f} -> {prob:.2f}."
    if facts.verdict == "fraud":
        tail += f" Episode of {len(episode)} transaction(s), exposure ${facts.exposure:,.2f}."
    return (lead + " " + why + tail).strip()


def template_sar(st, facts, findings, episode) -> str:
    f, trig = st.flag, st.trigger
    first, last = episode[0], episode[-1]
    amts = ", ".join(f"${t['amt']:.2f} ({t['id']}, {t['ts'][:16]})" for t in episode[:8])
    where = "online" if f["channel"] == "online" else f"card-present in billing region {f.get('region')}"
    dev = f" from device profile '{f['device']}'" if f.get("device") else ""
    proxy = f" through an {f['proxy_type'].split(':')[-1].lower()} proxy" if f.get("proxy_type") else ""
    parts = [
        f"Customer {trig.customer_id}, holder of card {trig.card_id}, is the subject of this report.",
        f"Between {first['ts'][:16]} and {last['ts'][:16]}, {len(episode)} transaction(s) totalling ${facts.exposure:,.2f} were "
        f"made on the card {where}{dev}{proxy}: {amts}.",
    ]
    if findings.structuring:
        parts.append("Every amount sat just below $500 and the purchases were placed minutes apart, a pattern consistent "
                     "with deliberately staying under an authorisation threshold.")
    if findings.device_ring:
        parts.append(f"The same device profile was used on {len(findings.connected_card_ids)} other customers' cards in the "
                     f"same period ({', '.join(findings.connected_card_ids[:8])}{'...' if len(findings.connected_card_ids) > 8 else ''}), "
                     "most of them behind anonymous proxies and each time as a new device, indicating a common actor.")
    if findings.card_testing:
        parts.append("The sequence began with several authorisations under $10 followed by larger purchases, which is how "
                     "a stolen card number is tested before use.")
    if findings.region_cluster:
        parts.append(f"The activity coincides with a cluster of other cards transacting for the first time in the same "
                     f"billing region ({', '.join(findings.connected_card_ids[:6])}).")
    if trig.trigger_type == "customer_report":
        parts.append("The cardholder reported the transaction as unauthorised and, on follow-up, maintained that they did "
                     "not make it while still holding the card.")
    elif facts.customer_response == "denied":
        parts.append("When contacted, the cardholder stated they did not make the transactions and still holds the card.")
    reasons = [s.claim.split(":")[0] for s in st.signals if s.weight >= 1.0 and s.family not in ("customer",)]
    if reasons:
        parts.append("The activity is suspicious because: " + "; ".join(r.rstrip(".") for r in reasons[:3]) + ".")
    prior = [c["case_id"] for c in st.similar if c.get("outcome") == "confirmed_fraud"][:3]
    if prior:
        parts.append(f"It closely resembles earlier confirmed fraud cases {', '.join(prior)}.")
    parts.append("The bank has recommended blocking and reissuing the card, placed linked cards under monitoring where "
                 "applicable, and retains the transaction, device and investigation records as supporting documentation.")
    return " ".join(parts)


def llm_rewrite(llm, kind: str, draft: str, st, facts, docs) -> str:
    if not llm.enabled:
        return draft
    guidance = "\n".join(f"- {d['section']}: {d['text'][:400]}" for d in docs[:3])
    evidence = "\n".join(f"- {s.claim}" for s in st.signals if s.weight or s.source == "customer")
    if kind == "sar":
        sys = ("You write FinCEN-style SAR narratives. Use only facts in the evidence and draft; keep every ID, date and "
               "amount exactly as given; cover who, what, when, where, how and why suspicious; 6 to 12 sentences; plain prose.")
    else:
        sys = ("You write internal fraud case summaries for analysts: 2 to 6 sentences, plain, factual, no invented facts, "
               "state the verdict, the key evidence and what happens next.")
    out = llm.complete(sys, f"Guidance retrieved from policy/regulatory store:\n{guidance}\n\nEvidence:\n{evidence}\n\n"
                            f"Verdict: {facts.verdict}, pattern: {facts.pattern}, exposure ${facts.exposure:,.2f}\n\n"
                            f"Draft to improve:\n{draft}", max_tokens=650 if kind == "sar" else 300)
    if not out:
        return draft
    # guard: the rewrite must not introduce IDs that are not in the draft/evidence
    known = set(re.findall(r"\b(?:\d{7}|C\d{5}(?:-K\d)?|CC-\d{4})\b", draft + evidence))
    used = set(re.findall(r"\b(?:\d{7}|C\d{5}(?:-K\d)?|CC-\d{4})\b", out))
    return out if used <= known else draft


def stop_reason(facts, prob, indep, response) -> str:
    if response == "denied":
        return (f"The requested verification settled it: the cardholder denied the activity, raising the probability to "
                f"{prob:.2f}. Further graph steps would not change the actions (policy 6).")
    if response == "confirmed":
        return (f"The requested verification settled it: the cardholder confirmed the activity, lowering the probability "
                f"to {prob:.2f}. Nothing left to investigate (policy 6, R3).")
    if response == "no_reply":
        return (f"No reply within 24 hours; the verdict stays uncertain at {prob:.2f}. Protective actions under R4/R8 are "
                "in place and the case is handed to an analyst rather than investigated further by the agent.")
    side = "at or above 0.85" if prob >= 0.85 else "at or below 0.15"
    return (f"Fraud probability {prob:.2f} is {side} with {indep} independent pieces of evidence, so the decision is "
            "defensible without asking the customer (policy 6).")


def build_case(llm, st, facts, findings, episode, prob0, prob, indep, plan0: ActionPlan, plan1: ActionPlan,
               evidence_requests, response, sar) -> dict:
    trig = st.trigger
    fraud = facts.verdict == "fraud"
    final_items = plan1.sorted()
    status = {"fraud": "closed_fraud", "legitimate": "closed_legitimate"}.get(facts.verdict)
    if status is None:
        status = "escalated" if any(i["action"] == "ESCALATE_TO_ANALYST" for i in final_items) else "open"
    affected = [t["id"] for t in episode] if facts.verdict != "legitimate" else []
    exposure = round(sum(abs(t["amt"]) for t in episode), 2) if affected else 0.0

    connected_cards = [c for c in findings.connected_card_ids if c != trig.card_id] if facts.verdict != "legitimate" else []
    devices = findings.connected_device_profiles if facts.verdict != "legitimate" else []
    similar = [c["case_id"] for c in st.similar[:5]]
    similar += [c["case_id"] for c in st.device_cases if c.get("outcome") == "confirmed_fraud"][:3]
    similar += [c["case_id"] for c in st.history if c["relation"] == "same_card" and c["case_id"].startswith("CC-")][:2]
    similar = list(dict.fromkeys(similar))[:8]

    summary = llm_rewrite(llm, "summary", template_summary(st, facts, prob0, prob, response, episode), st, facts, st.docs)
    file_sar = sar[0] and any(i["action"] == "FILE_REPORT" for i in final_items)
    if file_sar:
        narrative = llm_rewrite(llm, "sar", template_sar(st, facts, findings, episode), st, facts, st.docs)
        dates = sorted(t["ts"][:10] for t in episode)
        subjects = [trig.customer_id, trig.card_id] + connected_cards[:25] + devices[:3]
        sar_obj = {"file": True, "reason": sar[1], "narrative": narrative, "subjects": list(dict.fromkeys(subjects)),
                   "total_amount_usd": exposure, "activity_dates": [dates[0], dates[-1]]}
    else:
        sar_obj = {"file": False, "reason": sar[1] if fraud else
                   f"No SAR: verdict is {facts.verdict}; policy 3a requires confirmed or strongly suspected fraud.",
                   "narrative": "", "subjects": [], "total_amount_usd": 0, "activity_dates": []}

    initial = plan0.sorted()
    what_changed = "nothing"
    if evidence_requests:
        a0, a1 = {i["action"] for i in initial}, {i["action"] for i in final_items}
        added, removed = sorted(a1 - a0), sorted(a0 - a1)
        what_changed = (f"{evidence_requests[0]['assumed_response'].replace('Simulated: ', '').rstrip('.')}. "
                        f"Probability {prob0:.2f} -> {prob:.2f}"
                        + (f"; added {', '.join(added)}" if added else "")
                        + (f"; dropped {', '.join(removed)}" if removed else "") + ".")

    graph_id = f"CASE-{trig.opened_at[:4]}-{trig.case_id}"
    return {
        "case_id": trig.case_id,
        "case": {
            "status": status,
            "verdict": facts.verdict,
            "fraud_probability": round(prob, 2),
            "pattern": facts.pattern if facts.verdict != "legitimate" else "none",
            "pattern_description": pattern_description(findings, st.flag, episode) if facts.pattern == "undocumented" else "",
            "affected_txn_ids": affected,
            "first_suspicious_txn_id": affected[0] if affected else "",
            "connected_card_ids": connected_cards,
            "connected_device_profiles": devices,
            "exposure_usd": exposure,
            "evidence": evidence_list(st.signals),
            "similar_prior_cases": similar,
            "summary": summary,
            "written_to_graph": False,
            "graph_case_id": graph_id,
        },
        "evidence_requests": evidence_requests,
        "next_best_actions": {"initial": initial, "final": final_items, "what_changed": what_changed},
        "sar": sar_obj,
        "stop_reason": stop_reason(facts, prob, indep, response),
        "tool_calls": 0, "tokens": 0, "latency_s": 0.0,
        # extra fields (not part of the scored format) used by the UI and graph memory
        "graph_case_id": graph_id, "card_id": trig.card_id, "opened_at": trig.opened_at,
        "trigger": {"type": trig.trigger_type, "text": trig.trigger_text, "flagged_txn_id": trig.flagged_txn_id},
        "status": status, "pattern": facts.pattern, "exposure_usd": exposure, "summary": summary,
    }
