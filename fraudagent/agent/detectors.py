"""Pattern detectors and evidence scoring.

Each detector looks at graph evidence the agent already gathered and returns zero or more
`Signal`s. A signal is a claim with a source, the entity IDs it rests on, a log-odds weight
and a family. Signals from the same family are not independent (policy section 6 asks for
two *independent* pieces of evidence before stopping), so the scorer caps each family.

Weights were set from the closed-case history (Jul-Oct) and checked with a back-test on
October cases; see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median

TS = "%Y-%m-%d %H:%M:%S"


def _history_fraud_rate() -> float:
    """Share of closed cases that were confirmed fraud: the baseline a similar-case vote is compared against."""
    try:
        import pandas as pd

        from fraudagent.config import settings
        outcomes = pd.read_csv(settings.data_dir / "closed_cases_history.csv", usecols=["outcome"])["outcome"]
        return float((outcomes == "confirmed_fraud").mean())
    except Exception:
        return 0.84


HISTORY_FRAUD_RATE = _history_fraud_rate()
# Device profiles this common are browser/OS defaults, not devices: they link unrelated people.
GENERIC_DEVICE_CARDS = 60
PATTERNS = ("card_testing", "card_not_present_fraud", "card_not_present_new_device",
            "out_of_region_use", "account_takeover", "undocumented", "none")


def ts(s: str) -> datetime:
    return datetime.strptime(s[:19], TS)


@dataclass
class Signal:
    name: str
    claim: str
    weight: float                 # log-odds contribution (+ fraud, - legitimate)
    family: str                   # independence group
    source: str = "graph"         # graph | document | customer | external
    ref: str = ""
    entity_ids: list[str] = field(default_factory=list)
    pattern_hint: str = ""

    def as_evidence(self) -> dict:
        return {"claim": self.claim, "source": self.source, "ref": self.ref, "entity_ids": self.entity_ids}


@dataclass
class Findings:
    """What the detectors concluded, beyond the individual signals."""
    pattern: str = "none"
    pattern_description: str = ""
    affected_txn_ids: list[str] = field(default_factory=list)
    connected_card_ids: list[str] = field(default_factory=list)
    connected_device_profiles: list[str] = field(default_factory=list)
    card_testing: bool = False
    large_purchase_cleared: bool = False
    structuring: bool = False
    device_ring: bool = False
    region_cluster: bool = False
    recurring: bool = False
    shared_element: str = ""


GENERIC_INFO = {"windows", "ios device", "macos", "trident/7.0", "linux", ""}


def is_generic_device(device: str, all_time_cards: int) -> bool:
    """True for profiles that describe a platform rather than a device (shared by many strangers)."""
    info = device.split(" | ")[0].strip().lower() if device else ""
    return info in GENERIC_INFO or info.startswith("rv:") or all_time_cards > GENERIC_DEVICE_CARDS * 3


# ------------------------------------------------------------------------------------------
# Individual detectors
# ------------------------------------------------------------------------------------------
def case_model_signal(flag: dict) -> Signal:
    p = float(flag.get("cm_score", 0.0))
    if p >= 0.6:
        w, how = 2.0, "strongly resembles"
    elif p >= 0.3:
        w, how = 1.0, "resembles"
    elif p >= 0.1:
        w, how = 0.0, "is inconclusive against"
    elif p >= 0.03:
        w, how = -0.9, "does not resemble"
    else:
        w, how = -1.7, "clearly does not resemble"
    return Signal("case_memory_model", (
        f"Case-memory model score {p:.3f}: the transaction's feature profile (including Vesta's unnamed "
        f"V/C/D/M/id features) {how} transactions in confirmed-fraud closed cases (Jul-Oct); "
        f"bank risk score was {float(flag.get('risk', 0)):.2f}"), w, "model",
        ref="model:case_memory_gbm(holdout AUC 0.955 vs bank score 0.866)", entity_ids=[flag["id"]])


def risk_score_signal(flag: dict, trigger_type: str) -> Signal | None:
    r = float(flag.get("risk", 0))
    if trigger_type != "risk_score":
        return None
    w = 0.25 if r >= 0.8 else (0.1 if r >= 0.6 else 0.0)
    return Signal("bank_risk_score", f"Bank model scored the flagged transaction {r:.2f}. Treated as a reason to look, "
                  "not a verdict: above 0.7 most alerts in the closed-case history were cleared", w, "risk_score",
                  ref=f"txn:{flag['id']}.risk_score", entity_ids=[flag["id"]])


def card_testing(flag: dict, window: list[dict]) -> tuple[Signal | None, list[str], bool]:
    online = [t for t in window if t["channel"] == "online"]
    smalls = [t for t in online if t["amt"] < 10]
    for i, s0 in enumerate(smalls):
        run = [s for s in smalls[i:] if ts(s["ts"]) - ts(s0["ts"]) <= timedelta(hours=1)]
        if len(run) < 3:
            continue
        last = ts(run[-1]["ts"])
        bigs = [t for t in online if last < ts(t["ts"]) <= last + timedelta(hours=3) and t["amt"] >= 20]
        if not bigs:
            continue
        ids = [t["id"] for t in run] + [t["id"] for t in bigs]
        if flag["id"] not in ids:
            continue
        cleared = any(t["amt"] > 100 for t in bigs)
        amts = ", ".join(f"${t['amt']:.2f}" for t in run)
        return Signal("card_testing", (
            f"{len(run)} online authorisations under $10 ({amts}) within one hour, followed by "
            f"{len(bigs)} larger purchase(s) totalling ${sum(t['amt'] for t in bigs):,.2f}: a card-testing sequence"),
            3.0, "sequence", ref="query:card_window", entity_ids=ids, pattern_hint="card_testing"), ids, cleared
    return None, [], False


def structuring(flag: dict, window: list[dict]) -> tuple[Signal | None, list[str]]:
    band = [t for t in window if 400 <= t["amt"] < 500 and t["channel"] == "online"]
    for i, t0 in enumerate(band):
        run = [t for t in band[i:] if ts(t["ts"]) - ts(t0["ts"]) <= timedelta(minutes=90)]
        ids = [t["id"] for t in run]
        if len(run) >= 3 and flag["id"] in ids:
            span = (ts(run[-1]["ts"]) - ts(run[0]["ts"])).seconds // 60
            total = sum(t["amt"] for t in run)
            return Signal("sub_threshold_structuring", (
                f"{len(run)} online purchases within {span} minutes, each between $400 and $500 "
                f"({', '.join(f'${t['amt']:.2f}' for t in run)}; total ${total:,.2f}): amounts kept just under a "
                "$500 authorisation threshold"), 3.2, "sequence", ref="query:card_window",
                entity_ids=ids, pattern_hint="undocumented"), ids
    return None, []


def device_signals(flag: dict, baseline: dict, neighbors: dict | None) -> list[Signal]:
    out: list[Signal] = []
    dev = flag.get("device") or ""
    if not dev or flag["channel"] != "online":
        return out
    all_cards = int((neighbors or {}).get("all_time_cards", 0))
    generic = is_generic_device(dev, all_cards)
    seen_before = dev in (baseline.get("devices") or {})
    status = flag.get("dev_status", "")
    if status == "New" and not seen_before:
        w = 0.35 if generic else 0.8
        out.append(Signal("new_device", (
            f"Identity record marks the device as New for this account and the profile '{dev}' never appears in the "
            f"card's prior history" + (" (a common browser/OS profile, so a weak link)" if generic else "")), w, "device",
            ref="query:card_baseline + txn.id_15", entity_ids=[flag["id"], dev]))
    elif seen_before:
        out.append(Signal("known_device", f"Device profile '{dev}' was already used on this card before the alert "
                          f"({baseline['devices'][dev]} prior transactions)", -0.6, "device",
                          ref="query:card_baseline", entity_ids=[dev]))
    proxy = flag.get("proxy_type", "")
    if proxy and "TRANSPARENT" not in proxy.upper():
        out.append(Signal("proxy", f"Connection went through an {proxy.split(':')[-1].lower()} proxy ({proxy})", 0.5,
                          "device", ref=f"txn:{flag['id']}.id_23", entity_ids=[flag["id"]]))
    return out


def device_ring(flag: dict, neighbors: dict, ring: dict | None, device_cases: list[dict]) -> tuple[Signal | None, Findings]:
    f = Findings()
    dev = flag.get("device") or ""
    if not dev or not neighbors:
        return None, f
    if is_generic_device(dev, int(neighbors.get("all_time_cards", 0))):
        return None, f
    uses = neighbors.get("uses", [])
    by_card: dict[str, list[dict]] = defaultdict(list)
    for u in uses:
        by_card[u["card"]].append(u)
    others = [c for c in by_card if c != flag["card"]]
    anon = {c for c, us in by_card.items() if any("ANONYMOUS" in (u.get("proxy_type") or "").upper() or
                                                   "HIDDEN" in (u.get("proxy_type") or "").upper() for u in us)}
    new_on = {c for c, us in by_card.items() if any(u.get("dev_status") == "New" for u in us)}
    confirmed = [c for c in device_cases if c.get("outcome") == "confirmed_fraud"]
    undoc = [c for c in confirmed if c.get("pattern") == "undocumented"]
    # A ring is one specific handset driving many unrelated cards, hidden behind anonymising proxies.
    if len(others) < 4 or (len(anon) < 0.5 * len(by_card) and len(undoc) < 2):
        return None, f
    f.device_ring = True
    f.shared_element = f"device profile {dev}"
    # Connected cards are the ones that used this handset behind a proxy or as a New device;
    # the 2-hop ring expansion is reported as context only: those cards' other devices are their own phones.
    f.connected_card_ids = sorted((anon | new_on) - {flag["card"]})
    f.connected_device_profiles = [dev]
    claim = (f"Device profile '{dev}' was used by {len(by_card)} cards in a 60-day window around the alert; "
             f"{len(anon)} of them behind an anonymous/hidden proxy and {len(new_on)} flagged as a New device. "
             f"{len(confirmed)} closed case(s) on cards that used this profile were confirmed fraud"
             + (f", {len(undoc)} of them recorded as an undocumented pattern" if undoc else ""))
    ids = [flag["card"]] + f.connected_card_ids[:20] + [c["case_id"] for c in confirmed[:5]]
    return Signal("shared_device_ring", claim, 2.6 + (0.4 if confirmed else 0.0), "network",
                  ref="query:device_neighbors + query:device_ring + query:device_case_links",
                  entity_ids=ids, pattern_hint="undocumented"), f


def region_signals(flag: dict, baseline: dict, cluster: list[dict] | None) -> tuple[list[Signal], Findings]:
    out: list[Signal] = []
    f = Findings()
    region = flag.get("region") or ""
    if not region or flag["channel"] != "in_person":
        return out, f
    regions = baseline.get("regions") or {}
    n_prior, total = int(regions.get(region, 0)), max(1, sum(regions.values()))
    share = n_prior / total
    if n_prior == 0 or (share < 0.01 and n_prior <= 3):
        what = "no history" if n_prior == 0 else f"only {n_prior} of {total} prior transactions ({share:.1%})"
        out.append(Signal("new_region", f"Card-present purchase in billing region {region}, where this card has "
                          f"{what} before the alert", 1.1 if n_prior == 0 else 0.8, "region",
                          ref="query:card_baseline", entity_ids=[flag["id"]], pattern_hint="out_of_region_use"))
    elif n_prior >= 5 and share >= 0.02:
        out.append(Signal("familiar_region", f"Region {region} is familiar to this card: {n_prior} of "
                          f"{sum(regions.values())} prior transactions were billed there", -0.7, "region",
                          ref="query:card_baseline", entity_ids=[region]))
    if cluster:
        new_cards = [c for c in cluster if c["prior_in_region"] == 0 and c["prior_total"] >= 5 and c["card"] != flag["card"]]
        risky = [c for c in new_cards if c["max_risk"] >= 0.5]
        if len(new_cards) >= 8 and len(risky) >= 4:
            f.region_cluster = True
            f.shared_element = f"billing region {region}"
            f.connected_card_ids = sorted(c["card"] for c in risky)[:40]
            out.append(Signal("region_cluster", f"{len(new_cards)} other cards made card-present purchases in region "
                              f"{region} in the same 72h window with no prior history there; {len(risky)} of them were "
                              "scored 0.5+ by the bank model", 1.5, "network", ref="query:region_cluster",
                              entity_ids=[region] + f.connected_card_ids[:10]))
    return out, f


def amount_signals(flag: dict, baseline: dict) -> list[Signal]:
    amts = baseline.get("amounts") or []
    out: list[Signal] = []
    if len(amts) < 5:
        return out
    s = sorted(amts)
    p95 = s[int(0.95 * (len(s) - 1))]
    med = median(s)
    if flag["amt"] > max(p95 * 1.2, med * 4):
        out.append(Signal("amount_outlier", f"${flag['amt']:,.2f} is above the card's 95th percentile (${p95:,.2f}; "
                          f"median ${med:,.2f})", 0.5, "behaviour", ref="query:card_baseline", entity_ids=[flag["id"]]))
    elif flag["amt"] <= p95:
        out.append(Signal("amount_typical", f"${flag['amt']:,.2f} is within the card's usual range (median ${med:,.2f}, "
                          f"95th percentile ${p95:,.2f})", -0.3, "behaviour", ref="query:card_baseline",
                          entity_ids=[flag["id"]]))
    prods = baseline.get("products") or {}
    if prods and flag["product"] not in prods and flag["channel"] == "online":
        out.append(Signal("new_product", f"Product code {flag['product']} never used on this card before", 0.4,
                          "behaviour", ref="query:card_baseline", entity_ids=[flag["id"]]))
    return out


def recurring_signal(flag: dict, matches: list[dict], trigger_type: str) -> tuple[Signal | None, bool]:
    """R7: the disputed amount recurs on the card, month after month, from the same purchaser."""
    tight = [m for m in matches if abs(m["amt"] - flag["amt"]) <= max(0.05, 0.005 * flag["amt"])]
    same_src = [m for m in tight if m.get("p_email") and m["p_email"] == flag.get("p_email")]
    months = sorted({m["ts"][:7] for m in same_src} | {flag["ts"][:7]})
    idx = [int(m[:4]) * 12 + int(m[5:7]) for m in months]
    run = best = 1
    for a, b in zip(idx, idx[1:], strict=False):
        run = run + 1 if b - a == 1 else 1
        best = max(best, run)
    # monthly: at least three consecutive months (including the disputed one), not a burst of look-alikes
    if len(same_src) < 2 or best < 3 or len(same_src) > 3 * len(months):
        return None, False
    return Signal("recurring_charge", (
        f"{len(same_src)} earlier charges of ~${flag['amt']:.2f} (product {flag['product']}, purchaser email "
        f"{flag.get('p_email')}) in {len(months)} different months ({', '.join(months)}): the disputed charge matches "
        "the customer's own recurring pattern"), -2.0 if trigger_type == "customer_report" else -1.0, "recurring",
        ref="query:recurring_matches", entity_ids=[m["id"] for m in same_src[:6]]), True


def memory_signals(flag: dict, history: list[dict], similar: list[dict]) -> list[Signal]:
    out: list[Signal] = []
    t0 = ts(flag["ts"])
    recent = [c for c in history if c["relation"] == "same_card" and c["outcome"] == "confirmed_fraud"
              and timedelta(0) <= t0 - ts(c["opened_at"] + " 00:00:00") <= timedelta(days=45)]
    if recent:
        out.append(Signal("recent_fraud_on_card", f"Card had {len(recent)} confirmed-fraud case(s) in the 45 days before "
                          f"the alert ({', '.join(c['case_id'] for c in recent[:3])}); the number may still be compromised",
                          0.4, "memory", ref="query:card_case_history", entity_ids=[c["case_id"] for c in recent[:3]]))
    if similar:
        conf = [c for c in similar if c["outcome"] == "confirmed_fraud"]
        frac = len(conf) / len(similar)
        # the case history is mostly confirmed fraud, so a vote only carries information where it departs from it
        w = round((frac - HISTORY_FRAUD_RATE) * 1.5, 2)
        if abs(w) >= 0.1:
            out.append(Signal("similar_case_outcomes", f"{len(conf)} of the {len(similar)} most similar closed cases were "
                              f"confirmed fraud vs {HISTORY_FRAUD_RATE:.0%} across all closed cases "
                              f"({', '.join(c['case_id'] for c in similar[:4])})", w, "memory",
                              ref="query:similar_closed_cases (TigerVector)", entity_ids=[c["case_id"] for c in similar[:4]]))
    return out


def trigger_signal(trigger_type: str, flag: dict, text: str) -> Signal | None:
    if trigger_type == "customer_report":
        return Signal("customer_dispute", f"Cardholder reported they did not make the ${flag['amt']:.2f} transaction: "
                      f"\"{text.split('message:')[-1].strip()[:120]}\"", 0.8, "customer", source="customer",
                      ref="trigger:customer_report", entity_ids=[flag["id"]])
    if trigger_type == "analyst_request":
        return Signal("analyst_concern", f"Analyst requested review: {text[:160]}", 0.3, "analyst", source="external",
                      ref="trigger:analyst_request", entity_ids=[flag["id"]])
    return None


# ------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------
FAMILY_CAP = 3.2


def score(signals: list[Signal], prior_logit: float = 0.0) -> tuple[float, int, dict]:
    fam: dict[str, float] = defaultdict(float)
    for s in signals:
        fam[s.family] += s.weight
    capped = {k: max(-FAMILY_CAP, min(FAMILY_CAP, v)) for k, v in fam.items()}
    z = prior_logit + sum(capped.values())
    p = 1 / (1 + math.exp(-z))
    p = min(0.98, max(0.02, p))
    direction = 1 if p >= 0.5 else -1
    independent = sum(1 for v in capped.values() if v * direction >= 0.5)
    return round(p, 3), independent, capped


def classify_pattern(flag: dict, signals: list[Signal], episode: list[dict]) -> str:
    hints = [s.pattern_hint for s in signals if s.pattern_hint and s.weight > 0]
    if "undocumented" in hints:
        return "undocumented"
    if "card_testing" in hints:
        return "card_testing"
    channels = {t["channel"] for t in episode} or {flag["channel"]}
    if len(channels) > 1:
        return "account_takeover"
    if flag["channel"] == "in_person":
        return "out_of_region_use" if "out_of_region_use" in hints else "account_takeover"
    if any(t.get("dev_status") == "New" for t in episode) or any(s.name == "new_device" for s in signals):
        return "card_not_present_new_device"
    return "card_not_present_fraud"
