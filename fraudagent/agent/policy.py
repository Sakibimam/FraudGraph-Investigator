"""Fraud Policy v1.0 as code: actions, approval routes, rules R1-R10, SAR test, stopping rule.

The LLM never decides an action or a route. It can explain them, but every recommendation
goes through this module, and only `auto` actions can be executed by the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AUTO = "auto"
L1 = "L1"
L2 = "L2"

ACTIONS = {
    "ALLOW_TRANSACTION", "DECLINE_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS",
    "GENERATE_REPORT", "CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
}
AUTO_ACTIONS = {
    "ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH", "GENERATE_REPORT", "CREATE_CASE",
    "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
}
# Execution order when several actions are recommended ("order them by what happens first").
ORDER = [
    "DECLINE_TRANSACTION", "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS", "ALLOW_TRANSACTION",
    "CREATE_CASE", "VERIFY_WITH_CUSTOMER", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER", "FILE_REPORT", "ESCALATE_TO_ANALYST", "GENERATE_REPORT", "CLOSE_NO_FRAUD",
]

CASE_THRESHOLD = 0.30        # 3a: open a case at this probability
STOP_HIGH, STOP_LOW = 0.85, 0.15  # section 6
R1_BLOCK_FLOOR = 0.70        # R1: below this on a single signal, verify first
SAR_EXPOSURE = 1000.0        # 3a / R2
BLOCK_L2_EXPOSURE = 2500.0   # section 2
R4_ESCALATE_EXPOSURE = 500.0
R8_ESCALATE_EXPOSURE = 500.0


def route_for(action: str, exposure: float = 0.0) -> str:
    if action in AUTO_ACTIONS:
        return AUTO
    if action == "DECLINE_TRANSACTION":
        return L1
    if action == "BLOCK_CARD":
        return L2 if exposure > BLOCK_L2_EXPOSURE else L1
    return L2  # BLOCK_ALL_CARDS, FILE_REPORT


@dataclass
class Recommendation:
    action: str
    route: str
    reason: str

    def as_dict(self) -> dict:
        return {"action": self.action, "route": self.route, "reason": self.reason}


@dataclass
class ActionPlan:
    items: list[Recommendation] = field(default_factory=list)

    def add(self, action: str, reason: str, exposure: float = 0.0) -> None:
        assert action in ACTIONS, action
        for it in self.items:
            if it.action == action:
                if reason not in it.reason:
                    it.reason = f"{it.reason}; {reason}"
                return
        self.items.append(Recommendation(action, route_for(action, exposure), reason))

    def has(self, action: str) -> bool:
        return any(i.action == action for i in self.items)

    def sorted(self) -> list[dict]:
        return [i.as_dict() for i in sorted(self.items, key=lambda r: ORDER.index(r.action))]


@dataclass
class PolicyFacts:
    """Everything the policy needs to know, as established by the investigation."""
    trigger_type: str
    probability: float
    independent_signals: int
    verdict: str                       # fraud | legitimate | uncertain
    pattern: str
    exposure: float
    shared_origin: bool = False        # R6: shared device / region / recipient across cards
    coordinated: bool = False          # R9
    card_testing: bool = False         # R5
    large_purchase_cleared: bool = False
    recurring_dispute: bool = False    # R7
    evidence_conflicts: bool = False
    customer_response: str = ""        # "" | denied | confirmed | no_reply
    fraud_cards_for_customer: int = 1
    credentials_compromised: bool = False
    has_connected_cards: bool = False


def sar_required(f: PolicyFacts) -> tuple[bool, str]:
    """3a: file when fraud is confirmed/strongly suspected AND one qualifying condition holds."""
    if f.verdict != "fraud":
        return False, "No SAR: fraud is not confirmed or strongly suspected (3a)."
    reasons = []
    if f.exposure > SAR_EXPOSURE:
        reasons.append(f"exposure ${f.exposure:,.2f} exceeds $1,000 (3a, R2)")
    if f.shared_origin or f.has_connected_cards:
        reasons.append("activity connects to a shared device profile / other cards (3a, R2, R6)")
    if f.coordinated or f.pattern == "undocumented":
        reasons.append("pattern is coordinated or undocumented (3a, R9)")
    if not reasons:
        return False, (f"No SAR: fraud confirmed but exposure ${f.exposure:,.2f} is under $1,000, no shared "
                       "origin or connected cards, and the pattern is documented (3a). Case only.")
    return True, "File: " + "; ".join(reasons) + "."


def should_stop(prob: float, independent_signals: int) -> bool:
    return (prob >= STOP_HIGH or prob <= STOP_LOW) and independent_signals >= 2


def initial_plan(f: PolicyFacts, evidence_type: str | None) -> ActionPlan:
    """What the agent recommends before any requested evidence comes back."""
    p = ActionPlan()
    x = f.exposure
    if f.trigger_type == "customer_report" or f.probability >= CASE_THRESHOLD or evidence_type:
        p.add("CREATE_CASE", "3a: customer dispute / probability >= 0.30 / evidence requested")

    if f.recurring_dispute:
        p.add("VERIFY_WITH_CUSTOMER", "R7: disputed charge matches the customer's own recurring pattern")
        p.add("WARN_CUSTOMER", "R7: recurring charge reminder; do not block")
        return p

    if f.card_testing:
        p.add("DECLINE_TRANSACTION", "R5: small-authorisation testing sequence followed by a larger purchase", x)
        p.add("STEP_UP_AUTH", "R5: require OTP before further activity")
        if f.large_purchase_cleared and f.probability >= R1_BLOCK_FLOOR:
            p.add("BLOCK_CARD", "R5: a purchase over $100 already cleared", x)

    if evidence_type is None:
        return _decided_plan(f, p)

    # Uncertain: gather evidence first (R1). Protect the card without blocking it.
    if evidence_type == "step_up_auth":
        p.add("STEP_UP_AUTH", "R1: single/weak signal below 0.70, confirm identity before any block")
    elif evidence_type == "customer_validation":
        p.add("VERIFY_WITH_CUSTOMER", "R1: signal below 0.70 or conflicting, ask the cardholder before any block")
    else:
        p.add("ESCALATE_TO_ANALYST", "5: request information from an analyst")
    if f.probability >= 0.5 and not f.card_testing:
        p.add("MONITOR_CARD", "Raise monitoring for 72h while the verification is pending")
    if f.shared_origin:
        p.add("MONITOR_CONNECTED_CARDS", "R6: other cards share the same origin; monitor while pending")
    return p


def _decided_plan(f: PolicyFacts, p: ActionPlan) -> ActionPlan:
    x = f.exposure
    if f.verdict == "legitimate":
        if f.trigger_type == "customer_report":
            p.add("VERIFY_WITH_CUSTOMER", "R7/3a: dispute on activity that matches the cardholder; walk them through it")
            p.add("WARN_CUSTOMER", "Explain the charge and share a security tip")
        else:
            p.add("ALLOW_TRANSACTION", "Evidence supports a legitimate transaction")
        p.add("CLOSE_NO_FRAUD", "Stopping rule (6): probability <= 0.15 with independent evidence"
              if f.probability <= STOP_LOW else "R3: activity confirmed as the cardholder's")
        return p

    if f.verdict == "uncertain":
        if f.customer_response == "no_reply":
            p.add("MONITOR_CARD", "R4: no reply within 24 hours")
            p.add("DECLINE_TRANSACTION", "R4: decline pending authorisations", x)
            if x > R4_ESCALATE_EXPOSURE:
                p.add("ESCALATE_TO_ANALYST", f"R4: exposure ${x:,.2f} exceeds $500")
        if x > R8_ESCALATE_EXPOSURE or f.evidence_conflicts:
            p.add("ESCALATE_TO_ANALYST", "R8: verdict uncertain with exposure over $500 or conflicting evidence")
        p.add("MONITOR_CARD", "Keep monitoring while the case stays open")
        if f.shared_origin:
            p.add("MONITOR_CONNECTED_CARDS", "R6: shared origin with other cards")
        return p

    # fraud
    p.add("CREATE_CASE", "3a / R2: fraud case with evidence attached")
    if f.card_testing:
        p.add("DECLINE_TRANSACTION", "R5: testing sequence", x)
        p.add("STEP_UP_AUTH", "R5")
    if f.customer_response == "denied" or f.probability >= R1_BLOCK_FLOOR:
        why = "R2: customer denied the activity" if f.customer_response == "denied" else \
              f"Probability {f.probability:.2f} supported by independent evidence (R1 satisfied)"
        p.add("BLOCK_CARD", f"{why}; exposure ${x:,.2f} {'> $2,500 (L2)' if x > BLOCK_L2_EXPOSURE else '<= $2,500 (L1)'}", x)
    if (f.fraud_cards_for_customer >= 2 or f.credentials_compromised):
        p.add("BLOCK_ALL_CARDS", "R10: two or more of the customer's cards show fraud / credentials compromised")
    if f.shared_origin or f.has_connected_cards:
        p.add("MONITOR_CONNECTED_CARDS", "R6: monitor every card that shares the device profile / region / ring")
    file, why = sar_required(f)
    if file:
        p.add("FILE_REPORT", why)
    if f.pattern == "undocumented" or f.coordinated:
        p.add("ESCALATE_TO_ANALYST", "R9: undocumented / coordinated pattern, analyst review")
    if f.evidence_conflicts:
        p.add("ESCALATE_TO_ANALYST", "R8: evidence conflicts")
    return p


def final_plan(f: PolicyFacts) -> ActionPlan:
    """What the agent recommends after the (simulated) evidence response."""
    p = ActionPlan()
    p.add("CREATE_CASE", "3a: a case is opened whenever evidence is requested; the response is recorded in it")
    if f.recurring_dispute and f.verdict == "legitimate":
        p.add("CREATE_CASE", "R7 / 3a: disputed charge recorded")
        p.add("VERIFY_WITH_CUSTOMER", "R7: confirm the recurring merchant with the cardholder")
        p.add("WARN_CUSTOMER", "R7: recurring charge reminder; do not block")
        p.add("CLOSE_NO_FRAUD", "R3: cardholder recognised the recurring charge")
        return p
    if f.verdict == "legitimate":
        p.add("ALLOW_TRANSACTION", "R3: cardholder confirmed the transaction")
        p.add("CLOSE_NO_FRAUD", "R3: customer confirmation noted in the case file")
        return p
    return _decided_plan(f, p)
