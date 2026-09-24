"""Simulated evidence responses (policy section 5).

Customer and step-up replies are not part of the dataset, so the agent simulates them and
records the assumption. The simulation is deliberately transparent: the reply follows from the
evidence the agent already holds, which is what a real reply would most likely be, and the
probability update is a fixed likelihood ratio per reply type.

    posterior >= 0.55            -> cardholder denies / step-up fails
    0.42 <= posterior < 0.55     -> no reply within 24 hours (R4)
    posterior < 0.42             -> cardholder confirms / step-up passes
"""
from __future__ import annotations

DENY_LR, CONFIRM_LR = 6.0, 1 / 8.0


def _update(p: float, lr: float) -> float:
    odds = p / (1 - p) * lr
    return round(min(0.98, max(0.02, odds / (1 + odds))), 3)


class CustomerSimulator:
    def respond(self, kind: str, trigger_type: str, p: float, facts, flag: dict) -> tuple[str, str, float]:
        amt = f"${flag['amt']:.2f}"
        if kind == "step_up_auth":
            if p >= 0.55:
                return ("denied", f"Simulated: the one-time passcode sent to the cardholder's registered phone was not "
                        f"completed for the session that made the {amt} purchase; contacted separately, the cardholder "
                        "said they did not make it and still holds the card", _update(p, DENY_LR))
            if p >= 0.42:
                return ("no_reply", "Simulated: step-up challenge issued; no response from the cardholder within 24 hours",
                        p)
            return ("confirmed", f"Simulated: the cardholder passed the step-up challenge on their registered device and "
                    f"confirmed the {amt} purchase", _update(p, CONFIRM_LR))
        if kind == "analyst_info":
            if p >= 0.5:
                return ("denied", "Simulated: analyst review agrees the linked activity is not the cardholder's",
                        _update(p, DENY_LR))
            return ("confirmed", "Simulated: analyst found a benign explanation for the linked activity",
                    _update(p, CONFIRM_LR))
        # customer_validation
        if trigger_type == "customer_report":
            if facts.recurring_dispute:
                return ("confirmed", f"Simulated: shown the merchant descriptor and prior monthly charges, the cardholder "
                        f"recognised the {amt} charge as their own recurring subscription", _update(p, CONFIRM_LR))
            if p >= 0.5:
                return ("denied", f"Simulated: on follow-up the cardholder repeated that they did not make the {amt} "
                        "purchase, still has the card and has not shared the card details", _update(p, DENY_LR))
            if p >= 0.42:
                return ("no_reply", "Simulated: follow-up questions sent to the cardholder; no reply within 24 hours", p)
            return ("confirmed", f"Simulated: shown the merchant name, time and device of the {amt} purchase, the "
                    "cardholder recognised it as their own and withdrew the dispute",
                    _update(p, CONFIRM_LR))
        if p >= 0.55:
            return ("denied", f"Simulated: the cardholder replied that they did not make the {amt} transaction and still "
                    "holds the card", _update(p, DENY_LR))
        if p >= 0.42:
            return ("no_reply", "Simulated: verification message sent; no reply within 24 hours (R4)", p)
        return ("confirmed", f"Simulated: the cardholder confirmed they made the {amt} transaction", _update(p, CONFIRM_LR))
