"""Policy engine tests: every rule in the Fraud Policy v1.0 that the agent relies on."""
from fraudagent.agent.policy import (
    PolicyFacts,
    final_plan,
    initial_plan,
    route_for,
    sar_required,
    should_stop,
)


def facts(**kw):
    base = dict(trigger_type="risk_score", probability=0.5, independent_signals=1, verdict="uncertain",
                pattern="card_not_present_fraud", exposure=100.0)
    base.update(kw)
    return PolicyFacts(**base)


def actions(plan):
    return [a["action"] for a in plan.sorted()]


def test_routes():
    assert route_for("CREATE_CASE") == "auto"
    assert route_for("DECLINE_TRANSACTION") == "L1"
    assert route_for("BLOCK_CARD", 2500) == "L1"
    assert route_for("BLOCK_CARD", 2500.01) == "L2"
    assert route_for("BLOCK_ALL_CARDS") == "L2"
    assert route_for("FILE_REPORT") == "L2"


def test_r1_verify_before_block_on_weak_signal():
    p = initial_plan(facts(probability=0.45), "customer_validation")
    assert "VERIFY_WITH_CUSTOMER" in actions(p) and "BLOCK_CARD" not in actions(p)


def test_r2_denial_blocks_and_creates_case():
    p = final_plan(facts(verdict="fraud", probability=0.88, customer_response="denied"))
    assert {"BLOCK_CARD", "CREATE_CASE"} <= set(actions(p))


def test_r2_sar_when_exposure_over_1000_or_shared_device():
    assert sar_required(facts(verdict="fraud", exposure=1000.01))[0]
    assert not sar_required(facts(verdict="fraud", exposure=1000.0))[0]
    assert sar_required(facts(verdict="fraud", exposure=50, shared_origin=True))[0]


def test_r3_confirmation_closes():
    p = final_plan(facts(verdict="legitimate", probability=0.05, customer_response="confirmed"))
    assert "CLOSE_NO_FRAUD" in actions(p) and "BLOCK_CARD" not in actions(p)


def test_r4_no_reply_monitors_declines_and_escalates_over_500():
    p = final_plan(facts(verdict="uncertain", customer_response="no_reply", exposure=600))
    assert {"MONITOR_CARD", "DECLINE_TRANSACTION", "ESCALATE_TO_ANALYST"} <= set(actions(p))
    p = final_plan(facts(verdict="uncertain", customer_response="no_reply", exposure=100))
    assert "ESCALATE_TO_ANALYST" not in actions(p)


def test_r5_card_testing():
    p = initial_plan(facts(card_testing=True, probability=0.6), "step_up_auth")
    assert {"DECLINE_TRANSACTION", "STEP_UP_AUTH"} <= set(actions(p))
    p = initial_plan(facts(card_testing=True, large_purchase_cleared=True, probability=0.9, verdict="fraud"), None)
    assert "BLOCK_CARD" in actions(p)


def test_r6_shared_origin_monitors_connected_and_files():
    p = final_plan(facts(verdict="fraud", probability=0.9, shared_origin=True, has_connected_cards=True,
                         customer_response="denied"))
    assert {"MONITOR_CONNECTED_CARDS", "FILE_REPORT", "CREATE_CASE"} <= set(actions(p))


def test_r7_recurring_dispute_never_blocks():
    p = initial_plan(facts(trigger_type="customer_report", recurring_dispute=True, probability=0.2), "customer_validation")
    assert {"CREATE_CASE", "VERIFY_WITH_CUSTOMER", "WARN_CUSTOMER"} <= set(actions(p))
    assert "BLOCK_CARD" not in actions(p)


def test_r8_uncertain_and_exposed_escalates():
    p = final_plan(facts(verdict="uncertain", exposure=800))
    assert "ESCALATE_TO_ANALYST" in actions(p)


def test_r9_undocumented_files_and_escalates():
    p = final_plan(facts(verdict="fraud", probability=0.95, pattern="undocumented", customer_response="denied"))
    assert {"CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST"} <= set(actions(p))


def test_r10_never_block_all_cards_on_one_card():
    p = final_plan(facts(verdict="fraud", probability=0.98, exposure=5000, customer_response="denied"))
    assert "BLOCK_ALL_CARDS" not in actions(p)
    p = final_plan(facts(verdict="fraud", probability=0.98, fraud_cards_for_customer=2, customer_response="denied"))
    assert "BLOCK_ALL_CARDS" in actions(p)


def test_case_vs_report():
    ok, why = sar_required(facts(verdict="fraud", exposure=300))
    assert not ok and "Case only" in why
    assert not sar_required(facts(verdict="uncertain", exposure=5000))[0]


def test_stopping_rule():
    assert should_stop(0.9, 2) and should_stop(0.1, 2)
    assert not should_stop(0.9, 1) and not should_stop(0.5, 3)


def test_legitimate_never_blocks():
    for trig in ("risk_score", "customer_report", "analyst_request"):
        p = final_plan(facts(trigger_type=trig, verdict="legitimate", probability=0.05, customer_response="confirmed"))
        assert not {"BLOCK_CARD", "DECLINE_TRANSACTION", "FILE_REPORT", "BLOCK_ALL_CARDS"} & set(actions(p))
