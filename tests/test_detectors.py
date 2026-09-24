"""Detector tests on synthetic windows (no database needed)."""
from fraudagent.agent import detectors as D


def tx(i, ts, amt, channel="online", **kw):
    return {"id": str(i), "ts": ts, "amt": amt, "channel": channel, "product": "C", "device": "", "dev_status": "",
            "proxy_type": "", "region": "", **kw}


def test_card_testing_sequence():
    w = [tx(1, "2016-11-01 10:00:00", 1.1), tx(2, "2016-11-01 10:10:00", 2.4), tx(3, "2016-11-01 10:30:00", 0.95),
         tx(4, "2016-11-01 11:05:00", 259.98)]
    sig, ids, cleared = D.card_testing(w[3], w)
    assert sig and ids == ["1", "2", "3", "4"] and cleared


def test_card_testing_needs_small_run_before_purchase():
    w = [tx(4, "2016-11-01 09:00:00", 259.98), tx(1, "2016-11-01 10:00:00", 1.1), tx(2, "2016-11-01 10:10:00", 2.4),
         tx(3, "2016-11-01 10:30:00", 0.95)]
    assert D.card_testing(w[0], w)[0] is None


def test_structuring_band():
    w = [tx(i, f"2016-11-21 20:{m:02d}:00", a) for i, (m, a) in enumerate([(0, 478.95), (10, 456.96), (24, 488.04), (30, 482.12)])]
    sig, ids = D.structuring(w[-1], w)
    assert sig and len(ids) == 4 and sig.pattern_hint == "undocumented"
    assert D.structuring(w[0], w[:2])[0] is None


def test_generic_device_is_not_a_ring():
    assert D.is_generic_device("Windows |  | chrome 66.0 | ", 10)
    assert D.is_generic_device("iOS Device | iOS 11 | mobile safari | ", 10)
    assert not D.is_generic_device("SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080", 50)


def test_device_ring_requires_proxy_share():
    dev = "SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080"
    flag = tx(100, "2016-11-22 16:11:00", 74.96, device=dev, card="C1-K1")
    uses = [{"card": f"C{i}-K1", "txn": str(i), "ts": "2016-11-20 10:00:00", "amt": 50.0, "product": "C", "risk": .1,
             "dev_status": "New", "proxy_type": "IP_PROXY:ANONYMOUS", "region": ""} for i in range(2, 9)]
    sig, f = D.device_ring(flag, {"uses": uses, "all_time_cards": 8}, None, [])
    assert sig and f.device_ring and len(f.connected_card_ids) == 7
    clean = [{**u, "proxy_type": "", "dev_status": "Found"} for u in uses]
    assert D.device_ring(flag, {"uses": clean, "all_time_cards": 8}, None, [])[0] is None


def test_recurring_needs_consecutive_months():
    flag = tx(9, "2016-12-10 10:00:00", 49.99, p_email="gmail.com")
    m = lambda ts: {"id": ts, "ts": ts, "amt": 49.99, "p_email": "gmail.com", "channel": "online"}  # noqa: E731
    assert D.recurring_signal(flag, [m("2016-10-10 10:00:00"), m("2016-11-10 10:00:00")], "customer_report")[1]
    assert not D.recurring_signal(flag, [m("2016-08-10 10:00:00"), m("2016-10-10 10:00:00")], "customer_report")[1]


def test_family_cap_counts_correlated_evidence_once():
    sig = [D.Signal("a", "", 3.0, "device"), D.Signal("b", "", 3.0, "device")]
    p, indep, fam = D.score(sig)
    assert fam["device"] == D.FAMILY_CAP and indep == 1
