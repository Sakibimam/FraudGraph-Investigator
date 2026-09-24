"""Export a static, read-only snapshot of the dashboard for hosting (Vercel, GitHub Pages, any CDN).

The live system needs TigerGraph, the dataset and an LLM, none of which a static host can run.
This script calls the real API in-process and writes every response the UI needs to site/data/,
so the hosted page shows exactly what the TigerGraph run produced:

    python scripts/export_static.py        # writes site/

In the snapshot, "Run investigation" replays the recorded agent timeline, approvals are
role-checked in the browser but not persisted, and ad-hoc investigations are disabled.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRAPH_BACKEND", "local")  # the export only reads saved traces

from fastapi.testclient import TestClient  # noqa: E402

from fraudagent.api.server import app  # noqa: E402

SITE = ROOT / "site"


def main() -> None:
    client = TestClient(app)
    get = lambda path: client.get(path).json()  # noqa: E731
    if SITE.exists():
        shutil.rmtree(SITE)
    data = SITE / "data"
    for sub in ("case", "graph", "execution"):
        (data / sub).mkdir(parents=True)

    for name in ("index.html", "app.js", "styles.css"):
        shutil.copy(ROOT / "ui" / name, SITE / name)
    html = (SITE / "index.html").read_text().replace("/ui/", "")
    html = html.replace('<script src="app.js" defer></script>',
                        '<script>window.FG_STATIC = true;</script>\n  <script src="app.js" defer></script>')
    (SITE / "index.html").write_text(html)

    health = get("/api/health")
    live_llm = json.loads((ROOT / "runs" / "traces" / "HHG-014.json").read_text()).get("llm", health["llm"])
    health.update({"graph_backend": "snapshot of tigergraph-mcp run", "llm": live_llm, "static": True})
    (data / "health.json").write_text(json.dumps(health))
    cases = get("/api/cases")
    (data / "cases.json").write_text(json.dumps(cases))
    overview = get("/api/overview")
    overview["health"] = health
    (data / "overview.json").write_text(json.dumps(overview))
    monitor = get("/api/monitor")
    (data / "monitor.json").write_text(json.dumps(monitor))

    ids = [c["case_id"] for c in cases] + [m["case_id"] for m in monitor] + [a["case_id"] for a in overview["adhoc"]]
    for cid in ids:
        r = client.get(f"/api/cases/{cid}")
        if r.status_code != 200:
            continue
        (data / "case" / f"{cid}.json").write_text(json.dumps(r.json()))
        (data / "graph" / f"{cid}.json").write_text(json.dumps(get(f"/api/cases/{cid}/graph")))
        if cid.startswith("HHG-"):
            (data / "execution" / f"{cid}.json").write_text(json.dumps(get(f"/api/cases/{cid}/execution")))
    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"exported {len(ids)} cases to {SITE} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
