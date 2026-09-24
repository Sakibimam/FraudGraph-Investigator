"""Split the policy, typology and regulatory notes into retrievable chunks."""
from __future__ import annotations

import re
from pathlib import Path


def chunk_markdown(path: Path) -> list[dict]:
    """One chunk per heading (### / ##) and one per policy rule (**R1.** ...)."""
    text = path.read_text()
    source = path.stem
    chunks: list[dict] = []
    parts = re.split(r"\n(?=#{2,3} )", text)
    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        rules = re.split(r"\n(?=\*\*R\d+\.)", body)
        if len(rules) > 1:
            for r in rules:
                m = re.match(r"\*\*(R\d+)\.\s*([^*]+)\*\*", r.strip())
                sec = f"{m.group(1)} {m.group(2).strip()}" if m else heading
                chunks.append({"source": source, "section": sec, "text": r.strip()})
        else:
            for p in re.split(r"\n(?=\*\*\d\. )", body):
                m = re.match(r"\*\*(\d\. [^*]+)\*\*", p.strip())
                chunks.append({"source": source, "section": m.group(1).strip(".") if m else heading,
                               "text": p.strip() or heading})
    for i, c in enumerate(chunks):
        c["id"] = f"{source}#{i:02d}"
    return [c for c in chunks if len(c["text"]) > 40]
