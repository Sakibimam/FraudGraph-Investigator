"""The submitted answer files must pass the format / ID validator."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "cache" / "txn_enriched.parquet").exists(), reason="dataset not prepared")
def test_answer_files_valid():
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.validate_answers import main
    assert main(str(ROOT / "cases")) == 0
