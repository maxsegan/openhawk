"""The vendored cascade evidence and prompt modules are the frozen originals."""

import hashlib
import json
import os
from pathlib import Path

import pytest

FROZEN = Path(__file__).resolve().parent / "event_cascade_frozen"
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
SOURCES = {
    "fairlib.py": REPO / "cv/experiments/labeller_fair_evidence/fairlib.py",
    "render_evidence.py": REPO / "cv/experiments/labeller_fair_evidence/render_evidence.py",
    "run_label.py": REPO / "cv/experiments/labeller_fair_evidence/run_label.py",
    "guided_procedure.py": REPO / "cv/experiments/labeller_fair_evidence/guided_procedure.py",
    "or_call.py": REPO / "cv/experiments/vlm_labeller_axis/or_call.py",
    "trace.py": DATA / "processed/vlm_event_benchmark_v1/scripts/trace.py",
    "trace_from_track.py": DATA / "processed/vlm_event_benchmark_v1/scripts/trace_from_track.py",
}


def _logic(text: str) -> list[str]:
    """Lines other than the import plumbing the copy was allowed to change."""
    return [
        line
        for line in text.splitlines()
        if "sys.path.insert" not in line
        and not line.startswith(("import trace", "from fairlib", "from guided_procedure"))
        and not line.startswith(("from or_call", "from .", "from . import"))
    ]


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_copy_differs_from_its_source_only_in_imports(name):
    source = SOURCES[name]
    if not source.is_file():
        pytest.skip(f"source {source} not present")
    assert _logic((FROZEN / name).read_text()) == _logic(source.read_text())


RECORDED = DATA / "processed/cascade-mainline-20260927/live_panel_d"


@pytest.mark.skipif(not RECORDED.is_dir(), reason="recorded cascade evidence not present")
def test_rendered_evidence_is_byte_identical_to_the_recorded_run(tmp_path):
    from cv.pipeline import event_cascade_models as models

    row = json.loads((RECORDED / "candidates.json").read_text())["rows"][0]
    recorded = json.loads(
        (RECORDED / "adjudicate/evidence/one/pergrid" / row["clip"] / "manifest.json").read_text()
    )
    old = next(item for item in recorded["nominations"] if item["id"] == row["id"])
    new = models.render(row, tmp_path)
    for fresh, saved in zip(new["images"], old["images"], strict=True):
        assert (
            hashlib.sha256(Path(fresh["path"]).read_bytes()).hexdigest()
            == hashlib.sha256(Path(saved["path"]).read_bytes()).hexdigest()
        )
