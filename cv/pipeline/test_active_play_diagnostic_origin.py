"""The diagnostic CLI is explicit and cannot bless supplied boundaries as automatic."""

from pathlib import Path
import subprocess
import sys

import pytest

from cv.pipeline import provenance as p


def _run(tmp_path, mode, diagnostic):
    root = tmp_path / "inputs"
    root.mkdir()
    selection = tmp_path / "supplied_windows.json"
    selection.write_text('{"source_seconds":[10,20]}')
    ancestry = [dict(p.file_record(selection), role="supplied_native_attempt_selection")]
    parent = p.build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=mode,
        reviewed_inputs=ancestry if mode == p.DIAGNOSTIC_MODE else [],
        configuration={"supplied_segmentation": mode == p.DIAGNOSTIC_MODE},
    )
    p.write_provenance(root / "provenance.json", parent)
    output = tmp_path / "active.json"
    command = [
        sys.executable,
        "-m",
        "cv.pipeline.active_play_gate",
        "--processed",
        str(root),
        "--out",
        str(output),
        *(["--diagnostic-inputs"] if diagnostic else []),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result, output, root, ancestry


def test_explicit_diagnostic_cli_preserves_reviewed_boundaries(tmp_path):
    result, output, root, ancestry = _run(tmp_path, p.DIAGNOSTIC_MODE, True)
    assert result.returncode == 0, result.stderr
    document = p.load_provenance(output.with_name("active.provenance.json"))
    assert document["mode"] == p.DIAGNOSTIC_MODE
    assert document["reviewed_inputs"] == ancestry
    assert (
        p.file_record(root / "provenance.json", role="parent_provenance")
        in document["reused_artifacts"]
    )
    with pytest.raises(p.ProvenanceError, match="diagnostic"):
        p.load_provenance(output.with_name("active.provenance.json"), require_automatic=True)


@pytest.mark.parametrize("mode,diagnostic", [(p.DIAGNOSTIC_MODE, False), (p.AUTOMATIC_MODE, True)])
def test_origin_mismatch_refuses_before_outputs(tmp_path, mode, diagnostic):
    result, output, _, _ = _run(tmp_path, mode, diagnostic)
    assert result.returncode != 0
    assert not output.exists()


def test_default_automatic_cli_remains_automatic(tmp_path):
    result, output, _, _ = _run(tmp_path, p.AUTOMATIC_MODE, False)
    assert result.returncode == 0, result.stderr
    document = p.load_provenance(output.with_name("active.provenance.json"), require_automatic=True)
    assert document["reviewed_inputs"] == []
