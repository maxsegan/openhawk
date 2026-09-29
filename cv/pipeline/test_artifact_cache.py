import sys

import pytest

from cv.pipeline import artifact_cache

from cv.pipeline.artifact_cache import (
    stage_identity,
    stage_receipt_allows_dependency_narrowing,
    stage_receipt_matches,
    write_stage_receipt,
)


def test_receipt_rejects_changed_input_code_command_or_output(tmp_path) -> None:
    code = tmp_path / "stage.py"
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    code.write_text("print('one')\n")
    source.write_text("source-one")
    output.write_text("output-one")
    command = ["python", str(code), "--threshold", "1"]
    kwargs = {
        "out_dir": tmp_path,
        "stage": "example",
        "command": command,
        "inputs": [source],
        "outputs": [output],
    }
    write_stage_receipt(**kwargs)
    assert stage_receipt_matches(**kwargs)

    source.write_text("source-two")
    assert not stage_receipt_matches(**kwargs)
    source.write_text("source-one")
    code.write_text("print('two')\n")
    assert not stage_receipt_matches(**kwargs)
    code.write_text("print('one')\n")
    assert not stage_receipt_matches(**{**kwargs, "command": [*command, "--new"]})
    output.write_text("output-two")
    assert not stage_receipt_matches(**kwargs)


def test_receipt_includes_upstream_receipt_identity(tmp_path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    upstream = tmp_path / "upstream.json"
    source.write_text("source")
    output.write_text("output")
    upstream.write_text("one")
    kwargs = {
        "out_dir": tmp_path,
        "stage": "example",
        "command": ["true"],
        "inputs": [source],
        "outputs": [output],
        "upstream_receipts": [upstream],
    }
    write_stage_receipt(**kwargs)
    assert stage_receipt_matches(**kwargs)
    upstream.write_text("two")
    assert not stage_receipt_matches(**kwargs)


def test_function_configuration_cannot_be_ignored_by_reuse_or_dependency_narrowing(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_text("source")
    output.write_text("output")
    kwargs = dict(
        out_dir=tmp_path,
        stage="function",
        command=["python"],
        inputs=[source],
        outputs=[output],
        configuration={"mode": "first"},
    )
    write_stage_receipt(**kwargs)
    assert stage_receipt_matches(**kwargs)
    changed = {**kwargs, "configuration": {"mode": "second"}}
    assert not stage_receipt_matches(**changed)
    assert not stage_receipt_allows_dependency_narrowing(**changed)


def test_module_command_hashes_module_source() -> None:
    identity = stage_identity(
        stage="module",
        command=[sys.executable, "-m", "cv.pipeline.paths"],
        inputs=[],
    )

    assert len(identity["code"]) == 1
    assert identity["code"][0]["path"] == "cv/pipeline/paths.py"


def test_repository_code_closure_includes_local_imports() -> None:
    identity = stage_identity(
        stage="module",
        command=[
            sys.executable,
            "-m",
            "cv.pipeline.pose_player_crop",
        ],
        inputs=[],
    )

    paths = {row["path"] for row in identity["code"]}
    assert "cv/pipeline/pose_player_crop.py" in paths
    assert "cv/pipeline/pose_crop_infer.py" in paths


def test_tracker_receipts_include_every_declared_subprocess():
    from cv.pipeline.tracking_composition import RUNTIME_MODULE_DEPENDENCIES

    identity = stage_identity(
        stage="tracking",
        command=[sys.executable, "-m", "cv.pipeline.tracking_composition"],
        inputs=[],
    )
    paths = {row["path"] for row in identity["code"]}
    for module in RUNTIME_MODULE_DEPENDENCIES:
        assert module.replace(".", "/") + ".py" in paths


def test_changing_a_runtime_dependency_invalidates_the_parent_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_cache, "REPOSITORY_ROOT", tmp_path)
    entry, child = tmp_path / "stage.py", tmp_path / "child.py"
    entry.write_text('RUNTIME_MODULE_DEPENDENCIES = ("child",)\n')
    child.write_text("VERSION = 1\n")
    kwargs = dict(stage="example", command=[sys.executable, str(entry)], inputs=[])
    first = stage_identity(**kwargs)["fingerprint"]
    child.write_text("VERSION = 2\n")
    assert stage_identity(**kwargs)["fingerprint"] != first


@pytest.mark.parametrize("declaration", ['("missing",)', "compute_dependencies()", "(1,)"])
def test_invalid_runtime_dependency_declaration_fails_closed(tmp_path, monkeypatch, declaration):
    monkeypatch.setattr(artifact_cache, "REPOSITORY_ROOT", tmp_path)
    entry = tmp_path / "stage.py"
    entry.write_text(f"RUNTIME_MODULE_DEPENDENCIES = {declaration}\n")
    with pytest.raises(ValueError):
        artifact_cache.repository_code_closure([entry])


def test_receipt_can_drop_unrelated_upstream_dependencies(tmp_path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    required = tmp_path / "required.json"
    unrelated = tmp_path / "unrelated.json"
    for path, contents in (
        (source, "source"),
        (output, "output"),
        (required, "required"),
        (unrelated, "unrelated"),
    ):
        path.write_text(contents)
    base = {
        "out_dir": tmp_path,
        "stage": "example",
        "command": ["true"],
        "inputs": [source],
        "outputs": [output],
    }
    write_stage_receipt(
        **base,
        upstream_receipts=[required, unrelated],
    )

    assert stage_receipt_allows_dependency_narrowing(
        **base,
        upstream_receipts=[required],
    )
    required.write_text("changed")
    assert not stage_receipt_allows_dependency_narrowing(
        **base,
        upstream_receipts=[required],
    )
