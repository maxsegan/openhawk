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


def test_a_partly_regenerated_directory_matches_through_its_stored_listing(tmp_path, monkeypatch):
    import shutil

    from cv.pipeline import artifact_cache, provenance

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    frames = tmp_path / "processed" / "m" / "audit_frames_native_1080"
    for clip in ("pt0001", "pt0002", "pt0010"):
        (frames / clip).mkdir(parents=True)
        for n in (1, 2):
            (frames / clip / f"f_{n:04d}.jpg").write_bytes(f"{clip}-{n}".encode())
        (frames / clip / "extraction_receipt.json").write_text("{}")
    (frames / "pt0001.json").write_text("{}")  # string order differs from path order
    binding = artifact_cache.path_record(frames)
    rows = [
        (p.relative_to(frames).as_posix(), p.stat().st_size, provenance.file_sha256(p))
        for p in frames.rglob("*")
        if p.is_file()
    ]
    assert artifact_cache.listing_digest(rows) == binding["sha256"]
    listing = frames.with_name(frames.name + artifact_cache.LISTING_SUFFIX)
    listing.write_text("".join(f"{r}\t{s}\t{h}\n" for r, s, h in rows))
    shutil.rmtree(frames / "pt0002")
    assert artifact_cache.directory_matches(frames, binding)
    (frames / "pt0001" / "f_0001.jpg").write_bytes(b"changed")
    assert not artifact_cache.directory_matches(frames, binding)
    (frames / "pt0001" / "f_0001.jpg").write_bytes(b"pt0001-1")
    (frames / "pt0001" / "f_0003.jpg").write_bytes(b"added")
    assert not artifact_cache.directory_matches(frames, binding)
    (frames / "pt0001" / "f_0003.jpg").unlink()
    listing.write_text(listing.read_text().replace("pt0010/f_0002.jpg", "pt0010/f_0009.jpg"))
    assert not artifact_cache.directory_matches(frames, binding)


def test_hash_threads_give_the_same_directory_record(tmp_path, monkeypatch) -> None:
    from cv.pipeline import artifact_cache, provenance

    frames = tmp_path / "frames"
    for point in range(3):
        (frames / f"pt{point:04d}").mkdir(parents=True)
        for frame in range(7):
            (frames / f"pt{point:04d}" / f"f_{frame:04d}.jpg").write_bytes(
                bytes([point, frame]) * (1000 + frame)
            )
    monkeypatch.delenv("TENNIS_HASH_THREADS", raising=False)
    serial = artifact_cache.path_record(frames)
    artifact_cache._DIRECTORY_RECORDS.clear()
    provenance._HASH_CACHE.clear()
    monkeypatch.setenv("TENNIS_HASH_THREADS", "4")
    assert provenance.hash_threads() == 4
    assert artifact_cache.path_record(frames) == serial
    paths = sorted(frames.rglob("*.jpg"))
    assert provenance.bulk_sha256(paths) == [provenance.file_sha256(p) for p in paths]
