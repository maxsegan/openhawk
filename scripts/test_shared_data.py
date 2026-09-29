from pathlib import Path

from scripts import shared_data


def test_setup_moves_and_links_root_models(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    shared_root = tmp_path / "shared"
    repo.mkdir()
    (repo / "yolov8m.pt").write_bytes(b"weights")
    monkeypatch.setattr(shared_data, "tracked_layout", lambda _: (set(), set()))

    shared_data.migrate(repo, shared_root)
    shared_data.link(repo, shared_root)

    shared_model = shared_root / "models" / "pipeline" / "yolov8m.pt"
    local_model = repo / "yolov8m.pt"
    assert shared_model.read_bytes() == b"weights"
    assert local_model.is_symlink()
    assert local_model.resolve() == shared_model.resolve()


def test_link_replaces_identical_local_model(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    shared_root = tmp_path / "shared"
    repo.mkdir()
    shared_model = shared_root / "models" / "pipeline" / "yolov8m.pt"
    shared_model.parent.mkdir(parents=True)
    shared_model.write_bytes(b"weights")
    (repo / "yolov8m.pt").write_bytes(b"weights")
    monkeypatch.setattr(shared_data, "tracked_layout", lambda _: (set(), set()))

    shared_data.link(repo, shared_root)

    assert (repo / "yolov8m.pt").is_symlink()
    assert (repo / "yolov8m.pt").resolve() == shared_model.resolve()
