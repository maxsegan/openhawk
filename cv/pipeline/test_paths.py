from pathlib import Path

import pytest

from cv.pipeline.paths import data_relative, data_root, require_paths, tracker_root


def test_paths_use_environment_roots(monkeypatch, tmp_path: Path) -> None:
    data = tmp_path / "shared"
    tracker = tmp_path / "models"
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(data))
    monkeypatch.setenv("TENNIS_TRACKER_ROOT", str(tracker))

    assert data_root() == data
    assert tracker_root() == tracker
    assert data_relative(data / "processed" / "run") == "processed/run"


def test_require_paths_explains_external_configuration(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="TENNIS_DATA_ROOT"):
        require_paths({"source": tmp_path / "missing.mp4"})
