from __future__ import annotations

from cv.viz.portal_3d_src import build


def test_copy_static_installs_a_versioned_viewer(tmp_path) -> None:
    version = build.copy_static(tmp_path)
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "vendor" / "three.module.js").is_file()
    assert f"viewer.js?v={version}" in (tmp_path / "index.html").read_text()
    assert f"viewer_core.js?v={version}" in (tmp_path / "viewer.js").read_text()
