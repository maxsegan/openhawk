"""Install the static 3D flight viewer (HTML + JavaScript + vendored three.js) into a folder.

The product runner's ``portal`` step calls :func:`copy_static` and then writes one point
document per attempt under ``DEST/data/``. To view a run, serve that folder over HTTP:

    python -m cv.viz.portal_3d_src.build --output OUT/viewer
    python -m http.server -d RUN_OUT/portal/3d 8000   # then open http://localhost:8000/

Public-release version: the private repository's builder also assembled and deployed a
review portal from internal collections; only the static install is kept here.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent
STATIC = ("index.html", "viewer.js", "viewer_core.js")
VENDOR = ("three.module.js", "OrbitControls.js", "THREE_LICENSE.txt", "VERSION.txt")
HASHED = (
    "viewer.js",
    "viewer_core.js",
    "vendor/three.module.js",
    "vendor/OrbitControls.js",
)


def _build_hash() -> str:
    digest = hashlib.sha1()
    for relative in HASHED:
        digest.update((SRC / relative).read_bytes())
    return digest.hexdigest()[:12]


def _stamp_dest_files(version: str, destination: Path) -> None:
    viewer = destination / "viewer.js"
    text = viewer.read_text(encoding="utf-8")
    text = text.replace("'./viewer_core.js'", f"'./viewer_core.js?v={version}'")
    text = text.replace("'./vendor/OrbitControls.js'", f"'./vendor/OrbitControls.js?v={version}'")
    viewer.write_text(text, encoding="utf-8")

    index = destination / "index.html"
    text = index.read_text(encoding="utf-8")
    text = text.replace('src="viewer.js"', f'src="viewer.js?v={version}"')
    text = text.replace('"./vendor/three.module.js"', f'"./vendor/three.module.js?v={version}"')
    index.write_text(text, encoding="utf-8")


def copy_static(destination: Path) -> str:
    (destination / "vendor").mkdir(parents=True, exist_ok=True)
    (destination / "data").mkdir(parents=True, exist_ok=True)
    for name in STATIC:
        shutil.copy2(SRC / name, destination / name)
    for name in VENDOR:
        shutil.copy2(SRC / "vendor" / name, destination / "vendor" / name)
    version = _build_hash()
    _stamp_dest_files(version, destination)
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    version = copy_static(args.output.resolve())
    print(f"viewer {version}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
