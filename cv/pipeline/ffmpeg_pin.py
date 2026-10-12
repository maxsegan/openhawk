"""One ffmpeg build on every box (the product queue's pin).

Two ffmpeg builds write different JPEG bytes for the same frames: another machine's /usr/bin
6.1.1 against this machine's linuxbrew 8.0.1 moved 1-3% of ball candidates. The pin is
linuxbrew's 8.0.1_4 bottle with every library it loads, glibc and its loader included, in one
folder (``bin/ffmpeg`` and ``bin/ffprobe`` are wrappers that start the bundled loader), so every
box runs the same machine code. On a 2-min pilot span it is byte-identical to local
production's ffmpeg on all 842 outputs of the stages' command forms, on this machine and on the
5090, 3090 and 3080 boxes, and with SIMD limited to AVX2 or SSE4.2
(cv/experiments/gpu_decode/ffmpeg_parity.py, FINAL_REPORT_gpu_decode.md §10).

The queue puts ``<bundle>/bin`` first on PATH for every step (the stages call ``ffmpeg`` and
``ffprobe`` by name) and refuses a box whose bundle fails :func:`verify`. Step markers record
:func:`identity` of the ffmpeg each step ran.

    python -m cv.pipeline.ffmpeg_pin verify [DIR]     # default: the local bundle
    python -m cv.pipeline.ffmpeg_pin identity         # the ffmpeg on PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from functools import cache
from pathlib import Path

#: Folder name of the pinned bundle and the sha256 of its MANIFEST.sha256 (one line per file).
NAME = "ffmpeg-8.0.1_4-d959337b"
MANIFEST_SHA256 = "d959337b2d041bf8e8649301a44fa21af7fc5efadbcac2068edd1d609e734652"


def local_dir() -> Path:
    """The bundle on this machine (the copy ``product_boxes setup`` pushes to the boxes)."""
    data = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
    return data / "tools" / NAME


def verify_script(directory: str) -> str:
    """Shell that exits 0 only when ``directory`` holds the pinned bundle, every file intact."""
    return (
        f"cd {shlex.quote(directory)} && sha256sum -c --quiet --strict MANIFEST.sha256 && "
        f'test "$(sha256sum MANIFEST.sha256 | cut -d" " -f1)" = {MANIFEST_SHA256}'
    )


def verify(directory: Path | None = None) -> bool:
    directory = Path(directory or local_dir())
    if not (directory / "MANIFEST.sha256").is_file():
        return False
    return subprocess.run(["bash", "-c", verify_script(str(directory))]).returncode == 0


def with_pinned_path(environment: dict[str, str], directory: str | Path) -> dict[str, str]:
    """``environment`` with ``<directory>/bin`` first on PATH."""
    path = environment.get("PATH", os.defpath)
    return {**environment, "PATH": f"{directory}/bin:{path}"}


@cache
def identity(ffmpeg: str = "ffmpeg") -> dict:
    """Which ffmpeg a step runs: path, version line, and the pinned bundle's digest or the
    binary's sha256 when it is not a bundle."""
    found = shutil.which(ffmpeg)
    if found is None:
        return {"path": None}
    real = Path(found).resolve()
    record: dict = {"path": str(real)}
    try:
        done = subprocess.run([str(real), "-version"], capture_output=True, text=True, timeout=30)
        record["version"] = (done.stdout.splitlines() or [None])[0]
    except (OSError, subprocess.SubprocessError):
        record["version"] = None
    manifest = real.parent.parent / "MANIFEST.sha256"
    if manifest.is_file():
        record["bundle_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        record["pinned"] = record["bundle_manifest_sha256"] == MANIFEST_SHA256
    else:
        record["binary_sha256"] = hashlib.sha256(real.read_bytes()).hexdigest()
        record["pinned"] = False
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("directory", nargs="?", type=Path)
    sub.add_parser("identity")
    args = parser.parse_args()
    if args.command == "verify":
        ok = verify(args.directory)
        print(json.dumps({"directory": str(args.directory or local_dir()), "ok": ok}))
        raise SystemExit(0 if ok else 1)
    print(json.dumps(identity()))


if __name__ == "__main__":
    main()
