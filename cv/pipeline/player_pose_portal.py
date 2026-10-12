"""Write the match's 3D player pose into the staged portal (one layer per clip, bound per document).

For every source clip, the rows of ``player_pose_3d_v1.npz`` whose source clip and native frame
fall in the union of its documents' ``frame_range`` become a gzip layer with:

* ``players[side]``: the smoothed court root track per native frame (``frame``, ``t``, ``x``,
  ``y``, box ``conf``, ``track_id``, ``pose_status``); the viewer shows it in place of the
  document's sparse packet player states (whose ``names`` it keeps).
* ``skeletons[side]``: one compact row per native frame with any joint present,
  ``{"frame", "t", "j": [[x, y, z, status] x 17]}`` where x/y/z are integer centimetres
  relative to that frame's root (court axes, z up from the court) and status a
  ``player_pose_3d.STATUS`` code; a missing joint is ``null`` (2D confidences stay in the
  npz). ``skeleton_format`` declares the layout; the viewer expands it.

Times use the clip's native timebase (shared by its documents). Documents are rewritten
atomically; a document's own ``players``/``skeletons`` fields are left as exported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from cv.pipeline import player_pose_3d
from cv.viz import export_connected_3d as legacy

SIDES = player_pose_3d.SIDES
CAVEAT = (
    "Player skeletons are fitted 3D joints (stature-scaled limbs on a court-root camera model), "
    "not observations; faded joints are low-confidence or interpolated over a short gap."
)


def native_time(document: dict):
    timebase = document.get("native_timebase") or {}
    fps = float(timebase.get("fps") or document.get("fps") or 25.0)
    origin_t = float(timebase.get("origin_t") or 0.0)
    origin_frame = int(timebase.get("origin_frame") or 1)
    return lambda frame: round(origin_t + (int(frame) - origin_frame) / fps, 6)


def skeleton_format(names) -> dict:
    return {
        "schema": "player_pose_3d_compact_v1",
        "joint_names": [str(name) for name in names],
        "row": "[x_cm, y_cm, z_cm, status] relative to the frame's root",
        "status_codes": player_pose_3d.STATUS,
        "fitted": True,
    }


def document_layers(document: dict, pose: dict) -> tuple[dict, dict]:
    """``players`` and ``skeletons`` for one document from the loaded pose arrays."""
    clip = document.get("source_clip")
    lo, hi = (int(v) for v in document["frame_range"])
    time_of = native_time(document)
    names = (document.get("players") or {}).get("names") or {"near": None, "far": None}
    players: dict = {
        "names": names,
        "source": f"{player_pose_3d.SCHEMA} (automatic, smoothed box roots)",
    }
    skeletons: dict = {}
    mask = (pose["clip"] == clip) & (pose["frame"] >= lo) & (pose["frame"] <= hi)
    for side_index, side in enumerate(SIDES):
        rows = np.flatnonzero(mask & (pose["side"] == side_index))
        rows = rows[np.argsort(pose["frame"][rows], kind="stable")]
        track = {
            key: [] for key in ("frame", "t", "conf", "track_id", "pose_status", "root_jump_m")
        }
        # Court-plane metres from the lifter's root (not image pixels): kept apart from the
        # pixel-column contract and attached to the viewer's track under its "x"/"y" names.
        court_x, court_y = [], []
        skeleton_rows = []
        for i in rows:
            root = pose["root_xy"][i]
            if not np.isfinite(root).all():
                continue
            frame = int(pose["frame"][i])
            status = pose["joint_status"][i]
            present = status > 0
            track["frame"].append(frame)
            track["t"].append(time_of(frame))
            court_x.append(round(float(root[0]), 3))
            court_y.append(round(float(root[1]), 3))
            track["conf"].append(round(float(pose["box_conf"][i]), 3))
            track["track_id"].append(str(pose["track_id"][i]))
            track["pose_status"].append("pose" if present.any() else "no_pose_at_frame")
            track["root_jump_m"].append(None)
            if not present.any():
                continue
            joints = pose["joints_3d"][i]
            compact = []
            for k in range(len(status)):
                if not present[k] or not np.isfinite(joints[k]).all():
                    compact.append(None)
                    continue
                compact.append(
                    [
                        int(round((float(joints[k][0]) - float(root[0])) * 100)),
                        int(round((float(joints[k][1]) - float(root[1])) * 100)),
                        int(round(float(joints[k][2]) * 100)),
                        int(status[k]),
                    ]
                )
            skeleton_rows.append({"frame": frame, "t": time_of(frame), "j": compact})
        track.update(x=court_x, y=court_y)
        players[side] = track
        skeletons[side] = skeleton_rows
    return players, skeletons


def load(npz_path: Path) -> dict:
    return player_pose_3d.load_pose(npz_path)


def fill(portal: Path, npz_path: Path, receipt: dict | None = None) -> dict:
    """Write one gzip pose layer per source clip and bind every staged document to it.

    A parent attempt and its components share one window, so the layer is stored once per
    clip under ``data/pose/<clip>.json.gz`` (union of the documents' frame ranges) and each
    document gets ``pose_layer`` (its file, sha256 and frame range), ``skeleton_format`` and
    ``skeleton_caveat``. The viewer fetches the layer and slices it to the document's frames.
    """
    pose = load(npz_path)
    documents = player_pose_3d.portal_documents(portal)
    loaded = []
    windows: dict[str, dict] = {}
    for path in documents:
        document = json.loads(path.read_text())
        clip, frames = document.get("source_clip"), document.get("frame_range")
        if not clip or not frames:
            continue
        loaded.append((path, document))
        lo, hi = int(frames[0]), int(frames[1])
        if clip in windows:
            lo, hi = (
                min(lo, windows[clip]["frame_range"][0]),
                max(hi, windows[clip]["frame_range"][1]),
            )
            windows[clip]["frame_range"] = [lo, hi]
        else:
            windows[clip] = {"frame_range": [lo, hi], "document": document}
    layer_dir = portal / "data" / "pose"
    layer_dir.mkdir(parents=True, exist_ok=True)
    layers = {}
    layer_bytes = 0
    rows = 0
    format_ = skeleton_format(pose["joint_names"])
    for clip, window in sorted(windows.items()):
        scope = {**window["document"], "frame_range": window["frame_range"]}
        players, skeletons = document_layers(scope, pose)
        payload = {
            "schema": "player_pose_3d_layer_v1",
            "source_clip": clip,
            "frame_range": window["frame_range"],
            "skeleton_format": format_,
            "players": {side: players[side] for side in SIDES},
            # Near/far names as player_names decided them in the portal step, or nulls.
            "names": players["names"],
            "skeletons": skeletons,
        }
        path = layer_dir / f"{clip}.json.gz"
        temporary = path.with_suffix(".tmp")
        raw = json.dumps(payload, separators=(",", ":")).encode()
        temporary.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
        temporary.replace(path)
        layers[clip] = {
            "file": f"pose/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        layer_bytes += path.stat().st_size
        rows += sum(len(skeletons[side]) for side in SIDES)
    for path, document in loaded:
        layer = layers[document["source_clip"]]
        document["pose_layer"] = {
            "schema": "player_pose_3d_layer_v1",
            **layer,
            "frame_range": document["frame_range"],
            **({"pose_npz_sha256": receipt["output"]["sha256"]} if receipt else {}),
        }
        document["skeleton_format"] = format_
        document["skeleton_caveat"] = CAVEAT
        legacy._write_json(path, document)
    return {
        "filled": len(loaded),
        "skipped": len(documents) - len(loaded),
        "layers": len(layers),
        "skeleton_rows": rows,
        "layer_bytes": layer_bytes,
    }
