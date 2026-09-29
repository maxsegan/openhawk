"""Shot view classification: play camera vs replay, close-up, crowd, graphic.

Stage 1 only wants attempts from the behind-baseline play camera.  ``camera.py`` decides this
by k-means over 32x18 thumbnails at 1 fps, which cannot tell a tight player close-up from a
crowd shot and cannot see a two-second graphic at all.  Here every shot from
``shot_boundaries.py`` is represented by frozen DINOv3 ViT-S/16 embeddings of a few of its
frames plus its native-cadence motion statistics, and a logistic head maps that to one of
five view classes.

The embedding backbone is frozen and read from the local timm cache; only the logistic head
is fitted, on development-window shots labelled automatically by the local VLM.  Slow-motion
replay is separable from live play by inter-frame displacement, so the shot motion median,
p90 and near-duplicate fraction are appended to the embedding as explicit features.

RESOLUTION NOTE.  The backbone sees a 256x256 resize of the whole frame.  A whole-frame view
decision is a scene-type question, not a geometric measurement, and it never becomes tracking,
court, player or ball evidence; every geometric consumer downstream reads native pixels.  The
manifest records this as an explicit sub-native observation proxy, like the scoreboard frames.

    .venv/bin/python -m cv.pipeline.view_classifier predict --out out_dir --model head.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

VIEW_CLASSES = ("play", "replay", "closeup", "crowd", "graphic")
BACKBONE = "vit_small_patch16_dinov3.lvd1689m"
EMBED_SIZE = 256
MOTION_FEATURES = (
    "motion_median",
    "motion_p90",
    "motion_top_median",
    "motion_bottom_median",
    "duplicate_fraction",
)
VLM_URL = os.environ.get("VLM_URL", "http://localhost:8399/v1/chat/completions")

LABEL_PROMPT = """These are two frames from a tennis broadcast, half a second apart (the first
image is earlier). Classify the CAMERA VIEW of the shot they come from.
Answer with STRICT JSON only: {"view": "<one of play, replay, closeup, crowd, graphic>"}

- "play": THE main wide court camera used for live play - high above one baseline, looking
  down the length of the court, with the net across the middle and both service boxes visible.
  Use this even when nobody is hitting (players walking, towelling, bouncing the ball before a
  serve) as long as it is that high wide camera and the picture runs at normal speed.
  A LOW camera at courtside or player height is NOT "play", however much court it shows.
- "replay": a repeat of an action just played. Two strong signs: (i) the two frames are almost
  identical because the picture is in slow motion, while the players are clearly mid-action;
  (ii) an angle the live coverage does not use for play - net level, ground level behind the
  server, low courtside, tight on one player mid-stroke - or a ball-tracking animation, a
  "REPLAY" bug, or a wipe.
- "closeup": a tight shot of one or two people (player, coach, umpire, box), face or torso
  filling a large part of the frame, at normal speed.
- "crowd": spectators, stadium wide or aerial shots, ball kids, empty-court beauty shots, or
  any other live normal-speed camera that is neither the main play camera nor a person
  close-up - including low courtside cameras showing players between points.
- "graphic": a full-screen or near-full-screen title card, statistics table, sponsor slate,
  advertisement, or player-comparison graphic.

Decide "play" only for the main high wide court camera at normal speed. If the two frames are
nearly identical but the players are mid-stroke, prefer "replay".
Answer with JSON only."""


# ---------------- native frame sampling ----------------------------------------------------


def extract_native_frames(
    video: Path,
    out_dir: Path,
    *,
    fps: float,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    quality: int = 4,
) -> list[Path]:
    """Sample the span at --fps and write full 1920x1080 JPEGs (no downscale on disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("n_*.jpg"))
    if existing:
        return existing
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if start_seconds:
        command += ["-ss", f"{start_seconds:.6f}"]
    command += ["-i", str(video)]
    if duration_seconds is not None:
        command += ["-t", f"{duration_seconds:.6f}"]
    command += [
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(quality),
        str(out_dir / "n_%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    return sorted(out_dir.glob("n_*.jpg"))


def frame_times(count: int, *, fps: float, start_seconds: float) -> np.ndarray:
    return start_seconds + (np.arange(count, dtype=np.float64) + 0.5) / fps


# ---------------- frozen embedding ---------------------------------------------------------


def load_backbone(device: str = "cuda"):
    import timm
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.eval().to(device)
    if device.startswith("cuda"):
        model = model.half()
    return model, torch


def embed_frames(paths: list[Path], *, device: str = "cuda", batch: int = 128) -> np.ndarray:
    """Frozen DINOv3 embeddings, one row per frame (see the resolution note in the docstring)."""
    model, torch = load_backbone(device)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def read(path: Path) -> np.ndarray:
        image = cv2.imread(os.fspath(path))
        if image is None:
            raise ValueError(f"unreadable frame: {path}")
        image = cv2.resize(image, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - mean) / std).transpose(2, 0, 1)

    output = np.empty((len(paths), model.num_features), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=16) as pool, torch.no_grad():
        for start in range(0, len(paths), batch):
            chunk = list(pool.map(read, paths[start : start + batch]))
            tensor = torch.from_numpy(np.stack(chunk)).to(device)
            if device.startswith("cuda"):
                tensor = tensor.half()
            output[start : start + batch] = model(tensor).float().cpu().numpy()
    return output


# ---------------- shot representation ------------------------------------------------------


def shot_frame_indices(
    shots: list[dict], times: np.ndarray, *, per_shot: int = 4
) -> list[list[int]]:
    """Frame indices sampled inside each shot, avoiding the transition frames at its edges."""
    picks = []
    for shot in shots:
        start, end = float(shot["t_start"]), float(shot["t_end"])
        inset = min(0.25, 0.15 * (end - start))
        inside = np.where((times >= start + inset) & (times <= end - inset))[0]
        if not len(inside):
            centre = int(np.argmin(np.abs(times - 0.5 * (start + end))))
            picks.append([centre])
            continue
        chosen = np.unique(
            np.round(np.linspace(0, len(inside) - 1, min(per_shot, len(inside)))).astype(int)
        )
        picks.append([int(inside[index]) for index in chosen])
    return picks


def shot_features(shots: list[dict], embeddings: np.ndarray, picks: list[list[int]]) -> np.ndarray:
    """One row per shot: mean frame embedding, motion statistics, and their broadcast ratios.

    Slow-motion replay is separable from live play by how far things move between frames, but
    only relative to the broadcast: absolute motion energy varies threefold across sources.
    The last three columns are the shot's motion against this broadcast's own median, which is
    what makes a slow-motion repeat of a rally distinguishable from the rally.
    """
    reference = float(np.median([float(shot["motion_median"]) for shot in shots])) if shots else 1.0
    reference_high = (
        float(np.median([float(shot["motion_p90"]) for shot in shots])) if shots else 1.0
    )
    rows = []
    for shot, indices in zip(shots, picks):
        pooled = embeddings[indices].mean(axis=0)
        motion = np.array([float(shot[name]) for name in MOTION_FEATURES], dtype=np.float32)
        ratios = np.array(
            [
                float(shot["motion_median"]) / max(1e-3, reference),
                float(shot["motion_p90"]) / max(1e-3, reference_high),
                float(shot["frames"])
                / max(1.0, np.median([float(row["frames"]) for row in shots])),
            ],
            dtype=np.float32,
        )
        motion = np.concatenate([np.log1p(motion[:4]), motion[4:]])
        rows.append(
            np.concatenate([pooled, motion, [np.log1p(float(shot["frames"]))], np.log1p(ratios)])
        )
    return np.asarray(rows, dtype=np.float32)


# ---------------- automatic labelling via the local VLM ------------------------------------


def _query_view(paths: list[Path], index: int, timeout: float = 180.0) -> tuple[int, str | None]:
    """Label one shot from a pair of frames half a second apart.

    A single still cannot show slow motion, which is why the first pass found six replays in
    1107 shots.  Two frames can: in a slow-motion replay the players barely move between them
    while they are clearly mid-stroke.
    """
    content = []
    for path in paths:
        image = cv2.imread(os.fspath(path))
        if image is None:
            return index, None
        scaled = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        payload = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(payload).decode()
                },
            }
        )
    content.append({"type": "text", "text": LABEL_PROMPT})
    from cv.pipeline.vlm_policy import configured_vlm_model, reasoning_request_options

    model = configured_vlm_model()
    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": content}],
    }
    body.update(reasoning_request_options(model, "none", answer_tokens=64))
    try:
        request = urllib.request.Request(
            VLM_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = json.load(response)["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 - a server hiccup leaves this shot unlabelled
        return index, None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return index, None
    try:
        view = str(json.loads(match.group()).get("view", "")).strip().lower()
    except json.JSONDecodeError:
        return index, None
    return index, view if view in VIEW_CLASSES else None


def label_shots_with_vlm(
    frame_paths: list[Path], picks: list[list[int]], *, concurrency: int = 12
) -> list[str | None]:
    """One VLM read per shot, on its middle sampled frame. Training labels only."""
    pairs = []
    for indices in picks:
        middle = indices[len(indices) // 2]
        follower = min(middle + 1, len(frame_paths) - 1, max(indices))
        if follower == middle and middle > 0:
            follower = middle - 1
        pairs.append(sorted({middle, follower}))
    labels: list[str | None] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index, view in pool.map(
            lambda item: _query_view([frame_paths[j] for j in item[1]], item[0]),
            list(enumerate(pairs)),
        ):
            labels[index] = view
    return labels


# ---------------- logistic head ------------------------------------------------------------


class ViewHead:
    """Multinomial logistic head over frozen features; the only fitted part of this stage."""

    def __init__(self, classes: list[str], mean, scale, coefficients, intercept) -> None:
        self.classes = classes
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.coefficients = np.asarray(coefficients, dtype=np.float32)
        self.intercept = np.asarray(intercept, dtype=np.float32)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: list[str],
        *,
        seed: int = 0,
        class_weight: str | dict | None = None,
    ) -> "ViewHead":
        """Fit the head.  ``class_weight="balanced"`` is the replay-recall arm.

        Replay is 78 of 5,887 labelled shots, so an unweighted fit recovers 0.269 of them
        leave-one-broadcast-out.  Balancing the classes trades play and crowd recall for
        replay recall; the numbers are in ``cv.validation.s1_attempt_precision``.  The
        default is unweighted, which is the head that shipped.
        """
        from sklearn.linear_model import LogisticRegression

        mean = features.mean(axis=0)
        scale = features.std(axis=0) + 1e-6
        model = LogisticRegression(
            max_iter=4000, C=1.0, random_state=seed, class_weight=class_weight
        )
        model.fit((features - mean) / scale, labels)
        return cls(list(model.classes_), mean, scale, model.coef_, model.intercept_)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        scores = ((features - self.mean) / self.scale) @ self.coefficients.T + self.intercept
        if scores.shape[1] == 1:  # binary fallback
            scores = np.hstack([-scores, scores])
        scores -= scores.max(axis=1, keepdims=True)
        exponent = np.exp(scores)
        return exponent / exponent.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> list[str]:
        probabilities = self.probabilities(features)
        return [self.classes[index] for index in probabilities.argmax(axis=1)]

    def to_json(self) -> dict:
        return {
            "schema": "view_classifier_head_v1",
            "backbone": BACKBONE,
            "classes": self.classes,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
        }

    @classmethod
    def from_json(cls, document: dict) -> "ViewHead":
        return cls(
            document["classes"],
            document["mean"],
            document["scale"],
            document["coefficients"],
            document["intercept"],
        )


def read_shots(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_views(out_dir: Path, shots: list[dict], views: list[str], probabilities) -> Path:
    path = out_dir / "shot_views_v1.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot_index", "t_start", "t_end", "view", *VIEW_CLASSES])
        for shot, view, row in zip(shots, views, probabilities):
            writer.writerow(
                [
                    shot["shot_index"],
                    f"{float(shot['t_start']):.3f}",
                    f"{float(shot['t_end']):.3f}",
                    view,
                    *[f"{value:.4f}" for value in row],
                ]
            )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("frames", "embed", "label", "predict"))
    parser.add_argument("--out", type=Path, required=True, help="per-span artifact directory")
    parser.add_argument("--model", type=Path, help="trained head JSON (predict)")
    parser.add_argument("--video", type=Path, help="source broadcast (frames)")
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "frames":
        if args.video is None:
            parser.error("--video is required for frames")
        written = extract_native_frames(
            args.video,
            args.out / "frames_native",
            fps=args.frame_fps,
            start_seconds=args.start,
            duration_seconds=args.duration,
        )
        (args.out / "frames_native.json").write_text(
            json.dumps(
                {
                    "schema": "native_sampled_frames_v1",
                    "fps": args.frame_fps,
                    "start_seconds": args.start,
                    "frames": len(written),
                    "resolution_status": "NATIVE",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"{len(written)} native frames -> {args.out / 'frames_native'}")
        return 0
    shots = read_shots(args.out / "shot_boundaries_v1.csv")
    frames = sorted((args.out / "frames_native").glob("n_*.jpg"))
    meta = json.loads((args.out / "frames_native.json").read_text())
    times = frame_times(len(frames), fps=meta["fps"], start_seconds=meta["start_seconds"])
    picks = shot_frame_indices(shots, times)
    if args.action == "embed":
        embeddings = embed_frames(frames, device=args.device)
        np.savez_compressed(args.out / "frame_embeddings_v1.npz", embeddings=embeddings)
        print(f"{embeddings.shape} -> {args.out / 'frame_embeddings_v1.npz'}")
        return 0
    embeddings = np.load(args.out / "frame_embeddings_v1.npz")["embeddings"]
    if args.action == "label":
        labels = label_shots_with_vlm(frames, picks)
        (args.out / "shot_view_labels_v1.json").write_text(
            json.dumps({"schema": "shot_view_vlm_labels_v1", "labels": labels}, indent=2) + "\n"
        )
        print(f"labelled {sum(label is not None for label in labels)}/{len(labels)} shots")
        return 0
    head = ViewHead.from_json(json.loads(args.model.read_text()))
    features = shot_features(shots, embeddings, picks)
    probabilities = head.probabilities(features)
    ordered = [
        [
            float(probabilities[row][head.classes.index(name)]) if name in head.classes else 0.0
            for name in VIEW_CLASSES
        ]
        for row in range(len(shots))
    ]
    path = write_views(args.out, shots, head.predict(features), ordered)
    print(f"{len(shots)} shots -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
