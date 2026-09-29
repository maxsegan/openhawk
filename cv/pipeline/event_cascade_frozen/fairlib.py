"""Shared paths, nominations, prompt text and the 60-frame matcher.

The matcher is the one in ``score_labellers_multi.py``: greedy one-to-one, same
event type, within 60 frames. Abstain / no_event / none emit nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CROP_NATIVE = 160
ZOOM = 8
RADIUS = 12
CAP = 60
COLUMNS = 5

WORKTREE = Path(__file__).resolve().parents[3]
RULES = WORKTREE / ".claude/skills/tennis-event-label/references/decision_rules.md"
SHOTS = WORKTREE / "cv/validation/labels/s5_reasoned_nshot_v1/shots.json"
BENCHMARK = Path("data/processed/vlm_event_benchmark_v1/labeller_axis_workdir")
SCRIPTS = Path("data/processed/vlm_event_benchmark_v1/scripts")
CORPUS = Path("data/processed/postseg_pipeline_benchmark_0dc3bbb_clean_v1")
DEMO_PACK = Path(
    "data/processed/trajectory_summary_vlm_v1"
    "/s4_s5_sequence_residual_exploration_v2"
    "/s5_development_visual_candidate_pack_v11_sequence_audio"
)

# Frozen in PROTOCOL.md before any new score.
SLICE_CLIPS = (
    "rg2023f_w_swiatek_muchova__pt0004",
    "uso2020f_m_zverev_thiem__pt0001",
    "cincy2024f_m_sinner_tiafoe__pt0001",
    "uso2020f_m_zverev_thiem__pt0004",
)

EMIT_TYPES = ("contact", "bounce", "net_hit")
REJECT_TYPES = ("none", "no_event", "abstain")


def art_root() -> Path:
    root = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
    path = root / "processed" / "labeller_fair_evidence_20260923"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_truth() -> dict:
    return json.loads((BENCHMARK / "heldback2_truth.json").read_text())


def load_shots() -> list[dict]:
    return json.loads(SHOTS.read_text())["shots"]


def assert_demos_disjoint(shots: list[dict] | None = None, truth: dict | None = None) -> None:
    """Fail closed. A demo clip that is also a benchmark clip leaks the lesson."""
    shots = load_shots() if shots is None else shots
    truth = load_truth() if truth is None else truth
    demo = {str(shot["clip"]) for shot in shots}
    bench = set(truth)
    hit = sorted(demo & bench)
    if hit:
        raise SystemExit(f"demo clips overlap benchmark clips: {hit}")
    missing = [shot["case_id"] for shot in shots if "clip" not in shot or "reasoning" not in shot]
    if missing:
        raise SystemExit(f"demo shots missing clip or reasoning: {missing}")


def nominations(clips: list[str] | None = None) -> list[dict]:
    truth = load_truth()
    wanted = list(clips) if clips else sorted(truth)
    rows = []
    for key in wanted:
        if key not in truth:
            raise SystemExit(f"unknown clip {key}")
        manifest = json.loads((BENCHMARK / "refine" / key / "refine_manifest.json").read_text())
        match, point = key.split("__", 1)
        for board in manifest["storyboards"]:
            rows.append({
                "clip": key,
                "match": match,
                "point": point,
                "id": board["id"],
                "nominated_frame": int(board["nominated_frame"]),
                "nominated_type": board.get("nominated_type"),
                "old_panel_frames": board.get("panel_frames"),
            })
    return rows


def frames_dir(match: str, point: str) -> Path:
    return CORPUS / match / "audit_frames_native_1080" / point


def track_csv(match: str) -> Path:
    return CORPUS / match / "ball_track_joint_native1080_arc_augmented_v2.csv"


def decision_rules() -> str:
    return RULES.read_text().strip()


def task_preamble(*, presentation: str) -> str:
    trail = ""
    if presentation == "trail":
        trail = (
            "\nYou are then shown the SAME frames a second time, labelled "
            "FALLIBLE TRACKER HINT. Orange dots are earlier automatic-track positions "
            "and cyan dots are later ones, faded with distance. The nominated frame "
            "is left clean on purpose. The track is often wrong: a dot is not the ball, "
            "and the absence of a dot is not the absence of a ball. Judge the pixels.\n"
        )
    return f"""You are labelling one candidate tennis impact from native broadcast crops.

WHAT THE IMAGES ARE
Each crop is 160 native pixels around an automatic track, magnified ×8 with nearest-neighbour,
so one native pixel is an 8×8 block. The number in the top-left corner is the FRAME INDEX.
Nothing is drawn on the ball. Crops are centred on a tracker's estimate, which can drift,
so the ball may sit off-centre or leave the crop. Frames run from 12 before the nomination
to 12 after, skipping only frames the clip does not have.
The CLEAN images come first. Use those.{trail}
The nomination type and frame below are a hypothesis from an earlier automatic pass.
Retyping it or rejecting it is expected, not an error.

WHAT TO DECIDE
One physical event, or none, inside the frames you were shown:
  contact, bounce, net_hit — use the decision rules.
  no_event — the window supports no physical impact.
  abstain — the ball or the impact cannot be resolved. Do not guess.

Give the printed frame index of the impact. If the instant falls between two frames, give
the frame where the ball is closest to the racket or the court. For no_event or abstain,
repeat the nominated frame and set event_type accordingly.

{decision_rules()}

Return exactly one JSON object and nothing else:
{{"id":"<the id>","event_type":"contact|bounce|net_hit|no_event|abstain","frame":<int>,"confidence":"high|medium|low","evidence":"brief visible reason"}}
"""


def target_text(nom: dict) -> str:
    return (
        f"Unlabeled candidate id {nom['id']}. "
        f"Nominated as {nom['nominated_type']} near frame {nom['nominated_frame']}. "
        f"That nomination is not truth. Return the JSON object now."
    )
