"""The additive reference cannot silently become owner truth or timed ball centres."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cv.validation.ball_streak_reference import compare_owner, compare_owner_events, validate

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "cv/validation/labels/s6_owner_inputs_v1/agent_streak_reference_v1.json"
BENCHMARK = ROOT / "cv/validation/ball_label_comparisons/s6_return_audit_20260906_v1/protocol.json"


def inputs():
    return json.loads(LABELS.read_text()), json.loads(BENCHMARK.read_text())


def test_complete_additive_reference_and_immutable_owner():
    labels, benchmark = inputs()
    validate(labels, benchmark)
    provenance = labels["provenance"]
    owner = provenance["owner_reference"]
    assert hashlib.sha256((ROOT / owner["path"]).read_bytes()).hexdigest() == owner["sha256"]
    assert provenance["owner_blind"] is False
    frames = labels["records"][0]["frames"]
    assert len(frames) == 9
    assert sum(row["streak"]["status"] == "paired" for row in frames) == 7
    assert sum(row["streak"]["status"] == "partial" for row in frames) == 1
    assert sum(row["streak"]["status"] == "ambiguous" for row in frames) == 1


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing",
        "fractional",
        "owner",
        "nan",
        "infinite_radius",
        "zero_radius",
        "outside",
        "false_pair",
        "invented_shutter",
        "changed_convention",
    ],
)
def test_reject_malformed_reference(change):
    labels, benchmark = inputs()
    frames = labels["records"][0]["frames"]
    if change == "duplicate":
        frames.append(copy.deepcopy(frames[-1]))
    elif change == "missing":
        frames.pop(2)
    elif change == "fractional":
        frames[0]["frame"] = 305.5
    elif change == "owner":
        labels["provenance"]["owner_verified"] = True
    elif change == "false_pair":
        frames[3]["streak"]["status"] = "paired"
    elif change == "invented_shutter":
        frames[0]["streak"]["exposure_duration_seconds"] = 0.02
    elif change == "changed_convention":
        frames[0]["streak"]["leading"]["x1080"] += 1
    else:
        field, value = {
            "nan": ("x1080", float("nan")),
            "outside": ("x1080", 1920),
            "infinite_radius": ("uncertainty_radius_px1080", float("inf")),
            "zero_radius": ("uncertainty_radius_px1080", 0),
        }[change]
        frames[0]["streak"]["trailing"][field] = value
    with pytest.raises(ValueError):
        validate(labels, benchmark)


def test_frozen_reference_checksum_and_opened_owner_agreement():
    assert hashlib.sha256(LABELS.read_bytes()).hexdigest() == (
        "830493db786d5e2f144b069093f68a2c85ec71fdd4c2d9182e9efd4f67520c62"
    )
    labels, _ = inputs()
    owner = ROOT / labels["provenance"]["owner_reference"]["path"]
    score = compare_owner(LABELS, owner, "pt0002_ball")
    assert score["denominator_frames"] == 9
    assert score["compared_fronts"] == score["within_px"]["6"] == 8
    assert score["max_error_px"] == pytest.approx(5.5839502146777695)
    assert score["owner_blind"] is False
    assert score["trailing_endpoints_validated"] is False
    assert score["visibility"]["precision"] == 1
    assert score["visibility"]["recall"] == pytest.approx(8 / 9)
    assert score["exact_window_max_error_px"] is None
    assert not any(score["complete_window_within_px"].values())


def test_owner_comparison_requires_freeze_before_opening_owner(tmp_path):
    labels, _ = inputs()
    labels["annotation_status"] = "draft"
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="freeze"):
        compare_owner(draft, tmp_path / "nonexistent_owner.json", "pt0002_ball")


def test_frozen_early_serve_reference_and_opened_owner_agreement():
    labels_path = LABELS.with_name("agent_serve_streak_reference_v1.json")
    benchmark_path = ROOT / (
        "cv/validation/ball_label_comparisons/s6_serve_streak_20260906_v1/benchmark.json"
    )
    assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == (
        "d563d9e78f8bde94218169b04a76a90096b03ba54837746d9a16a0a8b1f842cd"
    )
    labels = json.loads(labels_path.read_text())
    validate(labels, json.loads(benchmark_path.read_text()))
    assert labels["annotation_status"] == "frozen_agent_reference"
    assert labels["provenance"]["owner_verified"] is False
    manifest = ROOT / labels["source_manifest"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == labels["source_manifest_sha256"]
    assert len(json.loads(manifest.read_text())["source_bindings"]) == 9
    frames = labels["records"][0]["frames"]
    assert [r["frame"] for r in frames] == list(range(284, 293))
    assert all(r["streak"]["status"] == "paired" for r in frames)
    owner = ROOT / labels["provenance"]["owner_reference"]["path"]
    score = compare_owner(labels_path, owner, "pt0002_ball")
    assert score["denominator_frames"] == score["compared_fronts"] == 9
    assert score["within_px"] == {"6": 9, "12": 9, "24": 9}
    assert score["mean_error_px"] == pytest.approx(1.2367888757383867)
    assert score["max_error_px"] == pytest.approx(1.8469434208984061)
    assert score["owner_blind"] is False
    assert score["trailing_endpoints_validated"] is False
    assert score["visibility"]["precision"] == score["visibility"]["recall"] == 1
    assert score["emitted_coverage"] == 1
    assert score["exact_window_max_error_px"] == score["max_error_px"]
    assert all(score["complete_window_within_px"].values())
    assert score["clean_frame_damage"] is None


def test_attempt_comparison_keeps_missing_owner_frames_in_denominator(tmp_path):
    path = (
        ROOT
        / "cv/validation/labels/s6_agent_inputs_v1/ao2023f_w_sabalenka_rybakina_pt0002_owner_overlap.json"
    )
    payload = json.loads(path.read_text())
    payload["ball"]["records"][0]["frames"] = [
        r for r in payload["ball"]["records"][0]["frames"] if r["frame"] != 284
    ]
    changed = tmp_path / "missing.json"
    changed.write_text(json.dumps(payload))
    owner = ROOT / payload["provenance"]["owner_reference"]["path"]
    score = compare_owner(changed, owner, "pt0002_ball")
    assert score["denominator_frames"] == 52
    assert score["compared_fronts"] == 51
    assert score["within_fraction_all_owner_frames"]["6"] == pytest.approx(51 / 52)
    assert score["visibility"]["recall"] == pytest.approx(51 / 52)
    assert score["frames"][0]["agent_status"] == "missing"
    assert score["exact_window_max_error_px"] is None
    payload["match_id"] = "different_match"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="match/clip"):
        compare_owner(changed, owner, "pt0002_ball")


def test_event_agreement_keeps_misses_extras_and_endings_distinct(tmp_path):
    owner = tmp_path / "events.csv"
    owner.write_text(
        "clip,event_type,labeled_frame,seed_id\nm__pt0001,contact,10,a\n"
        "m__pt0001,bounce,12,b\nm__pt0001,point_end,12,c\n"
        "m__pt0001,contact,30,outside_window\nother__pt0001,contact,10,other\n"
    )
    payload = {
        "annotation_status": "frozen_agent_reference",
        "match_id": "m",
        "attempt": {"clip": "pt0001", "native_window": [1, 20]},
        "events": {
            "records": [
                {
                    "event_type": "contact",
                    "frame": 10.5,
                    "frame_interval": [10, 11],
                    "status": "labeled",
                },
                {
                    "event_type": "bounce",
                    "frame": 12,
                    "frame_interval": [11.5, 12.5],
                    "status": "labeled",
                },
                {
                    "event_type": "contact",
                    "frame": 18,
                    "frame_interval": [17.5, 18.5],
                    "status": "labeled",
                },
            ]
        },
    }
    labels = tmp_path / "agent.json"
    labels.write_text(json.dumps(payload))
    digest = hashlib.sha256(owner.read_bytes()).hexdigest()
    score = compare_owner_events(labels, owner, digest)
    assert score["denominator_events"] == 3
    assert score["matched_events"] == 2
    assert score["within_frames"] == {"0": 1, "0.5": 2, "1": 2, "2": 2}
    assert len(score["unmatched_agent_events"]) == 1
    assert score["events"][-1]["agent_frame"] is None
    assert score["events"][0]["owner_in_agent_interval"]
    with pytest.raises(ValueError, match="source changed"):
        compare_owner_events(labels, owner, "wrong")
    payload["annotation_status"] = "draft"
    labels.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="freeze"):
        compare_owner_events(labels, tmp_path / "unopened.csv", digest)
