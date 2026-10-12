from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cv.viz import export_connected_3d as export


def test_mixed_input_overlay_names_ball_stream_without_promoting_run():
    label = {
        "native_size": [1920, 1080],
        "stream_origins": {"ball": "automatic", "events": "labeled"},
        "ball_convention": "detector_heatmap_nominal_centre",
        "automatic_inference_eligible": False,
        "ball": {"records": [{"clip": "pt0001", "frames": []}]},
    }
    before = json.dumps(label, sort_keys=True)
    overlay = export._video_overlay(
        label, {"measurement": {"native_projection": []}}, "pt0001", 1, 2
    )
    assert overlay["labeled_marker"] == "blue dot: automatic observed native detector center"
    assert json.dumps(label, sort_keys=True) == before
    assert "automatic_inference_eligible" not in overlay


def test_automatic_addresses_preserve_quantized_original_pts():
    images = [
        dict(
            frame=i + 1,
            clip="pt0001",
            source_pts=pts,
            source_time_base="1/1000",
            native_pts_seconds=pts / 1000,
        )
        for i, pts in enumerate([1000, 1033, 1067, 1100])
    ]
    label = {"source_pack": {"fps": 30000 / 1001, "images": images}}
    original = json.dumps(label, sort_keys=True)
    metadata, timestamp = export._native_timebase(label)
    assert metadata["native_frames"] == [1, 2, 3, 4]
    assert metadata["origin_frame"] == 1
    assert [timestamp(i) for i in range(1, 5)] == [1.0, 1.033, 1.067, 1.1]
    assert timestamp(2.5) == pytest.approx(1.05)
    assert json.dumps(label, sort_keys=True) == original
    with pytest.raises(ValueError, match="outside"):
        timestamp(4.1)
    images[1]["native_pts_seconds"] = 1.034
    with pytest.raises(ValueError, match="aliases conflict"):
        export._native_timebase(label)
    images[1]["native_pts_seconds"] = 1.033
    images[1]["frame"] = 3
    with pytest.raises(ValueError, match="gap or duplicate"):
        export._native_timebase(label)
    images[1]["frame"] = 2
    del images[1]["source_pts"]
    with pytest.raises(ValueError, match="every picture"):
        export._native_timebase(label)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict:
    positions = [
        [4.0 + index * 0.1, 1.0 + index, max(0.0325, 2.8 - index * 0.25)] for index in range(11)
    ]
    velocities = [[2.5, 25.0, -4.0] for _index in range(11)]
    return {
        "depth_hypothesis_m": 0.5,
        "stage": "refined",
        "measurement": {
            "native_projection": [
                {"frame": 2, "predicted": [101.0, 201.0], "error_px": 1.4, "split": "training"},
                {"frame": 3, "predicted": [111.0, 211.0], "error_px": 2.4, "split": "withheld"},
            ],
            "rms_px": {"training": 1.4, "withheld": 2.4},
            "contact_xyz": [[4.0, 1.0, 2.8]],
            "dense_flights": [
                {
                    "start_frame": 2.0,
                    "end_frame": 4.0,
                    "start_xyz": positions[0],
                    "end_xyz": positions[-1],
                    "positions": positions,
                    "velocities": velocities,
                    "bounces": [
                        {
                            "frame": 3.0,
                            "x": [4.5, 6.0, 0.0325],
                            "v_in": [2.5, 25.0, -4.0],
                            "v_out": [2.0, 18.0, 3.0],
                            "regime": "grip",
                        }
                    ],
                    "net_hits": [],
                }
            ],
            "physical": {
                "flights": [{"modeled_bounce_frames": [3.0]}],
                "terminal_completion": {"impact_frame": 3.0, "end_xyz": [4.5, 6.0, 0.0325]},
            },
            "fit": {},
        },
        "evidence": {"survived": True, "checks": {}, "death_reasons": []},
        "wall_seconds": 0.1,
    }


def test_export_attempt_preserves_native_time_family_and_video_overlay(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data_root = tmp_path / "tennis-data"
    source_dir = data_root / "processed" / "frames" / "pt0001"
    source_dir.mkdir(parents=True)
    images = []
    for frame in range(1, 6):
        path = source_dir / f"f_{frame:04d}.jpg"
        path.write_bytes(f"native-frame-{frame}".encode())
        if 2 <= frame <= 4:
            images.append(
                {
                    "clip": "pt0001",
                    "frame": frame,
                    "image_url": f"processed/frames/pt0001/{path.name}",
                    "native_pts_seconds": (frame - 1) / 25,
                    "source": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                }
            )
    label = {
        "annotation_origin": "agent",
        "native_size": [1920, 1080],
        "source_pack": {"fps": 25.0, "images": images, "native_size": [1920, 1080]},
        "ball": {
            "records": [
                {
                    "clip": "pt0001",
                    "frames": [
                        {
                            "frame": frame,
                            "status": "visible",
                            "x1080": 100.0 + frame,
                            "y1080": 200.0 + frame,
                            "uncertainty_radius_px1080": 4,
                        }
                        for frame in range(2, 5)
                    ],
                }
            ]
        },
    }
    label_path = repo / "cv" / "validation" / "labels" / "s6_agent_inputs_v1" / "label.json"
    _write_json(label_path, label)

    camera_path = data_root / "processed" / "sweep" / "inputs" / "cameras.json"
    camera = {
        "schema": "camera",
        "transport_reference_policy": "contact_anchor",
        "supported": 3,
        "total": 3,
        "airborne_metric_accuracy_certified": False,
        "anchor": {
            "frame": 2,
            "fit": {
                "focal_native_px": 3000.0,
                "camera_center_m": [5.0, -20.0, 9.0],
                "native_rms_px": 2.0,
            },
        },
    }
    camera_sha = _write_json(camera_path, camera)
    candidate = _candidate()
    report = {
        "attempt_id": "s6_agent_inputs_v1__test_match_pt0001_attempt01",
        "clip": "pt0001",
        "selected_arms": {"combined_toss_and_serve_prior": candidate},
        "selected": candidate,
        "refined_candidates": [candidate],
        "events": [
            {
                "event_type": "contact",
                "frame": 2.0,
                "frame_interval": [1.5, 2.5],
                "annotation_origin": "agent",
            },
            {
                "event_type": "bounce",
                "frame": 3.0,
                "frame_interval": [2.5, 3.5],
                "annotation_origin": "agent",
            },
        ],
        "player_states": [
            {"frame": 2, "side": "near", "court_centre_xy_m": [4.0, -0.5]},
        ],
        "inputs": [
            {
                "path_base": "TENNIS_DATA_ROOT",
                "path": "processed/sweep/inputs/cameras.json",
                "sha256": camera_sha,
            }
        ],
    }
    report_path = data_root / "processed" / "sweep" / "search" / "report.json"
    _write_json(report_path, report)
    aggregate_path = data_root / "processed" / "sweep" / "report.json"
    aggregate_raw = {
        "human_derived": True,
        "automatic_inference_eligible": False,
        "independent_xyz_truth_available": False,
        "fixed_configuration_sha256": "f" * 64,
    }
    _write_json(aggregate_path, aggregate_raw)
    aggregate = {**aggregate_raw, "_path": str(aggregate_path)}
    attempt = {
        "key": "demo",
        "reconstructed": True,
        "blocking_bounds": [],
        "selected_or_diagnostic_legal_checks": {"connected_input_physics": True},
        "rms_px": {"training": 1.4, "withheld": 2.4},
        "family": {
            "count": 1,
            "member_depths_m": [0.5],
            "selected_depth_y_m": 0.5,
            "depth_y_range_m": [0.5, 0.5],
            "width_m": 0.0,
            "midpoint_m": 0.5,
        },
    }
    topology = {
        "match_id": "test_w_alpha_beta",
        "topology_in_words": "One-shot fixture.",
        "shot_count": 1,
        "server": "Alpha",
        "server_camera_half": "near",
    }
    out_dir = data_root / "processed" / "portal" / "3d-real" / "data"
    doc, entry = export.export_attempt(
        attempt,
        aggregate,
        topology,
        report_path,
        label_path,
        out_dir,
        data_root,
        repo,
        1,
    )

    assert doc["frame_range"] == [1, 5]
    assert doc["review_frame_window"]["labeled"] == [2, 4]
    assert doc["native_timebase"]["origin_t"] == 0.0
    assert doc["frames"]["t"] == [0.0, 0.04, 0.08, 0.12, 0.16]
    assert doc["frames"]["x"][0] is None
    assert doc["frames"]["x"][1] == 4.0
    assert doc["quality"]["accepted"] is True
    assert doc["family"]["selected_depth_y_m"] == 0.5
    assert doc["video_overlay"]["frames"][1]["labeled_front"] == [102.0, 202.0]
    assert doc["video_overlay"]["frames"][1]["fitted_projection"] == [101.0, 201.0]
    assert len(doc["video_overlay"]["frames"]) == 5
    assert doc["timeline_ticks"] == [
        {"frame": 2.0, "t": 0.04, "kind": "contact", "source": "fit"},
        {"frame": 2.0, "t": 0.04, "kind": "contact", "source": "label"},
        {"frame": 3.0, "t": 0.08, "kind": "bounce", "source": "label"},
    ]
    assert doc["contacts"][0]["labeled_frame"] == 2.0
    assert doc["bounces"][0]["fitted_frame"] == 3.0
    assert doc["players"]["near"]["x"] == [4.0]
    assert entry["verdict"] == "reconstructed, owner-reviewed"
    assert doc["quality"]["footnote"].count("production") == 1
    assert (out_dir.parent / "frames" / "demo" / "f_0001.jpg").read_bytes() == b"native-frame-1"
    assert json.loads((out_dir / "connected_demo.json").read_text())["schema"] == "point3d_v1"


def test_held_blocker_text_reports_quantitative_excess() -> None:
    text = export._blocker_text(
        {
            "blocking_bounds": ["all_bounce_rays_agree", "all_player_reaches_plausible"],
            "bounce_ray_margin_m": -0.063,
            "player_reach_margins_m": [0.2, -1.25],
        }
    )
    assert "0.063 m" in text
    assert "1.250 m" in text
    assert "player reach" in text


def test_lifted_world_joints_become_local_xyz_without_flattening_depth() -> None:
    joints = {
        "left_ankle": (4.9, 7.8, 0.0, 0.9),
        "right_wrist": (5.6, 7.1, 1.2, 0.8),
    }
    local = export._local_joints(joints, (5.0, 8.0))
    assert local["left_ankle"] == [-0.1, -0.2, 0.0, 0.9]
    assert local["right_wrist"] == [0.6, -0.9, 1.2, 0.8]


def test_only_accepted_contacts_can_override_racket_face() -> None:
    contacts = [
        {"frame": 20.5, "side": "near", "x": 4.0, "y": -1.0, "z": 2.8},
        {"frame": 30.0, "side": "unknown", "x": 5.0, "y": 20.0, "z": 1.0},
    ]
    assert export._contact_frames_by_side(contacts, False) == {}
    accepted = export._contact_frames_by_side(contacts, True)
    assert accepted == {(21, "near"): contacts[0]}


def test_player_binding_accepts_native_sided_files_at_any_cadence() -> None:
    binding = {
        "path": "processed/demo/player_boxes_25_native_sided_v1.csv",
        "path_base": "TENNIS_DATA_ROOT",
    }
    assert export._player_source_binding({"inputs": [binding]}) == binding


def test_ground_pixel_backprojection_and_unordered_foot_pairing() -> None:
    projection = np.asarray(
        [
            [1000.0, 0.0, 0.0, 500.0],
            [0.0, 1000.0, 0.0, 300.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert export._ground_from_pixel(projection, (2500.0, 3300.0)) == (2.0, 3.0)
    errors = export._paired_foot_errors(
        [(1.0, 1.0), (3.0, 1.0)],
        [(3.1, 1.0), (0.9, 1.0)],
    )
    assert np.allclose(errors, [0.1, 0.1])
