import csv
import json
import sys

import numpy as np
import pytest

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_missing_first_player as model
from cv.experiments.connected_shooting import labeled_preparation_net_recovery as net
from cv.experiments.connected_shooting import player_state_fallback
from cv.experiments.connected_shooting import toss_witness

CLIP = "clip"
# Serve pictured at 77 with only the far receiver detected; the near server's
# nearest row is ten frames later.  Later contacts have both players in frame.
CONTACTS = [76.5, 100.0, 125.0, 150.0]
NEAR_BOX = (800.0, 700.0, 900.0, 950.0)  # image-low, court-near player
FAR_BOX = (900.0, 200.0, 950.0, 330.0)  # image-high, court-far player
BALL = {77: (850.0, 700.0), 100: (920.0, 220.0), 125: (850.0, 720.0), 150: (930.0, 210.0)}


def fixture():
    contacts = [dict(frame=t, frame_interval=[t - 0.5, t + 0.5]) for t in CONTACTS]
    labels = {f: np.array(p, float) for f, p in BALL.items()}
    cameras = {f: dict(P=np.eye(3, 4).tolist(), status="supported") for f in range(70, 160)}
    rows = {(77, "far"): {}, (87, "near"): {}}
    for frame in (100, 125, 150):
        rows[(frame, "near")] = {}
        rows[(frame, "far")] = {}

    def lookup(frame, pixel):
        return dict(side="near" if pixel[1] > 500 else "far")

    return contacts, labels, cameras, rows, lookup


def write_pose_csv(path, rows):
    fields = ["clip", "frame", "track_id", "side", "x0", "y0", "x1", "y1", "court_x", "court_y"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame, side, box, court in rows:
            writer.writerow(
                dict(
                    clip=CLIP,
                    frame=f"f_{frame:04d}.jpg",
                    track_id=side[0],
                    side=side,
                    x0=box[0],
                    y0=box[1],
                    x1=box[2],
                    y1=box[3],
                    court_x=court[0],
                    court_y=court[1],
                )
            )


def real_case_pose_csv(tmp_path):
    rows = [(77, "far", FAR_BOX, (4.0, 22.5)), (87, "near", NEAR_BOX, (5.0, 1.0))]
    for frame in (100, 125, 150):
        rows.append((frame, "near", NEAR_BOX, (5.0, 1.5)))
        rows.append((frame, "far", FAR_BOX, (4.0, 22.0)))
    path = tmp_path / "pose.csv"
    write_pose_csv(path, rows)
    return path


def test_later_consensus_recovers_absent_near_server():
    receipt = model.qualify(*fixture())
    assert receipt["implied_first_side"] == "near"
    assert receipt["sole_row_side_at_first_picture"] == "far"
    assert receipt["first_pose_frame"] == 77
    assert receipt["first_fitting_front_frame"] == 77
    assert [a["contact_index"] for a in receipt["later_associations"]] == [1, 2, 3]
    assert [a["side"] for a in receipt["later_associations"]] == ["far", "near", "far"]
    assert receipt["trajectory_observations_added"] == 0
    assert receipt["original_labels_changed"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "first_camera",
        "explicit_camera_status",
        "no_first_front",
        "target_present",
        "receiver_absent",
        "conflicting",
        "insufficient",
        "one_sided",
        "later_front_unsupported",
    ],
)
def test_rejects_unsupported_recovery(failure):
    contacts, labels, cameras, rows, lookup = fixture()
    if failure == "first_camera":
        cameras[77]["supported"] = False
    if failure == "explicit_camera_status":
        cameras[77]["status"] = "unsupported"
    if failure == "no_first_front":
        del labels[77]
    if failure == "target_present":
        rows[(77, "near")] = {}
    if failure == "receiver_absent":
        del rows[(77, "far")]
    if failure == "conflicting":
        labels[125] = np.array([920.0, 220.0])  # far hitter where grammar wants near
    if failure == "insufficient":
        contacts = contacts[:2]
    if failure == "one_sided":
        # Two later contacts show only one player; a lone row cannot witness a side.
        del rows[(125, "far")]
        del rows[(150, "near")]
    if failure == "later_front_unsupported":
        for frame in (100, 125):
            cameras[frame]["supported"] = False
    with pytest.raises(ValueError):
        model.qualify(contacts, labels, cameras, rows, lookup)


def test_recovered_states_substitute_only_first_and_alternate_from_it(tmp_path):
    contacts, labels, cameras, _, _ = fixture()
    pose_csv = real_case_pose_csv(tmp_path)
    names = ["Nadal", "Other", "Nadal", "Other"]
    statures = {"Nadal": 1.85, "Other": 1.90}
    before = whole.single.server_state
    result, receipt, first_state = model.recovered_player_states(
        pose_csv,
        CLIP,
        contacts,
        labels,
        cameras,
        image_coordinate_scale=1.0,
        player_names=names,
        player_statures_m=statures,
    )
    assert whole.single.server_state is before
    assert [r["side"] for r in result] == ["near", "far", "near", "far"]
    first = result[0]
    assert first["side"] == "near"
    assert first["player"] == "Nadal"
    assert first["stature_m"] == 1.85
    assert first["court_position_source"] == "nearest_sided_automatic_row"
    assert first["substitution"]["source_frame"] == 87
    assert first["court_position_sigma_m"] == pytest.approx(
        player_state_fallback.BOXES_ONLY_SIGMA_M
        + 10 * player_state_fallback.PLAYER_SPEED_M_PER_FRAME
    )
    assert first["court_position_sigma_m"] == pytest.approx(2.25)
    assert first["pose_wrist_witness"]["status"] == "abstained"
    # Later contacts are the ordinary pictured associations, not substitutes.
    assert all("substitution" not in r for r in result[1:])
    assert result[1]["player"] == "Other"
    assert receipt["result_sides"] == ["near", "far", "near", "far"]
    assert receipt["substituted_first_state"]["frame"] == 77
    assert receipt["pose_source"]["sha256"]
    assert first_state["side"] == "near"


def test_original_association_alone_adopts_sole_receiver(tmp_path):
    contacts, labels, _, _, _ = fixture()
    pose_csv = real_case_pose_csv(tmp_path)
    result = whole.alternating_player_states(pose_csv, CLIP, contacts, labels)
    assert [r["side"] for r in result] == ["far", "near", "far", "near"]


def test_server_state_restored_when_original_association_raises(tmp_path):
    contacts, labels, cameras, _, _ = fixture()
    pose_csv = real_case_pose_csv(tmp_path)
    before = whole.single.server_state

    def failing(*args, **kwargs):
        assert whole.single.server_state is not before
        raise RuntimeError("downstream failure")

    with pytest.raises(RuntimeError):
        model.recovered_player_states(
            pose_csv, CLIP, contacts, labels, cameras, original_players=failing
        )
    assert whole.single.server_state is before


def test_server_feet_proxy_discloses_source_and_claims_no_observed_feet(tmp_path):
    contacts, labels, cameras, _, _ = fixture()
    pose_csv = real_case_pose_csv(tmp_path)
    _, _, first_state = model.recovered_player_states(pose_csv, CLIP, contacts, labels, cameras)
    feet = model.recovered_server_feet(first_state)
    assert feet["side"] == "near"
    assert feet["court_xy_m"] == [5.0, 1.0]
    assert feet["history_frames"] == []
    assert feet["history_count"] == 0
    assert feet["history_status"] == model.FEET_HISTORY_STATUS
    assert feet["observed_at_contact_picture"] is False
    assert feet["court_position_source_frame"] == 87
    assert feet["court_position_sigma_m"] == pytest.approx(2.25)
    assert feet["precontact_track_grounding"] is False
    assert feet["release_position_synthesized"] is False
    assert feet["hand_height_prior_applied"] is False
    assert "87" in feet["fallback_reason"]
    for key in ("track_id", "association_distance_px", "image_coordinate_scale"):
        assert key in feet
    # The original witness on the same table would have chosen the far receiver.
    original = toss_witness.server_feet(
        pose_csv, CLIP, 76.5, np.array(BALL[77]), observation_fallback=True
    )
    assert original["side"] == "far"


def cli_inputs(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(dict(events=dict(records=[]), ball=dict(records=[]))))
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
        json.dumps(dict(cameras=[dict(frame=77, P=np.eye(3, 4).tolist(), status="supported")]))
    )
    output = tmp_path / "out" / "run"
    return ["--labels", str(labels), "--cameras", str(cameras), "--output", str(output)], output


@pytest.mark.parametrize("association", ["old", "recovered"])
def test_main_delegates_to_net_adapter_and_restores_patches(tmp_path, monkeypatch, association):
    remaining, output = cli_inputs(tmp_path)
    seen = {}
    originals = (whole.alternating_player_states, toss_witness.server_feet)

    def fake_net_main():
        seen["argv"] = list(sys.argv)
        seen["patched"] = (
            whole.alternating_player_states is not originals[0],
            toss_witness.server_feet is not originals[1],
        )
        raise RuntimeError("stop before any fit")

    monkeypatch.setattr(net, "main", fake_net_main)
    monkeypatch.setattr(
        sys, "argv", ["prog", "fit", "--association", association, "--", *remaining]
    )
    with pytest.raises(RuntimeError):
        model.main()
    assert seen["argv"][1:3] == ["fit", "--"]
    assert seen["argv"][3:] == remaining
    assert seen["patched"] == ((association == "recovered"),) * 2
    assert whole.alternating_player_states is originals[0]
    assert toss_witness.server_feet is originals[1]
    manifest = json.loads(
        (output.parent / (output.name + "_missing_first_player_manifest.json")).read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["association"] == association
    assert manifest["applications"] == []
    assert manifest["new_pixels_or_labels_added"] is False
    assert {m["role"] for m in manifest["module_dependencies"]} >= {
        "whole_point_search",
        "player_state_fallback",
        "toss_witness",
    }
