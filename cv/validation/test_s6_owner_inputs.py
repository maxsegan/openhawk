import copy
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from cv.validation import s6_owner_inputs as intake


def fixture():
    case = {
        "id": "ball",
        "kind": "ball",
        "clip": "pt0001",
        "frames": [1, 2, 3],
        "targets": [{"id": str(f), "frame": f} for f in [1, 2, 3]],
    }
    court = {
        "id": "court",
        "kind": "court",
        "clip": "pt0001",
        "frames": [2],
        "targets": [{"id": "net_center_top", "frame": 2}],
    }
    pack = {
        "schema": "s6_owner_input_pack_v1",
        "benchmark_id": "test",
        "match_id": "test_match",
        "native_size": [1920, 1080],
        "fps": 25,
        "ball_convention": "leading edge",
        "court_convention": "outside edge",
        "cases": [case, court],
        "images": [
            {
                "clip": "pt0001",
                "frame": f,
                "image_url": f"images/{f}.jpg",
                "source": {"sha256": str(f)},
            }
            for f in [1, 2, 3]
        ],
    }
    record = {
        "case_id": "ball",
        "clip": "pt0001",
        "window_status": "ambiguous",
        "complete": False,
        "frames": [],
        "notes": "unfinished",
    }
    labels = {
        **{
            k: pack[k]
            for k in (
                "benchmark_id",
                "match_id",
                "native_size",
                "ball_convention",
                "court_convention",
            )
        },
        "schema": "s6_owner_input_labels_v1",
        "ball": {
            "schema": "tennis_ball_track_sequence_labels_v1",
            "benchmark_id": "test",
            "records": [],
        },
        "ball_drafts": [record],
        "court": [{**record, "case_id": "court"}],
        "fractional_estimates": [],
    }
    return pack, labels


def visible(frame=1, target_id="1"):
    return {
        "frame": frame,
        "target_id": target_id,
        "status": "visible",
        "x1080": 500.25,
        "y1080": 250.5,
        "uncertainty_radius_px1080": 3.0,
    }


def test_partial_native_and_fractional_labels_remain_distinct_and_source_bound():
    pack, labels = fixture()
    labels["ball_drafts"][0]["frames"] = [visible()]
    labels["fractional_estimates"] = [
        {
            **visible(1.5),
            "case_id": "ball",
            "clip": "pt0001",
            "source_frames": [1, 2],
            "blend_weight": 0.5,
            "evidence_kind": "human_fractional_estimate_from_adjacent_native_frames",
        }
    ]
    before = copy.deepcopy(labels)
    result = intake.normalize(pack, labels)
    assert labels == before
    assert result["counts"]["native_ball_labels"] == 1
    assert result["counts"]["fractional_estimates"] == 1
    assert result["complete_ball_windows"] == []
    assert result["partial_ball_windows"][0]["frames"][0]["source_image_sha256"] == "1"
    assert result["fractional_estimates"][0]["source_image_sha256"] == ["1", "2"]
    assert result["human_derived"]


def test_full_window_needs_every_target_and_preserves_abstentions():
    pack, labels = fixture()
    record = labels["ball_drafts"].pop()
    record.update(complete=True, window_status="repaired", frames=[visible()])
    labels["ball"]["records"] = [record]
    with pytest.raises(ValueError, match="missing targets"):
        intake.normalize(pack, labels)
    record["frames"].extend(
        [visible(2, "2"), {"frame": 3, "target_id": "3", "status": "ambiguous"}]
    )
    result = intake.normalize(pack, labels)
    assert result["counts"]["complete_ball_windows"] == 1
    assert result["complete_ball_windows"][0]["frames"][2]["status"] == "ambiguous"


@pytest.mark.parametrize(
    "change",
    [
        {"frame": 1.5},
        {"frame": True},
        {"x1080": float("nan")},
        {"y1080": 1100},
        {"uncertainty_radius_px1080": 0},
        {"status": "occluded"},
        {"evidence_kind": "human_fractional_estimate_from_adjacent_native_frames"},
    ],
)
def test_invalid_or_non_native_coordinates_fail(change):
    pack, labels = fixture()
    labels["ball_drafts"][0]["frames"] = [{**visible(), **change}]
    with pytest.raises(ValueError):
        intake.normalize(pack, labels)


def test_duplicate_targets_and_missing_cases_cannot_change_denominators():
    pack, labels = fixture()
    labels["ball_drafts"][0]["frames"] = [visible(), visible()]
    with pytest.raises(ValueError, match="duplicate target"):
        intake.normalize(pack, labels)
    labels["ball_drafts"][0]["frames"] = []
    labels["court"] = []
    with pytest.raises(ValueError, match="denominator"):
        intake.normalize(pack, labels)


def test_court_frame_and_pack_identity_are_binding():
    pack, labels = fixture()
    labels["court"][0]["frames"] = [visible(1, "net_center_top")]
    with pytest.raises(ValueError, match="frame/target"):
        intake.normalize(pack, labels)
    labels["court"][0]["frames"][0]["frame"] = 2
    assert intake.normalize(pack, labels)["counts"]["court_landmarks"] == 1
    labels["match_id"] = "wrong_match"
    with pytest.raises(ValueError, match="mismatch"):
        intake.normalize(pack, labels)


def test_cli_checks_image_bytes_before_writing_and_never_overwrites(tmp_path, monkeypatch):
    pack, labels = fixture()
    (tmp_path / "images").mkdir()
    for image in pack["images"]:
        path = tmp_path / image["image_url"]
        Image.new("RGB", (1920, 1080)).save(path)
        image["source"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pack_path, labels_path, output = (
        tmp_path / n for n in ["manifest.json", "labels.json", "intake.json"]
    )
    pack_path.write_text(json.dumps(pack))
    labels_path.write_text(json.dumps(labels))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s6_owner_inputs",
            "--pack",
            str(pack_path),
            "--labels",
            str(labels_path),
            "--output",
            str(output),
        ],
    )
    intake.main()
    result = json.loads(output.read_text())
    assert len(result["input_bindings"]) == 6
    with pytest.raises(FileExistsError):
        intake.main()
    other_output = tmp_path / "other.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "s6_owner_inputs",
            "--pack",
            str(pack_path),
            "--labels",
            str(labels_path),
            "--output",
            str(other_output),
        ],
    )
    (tmp_path / "images/1.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from source"):
        intake.main()
    assert not other_output.exists()


def agent_fixture():
    directory = Path(__file__).parent / "labels/s6_agent_inputs_v1"
    labels = json.loads(
        (directory / "ao2023f_w_sabalenka_rybakina_pt0003_attempt01.json").read_text()
    )
    return labels["source_pack"], labels


def test_agent_intake_preserves_annotator_events_streaks_and_source_pts():
    pack, labels = agent_fixture()
    before = copy.deepcopy(labels)
    result = intake.normalize(pack, labels)
    assert labels == before
    assert result["annotation_origin"] == "agent"
    assert result["annotator_provenance"]["owner_verified"] is False
    assert result["events"]["records"][0]["frame"] == 62.5
    assert result["counts"]["native_ball_labels"] == 57
    assert result["counts"]["court_landmarks"] == 33
    assert result["complete_ball_windows"][0]["frames"] == labels["ball"]["records"][0]["frames"]


@pytest.mark.parametrize(
    "defect",
    [
        "source",
        "tail",
        "event_source",
        "event_interval",
        "event_frame",
        "event_ending",
        "origin",
        "missing_frame",
        "origin_typo",
        "origin_erased",
    ],
)
def test_agent_intake_rejects_corrupted_evidence(defect):
    pack, labels = agent_fixture()
    row = labels["ball"]["records"][0]["frames"][4]
    event = labels["events"]["records"][0]
    if defect == "source":
        row["source_image_sha256"] = "wrong"
    elif defect == "tail":
        row["streak"]["trailing"]["uncertainty_radius_px1080"] = float("nan")
    elif defect == "event_source":
        event["source_image_sha256"][0] = "wrong"
    elif defect == "event_interval":
        event["frame_interval"] = [64, 65]
    elif defect == "event_frame":
        event["frame"] = 62.25
    elif defect == "event_ending":
        labels["events"]["records"].pop()
    elif defect == "origin":
        labels["provenance"]["owner_verified"] = True
    elif defect == "origin_typo":
        labels["annotation_origin"] = "agnet"
    elif defect == "origin_erased":
        labels.pop("annotation_origin")
    else:
        labels["ball"]["records"][0]["frames"].pop()
    with pytest.raises(ValueError):
        intake.normalize(pack, labels)


def speed_fixture():
    pack, labels = agent_fixture()
    image = next(r for r in pack["images"] if r["frame"] == 75)
    labels["serve_speed_evidence"] = {
        "schema": "tennis_serve_speed_graphic_evidence_v1",
        "annotation_origin": "agent",
        "status": "visible",
        "value": 182,
        "unit": "km/h",
        "contact_event_id": "event_01",
        "visible_frames": [75],
        "note": "Test display update.",
        "observations": [
            {
                "clip": "pt0003",
                "frame": 75,
                "role": "current_serve",
                "value": 182,
                "unit": "km/h",
                "display_text": "182 km/h",
                "note": "Test native display.",
                "source_image_sha256": image["source"]["sha256"],
                "crop_native_xywh": [10, 20, 30, 40],
                "native_crop_sha256": "a" * 64,
                "crop_path_base": "TENNIS_DATA_ROOT",
                "crop_path": "speed.png",
            }
        ],
    }
    return pack, labels


def test_speed_is_separate_preserved_evidence():
    pack, labels = speed_fixture()
    before = copy.deepcopy(labels)
    result = intake.normalize(pack, labels)
    assert labels == before
    assert result["serve_speed_evidence"] == labels["serve_speed_evidence"]
    assert result["counts"]["native_ball_labels"] == 57


@pytest.mark.parametrize(
    "defect", ["stale", "contact", "hash", "inventory", "crop", "path", "value", "abstain"]
)
def test_speed_rejects_unbound_stale_or_inconsistent_witness(defect):
    pack, labels = speed_fixture()
    evidence = labels["serve_speed_evidence"]
    row = evidence["observations"][0]
    if defect == "stale":
        row["frame"] = 40
        row["source_image_sha256"] = next(r for r in pack["images"] if r["frame"] == 40)["source"][
            "sha256"
        ]
    elif defect == "contact":
        evidence["contact_event_id"] = "event_02"
    elif defect == "hash":
        row["source_image_sha256"] = "changed"
    elif defect == "inventory":
        evidence["visible_frames"] = [74, 75]
    elif defect == "crop":
        row["crop_native_xywh"] = [1900, 0, 40, 40]
    elif defect == "path":
        row["crop_path"] = "../speed.png"
    elif defect == "value":
        row["value"] = 175
    else:
        evidence["status"] = "ambiguous"
    with pytest.raises(ValueError):
        intake.normalize(pack, labels)


def test_speed_crop_must_match_native_pixels_even_with_updated_png_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    pack, labels = speed_fixture()
    evidence = labels["serve_speed_evidence"]
    row = evidence["observations"][0]
    binding = next(r for r in pack["images"] if r["frame"] == row["frame"])
    source = tmp_path / "source.jpg"
    Image.new("RGB", (1920, 1080), "blue").save(source)
    binding["image_url"] = source.name
    binding["source"]["sha256"] = row["source_image_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    crop = tmp_path / row["crop_path"]
    with Image.open(source) as native:
        native.crop((10, 20, 40, 60)).save(crop)
    row["native_crop_sha256"] = hashlib.sha256(crop.read_bytes()).hexdigest()
    assert intake.verify_speed_crops(pack, evidence, tmp_path) == [crop]
    Image.new("RGB", (30, 40), "red").save(crop)
    with pytest.raises(ValueError, match="bytes changed"):
        intake.verify_speed_crops(pack, evidence, tmp_path)
    row["native_crop_sha256"] = hashlib.sha256(crop.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="native pixels"):
        intake.verify_speed_crops(pack, evidence, tmp_path)
