from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import camera_geometry, human_audit, human_replay, model


def fixture():
    full, parameters = model.control()
    scenes = []
    for heldout in (False, True):
        selected = [((f % 3 == 0) if heldout else (f % 3 != 0)) for f in full.observation_frames]
        scenes.append(
            replace(
                full,
                observation_frames=tuple(f[m] for f, m in zip(full.observation_frames, selected)),
                pixels=tuple(p[m] for p, m in zip(full.pixels, selected)),
                cameras=tuple(p[m] for p, m in zip(full.cameras, selected)),
            )
        )
    query = tuple(
        np.unique(np.r_[np.arange(a, b, full.fps / 240), b])
        for a, b in zip(full.contact_frames, full.contact_frames[1:])
    )
    record = {
        "parameters": parameters,
        "dense_flights": model.chain(full, parameters, query_frames=query),
        "evidence": {
            "human_events": [
                {
                    "labeled_frame": "1",
                    "labeled_x540": str(full.pixels[0][0, 0] / 2 + 1),
                    "labeled_y540": str(full.pixels[0][0, 1] / 2),
                    "event_type": "contact",
                }
            ]
        },
    }
    return record, *scenes


def test_frozen_source_identity_survives_cross_worktree_path_display_not_byte_changes(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("frozen source")
    source = {
        "resolved_path": str(path),
        "record": {
            **human_audit.file_record(path),
            "path": "old/repository/source.txt",
            "path_base": "repository",
        },
    }
    assert human_audit.source_identity_matches(source)
    path.write_text("mutant source")
    assert not human_audit.source_identity_matches(source)
    del source["record"]["sha256"]
    with pytest.raises(KeyError):
        human_audit.source_identity_matches(source)


def test_defect_audit_keeps_bounce_mismatch_and_click_disagreement_without_accepting():
    record, scene, heldout = fixture()
    result = human_audit.measure(record, scene, heldout, [9, 21])
    assert result["maximum_contact_join_m"] == 0
    assert result["bounce_count_mismatch_flights"] == 2
    assert result["events"][0]["track_to_archived_coordinate_px"] == pytest.approx(2)
    assert result["events"][0]["fit_to_archived_coordinate_px"] == pytest.approx(2)
    assert not result["events"][0]["independent_coordinate_click_verified"]
    assert not result["complete_point_accepted"]
    assert not result["independent_3d_accuracy_measured"]


@pytest.mark.parametrize("verdict", ["new", "confirmed", "adjusted"])
def test_event_verdict_does_not_certify_independent_coordinate_click(verdict):
    record, scene, heldout = fixture()
    record["evidence"]["human_events"][0]["verdict"] = verdict
    event = human_audit.measure(record, scene, heldout, [9, 21])["events"][0]
    assert event["coordinate_origin"] == "not_preserved_by_csv"
    assert not event["independent_coordinate_click_verified"]


def test_changed_frozen_geometry_and_duplicate_native_ownership_fail():
    record, scene, heldout = fixture()
    with pytest.raises(ValueError, match="duplicated native"):
        human_audit.measure(record, scene, scene, [9, 21])
    record["dense_flights"][0]["positions"][0, 0] += 0.001
    with pytest.raises(ValueError, match="differs from frozen"):
        human_audit.measure(record, scene, heldout, [9, 21])


def test_missing_click_remains_in_event_inventory():
    record, scene, heldout = fixture()
    record["evidence"]["human_events"][0]["labeled_x540"] = ""
    result = human_audit.measure(record, scene, heldout, [9, 21])
    assert len(result["events"]) == 1
    assert result["events"][0]["status"] == "missing_native_observation_or_click"


def test_archived_coordinate_disagreement_uses_declared_radial_projection():
    record, scene, heldout = fixture()
    radial_scenes = []
    for target in (scene, heldout):
        lens = tuple(np.tile([2e-7, 960.0, 540.0], (len(f), 1)) for f in target.observation_frames)
        radial_scenes.append(
            replace(
                target,
                camera_distortion=lens,
                pixels=tuple(camera_geometry.distort(p, d) for p, d in zip(target.pixels, lens)),
            )
        )
    scene, heldout = radial_scenes
    event = record["evidence"]["human_events"][0]
    event["labeled_x540"], event["labeled_y540"] = map(str, scene.pixels[0][0] / 2)
    measured = human_audit.measure(record, scene, heldout, [9, 21])["events"][0]
    assert measured["track_to_archived_coordinate_px"] == pytest.approx(0, abs=1e-9)
    assert measured["fit_to_archived_coordinate_px"] == pytest.approx(0, abs=1e-9)
    assert not measured["independent_coordinate_click_verified"]


def test_native_renderer_keeps_original_and_distinguishes_three_coordinate_sources(tmp_path):
    record, scene, heldout = fixture()
    record.update(case="synthetic__pt0001", clip="pt0001", rebound_mode="fixed")
    original = tmp_path / "frames" / "pt0001" / "f_0001.jpg"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"unchanged original image reference")
    output = tmp_path / "new_view"
    human_replay.render(
        record,
        scene,
        heldout,
        {"inputs": {"frames": "frames"}, "native_size": [1920, 1080]},
        tmp_path,
        output,
    )
    assert original.read_bytes() == b"unchanged original image reference"
    page = output.with_suffix(".html").read_text()
    for marker in ('stroke="magenta"', 'stroke="orange"', 'stroke="cyan"'):
        assert marker in page
    assert 'viewBox="0 0 1920 1080"' in page
    assert "Open original native image" in page
    assert "not independently verified" in page
