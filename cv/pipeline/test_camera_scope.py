from cv.pipeline.camera_scope import build_contract, point_scope, window_scope


def _entry():
    return {
        "clip": "pt0001",
        "fps": 25.0,
        "active_spans": [[10.0, 100.0]],
        "event_spans": [[7.0, 103.0]],
    }


def test_nonstandard_camera_withholds_spatial_but_preserves_timing():
    point = point_scope(
        _entry(),
        {
            "court_arm_enabled": True,
            "box_arm_enabled": True,
            "proposed_trims": [
                {
                    "start_frame": 20,
                    "end_frame": 30,
                    "reasons": ["court_support_lost", "giant_player_box"],
                }
            ],
        },
    )
    verdict = window_scope(point, 15, 35)
    assert verdict["decision"] == "withhold_spatial"
    assert verdict["spatial_usable"] is False
    assert verdict["timing_usable"] is True
    assert point["timing_context_spans"] == [[7.0, 103.0]]


def test_missing_evidence_is_unknown_not_rejected():
    point = point_scope(_entry(), {"proposed_trims": []})
    verdict = window_scope(point, 20, 30)
    assert verdict == {
        "decision": "unknown",
        "spatial_usable": None,
        "timing_usable": True,
        "reasons": ["camera_scope_unknown"],
    }


def test_contract_does_not_load_labels_and_preserves_provenance():
    report = {
        "schema": "play_camera_leakage_v1",
        "match_id": "match",
        "points": {
            "pt0001": {
                "court_arm_enabled": True,
                "box_arm_enabled": False,
                "proposed_trims": [],
                "calibration_provenance": {
                    "source": "direct_camera",
                    "residuals": {"net_px": 2.0},
                    "frame_scope": [[10, 100]],
                    "fallback_ancestry": [],
                },
            }
        },
    }
    contract = build_contract({"match/pt0001": _entry()}, [report])
    assert contract["labels_loaded"] is False
    provenance = contract["points"]["match/pt0001"]["calibration_provenance"]
    assert provenance["source"] == "direct_camera"
    assert provenance["residuals"] == {"net_px": 2.0}
