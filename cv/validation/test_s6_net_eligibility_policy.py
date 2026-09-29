"""One manifest policy governs cold execution, loaded replay and displayed geometry."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from cv.pipeline import s6_labeled_stage as stage
from cv.experiments.connected_shooting import net_collision as net
from cv.experiments.connected_shooting import labeled_common_source as loader
from cv.viz import export_local_s6 as export


@pytest.mark.parametrize("enabled", ["off", "on"])
def test_whole_attempt_scope_and_exception_restoration(tmp_path, monkeypatch, enabled):
    expected = "physical_mesh_v1" if enabled == "on" else "timing_window_v1"

    def attempt(row, output, policy):
        assert net.active_policy() == expected
        output.mkdir()
        return {"net_collision_policy": net.active_policy()}

    monkeypatch.setattr(stage, "_run_attempt", attempt)
    result = stage.run_attempt(
        {"key": "control"}, tmp_path / "out", {"net_physical_eligibility": enabled}
    )
    assert result["net_collision_policy"] == expected
    assert net.active_policy() == "timing_window_v1"

    def fail(*args):
        assert net.active_policy() == expected
        raise ValueError("deliberate numerical control")

    monkeypatch.setattr(stage, "_run_attempt", fail)
    with pytest.raises(ValueError, match="deliberate"):
        stage.run_attempt(
            {"key": "control"}, tmp_path / "failure", {"net_physical_eligibility": enabled}
        )
    assert net.active_policy() == "timing_window_v1"


def test_recorded_source_policy_rejected_before_any_replay(tmp_path):
    args = {}
    for name in ["search_report", "labels", "packet", "cameras", "pose_csv"]:
        p = tmp_path / (name + ".json")
        p.write_text(
            json.dumps(
                {"net_collision_policy": "physical_mesh_v1"} if name == "search_report" else {}
            )
        )
        args[name] = p
    with pytest.raises(ValueError, match="active replay policy"):
        loader.load(SimpleNamespace(**args))
    with net.physical_eligibility():
        # Having the right mode reaches the ordinary input qualification next.
        with pytest.raises(ValueError, match="separated original packet"):
            loader.load(SimpleNamespace(**args))


def records():
    mode = "physical_mesh_v1"
    result = {
        "net_collision_policy": mode,
        "final_state_binding": {"sha256": {"net_collision_policy": export.value_sha256(mode)}},
    }
    search = {"net_collision_policy": mode}
    invocation = {"policy": {"net_physical_eligibility": "on"}}
    return [result, search, invocation, deepcopy(invocation)]


def test_export_checks_whole_policy_chain_and_legacy_compatibility():
    r = records()
    assert export.checked_net_policy(*r) == "physical_mesh_v1"
    assert export.checked_net_policy({}, {}, {"policy": {}}, {"policy": {}}) == "timing_window_v1"
    for i in range(4):
        changed = deepcopy(r)
        if i < 2:
            changed[i]["net_collision_policy"] = "timing_window_v1"
        else:
            changed[i]["policy"]["net_physical_eligibility"] = "off"
        with pytest.raises(ValueError, match="policy"):
            export.checked_net_policy(*changed)
    changed = deepcopy(r)
    changed[0]["final_state_binding"]["sha256"]["net_collision_policy"] = "tampered"
    with pytest.raises(ValueError, match="binding"):
        export.checked_net_policy(*changed)


def test_final_binding_includes_mode_and_settings_are_global():
    result = {
        k: {}
        for k in ["parameters", "measurement", "verdict", "net_response", "evaluation_context"]
    }
    result.update(
        initial_search={"parameters": []},
        selected_by={"selected_search": {}},
        net_collision_policy="physical_mesh_v1",
    )
    binding = stage.final_state_binding(result)
    assert binding["sha256"]["net_collision_policy"] == export.value_sha256("physical_mesh_v1")
    assert stage.shared_settings({})["net_physical_eligibility"] == "off"
    with pytest.raises(ValueError, match="net_physical_eligibility"):
        stage.shared_settings({"net_physical_eligibility": "unknown"})
