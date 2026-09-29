import pytest

from cv.experiments.connected_shooting import labeled_preparation_recovery_inputs as probe


def test_explicit_seed_control_and_exception_restore(monkeypatch):
    def original(*args, **kwargs):
        assert kwargs == {"recover_launch_feasibility": True}
        return [1, 2], {"scope": "input_only"}

    monkeypatch.setattr(probe.postbounce_initialization, "seed", original)
    receipts = []
    with pytest.raises(RuntimeError):
        with probe.recovery_scope(
            launch_feasibility=True, contact_player_feet=False, receipts=receipts
        ):
            assert probe.postbounce_initialization.seed()[0] == [1, 2]
            raise RuntimeError("stop")
    assert probe.postbounce_initialization.seed is original
    assert receipts[0]["evidence"] == {"scope": "input_only"}


def test_missing_optional_track_uses_actual_associated_player_only(monkeypatch):
    player = {"side": "far", "court_centre_xy_m": [4, 24], "frame": 40}
    monkeypatch.setattr(probe.whole, "alternating_player_states", lambda: [player])

    def absent(*args, **kwargs):
        raise ValueError("server feet lack a short pre-contact automatic track")

    monkeypatch.setattr(probe.whole.toss_witness, "server_feet", absent)
    receipts = []
    with probe.recovery_scope(
        launch_feasibility=False, contact_player_feet=True, receipts=receipts
    ):
        with pytest.raises(ValueError):
            probe.whole.toss_witness.server_feet()
        probe.whole.alternating_player_states()
        result = probe.whole.toss_witness.server_feet()
        assert result["court_xy_m"] == [4, 24]
        assert receipts[0]["precontact_track_available"] is False
    assert probe.whole.toss_witness.server_feet is absent
