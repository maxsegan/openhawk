import pytest

from cv.experiments.connected_shooting.labeled_preparation_net_recovery import net_bounded_frames


def test_post_net_ground_witness_cannot_use_pre_net_or_uncertain_impact_fronts():
    frames, receipt = net_bounded_frames(
        {"frame": 199.5}, range(190, 206), [{"frame_interval": [194, 195]}]
    )
    assert frames == list(range(196, 206))
    assert receipt["excluded_across_net_or_uncertain_frames"] == list(range(190, 196))


def test_earlier_ground_witness_cannot_consume_later_net_rebound():
    frames, _ = net_bounded_frames({"frame": 49.5}, range(45, 66), [{"frame_interval": [62, 63]}])
    assert frames == list(range(45, 62))


def test_net_and_ground_overlap_is_explicit_failure():
    with pytest.raises(ValueError, match="distinct ordered"):
        net_bounded_frames({"frame": 12}, [10, 11, 12, 13], [{"frame_interval": [11, 12]}])
