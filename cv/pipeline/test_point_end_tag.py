import pytest

from cv.pipeline import point_end_tag as tag

NET = 11.885
BOX = 6.40


def _row(kind, frame, xy=None):
    event = dict(event_type=kind, frame=float(frame), status="labeled")
    if xy is not None:
        event["automatic_location"] = dict(
            court_x_fraction=xy[0] / 10.97,
            court_y_fraction=xy[1] / 23.77,
            court_transport=dict(reliable_per_frame=True, static_fallback=False, held=False),
            fps=25.0,
        )
    return event


def _serve_rows(serve_y, landing_xy, *, net=False):
    rows = [_row("contact", 10, (5.0, serve_y)), _row("bounce", 20, landing_xy)]
    if net:
        rows.append(_row("net_hit", 15))
    return rows + [_row("contact", 30), _row("bounce", 50), _row("contact", 170)]


def _flights(*starts, serve_landing=(5.0, 5.0), serve_y=23.0):
    flights = [
        dict(role="serve" if i == 0 else "final", start_frame=float(s))
        for i, s in enumerate(starts)
    ]
    flights[0].update(start_xyz_m=[5.0, serve_y, 2.7], bounces_xyz_m=[[*serve_landing, 0.03]])
    return flights


def test_dead_serve_call_reads_a_long_serve_by_the_landing_half():
    """source08_r1_point002_a1: a far serve reads 0.31 m long on the near half, then a knock-back.

    Far-half readings of good serves were 0.3 m and 0.8 m long, so that half needs 1 m.
    """
    near = tag.dead_serve_call(_serve_rows(40.0, (2.0, NET - BOX - 0.31)))
    assert near["evidence"] == "serve_long" and near["landing_half"] == "near"
    assert near["out_by_m"] == pytest.approx(0.31)
    assert tag.dead_serve_call(_serve_rows(40.0, (2.0, NET - BOX - 0.2))) is None
    assert tag.dead_serve_call(_serve_rows(1.0, (5.0, NET + BOX + 0.8))) is None
    far = tag.dead_serve_call(_serve_rows(1.0, (5.0, NET + BOX + 1.1)))
    assert far["evidence"] == "serve_long" and far["landing_half"] == "far"
    # A landing on the server's own half is a missed row, not a fault.
    assert tag.dead_serve_call(_serve_rows(1.0, (5.0, 3.0))) is None


def test_net_serve_tags_the_knock_back_until_the_next_serve():
    """source07_case006_long_a1: the serve clips the net and lands in; the receiver hits it back."""
    rows = _serve_rows(40.0, (5.0, 7.4), net=True)
    flights = _flights(10, 30, 170)
    call = tag.tag(rows, flights)
    assert call["evidence"] == "net_serve" and call["frame"] == 20.0
    # Contact 170 follows 4.8 s of nothing and is the next serve.
    assert call["restart_frame"] == 170.0
    assert [f["after_point_end"] for f in flights] == [False, True, False]
    assert flights[1]["point_end_signal"] == "net_serve"


def test_fitted_serve_fault_tags_every_later_flight():
    """source08_medium_0625_a1: the far fault's row reads in, its fitted landing 0.87 m long."""
    rows = _serve_rows(40.0, (5.0, 7.4))
    flights = _flights(10, 30, serve_landing=(5.0, NET - BOX - 0.87))
    call = tag.tag(rows, flights)
    assert call["evidence"] == "fitted_serve_fault"
    assert [f["after_point_end"] for f in flights] == [False, True]
    good = _flights(10, 30, serve_landing=(5.0, NET - BOX - 0.5))
    assert tag.tag(rows, good) is None
    assert [f["after_point_end"] for f in good] == [False, False]


def test_a_low_first_flight_is_no_serve():
    """Paul-Vukic pt0106: an attempt starting mid-rally names its first groundstroke ``serve``."""
    rows = _serve_rows(40.0, (5.0, NET - BOX - 0.31))
    flights = _flights(10, 30, serve_landing=(5.0, NET - BOX - 1.9))
    flights[0]["start_xyz_m"][2] = 0.3
    assert tag.tag(rows, flights) is None
    assert not any(f["after_point_end"] for f in flights)
