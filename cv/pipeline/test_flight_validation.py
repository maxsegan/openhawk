import numpy as np

from cv.pipeline import flight_validation as fv


def _flight(start, end, z, kind="contact"):
    return {
        "start_frame": start,
        "end_frame": end,
        "accepted": True,
        "contacts": [{"end": "start", "kind": kind, "contact_xyz_m": [5.0, 2.0, z]}],
    }


def _dense(start, end, p0, p1, v0, v1, bounces=(), fps=50.0):
    return {
        "positions": [p0] * (1 + round((end - start) / fps * 240)),
        "start_frame": start,
        "end_frame": end,
        "start_xyz": p0,
        "end_xyz": p1,
        "velocities": [v0, v1],
        "bounces": [
            {"frame": b[0], "x": b[1]} if isinstance(b, tuple) else {"frame": b} for b in bounces
        ],
    }


def _result(flights, dense, events=()):
    return {
        "verdict": {"flights": flights, "accepted_flight_count": len(flights)},
        "measurement": {"dense_flights": dense},
        "source_plan": {"original_events": list(events)},
    }


def test_plausible_rally_passes():
    result = _result(
        [_flight(10, 60, 1.0), _flight(60, 110, 1.2)],
        [
            _dense(10, 60, [5, 2, 1], [5, 22, 1], [0, 20, 3], [0, 18, -3]),
            _dense(60, 110, [5, 22, 1.2], [5, 2, 1], [0, -20, 3], [0, -18, -3]),
        ],
    )
    rows = fv.validate(result)
    assert [r["ok"] for r in rows] == [True, True]


def test_each_reason():
    result = _result(
        [_flight(10, 20, 1.5), _flight(20, 70, 1.4), _flight(70, 90, 0.2), _flight(95, 99, 4.2)],
        [
            _dense(10, 20, [5, 2, 1.5], [6, 3, 2], [0, 5, 5], [0, 20, 0]),
            _dense(20, 70, [6, 3, 1.4], [5, 22, 0.1], [0, 20, 0], [0, 20, -2], bounces=[69.5]),
            _dense(70, 90, [5, 22, 0.2], [15, 30, 1], [12, 10, 3], [12, 10, -3]),
            _dense(95, 99, [5, 5, 4.2], [5, 9, 1], [0, 10, 0], [0, 10, 0]),
        ],
        events=[{"event_type": "bounce", "frame": 71.0}],
    )
    rows = fv.validate(result, reach=lambda f, xyz: {"excess_m": 9.0} if f == 95 else None)
    assert rows[0]["reasons"] == ["toss_like"]
    assert rows[1]["ok"]  # its junction was with the removed toss
    # A contact placed on a bounce keeps the ball going away from the net too.
    assert rows[2]["reasons"] == ["starts_at_bounce", "away_from_net"]
    assert rows[3]["reasons"] == ["launch_height", "far_from_player"]
    applied = fv.apply(result, rows)
    assert applied["verdict"]["accepted_flight_count"] == 1
    assert applied["verdict"]["flights"][2]["post_validation"]["rejected"] == [
        "starts_at_bounce",
        "away_from_net",
    ]
    assert result["verdict"]["flights"][2]["accepted"]  # input untouched


def test_mid_flight_split():
    result = _result(
        [_flight(10, 60, 1.0), _flight(60, 80, 1.4)],
        [
            _dense(10, 60, [5, 2, 1], [5, 20, 1.4], [0, 20, 3], [0, 20, 0]),
            _dense(60, 80, [5, 20, 1.4], [5, 24, 0.1], [1, 20, 0], [1, 20, -2]),
        ],
    )
    # A fake contact after the net keeps the ball going away from the net as well.
    assert fv.validate(result)[1]["reasons"] == ["mid_flight_split", "away_from_net"]


def test_away_from_net():
    # A post-point knock heads for the back fence; so does a mishit ending the point: a tag.
    result = _result(
        [_flight(10, 60, 1.0), _flight(200, 240, 2.2)],
        [
            _dense(10, 60, [5, 2, 1], [5, 22, 1], [0, 20, 3], [0, 18, -3]),
            _dense(200, 240, [4, 24.5, 2.2], [6, 33, 0], [1, 6, 3], [1, 6, -5]),
        ],
    )
    rows = fv.validate(result)
    assert rows[0]["ok"]
    assert rows[1]["ok"] and rows[1]["tags"] == ["away_from_net"]


def test_half_volley_and_low_launch():
    # Reverses at the bounce: a half volley, kept; 0.08 m alone is only a tag.
    result = _result(
        [_flight(10, 60, 1.0), _flight(60, 110, 0.08)],
        [
            _dense(10, 60, [5, 2, 1], [5, 22, 0.1], [0, 20, 3], [0, 18, -3], bounces=[59.0]),
            _dense(60, 110, [5, 22, 0.08], [5, 2, 1], [0, -20, 3], [0, -18, -3]),
        ],
        events=[{"event_type": "bounce", "frame": 59.0}],
    )
    rows = fv.validate(result)
    assert rows[1]["ok"] and rows[1]["tags"] == ["low_launch"]
    # A second implausibility makes the low launch count too.
    rows = fv.validate(result, reach=lambda f, xyz: {"excess_m": 9.0} if f == 60 else None)
    assert rows[1]["reasons"] == ["far_from_player", "low_launch"]


def test_own_side_bounce_and_removed_neighbour():
    # A launch at the near baseline that lands on its own side 0.06 s later is not a stroke;
    # the next flight's junction is then judged against the flight before it.
    result = _result(
        [_flight(10, 60, 1.0), _flight(60, 75, 1.2), _flight(75, 120, 1.0)],
        [
            _dense(10, 60, [5, 22, 1], [5, 2, 1], [0, -20, 3], [0, -18, -3]),
            _dense(
                60, 75, [5, 2, 1.2], [5, 4, 1], [0, 5, -3], [0, 5, 1], bounces=[(63.0, [5, 2.5, 0])]
            ),
            _dense(75, 120, [5, 4, 1], [5, 22, 1], [0, 6, 3], [0, 18, -3]),
        ],
    )
    rows = fv.validate(result)
    assert rows[1]["reasons"] == ["own_side_bounce"]
    assert rows[2]["ok"]


def test_plausibility_switch():
    serve = {**_flight(10, 60, 2.6), "role": "serve"}
    serve["contacts"][0]["contact_xyz_m"] = [5.0, 23.0, 2.6]
    fast = {**_flight(60, 70, 1.0), "role": "mid_rally"}
    short = _dense(10, 60, [5, 23, 2.6], [5, 13, 0.1], [0, -20, 0], [0, -10, 0], [(50, [5, 15, 0])])
    quick = _dense(60, 70, [5, 2, 1], [5, 22, 1], [0, 80, 0], [0, 80, 0])
    quick["positions"] = [[5, 2 + 20 * i / 48, 1] for i in range(49)]  # 20 m in 0.2 s
    result = _result([serve, fast], [short, quick])
    assert [r["ok"] for r in fv.validate(result, plausibility=False)] == [True, True]
    rows = fv.validate(result, plausibility=True)
    assert rows[0]["reasons"] == ["serve_short_own_side"]
    assert rows[1]["reasons"] == ["mean_speed"]
    assert rows[1]["measures"]["mean_speed_mps"] == 100.0
    short["net_hits"] = [{"frame": 40}]
    short["bounces"][0]["x"] = [5, 12.5, 0]
    assert fv.validate(result, plausibility=True)[0]["ok"]


def test_foot_reach_needs_both_players():
    P = np.array([[1000.0, 0, 960, 0], [0, 1000, 540, 0], [0, 0, 1, 10]])
    cams = {"cameras": [{"frame": 5, "supported": True, "P": P.tolist()}]}
    one = {5: [("near", [900, 500, 1000, 700])]}
    assert fv.foot_reach(cams, one, 5, [0, 0, 1]) is None


def _path(points, start, end, fps=50.0):
    """Dense flight through ``points`` (piecewise linear at 240 Hz) over ``start..end``."""
    n = 1 + round((end - start) / fps * 240)
    t = np.linspace(0, 1, n)
    knots = np.linspace(0, 1, len(points))
    path = np.stack([np.interp(t, knots, np.asarray(points, float)[:, i]) for i in range(3)], 1)
    return {
        "positions": path.tolist(),
        "start_frame": start,
        "end_frame": end,
        "start_xyz": list(points[0]),
        "end_xyz": list(points[-1]),
        "velocities": [[0, 1, 0], [0, 1, 0]],
        "bounces": [],
    }


def _trim_rows(role, dense, z=None, index_role=None):
    flight = {**_flight(dense["start_frame"], dense["end_frame"], z or dense["start_xyz"][2])}
    flight["role"] = role
    flight["contacts"][0]["contact_xyz_m"] = list(dense["start_xyz"])
    return fv.validate(_result([flight], [dense]), trim=True)[0]


def test_trim_switch_reasons():
    ok = _path([[5, 0, 2.8], [5, 12, 1.5], [5, 18, 0.03], [5, 24, 1.0]], 10, 60)
    ok["bounces"] = [{"frame": 45, "x": [5, 18, 0.03]}]
    assert _trim_rows("serve", ok)["ok"]
    assert fv.validate(_result([{**_flight(10, 60, 2.8), "role": "serve"}], [ok]), trim=False)[0][
        "ok"
    ]
    short = _path([[5, 0, 2.8], [5, 9, 0.03], [5, 20, 2.0]], 10, 60)
    short["bounces"] = [{"frame": 30, "x": [5, 9, 0.03]}]
    assert _trim_rows("serve", short)["reasons"] == ["serve_bounce_before_net"]
    short["net_hits"] = [{"frame": 20}]  # a net hit before the bounce: a net fault
    assert "serve_bounce_before_net" not in _trim_rows("serve", short)["reasons"]
    back = _path([[5, 2, 1.0], [5, 15, 1.5], [5, 10, 0.5]], 10, 60)
    assert _trim_rows("mid_rally", back)["reasons"][0] == "doubles_back"
    rising = _path([[5, 23, 1.4], [5, 12, 2.8], [5, 3, 4.2]], 10, 40)
    assert _trim_rows("final", rising)["reasons"] == ["never_descends"]
    away = _path([[5, -1, 1.0], [5, -15, 6.0], [5, -25, 9.0]], 10, 40)
    assert _trim_rows("final", away)["reasons"][:2] == ["never_descends", "away_high_or_deep"]
    deep = _path([[5, 34, 1.0], [5, 20, 1.5], [5, 10, 0.5]], 10, 40)
    assert _trim_rows("unknown_origin_flight", deep)["reasons"] == ["unknown_origin_start"]


def test_trims_keep_the_flight():
    serve = _path([[5, 0, 2.8], [5, 11.885, 0.8], [5, 11.0, 0.03], [5, 8.0, 1.0]], 10, 60)
    serve["net_hits"] = [{"frame": 26.7}]
    serve["bounces"] = [{"frame": 43.3, "x": [5, 11.0, 0.03]}]
    row = _trim_rows("serve", serve)
    assert row["ok"] and row["trim"] == {"end_frame": 26.7, "reason": "net_rebound"}
    long = _path([[5, 0, 1.0], [5, 18, 0.03], [5, 40, 1.0]], 10, 60)
    long["bounces"] = [{"frame": 30, "x": [5, 18, 0.03]}]
    row = _trim_rows("final", long)
    assert row["ok"] and row["trim"]["reason"] == "past_back_wall"
    assert 50 < row["trim"]["end_frame"] < 60
