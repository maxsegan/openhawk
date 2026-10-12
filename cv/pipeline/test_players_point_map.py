from players import rally_points


def test_rally_points_reads_requested_versioned_map(tmp_path) -> None:
    (tmp_path / "point_video_map.csv").write_text(
        "pt,t_start,t_end,rally_t_start,rally_t_end,n_runs\n1,1,2,,,0\n"
    )
    (tmp_path / "point_video_map_v2.csv").write_text(
        "pt,t_start,t_end,rally_t_start,rally_t_end,n_runs\n1,1,2,10,12,1\n"
    )

    assert rally_points(str(tmp_path), "point_video_map_v2.csv") == [
        {"pt": 1, "t0": 10.0, "t1": 12.0}
    ]
