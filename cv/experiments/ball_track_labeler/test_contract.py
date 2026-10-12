import copy

import pytest

from cv.experiments.ball_track_labeler.contract import (
    BENCHMARK_SCHEMA,
    LABEL_SCHEMA,
    validate_benchmark,
    validate_labels,
)


def _benchmark() -> dict:
    return {
        "schema": BENCHMARK_SCHEMA,
        "benchmark_id": "test",
        "status": "development",
        "resolution_contract": {"authority": "native_1920x1080"},
        "cases": [
            {
                "case_id": "m__pt1__f10",
                "match_id": "m",
                "clip": "pt1",
                "center_frame": 10,
                "frames": [
                    {
                        "frame": frame,
                        "image_url": f"f{frame}.jpg",
                        "source_size": [1920, 1080],
                    }
                    for frame in (9, 10, 11)
                ],
            }
        ],
    }


def _labels() -> dict:
    return {
        "schema": LABEL_SCHEMA,
        "benchmark_id": "test",
        "resolution_contract": {
            "authority": "native_1920x1080",
            "legacy_status": "LEGACY_BAD_SHOULD_UPDATE",
        },
        "records": [
            {
                "case_id": "m__pt1__f10",
                "window_status": "repaired",
                "frames": [
                    {
                        "frame": 9,
                        "status": "visible",
                        "x1080": 2.0,
                        "y1080": 4.0,
                        "x540": 1.0,
                        "y540": 2.0,
                    },
                    {
                        "frame": 10,
                        "status": "occluded",
                        "x1080": None,
                        "y1080": None,
                        "x540": None,
                        "y540": None,
                    },
                    {
                        "frame": 11,
                        "status": "visible",
                        "x1080": 6.0,
                        "y1080": 8.0,
                        "x540": 3.0,
                        "y540": 4.0,
                    },
                ],
            }
        ],
    }


def test_public_benchmark_rejects_truth_fields() -> None:
    benchmark = _benchmark()
    validate_benchmark(benchmark)
    benchmark["cases"][0]["frames"][0]["expected_x540"] = 1
    with pytest.raises(ValueError, match="forbidden truth field"):
        validate_benchmark(benchmark)


def test_labels_require_every_frame_and_null_nonvisible_xy() -> None:
    benchmark = _benchmark()
    labels = _labels()
    validate_labels(labels, benchmark)
    incomplete = copy.deepcopy(labels)
    incomplete["records"][0]["frames"].pop()
    with pytest.raises(ValueError, match="every benchmark frame"):
        validate_labels(incomplete, benchmark)
    invalid = copy.deepcopy(labels)
    invalid["records"][0]["frames"][1]["x540"] = 2
    with pytest.raises(ValueError, match="non-visible xy"):
        validate_labels(invalid, benchmark)


def test_labels_restrict_inferred_coordinates_to_occlusion_and_frame() -> None:
    benchmark = _benchmark()
    labels = _labels()
    labels["records"][0]["frames"][1]["inferred_x1080"] = 4.0
    labels["records"][0]["frames"][1]["inferred_y1080"] = 8.0
    validate_labels(labels, benchmark)
    labels["records"][0]["frames"][1]["status"] = "outside_frame"
    with pytest.raises(ValueError, match="require occluded"):
        validate_labels(labels, benchmark)
    labels["records"][0]["frames"][1]["status"] = "occluded"
    labels["records"][0]["frames"][1]["inferred_x1080"] = -1.0
    with pytest.raises(ValueError, match="inferred x outside"):
        validate_labels(labels, benchmark)
