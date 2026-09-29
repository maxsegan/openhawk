"""Fail-closed contracts for sequence-level 2D ball labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BENCHMARK_SCHEMA = "tennis_ball_track_window_benchmark_v1"
LABEL_SCHEMA = "tennis_ball_track_sequence_labels_v1"
FRAME_STATUSES = frozenset({"visible", "occluded", "not_visible", "outside_frame", "ambiguous"})
WINDOW_STATUSES = frozenset({"correct", "repaired", "unusable", "ambiguous"})
FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "truth",
        "expected",
        "expected_x540",
        "expected_y540",
        "owner_note",
        "corrected_x540",
        "corrected_y540",
    }
)
NATIVE_WIDTH = 1920
NATIVE_HEIGHT = 1080
LEGACY_STATUS = "LEGACY_BAD_SHOULD_UPDATE"
NATIVE_RESOLUTION_CONTRACT = {
    "authority": "native_1920x1080",
    "minimum_source_resolution": [NATIVE_WIDTH, NATIVE_HEIGHT],
    "legacy_mirrors": ["x540", "y540"],
    "legacy_status": LEGACY_STATUS,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_truth_fields(value: Any, path: str = "benchmark") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require(key not in FORBIDDEN_PUBLIC_FIELDS, f"forbidden truth field at {path}.{key}")
            _reject_truth_fields(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_truth_fields(nested, f"{path}[{index}]")


def validate_benchmark(payload: Mapping[str, Any]) -> None:
    """Validate an agent-visible benchmark manifest and reject truth leakage."""
    _require(payload.get("schema") == BENCHMARK_SCHEMA, "unsupported benchmark schema")
    _require(bool(payload.get("benchmark_id")), "benchmark_id is required")
    _require(payload.get("status") in {"development", "sealed_transfer"}, "invalid status")
    resolution = payload.get("resolution_contract")
    _require(isinstance(resolution, Mapping), "native resolution_contract is required")
    _require(
        resolution.get("authority") == "native_1920x1080",
        "tracking authority must be native_1920x1080",
    )
    cases = payload.get("cases")
    _require(isinstance(cases, list) and cases, "benchmark must contain cases")
    _reject_truth_fields(payload)
    seen: set[str] = set()
    for case in cases:
        _require(isinstance(case, Mapping), "case must be an object")
        case_id = str(case.get("case_id") or "")
        _require(case_id and case_id not in seen, f"duplicate or missing case_id: {case_id}")
        seen.add(case_id)
        frames = case.get("frames")
        _require(isinstance(frames, list) and len(frames) >= 3, f"{case_id}: too few frames")
        frame_numbers = [int(frame["frame"]) for frame in frames]
        _require(
            frame_numbers == sorted(set(frame_numbers)), f"{case_id}: frames not unique/sorted"
        )
        _require(int(case["center_frame"]) in frame_numbers, f"{case_id}: center frame absent")
        for frame in frames:
            _require(bool(frame.get("image_url")), f"{case_id}: frame image_url missing")
            source_size = frame.get("source_size")
            _require(
                isinstance(source_size, list)
                and len(source_size) == 2
                and int(source_size[0]) >= NATIVE_WIDTH
                and int(source_size[1]) >= NATIVE_HEIGHT,
                f"{case_id}: source evidence below 1920x1080",
            )
            estimate = frame.get("estimate")
            if estimate is not None:
                _require(
                    0 <= float(estimate["x1080"]) <= NATIVE_WIDTH,
                    f"{case_id}: estimate x outside image",
                )
                _require(
                    0 <= float(estimate["y1080"]) <= NATIVE_HEIGHT,
                    f"{case_id}: estimate y outside image",
                )


def validate_labels(payload: Mapping[str, Any], benchmark: Mapping[str, Any]) -> None:
    """Validate complete sequence labels against the public benchmark contract."""
    validate_benchmark(benchmark)
    _require(payload.get("schema") == LABEL_SCHEMA, "unsupported label schema")
    _require(payload.get("benchmark_id") == benchmark.get("benchmark_id"), "benchmark mismatch")
    resolution = payload.get("resolution_contract")
    _require(isinstance(resolution, Mapping), "label resolution_contract is required")
    _require(
        resolution.get("authority") == "native_1920x1080",
        "label tracking authority must be native_1920x1080",
    )
    _require(
        resolution.get("legacy_status") == LEGACY_STATUS,
        f"legacy mirrors must be marked {LEGACY_STATUS}",
    )
    records = payload.get("records")
    _require(isinstance(records, list), "records must be a list")
    by_case = {case["case_id"]: case for case in benchmark["cases"]}
    seen: set[str] = set()
    for record in records:
        case_id = str(record.get("case_id") or "")
        _require(
            case_id in by_case and case_id not in seen, f"unknown or duplicate case: {case_id}"
        )
        seen.add(case_id)
        _require(
            record.get("window_status") in WINDOW_STATUSES, f"{case_id}: invalid window status"
        )
        frames = record.get("frames")
        _require(isinstance(frames, list), f"{case_id}: frames must be a list")
        expected_frames = {int(row["frame"]) for row in by_case[case_id]["frames"]}
        actual_frames = {int(row["frame"]) for row in frames}
        _require(
            actual_frames == expected_frames, f"{case_id}: every benchmark frame must be labeled"
        )
        for frame in frames:
            status = frame.get("status")
            _require(status in FRAME_STATUSES, f"{case_id}: invalid frame status {status}")
            x_value, y_value = frame.get("x540"), frame.get("y540")
            x_native, y_native = frame.get("x1080"), frame.get("y1080")
            if status == "visible":
                _require(
                    x_value is not None and y_value is not None, f"{case_id}: visible needs xy"
                )
                _require(0 <= float(x_value) <= 960, f"{case_id}: x540 outside image")
                _require(0 <= float(y_value) <= 540, f"{case_id}: y540 outside image")
                _require(
                    x_native is not None and y_native is not None,
                    f"{case_id}: native xy is authoritative and required",
                )
                _require(0 <= float(x_native) <= 1920, f"{case_id}: x1080 outside image")
                _require(0 <= float(y_native) <= 1080, f"{case_id}: y1080 outside image")
                _require(
                    abs(float(x_native) - 2 * float(x_value)) <= 0.02,
                    f"{case_id}: x1080/x540 mismatch",
                )
                _require(
                    abs(float(y_native) - 2 * float(y_value)) <= 0.02,
                    f"{case_id}: y1080/y540 mismatch",
                )
            else:
                _require(
                    x_value is None and y_value is None, f"{case_id}: non-visible xy must be null"
                )
                _require(
                    x_native is None and y_native is None,
                    f"{case_id}: non-visible native xy must be null",
                )
            inferred_x = frame.get("inferred_x1080")
            inferred_y = frame.get("inferred_y1080")
            _require(
                (inferred_x is None) == (inferred_y is None),
                f"{case_id}: inferred native coordinate pair is incomplete",
            )
            if inferred_x is not None:
                _require(
                    status == "occluded",
                    f"{case_id}: inferred coordinates require occluded status",
                )
                _require(0 <= float(inferred_x) <= 1920, f"{case_id}: inferred x outside image")
                _require(0 <= float(inferred_y) <= 1080, f"{case_id}: inferred y outside image")
            radius = frame.get("uncertainty_radius_px540")
            if radius is not None:
                _require(float(radius) >= 0, f"{case_id}: negative uncertainty")
