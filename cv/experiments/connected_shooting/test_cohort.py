"""The discovered cohort must be honest about where every field came from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv.experiments.connected_shooting import cohort, labeled_attempt_sweep as sweep

LABELS = Path("cv/validation/labels/s6_agent_inputs_v1")


@pytest.fixture(scope="module")
def discovered() -> list[cohort.DiscoveredAttempt]:
    return cohort.discover(LABELS.resolve())


def test_every_complete_attempt_is_found_once(discovered):
    keys = [item.key for item in discovered]
    assert len(keys) == len(set(keys))
    assert len(discovered) >= 80
    assert len({item.match_id for item in discovered}) >= 30


def test_the_frozen_seventeen_keep_their_published_configuration(discovered):
    published = {case.label_file: case for case in (*sweep.CASES, *sweep.NEW_CASES)}
    by_key = {item.key: item for item in discovered}
    matched = 0
    for item in by_key.values():
        names = [item.label_file, *item.provenance["superseded_revisions"]]
        case = next((published[name] for name in names if name in published), None)
        if case is None:
            continue
        matched += 1
        assert item.serve_number == case.serve_number
        assert item.surface == case.surface
        assert [name for name, _ in item.player_statures_m] == [
            name for name, _ in case.player_statures_m
        ]
        assert item.provenance["serve_number_source"] == "frozen_reference_configuration"
    assert matched == len(published)


def test_a_one_contact_attempt_names_only_its_server(discovered):
    for item in discovered:
        if item.contact_count == 1:
            assert len(item.player_statures_m) == 1
        assert len(item.player_statures_m) <= 2


def test_every_derived_field_records_its_derivation(discovered):
    for item in discovered:
        for field in (
            "surface_source",
            "gender_source",
            "serve_number_source",
            "player_order_source",
            "pose_image_scale_source",
        ):
            assert item.provenance[field]
        assert item.surface in cohort.SURFACES
        assert item.gender in ("men", "women")
        assert item.serve_number in (1, 2)


def test_a_named_dense_revision_exists_and_is_not_the_attempt_itself(discovered):
    for item in discovered:
        if item.dense_label_file is None:
            continue
        assert item.dense_label_file != item.label_file
        assert (LABELS / item.dense_label_file).is_file()


def test_the_cohort_manifest_is_serialisable(discovered, tmp_path):
    from dataclasses import asdict

    path = tmp_path / "cohort.json"
    path.write_text(json.dumps([asdict(item) for item in discovered], allow_nan=False))
    assert json.loads(path.read_text())


def test_grass_is_fitted_on_a_measured_profile_only_when_asked():
    case = sweep.AttemptCase("k", "f.json", "grass", 2)
    assert sweep.fitted_surface(case, "hold") == "grass"
    assert sweep.fitted_surface(case, "hard") == "hard"
    for surface in sweep.MEASURED_BOUNCE_SURFACES:
        measured = sweep.AttemptCase("k", "f.json", surface, 2)
        assert sweep.fitted_surface(measured, "hard") == surface
