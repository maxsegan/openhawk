"""Lean component receipts reference the labels slot by file record; the guard follows it."""

import hashlib
import json

from cv.pipeline import s6_input_origin as origin


def _record(tmp_path, monkeypatch, document: dict) -> dict:
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    data = json.dumps(document).encode()
    path = tmp_path / "processed" / "doc.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path_base": "TENNIS_DATA_ROOT",
        "path": "processed/doc.json",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_referenced_automatic_consumer_document_is_an_automatic_slot(tmp_path, monkeypatch):
    record = _record(
        tmp_path,
        monkeypatch,
        {"schema": origin.CONSUMER_SCHEMA, "annotation_origin": origin.AUTOMATIC},
    )
    view = origin.guard_view({"contact_component_source": {"labels": record}})
    assert "automatic_consumer_slot" in view["contact_component_source"]
    assert "labels" not in view["contact_component_source"]


def test_referenced_human_labels_stay_in_the_labels_slot(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch, {"schema": "human_labels_v1"})
    view = origin.guard_view({"contact_component_source": {"labels": record}})
    assert "labels" in view["contact_component_source"]


def test_a_reference_whose_bytes_changed_stays_in_the_labels_slot(tmp_path, monkeypatch):
    record = _record(
        tmp_path,
        monkeypatch,
        {"schema": origin.CONSUMER_SCHEMA, "annotation_origin": origin.AUTOMATIC},
    )
    record = dict(record, sha256="0" * 64)
    view = origin.guard_view({"contact_component_source": {"labels": record}})
    assert "labels" in view["contact_component_source"]
