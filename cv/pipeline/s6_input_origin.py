"""Validate automatic ancestry at the shared S6 observation boundary.

Legacy input-slot names describe compatibility shapes, not their origin. Automatic
inputs require producer-bound artifacts; a caller-supplied eligibility flag is insufficient.
"""

from __future__ import annotations

import json
from pathlib import Path

from cv.pipeline import provenance

AUTOMATIC = "automatic"
LABELED = "labeled"
STREAMS = {"ball", "events", "camera", "players"}


def origin(row: dict) -> str:
    value = row.get("observation_origin", LABELED)
    if value not in (LABELED, AUTOMATIC):
        raise ValueError("explicit labeled or automatic observation origin required")
    if value == LABELED and row.get("automatic_provenance") is not None:
        raise ValueError("labeled observations cannot declare automatic provenance")
    return value


CONSUMER_SCHEMA = "s6_automatic_observation_document_v1"


def _automatic_slot(parent: str, key: str, value) -> bool:
    """An automatic consumer document (or its digest) filed under the legacy ``labels`` slot.

    Component contracts bind their source documents by slot name (``packet``, ``labels``,
    ``cameras``). For automatic rows that slot holds the automatic consumer document.
    """
    if key != "labels":
        return False
    if isinstance(value, dict):
        return (
            value.get("schema") == CONSUMER_SCHEMA and value.get("annotation_origin") == AUTOMATIC
        )
    return (
        parent == "source_bindings"
        and isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def guard_view(document, parent: str = ""):
    """The document for the name-based human-input guard, with automatic slots renamed.

    Renamed, not removed: the contents of an automatic consumer document are still checked.
    """
    if isinstance(document, list):
        return [guard_view(value, parent) for value in document]
    if not isinstance(document, dict):
        return document
    return {
        ("automatic_consumer_slot" if _automatic_slot(parent, str(key), value) else key): (
            guard_view(value, str(key))
        )
        for key, value in document.items()
    }


def _identity(record: dict) -> tuple:
    return tuple(record.get(key) for key in ("path_base", "path", "sha256"))


def derived_producer(
    row: dict,
    output: Path,
    added: list[dict],
    configuration: dict,
    resolve,
    models: list[dict] | None = None,
) -> Path:
    """Provenance of automatic inputs rewritten from an automatic row.

    The original producer stays the parent: same source video and artifacts, except the
    row's labels and packet, which are replaced by the ``added`` records (the rewritten
    pair plus whatever the rewrite read). No human or reviewed input is added.
    """
    parent_path = resolve(row["automatic_provenance"])
    if provenance.file_sha256(parent_path) != row["automatic_provenance"].get("sha256"):
        raise ValueError("parent automatic provenance digest changed")
    parent = provenance.load_provenance(parent_path, require_automatic=True)
    replaced = {_identity(row["labels"]), _identity(row["packet"])}
    reused = [record for record in parent["reused_artifacts"] if _identity(record) not in replaced]
    reused += [
        *added,
        provenance.file_record(parent_path, role="parent_automatic_provenance"),
    ]
    document = provenance.build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=provenance.AUTOMATIC_MODE,
        source_videos=parent["source_videos"],
        models=list(parent["models"]) + list(models or []),
        configuration={**configuration, "parent_configuration": parent["configuration"]},
        reused_artifacts=reused,
        fallbacks=parent["fallbacks"],
    )
    return provenance.write_provenance(output, document)


def validate_automatic(row: dict, inputs: dict[str, Path], resolve) -> None:
    """Validate bound producer provenance and the exact four executed input artifacts."""
    if origin(row) != AUTOMATIC:
        return
    if row.get("declared_flights") is not None:
        raise ValueError("automatic inference cannot accept a reference-flight denominator")
    binding = row.get("automatic_provenance")
    if not isinstance(binding, dict):
        raise ValueError("automatic observations require producer provenance")
    path = resolve(binding)
    if provenance.file_sha256(path) != binding.get("sha256"):
        raise ValueError("automatic producer provenance digest changed")
    producer = provenance.load_provenance(path, require_automatic=True)
    if not producer["source_videos"]:
        raise ValueError("automatic observation provenance requires a source video")
    bound = {_identity(record) for record in producer["reused_artifacts"]}
    for name in inputs:
        if _identity(row[name]) not in bound:
            raise ValueError(f"automatic producer does not bind executed {name} artifact")
    sidecar = inputs["pose_csv"].with_name(inputs["pose_csv"].name + ".coordinates.json")
    # The player producer declares coordinate scale and source; implicit dimensions
    # would allow ambient files to change the same supposedly frozen observation row.
    if not sidecar.is_file() or _identity(provenance.file_record(sidecar)) not in bound:
        raise ValueError("automatic producer must bind original player coordinates")
    consumer, packet, cameras = (
        json.loads(inputs[name].read_text()) for name in ("labels", "packet", "cameras")
    )
    if consumer.get("schema") != "s6_automatic_observation_document_v1":
        raise ValueError("automatic consumer document schema required")
    if consumer.get("annotation_origin") != AUTOMATIC or packet.get("human_derived") is not False:
        raise ValueError("automatic observations cannot contain human-derived streams")
    for document in (consumer, packet, cameras):
        if document.get("human_derived") or document.get("human_derived_inputs"):
            raise ValueError("automatic observation artifact declares human-derived evidence")
        provenance.assert_automatic_document(
            guard_view(document), context="shared S6 automatic observations"
        )
        if document.get("automatic_inference_eligible") is not True:
            raise ValueError("automatic observation artifact is not eligible")
    attempt = packet["attempts"][0]
    for document in (consumer, packet, attempt):
        streams = document.get("stream_origins", {})
        if any(streams.get(name) != AUTOMATIC for name in STREAMS):
            raise ValueError("all executed S6 streams must have automatic origin")
    from cv.experiments.connected_shooting import observation_operator

    if observation_operator.from_packet(packet) != observation_operator.declaration(
        "nominal_center"
    ):
        raise ValueError("automatic detector observations require their native center operator")


def serve_prior_path(row: dict, policy: dict, resolve) -> Path | None:
    """An automatic invocation names its trained prior, never an ambient default."""
    record = policy.get("serve_location_prior")
    if record is None:
        if origin(row) == AUTOMATIC:
            raise ValueError("automatic S6 requires an explicitly bound serve-location prior")
        return None
    if not isinstance(record, dict):
        raise ValueError("serve-location prior must be a bound model record")
    path = resolve(record)
    if provenance.file_sha256(path) != record.get("sha256"):
        raise ValueError("declared serve-location prior digest changed")
    return path
