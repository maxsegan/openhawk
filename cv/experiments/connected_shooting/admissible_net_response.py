"""Admissible-set net response: a searched v_out in the box, replayed by using_response.

The stored outgoing velocity is not a material law. Replay must wrap
``using_response`` from the verdict flight witness; a configuration receipt
alone is how the first gate wave voided itself.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import numpy as np

from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity
from cv.experiments.connected_shooting.labeled_passive_tape import using_response
from cv.pipeline import net_cord_response as cord

MODEL = "admissible_net_response_v1"


class AdmissibleSetResponse(FreeNetVelocity):
    """Assigned outgoing velocity that must stay inside the admissible box."""

    @property
    def key(self):
        return (MODEL,) + self.outgoing_velocity_mps

    def velocity(self, incoming):
        outgoing = super().velocity(incoming)
        if cord.in_box(incoming, outgoing):
            return outgoing
        return cord.project_to_box(incoming, outgoing)

    def record(self, incoming, outgoing):
        receipt = super().record(incoming, outgoing)
        incoming = np.asarray(incoming, float)
        outgoing = np.asarray(outgoing, float)
        receipt.update(
            model=MODEL,
            speed_ratio=cord.speed_ratio(incoming, outgoing),
            in_box=bool(cord.in_box(incoming, outgoing)),
            crossing_sign=float(np.sign(outgoing[1])),
            spin_unchanged=True,
            contact_geometry=(
                "admissible-set outgoing velocity at the tape band; not a VR tape clip"
            ),
        )
        return receipt


def from_record(record: dict) -> AdmissibleSetResponse:
    if not isinstance(record, dict) or "outgoing_velocity_mps" not in record:
        raise ValueError("admissible net response record required")
    return AdmissibleSetResponse(record["outgoing_velocity_mps"])


def from_existence(result: cord.ExistenceResult) -> AdmissibleSetResponse:
    return AdmissibleSetResponse(tuple(float(v) for v in result.v_out))


def search_from_chain(scene, parameters, fps: float) -> dict | None:
    """Search an admissible v_out from a fitted incoming net hit. None if no hit."""
    from cv.experiments.connected_shooting import model

    try:
        flights = model.chain(scene, np.asarray(parameters, float))
    except (ValueError, np.linalg.LinAlgError):
        return None
    for index, flight in enumerate(flights):
        hits = flight.get("net_hits") or []
        if len(hits) != 1:
            continue
        hit = hits[0]
        frames = np.asarray(scene.observation_frames[index], float)
        positions = np.asarray(flight["positions"], float)
        if frames.size != len(positions):
            frames = np.linspace(flight["start_frame"], flight["end_frame"], len(positions))
        post = [
            (float(frame), np.asarray(xyz, float))
            for frame, xyz in zip(frames, positions, strict=True)
            if float(frame) > float(hit["frame"]) + 1e-6
        ]
        if len(post) < 2:
            continue
        result = cord.search_admissible_v_out(
            hit["v_in"],
            hit["x"],
            float(hit["frame"]),
            post[:12],
            float(fps),
            hit.get("w_in"),
        )
        response = from_existence(result)
        record = response.record(hit["v_in"], result.v_out)
        record["outgoing_velocity_mps"] = list(response.outgoing_velocity_mps)
        record["net_cord_response"] = cord.flight_witness(hit, result, cord.ADMISSIBLE_SET)
        return record
    return None


@contextmanager
def using_admissible(
    response: AdmissibleSetResponse | None,
    mode: str = cord.ADMISSIBLE_SET,
    h_tol: float | None = None,
):
    """Tape-band eligibility plus the searched outgoing velocity."""
    if response is None:
        with cord.using_mode(mode, h_tol=h_tol):
            yield
        return
    with cord.using_mode(mode, h_tol=h_tol), using_response(response):
        yield


def widen_image_residual(image, scene):
    """Divide post-contact image residual rows by 3 when the admissible set is on."""
    if cord.active_mode() != cord.ADMISSIBLE_SET or scene.net_hit_frames is None:
        return image
    scaled = np.asarray(image, float).copy()
    offset = 0
    for index, frames in enumerate(scene.observation_frames):
        nets = scene.net_hit_frames[index]
        count = len(frames)
        if len(nets) and scaled.ndim == 2:
            net_frame = float(nets[0])
            for j, frame in enumerate(frames):
                if float(frame) > net_frame:
                    scaled[offset + j] /= cord.POST_CONTACT_WIDENING
        offset += count
    return scaled


def replay_from_verdict(report: dict):
    """Read the response off verdict flight rows. Empty if the search wrote none.

    Keying this off ``report['configuration']`` is the void-wave failure: several
    writers build that dict independently, so a report can carry new targets and
    none of the keys. The flight witness is self-describing.
    """
    flights = (report.get("verdict") or {}).get("flights") or []
    witnesses = [row.get("net_cord_response") for row in flights if row.get("net_cord_response")]
    if not witnesses:
        return nullcontext()
    evidence = [row for row in witnesses if row.get(cord.FIELD) == cord.EVIDENCE_BOUND]
    if evidence:
        admitted_rows = [row for row in evidence if row.get("admitted")]
        if not admitted_rows:
            return nullcontext()
        bounds = cord.bounds_from_records(admitted_rows)
        admitted = [bound for bound in bounds if bound.admitted]
        from cv.experiments.connected_shooting.labeled_passive_tape import using_response
        from cv.experiments.connected_shooting.observation_net_seed import _compose

        velocity = admitted_rows[-1].get("outgoing_velocity_mps") or admitted_rows[-1].get(
            "v_out_mps"
        )
        locked = nullcontext()
        if velocity is not None:
            locked = using_response(from_record({"outgoing_velocity_mps": velocity}))
        width = admitted_rows[-1].get("h_tol_m")
        return _compose(cord.using_evidence(admitted, h_tol=width), locked)
    record = witnesses[-1]
    if record.get(cord.FIELD) != cord.ADMISSIBLE_SET and record.get("model") != MODEL:
        return nullcontext()
    response = from_record(record if "outgoing_velocity_mps" in record else {
        "outgoing_velocity_mps": record["v_out_mps"]
    })
    return using_admissible(response, h_tol=record.get("h_tol_m"))


def attach_flight_witness(flights: list[dict], report_or_mode) -> list[dict]:
    """Copy the search's response onto every net-hit verdict row."""
    mode = (
        report_or_mode
        if isinstance(report_or_mode, str)
        else (report_or_mode.get("configuration") or {}).get(cord.FIELD, cord.TAPE_CLIP)
    )
    if mode != cord.ADMISSIBLE_SET:
        return flights
    source = None
    if not isinstance(report_or_mode, str):
        source = (report_or_mode.get("net_response") or {}).copy()
    attached = []
    for row in flights:
        hits = row.get("modeled_net_hits") or row.get("net_hits") or []
        if not hits:
            attached.append(row)
            continue
        witness = row.get("net_cord_response")
        if witness is None and source is not None:
            witness = {
                **cord.receipt(mode),
                "v_out_mps": source.get("outgoing_velocity_mps") or source.get("v_out_mps"),
                "outgoing_velocity_mps": source.get("outgoing_velocity_mps")
                or source.get("v_out_mps"),
                "model": MODEL,
            }
        attached.append(row if witness is None else {**row, "net_cord_response": witness})
    return attached
