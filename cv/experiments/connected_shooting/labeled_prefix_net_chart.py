"""Declared-net chart coordinates for movable following flights of the serve prefix.

Opened labeled experiment option, off by default. A movable following flight that
carries exactly one declared net whose original interval lies inside it trades its
raw (Vy, Vz) launch coordinates for the net stage's own chart coordinates: the net
epoch offset inside the original interval and the mesh-height fraction. Every trial
then reaches the plane at a declared epoch at a height inside the mesh, so objective
finite differences never straddle the hit/miss eligibility edge. The unchanged
collision simulator still decides the branch on replay: a trial without exactly one
hit at the chart epoch is invalid, never a penalty win. Topology-only; abstains
with a recorded reason on other topologies. No case or player branches.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting import full_native_continuation as full
from cv.experiments.connected_shooting.labeled_net_height_chart import project_net_height
from cv.experiments.connected_shooting.model import R_BALL
from cv.pipeline.physics_knot_solver import net_tape_height

# Same lower numerical margin as the net stage's mesh-height coordinate (about 10um
# above simultaneous ground/net contact); the same margin below the tape top keeps a
# converged chart height on the eligible side of `net_collision._collision_eligible`.
FRACTION_BOUNDS = (1e-5, 1.0 - 1e-5)
NET_CENTRE_X_M = 5.485
NET_HALF_WIDTH_M = 6.4
EPOCH_TOLERANCE_FRAMES = 1e-6


def coordinate(flight: int) -> int:
    """First (Vy) slot of the following-launch block that the chart replaces."""
    return 10 + 6 * (flight - 1) + 1


def specs(context, movable: int) -> list[dict]:
    """Inventory movable following flights with one whole declared net interval."""
    scene = context["scene"]
    groups = scene.net_hit_frames
    rows = []
    for flight in range(1, movable):
        start, end = map(float, scene.contact_frames[flight : flight + 2])
        declared = () if groups is None else tuple(float(t) for t in groups[flight])
        reason, frame, interval = None, None, None
        if len(declared) != 1:
            reason = (
                "no declared net in flight"
                if not declared
                else "more than one declared net in flight"
            )
        else:
            frame = declared[0]
            events = [
                e
                for e in context["events"]
                if e["event_type"] == "net_hit" and abs(float(e["frame"]) - frame) <= 1e-8
            ]
            if len(events) != 1:
                reason = "declared net without exactly one original event"
            else:
                low, high = map(float, events[0]["frame_interval"])
                if not (
                    np.isfinite([low, high]).all()
                    and low < high
                    and start < low
                    and high < end
                    and low <= frame <= high
                ):
                    reason = "original net interval must lie strictly inside the flight"
                else:
                    interval = [low, high]
        row = dict(flight=flight, supported=reason is None, reason=reason)
        if getattr(scene, "terminal_net_tail", None) is not None and flight == len(scene.pixels) - 1:
            row.update(
                supported=False,
                reason="terminal-tail ground/net ordering is unresolved; airborne net chart unsupported",
            )
            reason = row["reason"]
        if reason is None:
            row.update(frame=frame, interval=interval, coordinate=coordinate(flight))
        rows.append(row)
    return rows


class DeclaredNetChart:
    """Solve a following flight's Vy/Vz so it reaches the declared net inside the mesh."""

    def __init__(self, scene, spec: dict):
        self.scene = scene
        self.flight = int(spec["flight"])
        self.mid = float(spec["frame"])
        low, high = map(float, spec["interval"])
        self.offset_bounds = (low - self.mid, high - self.mid)
        self.n = len(scene.contact_frames) - 1

    def project(self, parameters, offset: float, fraction: float):
        p = np.asarray(parameters, float).copy()
        n, f = self.n, self.flight
        if p.shape != (5 + 6 * n,) or not np.isfinite(p).all():
            raise ValueError("finite 5+6n single-shooting parameters required")
        if not self.offset_bounds[0] <= offset <= self.offset_bounds[1]:
            raise ValueError("net epoch offset outside the original interval")
        if not FRACTION_BOUNDS[0] <= fraction <= FRACTION_BOUNDS[1]:
            raise ValueError("mesh fraction outside the chart bounds")
        # The same single-shooting prefix the objective integrates supplies the start;
        # the chart flight's own launch cannot influence it. Earlier-flight traces are
        # cached, so the objective's later chain call reuses them.
        start = np.asarray(full.model.chain(self.scene, p)[f - 1]["end_xyz"], float)
        theta = np.r_[start, p[3 + 3 * f : 6 + 3 * f], p[3 + 3 * n + 3 * f : 6 + 3 * n + 3 * f]]
        epoch = self.mid + float(offset)
        projected, receipt = project_net_height(
            theta,
            float(self.scene.contact_frames[f]),
            epoch,
            float(self.scene.fps),
            self.scene.surface,
            mesh_fraction=float(fraction),
            bounce_profile=self.scene.bounce_profile,
            rebound_scales=p[-2:],
        )
        p[4 + 3 * f] = projected[4]
        p[5 + 3 * f] = projected[5]
        x = float(receipt["net_xyz_m"][0])
        return p, dict(
            receipt,
            flight=f,
            net_epoch=epoch,
            epoch_offset=float(offset),
            mesh_fraction=float(fraction),
            connected_start_xyz_m=start.tolist(),
            width_slack_m=NET_HALF_WIDTH_M - abs(x - NET_CENTRE_X_M),
        )


def seed(chain, spec: dict) -> dict:
    """Chart coordinates of the same-invocation physical net hit, restored into bounds.

    Mirrors the net stage's same_invocation_net_state_or_mesh_midpoint initialization.
    A hit at the mesh top or interval edge is clipped into the chart bounds and the
    shift is disclosed; such a source is not an exactly representable incumbent.
    """
    flight = int(spec["flight"])
    hits = chain[flight].get("net_hits", [])
    low, high = map(float, spec["interval"])
    mid = float(spec["frame"])
    record = dict(
        flight=flight,
        coordinate=spec["coordinate"],
        physical_hit_count=len(hits),
        interval=[low, high],
        fraction_bounds=list(FRACTION_BOUNDS),
    )
    if len(hits) != 1:
        record.update(
            mode="mesh_midpoint",
            source_representable=False,
            restored=False,
            epoch_offset=0.0,
            mesh_fraction=0.5,
            reason="source does not have exactly one eligible net hit in the flight",
        )
        return record
    hit = hits[0]
    frame = float(hit["frame"])
    xyz = np.asarray(hit["x"], float)
    tape = float(net_tape_height(float(xyz[0])))
    fraction = (float(xyz[2]) - R_BALL) / tape
    offset = frame - mid
    inside = low <= frame <= high and FRACTION_BOUNDS[0] <= fraction <= FRACTION_BOUNDS[1]
    restored_offset = float(np.clip(offset, low - mid, high - mid))
    restored_fraction = float(np.clip(fraction, *FRACTION_BOUNDS))
    record.update(
        mode="same_invocation_net_state",
        source_net_frame=frame,
        source_net_xyz_m=xyz.tolist(),
        source_mesh_fraction=float(fraction),
        source_representable=bool(inside),
        restored=not inside,
        epoch_offset=restored_offset,
        mesh_fraction=restored_fraction,
        reason=None if inside else "source net hit at a chart bound; restored inside the bounds",
        restoration=None
        if inside
        else dict(
            epoch_shift_frames=restored_offset - offset,
            height_shift_m=(restored_fraction - fraction) * tape,
        ),
    )
    return record


def replay_check(chain, spec: dict, chart_epoch: float) -> dict:
    """Require the replayed flight to hit the declared net at the chart epoch, in interval."""
    flight = int(spec["flight"])
    hits = chain[flight].get("net_hits", [])
    if len(hits) != 1:
        raise ValueError("declared-net chart trial left the supported collision branch")
    hit = hits[0]
    frame = float(hit["frame"])
    if abs(frame - float(chart_epoch)) > EPOCH_TOLERANCE_FRAMES:
        raise ValueError("replayed net epoch differs from the chart epoch")
    low, high = map(float, spec["interval"])
    if not low <= frame <= high:
        raise ValueError("replayed net hit outside the original interval")
    xyz = np.asarray(hit["x"], float)
    tape = float(hit["tape_height_m"])
    after = [b for b in chain[flight].get("bounces", []) if float(b["frame"]) > frame]
    return dict(
        flight=flight,
        frame=frame,
        epoch_error_frames=frame - float(chart_epoch),
        xyz_m=xyz.tolist(),
        tape_height_m=tape,
        margin_below_top_m=tape + R_BALL - float(xyz[2]),
        margin_above_bottom_m=float(xyz[2]) - R_BALL,
        width_slack_m=NET_HALF_WIDTH_M - abs(float(xyz[0]) - NET_CENTRE_X_M),
        following_ground_count=len(after),
        first_following_ground_frame=float(after[0]["frame"]) if after else None,
    )
