"""Hard serve-contact reach cylinder: a cut of the contact image ray, not a soft term.

The soft stature prior in ``athlete_priors`` ranks a serve height; it cannot stop
the depth search from placing the first contact at the toss apex or below the
strike zone, and a serve fitted there drags every later flight of the connected
point with it.  This module states the same physical fact as a **hard cut**: the
serve contact must lie inside a cylinder around the server's own sided box --
horizontal radius scaled by stature about the box's court centre, height inside a
band of stature multiples.

The server's native box height at that court-Y is the local metric scale, so the
band is also reported in native pixels on the same picture the fit is judged on.

Because the depth search pins the first contact's court-Y to the branch depth,
the cylinder reduces at each branch to a box bound on the fitted contact's X and
Z.  That is what makes this a cut of the *image ray* rather than a verdict on the
fit: the contact stays pinned to its native pixel through the image residual, so
bounding X and Z keeps only the segment of that ray which is inside the cylinder,
inside the optimizer.  Nothing here is a calibrated anatomical limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cv.experiments.connected_shooting import contact_geometry, player_position


@dataclass(frozen=True)
class ServeReachConfig:
    """Band and radius in statures, plus the margin that makes the band a hard cut."""

    height_low_statures: float = 1.50
    height_high_statures: float = 1.75
    margin_statures: float = 0.10
    radius_statures: float = 1.35

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), float)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or self.height_low_statures <= 0
            or self.radius_statures <= 0
            or self.height_low_statures >= self.height_high_statures
            or self.margin_statures >= self.height_low_statures
        ):
            raise ValueError("finite ordered positive serve-reach configuration required")


@dataclass(frozen=True)
class ServeReachCylinder:
    """One server's hard reach cylinder, with the local metric scale it was read at."""

    player: str | None
    side: str
    stature_m: float
    centre_xy_m: tuple[float, float]
    radius_m: float
    height_interval_m: tuple[float, float]
    zero_penalty_height_interval_m: tuple[float, float]
    box_height_native_px: float | None
    metres_per_native_pixel: float | None
    config: ServeReachConfig

    def depth_slice(self, first_contact_y_m: float) -> dict[str, Any]:
        """Cut the cylinder at one pinned court-Y; empty when the cylinder cannot reach."""
        y = float(first_contact_y_m)
        if not np.isfinite(y):
            raise ValueError("finite pinned first-contact court-Y required")
        offset = abs(y - self.centre_xy_m[1])
        half_width = float(np.sqrt(max(self.radius_m**2 - offset**2, 0.0)))
        reachable = bool(offset < self.radius_m)
        return {
            "first_contact_y_m": y,
            "depth_offset_from_server_m": float(offset),
            "cylinder_reaches_this_depth": reachable,
            "x_interval_m": (
                [self.centre_xy_m[0] - half_width, self.centre_xy_m[0] + half_width]
                if reachable
                else None
            ),
            "z_interval_m": list(self.height_interval_m),
            "half_width_m": half_width,
        }

    def evaluate(self, contact_xyz_m) -> dict[str, Any]:
        """Membership of one fitted serve contact; excesses are reported, never clipped."""
        contact = np.asarray(contact_xyz_m, float)
        if contact.shape != (3,) or not np.isfinite(contact).all():
            raise ValueError("finite 3D serve contact required")
        low, high = self.height_interval_m
        height_excess = float(max(low - contact[2], contact[2] - high, 0.0))
        radial = float(np.linalg.norm(contact[:2] - np.asarray(self.centre_xy_m, float)))
        radial_excess = float(max(radial - self.radius_m, 0.0))
        return {
            "schema": "connected_serve_reach_cylinder_v1",
            "hard_cut": True,
            "player": self.player,
            "side": self.side,
            "stature_m": self.stature_m,
            "contact_height_m": float(contact[2]),
            "contact_height_statures": float(contact[2] / self.stature_m),
            "height_interval_m": [low, high],
            "height_interval_statures": [
                low / self.stature_m,
                high / self.stature_m,
            ],
            "height_excess_m": height_excess,
            "root_to_contact_m": radial,
            "radius_m": self.radius_m,
            "radial_excess_m": radial_excess,
            "inside": bool(height_excess == 0.0 and radial_excess == 0.0),
            "box_height_native_px": self.box_height_native_px,
            "metres_per_native_pixel": self.metres_per_native_pixel,
            "band_native_px_above_box_bottom": (
                None
                if self.metres_per_native_pixel is None
                else [
                    low / self.metres_per_native_pixel,
                    high / self.metres_per_native_pixel,
                ]
            ),
        }

    def record(self) -> dict[str, Any]:
        return {
            "schema": "connected_serve_reach_cylinder_v1",
            "hard_cut": True,
            "player": self.player,
            "side": self.side,
            "stature_m": self.stature_m,
            "centre_xy_m": list(self.centre_xy_m),
            "centre_source": "server sided player box court centre at the contact exposure",
            "radius_m": self.radius_m,
            "radius_statures": self.config.radius_statures,
            "height_interval_m": list(self.height_interval_m),
            "zero_penalty_height_interval_m": list(self.zero_penalty_height_interval_m),
            "height_interval_statures": [
                self.config.height_low_statures - self.config.margin_statures,
                self.config.height_high_statures + self.config.margin_statures,
            ],
            "margin_statures": self.config.margin_statures,
            "box_height_native_px": self.box_height_native_px,
            "metres_per_native_pixel": self.metres_per_native_pixel,
            "interpretation": (
                "hard cut of the serve contact image ray inside the depth search; the "
                "server's own sided box supplies the centre and the local metric scale"
            ),
        }


def build(player_state: dict[str, Any], config: ServeReachConfig = ServeReachConfig()):
    """Read one server's cylinder from the sided box state the fit already uses."""
    config.validate()
    stature = player_state.get("stature_m")
    if stature is None or not np.isfinite(float(stature)) or float(stature) <= 0:
        raise ValueError("serve reach cut requires the server's positive finite stature")
    stature = float(stature)
    # This is a hard cut of the serve contact ray, so an absent root refuses
    # outright: there is no honest wider cylinder, only no cylinder at all.
    centre = player_position.require_root(player_state, "the serve reach cylinder")
    box_height = player_state.get("box_height_native_px")
    box_height = (
        None
        if box_height is None or not np.isfinite(float(box_height)) or float(box_height) <= 0
        else float(box_height)
    )
    return ServeReachCylinder(
        player=player_state.get("player"),
        side=str(player_state["side"]),
        stature_m=stature,
        centre_xy_m=(float(centre[0]), float(centre[1])),
        radius_m=config.radius_statures * stature,
        height_interval_m=(
            (config.height_low_statures - config.margin_statures) * stature,
            (config.height_high_statures + config.margin_statures) * stature,
        ),
        zero_penalty_height_interval_m=(
            config.height_low_statures * stature,
            config.height_high_statures * stature,
        ),
        box_height_native_px=box_height,
        metres_per_native_pixel=None if box_height is None else stature / box_height,
        config=config,
    )


def image_ray(camera: np.ndarray, radial: np.ndarray | None, pixel: np.ndarray) -> dict[str, Any]:
    """Two world points of the native contact ray, from two assumed heights."""
    base = contact_geometry.plane_proxy(camera, radial, pixel, 0.0)
    unit = contact_geometry.plane_proxy(camera, radial, pixel, 1.0)
    delta = np.r_[unit - base, 1.0]
    if not np.isfinite(delta).all() or abs(delta[1]) < 1e-9:
        raise ValueError("contact ray is degenerate in court depth")
    return {"base_xyz_m": np.r_[base, 0.0], "per_metre_of_height": delta}


def ray_at_depth(ray: dict[str, Any], first_contact_y_m: float) -> np.ndarray:
    """Where the native contact ray sits when the search pins this court-Y."""
    base, delta = ray["base_xyz_m"], ray["per_metre_of_height"]
    height = (float(first_contact_y_m) - base[1]) / delta[1]
    return base + height * delta


def ray_cut(ray: dict[str, Any], cylinder: ServeReachCylinder, depths) -> dict[str, Any]:
    """Report where the native contact ray enters the cylinder, branch by branch.

    This is the geometry the bound implements.  It is reported rather than used to
    drop branches: an excluded branch produces no evidence, and a serve front one
    exposure away from the physical contact moves the ray by the ball's own travel.
    """
    rows = []
    for depth in np.asarray(depths, float):
        point = ray_at_depth(ray, float(depth))
        verdict = cylinder.evaluate(point)
        rows.append(
            {
                "depth_hypothesis_m": float(depth),
                "ray_contact_xyz_m": point.tolist(),
                "ray_height_m": float(point[2]),
                "ray_height_statures": float(point[2] / cylinder.stature_m),
                "inside_cylinder": verdict["inside"],
                "height_excess_m": verdict["height_excess_m"],
                "radial_excess_m": verdict["radial_excess_m"],
            }
        )
    inside = [row["depth_hypothesis_m"] for row in rows if row["inside_cylinder"]]
    return {
        "schema": "connected_serve_reach_ray_cut_v1",
        "branches": rows,
        "depths_inside_cylinder_m": inside,
        "branch_count_inside": len(inside),
        "branch_count": len(rows),
    }


def contact_hypotheses(
    ray: dict[str, Any],
    location_prior: dict[str, Any],
    feet: dict[str, Any],
    stature_m: float,
    *,
    count: int = 3,
) -> list[dict[str, Any]]:
    """Resolve a toss/contact ray with the external prior and grounded server feet.

    The outputs are starts, never cuts: no interval or inequality is returned.  The
    prior supplies the otherwise weak camera-ray coordinate, while the feet supply a
    broad stature-scaled reach observation.  Multiple mixture components and, when
    needed, one-sigma positions along the ray preserve depth alternatives.
    """
    if location_prior.get("status") != "supported" or count not in {2, 3}:
        raise ValueError("supported location prior and two or three hypotheses required")
    base = np.asarray(ray["base_xyz_m"], float)
    direction = np.asarray(ray["per_metre_of_height"], float)
    feet_xy = np.asarray(feet["court_xy_m"], float)
    if (
        base.shape != (3,)
        or direction.shape != (3,)
        or feet_xy.shape != (2,)
        or not np.isfinite(np.r_[base, direction, feet_xy, stature_m]).all()
        or np.linalg.norm(direction) < 1e-9
        or stature_m <= 0
    ):
        raise ValueError("finite contact ray, grounded feet and positive stature required")
    feet_mean = np.r_[feet_xy, 1.62 * stature_m]
    feet_sigma = np.asarray([0.55, 0.38, 0.18]) * stature_m
    feet_precision = np.diag(1.0 / feet_sigma**2)

    ranked = []
    for index, component in enumerate(location_prior["components"]):
        mean = np.asarray(component["mean_xyz_m"], float)
        covariance = np.asarray(component["covariance_xyz_m2"], float)
        if mean.shape != (3,) or covariance.shape != (3, 3):
            continue
        covariance = 0.5 * (covariance + covariance.T)
        try:
            prior_precision = np.linalg.inv(covariance)
            precision = prior_precision + feet_precision
            fused_covariance = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            continue
        fused_mean = fused_covariance @ (prior_precision @ mean + feet_precision @ feet_mean)
        ray_parameter = float(
            direction @ precision @ (fused_mean - base) / (direction @ precision @ direction)
        )
        point = base + ray_parameter * direction
        ray_sigma = float(1.0 / np.sqrt(direction @ precision @ direction))
        peak = float(
            component.get("weight", 1.0) / np.sqrt(max(float(np.linalg.det(covariance)), 1e-15))
        )
        ranked.append(
            {
                "component_index": index,
                "peak": peak,
                "xyz_m": point,
                "ray_parameter": ray_parameter,
                "ray_parameter_sigma": ray_sigma,
                "covariance": fused_covariance,
                "prior_mean": mean,
                "feet_mean": feet_mean,
                "feet_sigma": feet_sigma,
            }
        )
    if not ranked:
        raise ValueError("no finite serve-location component projects onto the contact ray")
    ranked.sort(key=lambda row: (-row["peak"], row["component_index"]))
    selected = ranked[:count]
    # A one-component external cell is common.  Preserve the shared camera-ray
    # ambiguity explicitly instead of pretending it produced one exact contact.
    if len(selected) < count:
        mode = selected[0]
        for sign in (-1.0, 1.0):
            if len(selected) >= count:
                break
            clone = dict(mode)
            clone["component_index"] = mode["component_index"]
            clone["ray_parameter"] = mode["ray_parameter"] + sign * mode["ray_parameter_sigma"]
            clone["xyz_m"] = base + clone["ray_parameter"] * direction
            clone["depth_offset_sigma"] = sign
            clone["peak"] = mode["peak"] * 0.5
            selected.append(clone)

    outputs = []
    for rank, row in enumerate(selected[:count]):
        point = np.asarray(row["xyz_m"], float)
        if np.any(point < [-10.0, -15.0, 0.0325]) or np.any(point > [21.0, 40.0, 12.0]):
            continue
        outputs.append(
            {
                "rank": rank,
                "source_component_index": int(row["component_index"]),
                "contact_xyz_m": point.tolist(),
                "contact_sigma_m": np.sqrt(np.maximum(np.diag(row["covariance"]), 0.0)).tolist(),
                "ray_parameter": float(row["ray_parameter"]),
                "ray_parameter_sigma": float(row["ray_parameter_sigma"]),
                "depth_offset_sigma": float(row.get("depth_offset_sigma", 0.0)),
                "external_prior_mean_xyz_m": row["prior_mean"].tolist(),
                "grounded_feet_target_xyz_m": row["feet_mean"].tolist(),
                "grounded_feet_sigma_m": row["feet_sigma"].tolist(),
                "candidate_generator_only": True,
                "optimizer_bound": False,
                "optimizer_inequality": False,
            }
        )
    if len(outputs) < 2:
        raise ValueError("contact ray and prior did not yield two in-range hypotheses")
    return outputs


def same_player_regularization(
    history: list[dict[str, Any]],
    player: str | None,
    side: str,
    serve_number: int | None,
    location_prior: dict[str, Any],
) -> dict[str, Any]:
    """Running accepted-contact mean with Hawk-Eye component widths.

    First and second serves never share a cell.  With no earlier accepted contact
    this abstains; callers add its normalized residual to ranking only.
    """
    rows = [
        row
        for row in history
        if row.get("player") == player
        and row.get("side") == side
        and row.get("serve_number") == serve_number
        and row.get("accepted") is True
    ]
    if not rows:
        return {
            "status": "abstained",
            "abstention_reason": "no_earlier_accepted_same_player_side_serve_number_contact",
            "soft_residual_only": True,
            "history_count": 0,
        }
    values = np.asarray([row["contact_xyz_m"] for row in rows], float)
    component = location_prior["components"][location_prior["mode_component_index"]]
    sigma = np.sqrt(np.maximum(np.diag(np.asarray(component["covariance_xyz_m2"], float)), 0.0))
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or np.any(sigma <= 0)
    ):
        raise ValueError("finite accepted contact history and Hawk-Eye widths required")
    return {
        "status": "supported",
        "target_xyz_m": np.mean(values, axis=0).tolist(),
        "sigma_m": sigma.tolist(),
        "history_count": len(values),
        "history_attempt_ids": [row.get("attempt_id") for row in rows],
        "serve_number_aware": True,
        "soft_residual_only": True,
        "sigma_source": "external_Hawk-Eye_player/side/end/serve-number_component_width",
    }


def evaluate_same_player_regularization(contact_xyz_m, regularization: dict[str, Any]) -> dict:
    if regularization.get("status") != "supported":
        return {**regularization, "selector_penalty": 0.0, "hard_gate": False}
    contact = np.asarray(contact_xyz_m, float)
    target = np.asarray(regularization["target_xyz_m"], float)
    sigma = np.asarray(regularization["sigma_m"], float)
    residual = (contact - target) / sigma
    return {
        **regularization,
        "contact_xyz_m": contact.tolist(),
        "residual_sigma": residual.tolist(),
        "selector_penalty": float(0.5 * residual @ residual),
        "hard_gate": False,
        "survived": True,
    }
