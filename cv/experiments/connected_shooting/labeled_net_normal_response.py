"""Optional direct normal restitution at the first ground impact after a net.

Observed net/ground/rebound evidence qualifies this local S6 refinement. The
coefficient is independent of the sparsely calibrated incidence extrapolation.
Tangential response defaults to nominal; explicit nested research modes add a
speed multiplier and optionally transverse velocity. Nominal outgoing spin,
position and dwell remain unchanged; this is not a coupled friction/spin law.
"""

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

from dataclasses import dataclass
from types import FunctionType, SimpleNamespace

import numpy as np

from cv.experiments.connected_shooting import measured_dynamics, net_collision
from physics import bounce_reference

BOUNDS = (0.05, 1.0)
HORIZONTAL_MODES = ("off", "retention", "heading")


@dataclass(frozen=True)
class NetNormalResponse:
    net_response: object
    first_ground_normal_restitution: float
    horizontal_mode: str = "off"
    tangential_multiplier: float = 1.0
    transverse_velocity_mps: float = 0.0

    def __post_init__(self):
        value = float(self.first_ground_normal_restitution)
        if not np.isfinite(value) or not BOUNDS[0] <= value <= BOUNDS[1]:
            raise ValueError("first-ground normal restitution must be in [.05,1]")
        if hasattr(self.net_response, "outgoing_spin_rad_s"):
            raise ValueError("direct normal arm preserves nominal net spin")
        object.__setattr__(self, "first_ground_normal_restitution", value)
        if self.horizontal_mode not in HORIZONTAL_MODES:
            raise ValueError("explicit supported horizontal response mode required")
        multiplier, transverse = (
            float(self.tangential_multiplier),
            float(self.transverse_velocity_mps),
        )
        if (
            not np.isfinite([multiplier, transverse]).all()
            or not 0 <= multiplier <= 4
            or abs(transverse) > 20
        ):
            raise ValueError(
                "horizontal multiplier in [0,4] and transverse velocity in [-20,20] required"
            )
        if self.horizontal_mode == "off" and (multiplier != 1 or transverse != 0):
            raise ValueError("disabled horizontal response must retain nominal values")
        if self.horizontal_mode != "heading" and transverse != 0:
            raise ValueError("transverse velocity requires explicit heading mode")
        object.__setattr__(self, "tangential_multiplier", multiplier)
        object.__setattr__(self, "transverse_velocity_mps", transverse)

    @property
    def outgoing_velocity_mps(self):
        return self.net_response.outgoing_velocity_mps

    @property
    def key(self):
        original = (
            "net_first_ground_normal_v1",
            self.net_response.key,
            self.first_ground_normal_restitution,
        )
        return (
            original
            if self.horizontal_mode == "off"
            else (
                "net_first_ground_horizontal_v1",
                *original,
                self.horizontal_mode,
                self.tangential_multiplier,
                self.transverse_velocity_mps,
            )
        )

    def velocity(self, incoming):
        return self.net_response.velocity(incoming)

    def record(self, incoming, outgoing):
        record = self.net_response.record(incoming, outgoing) | {
            "first_ground_normal_restitution": self.first_ground_normal_restitution,
            "ground_response_scope": "first detected ground after net; nominal tangential/spin response",
        }
        if self.horizontal_mode != "off":
            record.update(
                first_ground_horizontal_mode=self.horizontal_mode,
                first_ground_tangential_multiplier=self.tangential_multiplier,
                first_ground_transverse_velocity_mps=self.transverse_velocity_mps,
                ground_response_scope="first detected ground after net; effective translation; nominal spin/dwell",
            )
        return record

    def horizontal_velocity(self, state, outgoing):
        """Optional effective XY map; no contact position, dwell or spin mutation."""
        velocity = outgoing.copy()
        horizontal = float(np.linalg.norm(state.velocity[:2]))
        if horizontal <= 1e-8:
            raise ValueError("horizontal heading unsupported at near-zero incoming speed")
        direction = state.velocity[:2] / horizontal
        # Branches preserve exact normal-law behavior at both arms' nominal start.
        if self.tangential_multiplier != 1:
            velocity[:2] *= self.tangential_multiplier
        if self.transverse_velocity_mps != 0:
            velocity[:2] += self.transverse_velocity_mps * np.array([-direction[1], direction[0]])
        incoming = np.r_[
            state.velocity[:2] / state.horizontal_retention, -state.velocity[2] / state.restitution
        ]
        speed_in = float(np.linalg.norm(incoming[:2]))
        xy = velocity[:2]
        angle = float(
            np.degrees(np.arctan2(direction[0] * xy[1] - direction[1] * xy[0], direction @ xy))
        )
        return velocity, dict(
            first_ground_horizontal_mode=self.horizontal_mode,
            first_ground_tangential_multiplier=self.tangential_multiplier,
            first_ground_transverse_velocity_mps=self.transverse_velocity_mps,
            normal_only_outgoing_velocity_mps=outgoing.tolist(),
            effective_outgoing_velocity_mps=velocity.tolist(),
            effective_horizontal_retention=float(np.linalg.norm(xy) / speed_in),
            heading_change_deg=angle,
            translational_energy_ratio=float(velocity @ velocity / (incoming @ incoming)),
            response_is_coupled_friction_spin_model=False,
            ground_spin_unchanged=True,
        )


def prior_from_observations(context, parameters):
    """Qualify from native evidence, never fitted incidence or acceptance bits."""
    scene = context["scene"]
    start, end = scene.contact_frames[-2:]
    nets = [
        e for e in context["events"] if e["event_type"] == "net_hit" and start < e["frame"] < end
    ]
    grounds = sorted(
        [e for e in context["events"] if e["event_type"] == "bounce" and start < e["frame"] <= end],
        key=lambda e: float(e["frame"]),
    )
    if len(nets) != 1 or len(grounds) not in (1, 2):
        raise ValueError("one observed terminal net and one or two following grounds required")
    net, ground = nets[0], grounds[0]
    if not occurrence.resolved_membership(net) or not occurrence.resolved_membership(ground):
        raise ValueError("resolved net and ground membership required")
    nlo, nhi = map(float, net["frame_interval"])
    glo, ghi = map(float, ground["frame_interval"])
    if not start < nlo <= nhi < glo <= ghi < end:
        raise ValueError("ordered original net, ground and rebound horizon required")
    frames = np.unique(
        np.r_[scene.observation_frames[-1], context["heldout"].observation_frames[-1]]
    )
    approach = frames[(frames > nhi) & (frames <= glo)]
    rebound_end = float(grounds[1]["frame_interval"][0]) if len(grounds) == 2 else end
    rebound = frames[
        (frames > ghi) & (frames < rebound_end if len(grounds) == 2 else frames <= end)
    ]
    if len(approach) < 2 or len(rebound) < 3:
        raise ValueError("two observed post-net approach and three rebound pictures required")
    surface = str(scene.surface)
    row = bounce_reference.MEASURED[surface]
    incidence = row["incidence_deg_median"]
    multiplier = float(parameters[-2])
    center = float(np.clip(bounce_reference.restitution(incidence, surface) * multiplier, *BOUNDS))
    sigma = max(0.15, row["restitution_residual_std_at_model_spin"] * multiplier)
    return {
        "center": center,
        "sigma": float(sigma),
        "bounds": list(BOUNDS),
        "surface": surface,
        "prior_incidence_deg": incidence,
        "prior_source": "in-support surface median and broad residual scatter; weak modeling prior",
        "prior_is_calibrated_uncertainty": False,
        "net_interval": [nlo, nhi],
        "ground_interval": [glo, ghi],
        "approach_frames": approach.tolist(),
        "rebound_frames": rebound.tolist(),
        "qualification_uses_fitted_incidence_or_gates": False,
    }


def simulator(response):
    """Immutable response-specific integration and cache, with per-trace impact state."""
    # Nested response scopes must clone the numerical function, not the outer
    # response wrapper (whose closure and response-specific cache belong to it).
    original = getattr(net_collision.simulate, "_base_net_simulator", net_collision.simulate)

    def integrate(*args):
        seen_net, applied = False, False

        def net_velocity(incoming):
            nonlocal seen_net
            seen_net = True
            return response.velocity(incoming)

        def rebound(state, profile, scales=(1.0, 1.0)):
            nonlocal applied
            velocity, receipt = measured_dynamics.rebound_velocity(state, profile, scales)
            if seen_net and not applied:
                nominal = velocity.copy()
                coefficient = response.first_ground_normal_restitution
                velocity = velocity.copy()
                velocity[2] = coefficient * float(state.velocity[2] / state.restitution)
                receipt = {
                    **receipt,
                    "nominal_applied_restitution": receipt["applied_restitution"],
                    "requested_restitution": coefficient,
                    "applied_restitution": coefficient,
                    "first_ground_normal_restitution": coefficient,
                    "nominal_outgoing_velocity_mps": nominal.tolist(),
                    "measured_surface_law": False,
                    "coefficient_clipped": bool(
                        receipt["requested_horizontal_retention"]
                        != receipt["applied_horizontal_retention"]
                    ),
                }
                if response.horizontal_mode != "off":
                    velocity, horizontal_record = response.horizontal_velocity(state, velocity)
                    receipt.update(
                        nominal_applied_horizontal_retention=receipt[
                            "applied_horizontal_retention"
                        ],
                        nominal_requested_horizontal_retention=receipt[
                            "requested_horizontal_retention"
                        ],
                        nominal_coefficient_clipped=receipt["coefficient_clipped"],
                        requested_horizontal_retention=horizontal_record[
                            "effective_horizontal_retention"
                        ],
                        applied_horizontal_retention=horizontal_record[
                            "effective_horizontal_retention"
                        ],
                        coefficient_clipped=False,
                    )
                    receipt.update(horizontal_record)
                applied = True
            return velocity, receipt

        cloned = FunctionType(
            net_collision.integrate.__code__,
            {
                **net_collision.integrate.__globals__,
                "net_impact_velocity": net_velocity,
                "rebound_velocity": rebound,
            },
            argdefs=net_collision.integrate.__defaults__,
        )
        return cloned(*args)

    def cached(key, function, *args):
        return measured_dynamics.cached_integration((response.key, *key), function, *args)

    simulated = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "integrate": integrate,
            "measured_dynamics": SimpleNamespace(cached_integration=cached),
        },
        argdefs=original.__defaults__,
    )
    simulated.__kwdefaults__ = original.__kwdefaults__

    def run(*args, **kwargs):
        result = simulated(*args, **kwargs)
        for hit in result[-1]:
            hit["experimental_response"] = response.record(hit["v_in"], hit["v_out"])
        return result

    run._base_net_simulator = original
    return run
