"""Construct input-only contact/net seeds with the actual measured flight model.

Status: bounded labeled-development initialization control. Generic contact heights
and the final incoming native ray define seed targets, not new observations. Each
flight is solved separately so a known net response never contaminates its incoming
velocity initializer. The subsequent fit uses the original observations/objective.
Run --source-dir PREPARED_CASE --output NEW_DIRECTORY; all failed seeds are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import labeled_preparation_net_followup as source_loader
from cv.experiments.connected_shooting import measured_dynamics, model
from cv.pipeline import provenance


def ray_at_plane(camera, pixel, axis: int, coordinate: float) -> np.ndarray:
    """Intersect an undistorted native ray with one explicitly fixed world plane."""
    camera, pixel = np.asarray(camera, float), np.asarray(pixel, float)
    if camera.shape != (3, 4) or pixel.shape != (2,) or axis not in (0, 1, 2):
        raise ValueError("3x4 camera, two-pixel coordinate and XYZ axis required")
    if not np.isfinite(camera).all() or not np.isfinite(pixel).all() or not np.isfinite(coordinate):
        raise ValueError("finite ray and plane required")
    equations = camera[:2] - pixel[:, None] * camera[2]
    free = [i for i in range(3) if i != axis]
    xyz = np.zeros(3)
    xyz[axis] = coordinate
    xyz[free] = np.linalg.solve(
        equations[:, free], -equations[:, axis] * coordinate - equations[:, 3]
    )
    if (camera @ np.r_[xyz, 1])[2] <= 0:
        raise ValueError("ray-plane point behind camera")
    return xyz


def solve_flight(start, velocity, target, first, last, fps, surface, expected_bounces):
    """Solve only outgoing velocity, retaining bounded failure and impact topology."""
    start, target = np.asarray(start, float), np.asarray(target, float)

    def simulate(v):
        return measured_dynamics.simulate(
            np.r_[start, v, np.zeros(3)], first, np.asarray([first, last]), fps, surface
        )

    def residual(v):
        return simulate(v)[0][-1] - target

    fit = least_squares(
        residual,
        np.asarray(velocity, float),
        bounds=(-np.full(3, 75.0), np.full(3, 75.0)),
        max_nfev=200,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    positions, _, _, impacts = simulate(fit.x)
    error = float(np.linalg.norm(positions[-1] - target))
    return (
        fit.x,
        positions[-1],
        {
            "success": bool(fit.success),
            "usable": bool(fit.success and error < 1e-4 and len(impacts) == expected_bounces),
            "nfev": fit.nfev,
            "endpoint_error_m": error,
            "expected_bounces": expected_bounces,
            "actual_bounces": len(impacts),
            "bounce_frames": [b["frame"] for b in impacts],
            "start_xyz": start.tolist(),
            "target_seed_only_xyz": target.tolist(),
            "end_xyz": positions[-1].tolist(),
            "velocity": fit.x.tolist(),
        },
    )


def solve_bounce_ray(start, velocity, camera, pixel, first, last, bounce, fps, surface):
    """Seed-only contact ray and supplied impact epoch, with no contact-height target."""
    camera, pixel = np.asarray(camera), np.asarray(pixel)

    def simulate(v):
        return measured_dynamics.simulate(
            np.r_[start, v, np.zeros(3)], first, np.array([first, last]), fps, surface
        )

    def residual(v):
        x, _, _, impacts = simulate(v)
        projected = camera @ np.r_[x[-1], 1]
        if len(impacts) != 1:
            raise ValueError("bounce-ray seed left its single-impact branch")
        return np.r_[
            (projected[:2] / projected[2] - pixel) / 100, (impacts[0]["frame"] - bounce) / fps
        ]

    fit = least_squares(
        residual,
        velocity,
        bounds=(-np.full(3, 75.0), np.full(3, 75.0)),
        max_nfev=200,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    x, _, _, impacts = simulate(fit.x)
    error = residual(fit.x)
    return (
        fit.x,
        x[-1],
        dict(
            success=bool(fit.success),
            usable=bool(fit.success and np.max(np.abs(error)) < 1e-6),
            nfev=fit.nfev,
            seed_mode="existing_bounce_epoch_and_contact_ray",
            scaled_residual=error.tolist(),
            expected_bounces=1,
            actual_bounces=len(impacts),
            bounce_frames=[b["frame"] for b in impacts],
            start_xyz=np.asarray(start).tolist(),
            end_xyz=x[-1].tolist(),
            velocity=fit.x.tolist(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--late-bounce-ray", action="store_true")
    parser.add_argument("--net-extrapolate", action="store_true")
    parser.add_argument("--heights-m", type=float, nargs="+", default=[0.75, 1.25, 1.75])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    context, old, _, _, _, source, labels, *_ = source_loader.load(args.source_dir)
    scene = context["scene"]
    if scene.dynamics != "measured_240hz" or scene.bounce_profile != "nominal":
        raise ValueError("this seed control requires measured_240hz with nominal bounce profile")
    if scene.camera_distortion is not None:
        raise ValueError("explicit undistortion required for radial cameras")
    packet = json.loads((args.source_dir / "recovery_packet.json").read_text())
    attempt = packet["attempts"][0]
    cameras_path = args.source_dir / "inputs/cameras.json"
    cameras = json.loads(cameras_path.read_text())
    camera_rows = {
        r["frame"]: r for r in cameras["cameras"] if r.get("supported", True) and "P" in r
    }
    events = labels["events"]["records"]
    frames = scene.contact_frames
    count = len(frames) - 1
    nets = [e for e in events if e["event_type"] == "net_hit" and e["status"] == "labeled"]
    if len(nets) != 1 or not frames[-2] < nets[0]["frame"] < frames[-1]:
        raise ValueError("one known final-flight net impact required")
    rows = attempt["owner_ball_labels"]
    boundaries = []
    for i in range(count):
        final = i == count - 1
        epoch = float(nets[0]["frame"] if final else frames[i + 1])
        eligible = [
            r
            for r in rows
            if r["status"] == "visible"
            and frames[i] < r["frame"] <= epoch
            and r["frame"] in camera_rows
        ]
        native = max(eligible, key=lambda r: r["frame"])
        pixel = [native["x1080"], native["y1080"]]
        extrapolation = None
        if final and args.net_extrapolate:
            front = sorted(eligible, key=lambda r: r["frame"])[-3:]
            relative = np.array([r["frame"] - epoch for r in front])
            measured = np.array([[r["x1080"], r["y1080"]] for r in front])
            pixel = np.linalg.lstsq(np.c_[np.ones(len(front)), relative], measured, rcond=None)[0][
                0
            ].tolist()
            extrapolation = dict(
                native_frames=[r["frame"] for r in front],
                native_pixels=measured.tolist(),
                seed_epoch=epoch,
                seed_pixel=pixel,
                camera_policy="last_real_native_camera_for_seed_only",
            )
        expected = sum(
            e["event_type"] == "bounce"
            and e["status"] == "labeled"
            and frames[i] < e["frame"] < epoch
            for e in events
        )
        boundaries.append(
            dict(
                flight=i,
                epoch=epoch,
                native_frame=native["frame"],
                pixel=pixel,
                seed_extrapolation=extrapolation,
                camera=camera_rows[native["frame"]]["P"],
                expected_bounces=int(expected),
                plane="net_Y" if final else "generic_contact_Z",
            )
        )
    plan = dict(
        status="frozen_before_seed_solve",
        question="Can physics-respecting contact and pre-net initialization recover an existing prepared point?",
        source=provenance.file_record(args.source_dir / "search/report.json"),
        inputs=[
            provenance.file_record(args.source_dir / "recovery_packet.json"),
            provenance.file_record(cameras_path),
        ],
        producer=provenance.file_record(Path(__file__)),
        heights_m=args.heights_m,
        late_bounce_ray=args.late_bounce_ray,
        net_extrapolate=args.net_extrapolate,
        boundary_rays=boundaries,
        source_start_seed=source["initialization"]["flights"][0]["local_start_xyz_seed"],
        final_fit_objective_unchanged=True,
        no_truth_input=True,
        seed_only_approximation="Incoming native ray used at later contact/net epoch only for initialization. Observations and their real exposure times stay unchanged.",
        spin_seed="zero, subsequently free",
        rebound_scales_seed=[1, 1],
        cohort_denominator=89,
    )
    (args.output / "preregistered.json").write_text(json.dumps(plan, indent=2) + "\n")
    records = []
    for name, height in [("continuation", None), *[(f"height_{z:g}", z) for z in args.heights_m]]:
        row = dict(name=name, height_m=height, status="ready", steps=[])
        try:
            p = old.copy()
            if height is not None:
                p[:3] = plan["source_start_seed"]
                p[3 + 3 * count : 3 + 6 * count] = 0
                p[-2:] = 1
                start = p[:3].copy()
                for boundary in boundaries:
                    i = boundary["flight"]
                    axis, coordinate = (1, 11.885) if boundary["plane"] == "net_Y" else (2, height)
                    target = ray_at_plane(boundary["camera"], boundary["pixel"], axis, coordinate)
                    v0 = source["initialization"]["flights"][i]["local_velocity_seed"]
                    if boundary["plane"] == "net_Y":
                        dt = (boundary["epoch"] - frames[i]) / scene.fps
                        v0 = (target - start) / dt
                        v0[2] += 0.5 * 9.81 * dt
                    velocity, start, step = solve_flight(
                        start,
                        v0,
                        target,
                        frames[i],
                        boundary["epoch"],
                        scene.fps,
                        scene.surface,
                        boundary["expected_bounces"],
                    )
                    if (
                        args.late_bounce_ray
                        and i == count - 2
                        and boundary["expected_bounces"] == 1
                    ):
                        original_step = step
                        bounce = next(
                            e["frame"]
                            for e in events
                            if e["event_type"] == "bounce"
                            and frames[i] < e["frame"] < boundary["epoch"]
                        )
                        velocity, start, step = solve_bounce_ray(
                            np.asarray(step["start_xyz"]),
                            velocity,
                            boundary["camera"],
                            boundary["pixel"],
                            frames[i],
                            boundary["epoch"],
                            bounce,
                            scene.fps,
                            scene.surface,
                        )
                        step["generic_height_seed_attempt"] = original_step
                    row["steps"].append(step)
                    if not step["usable"]:
                        raise ValueError(f"flight {i} endpoint or bounce topology failed")
                    p[3 + 3 * i : 6 + 3 * i] = velocity
            paths = model.chain(scene, p)
            if height is not None:
                for i, flight in enumerate(paths):
                    expected = boundaries[i]["expected_bounces"] + int(i == count - 1)
                    if len(flight["bounces"]) != expected or len(flight.get("net_hits", [])) != int(
                        i == count - 1
                    ):
                        raise ValueError(
                            "full original-law replay changed the required impact topology"
                        )
            row["physical_replay"] = [
                dict(
                    start=f["start_xyz"].tolist(),
                    end=f["end_xyz"].tolist(),
                    bounce_count=len(f["bounces"]),
                    net_count=len(f.get("net_hits", [])),
                )
                for f in paths
            ]
            row["fit"] = dict(parameters=p.tolist())
            seed = args.output / (name + ".json")
            seed.write_text(json.dumps(row, indent=2) + "\n")
            row["baseline"] = provenance.file_record(seed)
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            row.update(status="failed", error=str(exc))
        records.append(row)
    (args.output / "report.json").write_text(
        json.dumps(dict(plan=plan, cases=records), indent=2) + "\n"
    )
    print(
        json.dumps(
            [dict(name=r["name"], status=r["status"], error=r.get("error")) for r in records]
        )
    )


if __name__ == "__main__":
    main()
