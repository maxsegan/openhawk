"""Run the frozen connected-family sweep over a labeled attempt cohort on CPU.

Question: how many fit-ready human-labeled broadcast attempts form at least one
complete, continuous, legal connected family under one fixed configuration?
This evaluation-only entrypoint prepares qualified cameras, refines every depth
branch inside the fixed serve-region beam, renders native/court/side evidence,
and writes one aggregate report.  Human labels never become automatic inputs.

``--cohort nine`` reproduces the frozen nine-attempt reference exactly.
``--cohort labeled17`` adds the eight later reviewed broadcasts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from cv.experiments.connected_shooting import (
    agent_attempt_prepare,
    camera_geometry,
    cohort,
    toss_witness,
)
from cv.pipeline import paths, provenance
from physics import grass_surface_model, surface_model


@dataclass(frozen=True)
class AttemptCase:
    """One labeled attempt and the explicit automatic inputs it is fitted with.

    ``player_localization`` names an automatic player artifact relative to
    ``TENNIS_DATA_ROOT/processed``.  It is explicit rather than discovered when
    the default benchmark root has no sided court-projected rows covering the
    attempt's clip.  ``serve_number_resolved`` is false when the frozen label
    document says the serve number is unresolved; the soft prior then uses the
    wider second-serve width and never gates a branch.
    """

    key: str
    label_file: str
    surface: str
    serve_number: int
    pose_image_scale: float = 1.0
    player_statures_m: tuple[tuple[str, float], ...] = ()
    player_localization: str | None = None
    serve_number_resolved: bool = True
    dense_label_file: str | None = None
    match_id: str | None = None
    clip: str | None = None
    gender: str | None = None
    lane: str | None = None
    ending_kind: str | None = None
    contact_count: int | None = None
    discovery: dict[str, Any] | None = None


CASES = (
    AttemptCase(
        "ao1",
        "ao2023f_w_sabalenka_rybakina_pt0001_attempt01_fit_v2.json",
        "hard",
        1,
        player_statures_m=(("Rybakina", 1.84), ("Sabalenka", 1.82)),
    ),
    AttemptCase(
        "ao2",
        "ao2023f_w_sabalenka_rybakina_pt0002_attempt01_fit_v2.json",
        "hard",
        2,
        player_statures_m=(("Rybakina", 1.84), ("Sabalenka", 1.82)),
    ),
    AttemptCase(
        "ao3",
        "ao2023f_w_sabalenka_rybakina_pt0003_attempt01_fit_v2.json",
        "hard",
        1,
        player_statures_m=(("Sabalenka", 1.82),),
    ),
    AttemptCase(
        "clay2",
        "rg2024f_w_swiatek_paolini_pt0002_attempt01_fit_v2.json",
        "clay",
        1,
        player_statures_m=(("Swiatek", 1.76), ("Paolini", 1.63)),
    ),
    AttemptCase(
        "men2",
        "uso2020f_m_zverev_thiem_pt0002_attempt01_fit_v2.json",
        "hard",
        1,
        player_statures_m=(("Thiem", 1.85), ("Zverev", 1.98)),
    ),
    AttemptCase(
        "men1",
        "uso2020f_m_zverev_thiem_pt0001_attempt01_astrareview_v1.json",
        "hard",
        1,
        2.0,
        (("Zverev", 1.98), ("Thiem", 1.85)),
    ),
    AttemptCase(
        "clay3",
        "rg2024f_w_swiatek_paolini_pt0003_attempt01_clay3labels_v1.json",
        "clay",
        1,
        player_statures_m=(("Paolini", 1.63), ("Swiatek", 1.76)),
    ),
    AttemptCase(
        "osaka3",
        "ao2019f_w_osaka_kvitova_pt0003_attempt02.json",
        "hard",
        2,
        player_statures_m=(("Osaka", 1.80), ("Kvitova", 1.82)),
    ),
    AttemptCase(
        "ao3b",
        "ao2023f_w_sabalenka_rybakina_pt0003_attempt02_fit_v2.json",
        "hard",
        2,
        player_statures_m=(("Sabalenka", 1.82), ("Rybakina", 1.84)),
    ),
)

NEW_CASES = (
    AttemptCase(
        "muchova1",
        "rg2023f_w_swiatek_muchova_pt0001_attempt01.json",
        "clay",
        2,
        player_statures_m=(("Muchova", 1.80), ("Swiatek", 1.76)),
        player_localization=(
            "postseg_pipeline_benchmark_766a967/rg2023f_w_swiatek_muchova/"
            "player_pose_tracked_crop_native_backfill_v1.csv"
        ),
        serve_number_resolved=False,
    ),
    AttemptCase(
        "wim3",
        "wim2023f_m_alcaraz_djokovic_pt0003_attempt01.json",
        "grass",
        2,
        player_statures_m=(("Djokovic", 1.88), ("Alcaraz", 1.83)),
        serve_number_resolved=False,
    ),
    AttemptCase(
        "nadal2",
        "atp_2024_520_r128_335_alexander_zverev_rafael_nadal_pt0002_attempt01.json",
        "clay",
        2,
        player_statures_m=(("Nadal", 1.85), ("Zverev", 1.98)),
        serve_number_resolved=False,
    ),
    AttemptCase(
        "fritz2",
        "uso2024sf_m_fritz_tiafoe_pt0002_attempt01.json",
        "hard",
        2,
        player_statures_m=(("Fritz", 1.96), ("Tiafoe", 1.88)),
        serve_number_resolved=False,
    ),
    AttemptCase(
        "garcia2",
        "wtaf2022f_w_garcia_sabalenka_pt0002_attempt01.json",
        "hard",
        2,
        player_statures_m=(("Garcia", 1.77), ("Sabalenka", 1.82)),
        serve_number_resolved=False,
    ),
    AttemptCase(
        "moutet2",
        "atp_2026_580_r128_103_corentin_moutet_tristan_schoolkate_pt0002_attempt01.json",
        "hard",
        2,
        player_statures_m=(("Schoolkate", 1.83), ("Moutet", 1.80)),
        serve_number_resolved=False,
    ),
    AttemptCase(
        "cincy1",
        "cincy2024f_m_sinner_tiafoe_pt0001_attempt01.json",
        "hard",
        2,
        player_statures_m=(("Tiafoe", 1.88), ("Sinner", 1.91)),
        player_localization=(
            "postseg_pipeline_benchmark_766a967/cincy2024f_m_sinner_tiafoe/"
            "player_pose_tracked_crop_native_backfill_v1.csv"
        ),
        serve_number_resolved=False,
    ),
    AttemptCase(
        "rome2",
        "rome2026f_w_svitolina_gauff_pt0002_attempt01.json",
        "clay",
        2,
        2.0,
        (("Gauff", 1.75), ("Svitolina", 1.74)),
        player_localization=(
            "postseg_pipeline_benchmark_766a967/rome2026f_w_svitolina_gauff/"
            "player_boxes_24_native_sided_v1.csv"
        ),
        serve_number_resolved=False,
    ),
)


def discovered_cases(labels_root: Path) -> tuple[AttemptCase, ...]:
    """Every complete labeled attempt in one label root, with explicit inputs.

    The frozen seventeen keep their published serve ordinal, hitter order,
    surface and player artifact; everything else is derived from the label
    document and recorded with its derivation in ``discovery``.
    """
    return tuple(
        AttemptCase(
            key=item.key,
            label_file=item.label_file,
            surface=item.surface,
            serve_number=item.serve_number,
            pose_image_scale=item.pose_image_scale,
            player_statures_m=item.player_statures_m,
            player_localization=item.player_localization,
            serve_number_resolved=item.serve_number_resolved,
            dense_label_file=item.dense_label_file,
            match_id=item.match_id,
            clip=item.clip,
            gender=item.gender,
            lane=item.lane,
            ending_kind=item.ending_kind,
            contact_count=item.contact_count,
            discovery={
                "first_contact_frame": item.first_contact_frame,
                "ending_frame": item.ending_frame,
                "bounce_count": item.bounce_count,
                "net_hit_count": item.net_hit_count,
                "labeled_native_frames": item.labeled_native_frames,
                "visible_native_frames": item.visible_native_frames,
                **item.provenance,
            },
        )
        for item in cohort.discover(labels_root)
    )


COHORTS = {
    "nine": CASES,
    "labeled17": CASES + NEW_CASES,
    "all": None,
}

FIXED_CONFIGURATION = {
    "schema": "connected_labeled_attempt_s6_reference_configuration_v3",
    "cohort": "cv/validation/labels/s6_agent_inputs_v1 fit-ready reviewed attempts",
    "opened_development_data": True,
    "automatic_inference_eligible": False,
    "workers": 8,
    "math_threads_per_worker": 1,
    "camera_transport_reference_policy": "contact_anchor",
    "camera_landmark_frames_per_attempt": "six prior; additive Astra clay3 revision has eight",
    "depth_hypotheses": "ten 0.5 m branches from agent_single_flight_search.limits",
    "coarse_iterations": 100,
    "refine_iterations": 200,
    "depth_beam": "fit only fixed first-contact Y values inside the serve-region bounds",
    "refine_all_in_beam_depth_hypotheses": True,
    "five_plus_flight_in_serve_bounds_structure_seed_max_nfev_per_local_solve": 40,
    "exposure_duration_frames": 0.25,
    "exposure_duration_measured": False,
    "directional_horizons_frames": [4, 10],
    "directional_rms_limit_px": 16.0,
    "directional_optimizer_interior_margin_px": 0.5,
    "bounce_ray_limit_m": 0.9144,
    "bounce_witness_mode": "subframe_graded_circle",
    "athlete_prior_mode": "stature_pose_soft",
    "player_image_coordinate_policy": "per-attempt explicit scale; USO men1 half-native CSV uses 2.0",
    "terminal_optimizer_interior_margin_frames": 0.01,
    "speed_relation": {
        "hard_gate": False,
        "sigma_formula": "sqrt(1.0^2 + (0.04 * speed_mps)^2)",
        "sigma_at_50_mps": 5**0.5,
        "evaluated_at": "fitted first-contact instant",
    },
    "toss_witness_hard_gate": False,
    "serve_contact_prior_hard_gate": False,
    "stature_sources": {
        "Zverev": "https://www.atptour.com/en/players/alexanderzverev/z355/overview",
        "Thiem": "https://www.atptour.com/en/players/xx/tb69/overview",
        "Rybakina": "https://www.wtatennis.com/players/324166/elenarybakina/",
        "Sabalenka": "https://www.wtatennis.com/players/320760/aryna-sabalenka/",
        "Swiatek": "https://www.wtatennis.com/players/326408/igaswiatek/",
        "Paolini": "https://www.wtatennis.com/players/319280/jasmine-paolini",
        "Osaka": "https://www.wtatennis.com/players/319998/naomi_osaka",
        "Kvitova": "https://www.wtatennis.com/players/314206/petra-kvitova",
        "Muchova": "https://www.wtatennis.com/players/322191/karolina-muchova",
        "Djokovic": "https://www.atptour.com/en/players/novak-djokovic/d643/overview",
        "Alcaraz": "https://www.atptour.com/en/players/carlos-alcaraz/a0e2/overview",
        "Nadal": "https://www.atptour.com/en/players/rafael-nadal/n409/overview",
        "Fritz": "https://www.atptour.com/en/players/taylor-fritz/fb98/overview",
        "Tiafoe": "https://www.atptour.com/en/players/frances-tiafoe/td51/overview",
        "Garcia": "https://www.wtatennis.com/players/315391/caroline-garcia",
        "Moutet": "https://www.atptour.com/en/players/corentin-moutet/mw02/overview",
        "Schoolkate": "https://www.atptour.com/en/players/tristan-schoolkate/s0n0/overview",
        "Sinner": "https://www.atptour.com/en/players/jannik-sinner/s0ag/overview",
        "Svitolina": "https://www.wtatennis.com/players/316738/elina-svitolina",
        "Gauff": "https://www.wtatennis.com/players/328560/coco-gauff",
    },
    "stature_source_note": (
        "official tour-listed statures read from the profiles above; they are explicit "
        "soft evidence, not measured anthropometry for these specific broadcasts"
    ),
}


def _environment(overlay: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    environment.update(overlay or {})
    return environment


def _run(
    command: list[str],
    log_path: Path,
    check: bool = True,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    began = time.monotonic()
    with log_path.open("w") as handle:
        completed = subprocess.run(
            command,
            cwd=paths.REPO_ROOT,
            env=_environment(environment),
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    output = log_path.read_text()
    if completed.returncode and check:
        raise RuntimeError(
            f"command failed ({completed.returncode}); inspect {log_path}: {' '.join(command)}"
        )
    return {
        "wall_seconds": time.monotonic() - began,
        "returncode": completed.returncode,
        "output_tail": output.splitlines()[-4:],
    }


def _covers_contacts(path: Path, labels: dict[str, Any]) -> bool:
    """Require a court-projected sided row at every rounded contact picture."""
    clip = labels["attempt"]["clip"]
    contacts = {
        f"f_{int(np.ceil(float(row['frame']))):04d}.jpg"
        for row in labels["events"]["records"]
        if row.get("status") == "labeled" and row["event_type"] == "contact"
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"side", "court_x", "court_y"}.issubset(reader.fieldnames):
            return False
        seen = {row["frame"] for row in reader if row.get("clip") == clip}
    return contacts.issubset(seen)


def _pose_path(
    labels: dict[str, Any],
    case: AttemptCase | None = None,
    *,
    require_every_contact: bool = True,
) -> Path:
    """Resolve one automatic player artifact; explicit choices are recorded.

    Discovery keeps the frozen nine on exactly the artifacts they were fitted
    with.  Attempts whose clip has no sided rows in that benchmark root name an
    explicit alternative instead of silently falling back to an unsided file
    that carries no court position.

    ``require_every_contact`` is the old all-or-nothing rule.  Under the
    observation fallback the runner substitutes a neighbouring sided row for a
    contact this artifact does not cover, with a widened declared sigma, so a
    detector gap at one picture no longer costs the whole attempt.
    """
    if case is not None and case.player_localization is not None:
        explicit = paths.data_root() / "processed" / case.player_localization
        if require_every_contact and not _covers_contacts(explicit, labels):
            raise ValueError(f"explicit player localization misses a contact: {explicit}")
        return explicit
    match_root = (
        paths.data_root() / "processed/postseg_pipeline_benchmark_969657a" / labels["match_id"]
    )
    clip = labels["attempt"]["clip"]
    for name in ("player_pose_tracked_crop_native_v1.csv", "player_boxes_30_native_sided_v1.csv"):
        candidate = match_root / name
        if not candidate.is_file():
            continue
        with candidate.open(newline="") as handle:
            if any(row.get("clip") == clip for row in csv.DictReader(handle)):
                return candidate
    raise ValueError(f"no automatic player localization for {labels['match_id']} {clip}")


def _timing_uncertain_events_match(packet: dict, labels: dict) -> bool:
    """Do not reuse pre-fix packets that erased a timing-uncertain bounce."""
    uncertain = [
        row
        for row in labels.get("events", {}).get("records", [])
        if agent_attempt_prepare.timing_uncertain_bounce(row)
    ]
    if not uncertain:
        return True
    attempts = packet.get("attempts", [])
    if len(attempts) != 1:
        return False
    events = attempts[0].get("events", [])
    return all(
        sum(
            event.get("event_type") == "bounce"
            and event.get("frame") == row["frame"]
            and event.get("frame_interval") == row["frame_interval"]
            and event.get("status") == row["status"]
            and event.get("timing_status") == row["timing_status"]
            and event.get("exact_epoch_observed") is False
            for event in events
        )
        == 1
        for row in uncertain
    )


def _cached_preparation(
    case: AttemptCase, labels_path: Path, cache_root: Path, destination: Path
) -> dict[str, Any] | None:
    source = cache_root / case.key / "inputs"
    required = ("packet.json", "cameras.json", "qualification.json")
    if any(not (source / name).is_file() for name in required):
        return None
    packet = json.loads((source / "packet.json").read_text())
    if packet.get("external_label_binding", {}).get("record") != provenance.file_record(
        labels_path
    ):
        return None
    if packet.get("configuration", {}).get("transport_reference_policy") != "contact_anchor":
        return None
    if not _timing_uncertain_events_match(packet, json.loads(labels_path.read_text())):
        return None
    shutil.copytree(source, destination, copy_function=os.link)
    return {
        "status": "reused_provenance_matched_cache",
        "cache_root": str(cache_root),
        "source_packet": provenance.file_record(source / "packet.json"),
        "source_cameras": provenance.file_record(source / "cameras.json"),
        "copy_mode": "hardlink_immutable_inputs",
    }


def prepare_case(*arguments: Any, **keywords: Any) -> dict[str, Any]:
    """Prepare one attempt's cameras, or record why it could not be prepared."""
    try:
        return _prepare_case(*arguments, **keywords)
    except Exception as error:  # noqa: BLE001 - deliberately per-attempt
        return {
            "status": "held_preparation_precondition",
            "blocker": f"{type(error).__name__}: {error}",
        }


def _prepare_case(
    case: AttemptCase,
    output: Path,
    labels_root: Path,
    preparation_cache: Path | None = None,
    jobs: int = 8,
) -> dict[str, Any]:
    labels_path = labels_root / case.label_file
    destination = output / case.key / "inputs"
    log = output / case.key / "prepare.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / "cameras.json").exists():
        packet = json.loads((destination / "packet.json").read_text())
        if not _timing_uncertain_events_match(packet, json.loads(labels_path.read_text())):
            raise ValueError("prepared packet erased uncertain bounce timing; use a fresh output")
        return {"status": "reused_complete", "log": str(log)}
    if destination.exists():
        raise FileExistsError(f"incomplete preparation output: {destination}")
    if preparation_cache is not None:
        began = time.monotonic()
        cached = _cached_preparation(case, labels_path, preparation_cache, destination)
        if cached is not None:
            return {"log": None, "wall_seconds": time.monotonic() - began, **cached}
    receipt = _run(
        [
            sys.executable,
            "-m",
            "cv.experiments.connected_shooting.agent_attempt_prepare",
            "--labels",
            str(labels_path),
            "--jobs",
            str(jobs),
            "--transport-reference-policy",
            "contact_anchor",
            "--output",
            str(destination),
        ],
        log,
        check=False,
    )
    if receipt["returncode"]:
        # A label document the preparation adapter cannot accept is a fail-closed
        # abstention for that attempt with its reason recorded.  It must not cost
        # the whole cohort, and it stays in the denominator.
        if destination.exists():
            shutil.rmtree(destination)
        return {
            "status": "held_preparation_precondition",
            "log": str(log),
            "blocker": next(
                (line for line in reversed(log.read_text().splitlines()) if line.strip()),
                "unknown",
            ),
            **receipt,
        }
    return {"status": "completed", "log": str(log), **receipt}


MEASURED_BOUNCE_SURFACES = ("hard", "clay")

SURFACE_MODEL_MODES = ("off", "lines", "full", "measured")

#: Round index of each labeled broadcast, read from its own key: a slam final is round 7 and
#: an ``r128`` is round 1.  A round is a proxy for tournament day; it is the only day evidence
#: these broadcasts carry, and the corpus day slope is what consumes it.
ROUND_BY_SUFFIX = {
    "r128": 1,
    "r64": 2,
    "r32": 3,
    "r16": 4,
    "qf": 5,
    "sf": 6,
    "f": 7,
    "rr": 3,
}


def broadcast_round(match_id: str | None) -> float | None:
    """The round this broadcast is, or ``None`` when the key does not say.

    Two key shapes carry it: ``ao2019f`` and ``wim2025r32`` put it on the leading token, and
    ``atp_2024_540_r16_217_...`` puts it in its own token.  A bare suffix match is only tried
    on the leading token, because otherwise a player called Fritz would be a final.
    """
    if not match_id:
        return None
    tokens = str(match_id).split("_")
    for suffix, index in sorted(ROUND_BY_SUFFIX.items(), key=lambda row: -len(row[0])):
        if tokens[0].endswith(suffix):
            return float(index)
    for token in tokens[1:]:
        if token in ROUND_BY_SUFFIX:
            return float(ROUND_BY_SUFFIX[token])
    return None


def surface_model_spec(
    case: AttemptCase,
    grass_bounce_profile: str,
    surface_model_mode: str,
    broadcasts: dict[str, Any] | None,
    surface_model_surfaces: tuple[str, ...] = MEASURED_BOUNCE_SURFACES,
) -> str | None:
    """The ``physics.surface_model`` spec this attempt's surface name stands for, if any.

    ``None`` means the unqualified surface: the fitter as it stands, arithmetically
    untouched.  A spec means the region- and day-aware court -- a per-broadcast wear scalar,
    the fixed spatial wear kernel and the line-skid switch of
    ``docs/brainstorm/surface_bounce_literature.md``.

    Grass and the measured surfaces are selected separately because they are different
    claims.  On grass the model also replaces the *base* pair, because grass has no corpus
    row at all and its broadcast-measured row is a retained negative.  On hard and clay the
    base stays the Hawk-Eye row and the model only adds the region and line terms, so
    ``lines`` and ``full`` can only move an attempt through those terms.
    """
    surface = fitted_surface(case, grass_bounce_profile)
    if surface == "grass" and grass_bounce_profile == "regional_v2":
        return surface_model.format_spec(grass_surface_model.regional_v2_model(case.match_id))
    if surface == "grass" and grass_bounce_profile in {"regional", "measured"}:
        base = "literature"
        mode = "measured" if grass_bounce_profile == "measured" else "full"
    elif (
        surface in MEASURED_BOUNCE_SURFACES
        and surface in surface_model_surfaces
        and surface_model_mode != "off"
    ):
        base, mode = "reference", surface_model_mode
    else:
        return None
    fitted = (broadcasts or {}).get(case.match_id or "", {})
    wear = float(fitted.get("wear", 1.0)) if mode == "full" else 1.0
    amplitude = (
        float(fitted.get("amplitude", surface_model.LITERATURE[surface].amplitude_prior_mean))
        if mode == "full"
        else 0.0
    )
    return surface_model.format_spec(
        surface_model.SurfaceModel(
            surface=surface,
            wear=wear,
            amplitude=amplitude,
            base_source=base,
            region_map="measured" if mode == "measured" else "literature",
            day=broadcast_round(case.match_id) if mode == "measured" else None,
            # A broadcast row may switch the line term off, which is what a null control
            # needs: a bounce perturbation of comparable size that claims no physics.
            line_switch=bool(fitted.get("line_switch", True)),
            provenance=str(fitted.get("provenance", "prior")),
        )
    )


def surface_model_environment(spec: str | None) -> dict[str, str]:
    """The one environment entry that selects a court model for a search subprocess.

    The surface reaches ``physics.bounce_reference`` as a bare name through a subprocess
    this sweep launches and modules it does not own, so the arm is selected here and read
    once there (``physics.surface_model.OVERRIDE_VARIABLE``).  It is recorded in this
    sweep's frozen configuration and in every attempt row.
    """
    if spec is None:
        return {}
    return {surface_model.OVERRIDE_VARIABLE: json.dumps({spec.split("@")[0]: spec})}


def fitted_surface(case: AttemptCase, grass_bounce_profile: str) -> str:
    """Name the measured bounce profile this attempt is actually fitted with.

    The Hawk-Eye corpus behind ``physics.bounce_reference`` has clay and hard and
    no grass.  ``hard`` substitutes the hard profile, which measures the rest of
    the chain at the cost of a bounce model that is not this surface; ``grass``
    uses the thin broadcast-label-derived grass row, which is this surface but is
    eleven impacts wide; ``hold`` refuses and measures nothing.  The choice is
    explicit and recorded, and the attempt's real surface is never rewritten.
    """
    if case.surface in MEASURED_BOUNCE_SURFACES:
        return case.surface
    if grass_bounce_profile == "hold":
        return case.surface
    if grass_bounce_profile in {"regional", "regional_v2", "measured"}:
        # The surface *name* is still grass; what changes is which court model that name
        # stands for, which travels beside it (:func:`surface_model_spec`).
        return "grass"
    return grass_bounce_profile


def player_ledger_path(case: AttemptCase) -> Path | None:
    """Locate this attempt's ``tennis_player_state_v1`` sidecar, if one shipped."""
    if case.match_id is None or case.clip is None:
        return None
    candidate = (
        paths.data_root()
        / "processed/playerledger/ledger_v1/states"
        / case.match_id
        / f"{case.clip}_player_state_v1.csv"
    )
    return candidate if candidate.is_file() else None


def toss_label_path(case: AttemptCase, labels_root: Path) -> Path | None:
    """Resolve the attempt's additive toss file, then its dense fallback."""
    if case.match_id is not None and case.clip is not None:
        candidates = sorted(labels_root.glob(f"{case.match_id}_{case.clip}_*_toss_v1.json"))
        if len(candidates) > 1:
            attempt = re.search(r"_((?:attempt|fault)\d+)", case.label_file)
            if attempt is not None:
                token = f"_{attempt.group(1)}_"
                candidates = [path for path in candidates if token in path.name]
        if len(candidates) > 1:
            raise ValueError(f"multiple toss witnesses for {case.key}")
        if candidates:
            return candidates[0]
    if case.dense_label_file is not None:
        candidate = labels_root / case.dense_label_file
        if candidate.is_file():
            return candidate
    return None


def search_case(*arguments: Any, **keywords: Any) -> dict[str, Any]:
    """Fit one attempt, or record why it could not be fitted.

    No single attempt may cost the cohort.  Every failure inside the runner --
    a missing explicit input, an unreadable artifact, a raised precondition --
    becomes a fail-closed abstention for that attempt with its reason, and the
    attempt stays in the denominator.
    """
    try:
        return _search_case(*arguments, **keywords)
    except Exception as error:  # noqa: BLE001 - deliberately per-attempt
        return {
            "status": "held_runner_precondition",
            "blocker": f"{type(error).__name__}: {error}",
        }


def _search_case(
    case: AttemptCase,
    output: Path,
    configuration_path: Path,
    bounce_witness_mode: str,
    athlete_prior_mode: str,
    labels_root: Path,
    event_recovery: str = "off",
    max_topology_branches: int = 6,
    observation_fallback: str = "off",
    player_ledger: bool = False,
    dense_labels: bool = False,
    dense_label_retry: bool = False,
    grass_bounce_profile: str = "hold",
    serve_reach_cut: str = "off",
    serve_reach_margin_statures: float = 0.10,
    serve_contact_epoch: str = "off",
    serve_start_prior: str = "off",
    serve_contact_hypotheses: str = "off",
    serve_contact_history: Path | None = None,
    serve_ending_consistency: str = "off",
    refine_inequalities: str = "terminal_only",
    anchor_residuals: str = "present",
    seed_restarts: str = "three",
    reference_restart: str = "off",
    shooting_parameterization: str = "single",
    local_flight_refits: str = "off",
    shared_state_second_stage_cache: Path | None = None,
    coarse_iterations: int = 150,
    surface_spec: str | None = None,
    bounce_bracket_frames: float = 1.0,
) -> dict[str, Any]:
    labels_path = labels_root / case.label_file
    labels = json.loads(labels_path.read_text())
    destination = output / case.key / "search"
    log = output / case.key / "search.log"
    if (destination / "report.json").exists():
        return {"status": "reused_complete", "log": str(log)}
    discarded = None
    if destination.exists():
        # A previous run of this same sweep left a search that raised before it
        # wrote its report.  That is derived output of this directory, so it is
        # discarded and retried rather than blocking the whole cohort.
        discarded = sorted(item.name for item in destination.iterdir())
        shutil.rmtree(destination)
    inputs = output / case.key / "inputs"
    if not (inputs / "cameras.json").is_file():
        return {
            "status": "held_preparation_precondition",
            "log": str(log),
            "blocker": "the preparation adapter did not accept this label document",
        }
    command = [
        sys.executable,
        "-m",
        "cv.experiments.connected_shooting.agent_whole_point_search",
        "--labels",
        str(labels_path),
        "--packet",
        str(inputs / "packet.json"),
        "--cameras",
        str(inputs / "cameras.json"),
        "--pose-csv",
        str(_pose_path(labels, case, require_every_contact=observation_fallback == "off")),
        "--coarse-iterations",
        str(coarse_iterations),
        "--refine-iterations",
        "200",
        "--serve-number",
        str(case.serve_number),
        "--serve-number-evidence",
        str(configuration_path),
        "--surface",
        fitted_surface(case, grass_bounce_profile),
        "--bounce-witness-mode",
        bounce_witness_mode,
        "--pose-image-scale",
        str(case.pose_image_scale),
        "--athlete-prior-mode",
        athlete_prior_mode,
        "--event-recovery",
        event_recovery,
        "--max-topology-branches",
        str(max_topology_branches),
        "--observation-fallback",
        observation_fallback,
        "--serve-reach-cut",
        serve_reach_cut,
        "--serve-reach-margin-statures",
        str(serve_reach_margin_statures),
        "--output",
        str(destination),
    ]
    if serve_contact_epoch != "off":
        command.extend(["--serve-contact-epoch", serve_contact_epoch])
    if serve_start_prior != "off":
        command.extend(["--serve-start-prior", serve_start_prior])
    if serve_contact_hypotheses != "off":
        command.extend(["--serve-contact-hypotheses", serve_contact_hypotheses])
    if serve_start_prior != "off" or serve_contact_hypotheses != "off":
        toss = toss_label_path(case, labels_root)
        if toss is not None:
            command.extend(["--toss-labels", str(toss)])
        if not case.serve_number_resolved:
            command.append("--serve-number-uncertain")
    if serve_contact_history is not None:
        command.extend(["--serve-contact-history", str(serve_contact_history)])
    if serve_ending_consistency != "off":
        command.extend(["--serve-ending-consistency", serve_ending_consistency])
        if case.ending_kind:
            command.extend(["--ending-kind", case.ending_kind])
    if seed_restarts != "off":
        command.extend(["--seed-restarts", seed_restarts])
    if reference_restart != "off":
        command.extend(["--reference-restart", reference_restart])
    if shooting_parameterization != "single":
        command.extend(["--shooting-parameterization", shooting_parameterization])
    if local_flight_refits != "off":
        command.extend(["--local-flight-refits", local_flight_refits])
    if shared_state_second_stage_cache is not None:
        command.extend(
            [
                "--shared-state-second-stage-reference",
                str(shared_state_second_stage_cache / case.key / "search/report.json"),
            ]
        )
    if bounce_bracket_frames != 1.0:
        command.extend(["--bounce-bracket-frames", str(bounce_bracket_frames)])
    if refine_inequalities != "in" or anchor_residuals != "absent":
        # Only an explicit non-reference arm adds a flag, so the reference run
        # invokes the search with exactly its published command line.
        command.extend(
            [
                "--refine-inequalities",
                refine_inequalities,
                "--anchor-residuals",
                anchor_residuals,
            ]
        )
    if dense_labels and case.dense_label_file is not None:
        command.extend(["--dense-labels", str(labels_root / case.dense_label_file)])
    ledger = player_ledger_path(case)
    if player_ledger and ledger is not None:
        command.extend(["--player-ledger", str(ledger)])
    for player, stature in case.player_statures_m:
        command.extend(["--player-stature", f"{player}={stature}"])
        command.extend(["--player-order", player])
    environment = surface_model_environment(surface_spec)
    receipt = _run(command, log, check=False, environment=environment)
    dense_retry = None
    if (
        receipt["returncode"]
        and dense_label_retry
        and case.dense_label_file is not None
        and "--dense-labels" not in command
    ):
        # The base export abstains where this attempt needs an observation. The
        # additive dense revision is an independent second annotation, so it is
        # not a default; but an attempt that fails closed without it has nothing
        # to lose, and the retry can only turn a hold into a measurement.
        if destination.exists():
            shutil.rmtree(destination)
        dense_command = [
            *command,
            "--dense-labels",
            str(labels_root / case.dense_label_file),
        ]
        dense_receipt = _run(dense_command, log, check=False, environment=environment)
        dense_retry = {
            "attempted": True,
            "dense_label_file": case.dense_label_file,
            "returncode": dense_receipt["returncode"],
        }
        if not dense_receipt["returncode"]:
            return {
                "status": "completed",
                "log": str(log),
                "discarded_incomplete_output": discarded,
                "dense_label_retry": dense_retry,
                **dense_receipt,
            }
        receipt = dense_receipt
    if receipt["returncode"]:
        # A precondition the current arm cannot satisfy is a fail-closed
        # abstention for this attempt, not a reason to lose the whole cohort.
        return {
            "status": "held_input_precondition",
            "log": str(log),
            "blocker": next(
                (line for line in reversed(log.read_text().splitlines()) if line.strip()),
                "unknown",
            ),
            "discarded_incomplete_output": discarded,
            "dense_label_retry": dense_retry,
            **receipt,
        }
    return {
        "status": "completed",
        "log": str(log),
        "discarded_incomplete_output": discarded,
        **receipt,
    }


def _flatten_bounce_errors(value: Any) -> list[float]:
    rows: list[float] = []
    if isinstance(value, list):
        for item in value:
            rows.extend(_flatten_bounce_errors(item))
    elif value is not None:
        rows.append(float(value))
    return rows


def _selected_candidate(report: dict[str, Any]) -> dict[str, Any]:
    return (
        report.get("selected_arms", {}).get("combined_toss_and_serve_prior")
        or report.get("selected")
        or report["diagnostic_candidate"]
    )


def held_case(
    case: AttemptCase,
    output: Path,
    receipt: dict[str, Any],
    grass_bounce_profile: str = "hold",
    surface_spec: str | None = None,
) -> dict[str, Any]:
    """Keep an unattemptable attempt in the denominator with its exact blocker."""
    return {
        "key": case.key,
        "metadata": _case_metadata(case, grass_bounce_profile, surface_spec),
        "attempt_id": None,
        "status": receipt.get("status", "held_input_precondition"),
        "reconstructed": False,
        "reconstructed_on_the_labeled_topology": False,
        "topology": {"name": None, "shortened": False},
        "family": {"count": 0, "depth_y_range_m": None, "width_m": None, "midpoint_m": None},
        "blocking_bounds": [receipt.get("status", "held_input_precondition")],
        "input_precondition_blocker": receipt.get("blocker"),
        "search_log": receipt.get("log"),
        "pictures": [],
        "report": None,
    }


def selected_topology(report: dict[str, Any]) -> dict[str, Any]:
    """Name the topology this fit actually used against the one the labels carry.

    Event recovery may demote a supplied contact or end the point at an earlier
    bounce.  Either drops a labeled shot from the modeled window, so the fit is a
    partial point: real evidence about the flights it kept, and not a
    reconstruction of the labeled point.
    """
    recovery = report.get("event_recovery") or {}
    labeled = sum(1 for row in report.get("supplied_events", []) if row["event_type"] == "contact")
    modeled = sum(1 for row in report.get("events", []) if row["event_type"] == "contact")
    return {
        "name": recovery.get("selected_topology", "supplied"),
        "labeled_contacts": labeled,
        "modeled_contacts": modeled,
        "shortened": bool(labeled and modeled < labeled),
        "recovered_events": recovery.get("recovered_events", []),
        "demoted_events": recovery.get("demoted_events", []),
        "supplied_topology_blocker": recovery.get("supplied_topology_blocker"),
    }


def branch_optimizer_exits(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One optimizer exit receipt per fitted depth branch, coarse and refined.

    A stalled solve and a converged one that misses a gate are different
    diagnoses and were previously indistinguishable in the sweep report, which
    published only the selected branch's exit.  This adds the primary SLSQP
    status/message/iteration count, the training and withheld RMS at that
    stage, the separate feasibility solve at the final published limits, and
    whether the branch would pass the unchanged acceptance gates.  It reads
    receipts the search already wrote; no fit changes.
    """
    rows = []
    for stage in ("coarse_candidates", "refined_candidates"):
        for candidate in report.get(stage) or []:
            fit = candidate["measurement"]["fit"]
            primary = fit.get("primary_optimizer") or {
                "success": fit["success"],
                "status": fit["status"],
                "message": fit["message"],
                "iterations": fit["iterations"],
            }
            feasibility = fit.get("final_gate_feasibility")
            evidence = candidate["evidence"]
            rows.append(
                {
                    "stage": candidate["stage"],
                    "depth_hypothesis_m": candidate["depth_hypothesis_m"],
                    "primary_optimizer": primary,
                    "final_gate_feasibility": (
                        None
                        if feasibility is None
                        else {
                            "method": feasibility["method"],
                            "success": feasibility["success"],
                            "status": feasibility["status"],
                            "message": feasibility["message"],
                            "iterations": feasibility["iterations"],
                        }
                    ),
                    "resolved_success": fit["success"],
                    "resolved_status": fit["status"],
                    "resolved_iterations": fit["iterations"],
                    "initial_cost": fit["initial_cost"],
                    "final_cost": fit["final_cost"],
                    "training_rms_px": candidate["measurement"]["rms_px"]["training"],
                    "withheld_rms_px": candidate["measurement"]["rms_px"]["withheld"],
                    "passes_unchanged_acceptance_gates": bool(evidence["survived"]),
                    "death_reasons": evidence["death_reasons"],
                    "solver_experimental_arm": fit.get("experimental_arm"),
                    "wall_seconds": candidate["wall_seconds"],
                }
            )
    return rows


def summarize_case(
    case: AttemptCase,
    output: Path,
    grass_bounce_profile: str = "hold",
    surface_spec: str | None = None,
) -> dict[str, Any]:
    report_path = output / case.key / "search/report.json"
    report = json.loads(report_path.read_text())
    candidate = _selected_candidate(report)
    evidence = candidate["evidence"]
    family = report["families"]["combined_toss_and_serve_prior"]
    errors = _flatten_bounce_errors(evidence["bounce_horizontal_errors_m"])
    gate_errors = _flatten_bounce_errors(evidence.get("bounce_gate_errors_m", errors))
    maximum_bounce = max(errors) if errors else None
    maximum_bounce_gate = max(gate_errors) if gate_errors else None
    maximum_direction = evidence["directional_support"]["maximum_window_rms_px"]
    legal = [row for row in report["refined_candidates"] if row["evidence"]["survived"]]
    toss_sensitivity = []
    for cutoff in (2.0, 2.5, 3.0, 3.5):
        members = [
            row["depth_hypothesis_m"]
            for row in legal
            if row["evidence"]["toss_witness"]["weighted_image_rms"] is None
            or row["evidence"]["toss_witness"]["weighted_image_rms"] <= cutoff
        ]
        toss_sensitivity.append(
            {"cutoff": cutoff, "diagnostic_count": len(members), "member_depths_m": sorted(members)}
        )
    prior_sensitivity = {}
    side = report["toss_server_feet"]["side"]
    for serve_number in (1, 2):
        ranked = sorted(
            legal,
            key=lambda row: (
                row["evidence"]["input_only_rank_score"]
                + row["evidence"]["toss_witness"]["selector_penalty"]
                + toss_witness.serve_contact_prior(
                    row["evidence"]["contact_xyz_m"][0][1], side, serve_number
                )["selector_penalty"],
                row["depth_hypothesis_m"],
            ),
        )
        prior_sensitivity[str(serve_number)] = (
            None if not ranked else ranked[0]["depth_hypothesis_m"]
        )
    toss_weight_sensitivity = []
    for multiplier in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        ranked = sorted(
            legal,
            key=lambda row: (
                row["evidence"]["input_only_rank_score"]
                + multiplier * row["evidence"]["toss_witness"]["selector_penalty"]
                + row["evidence"]["serve_contact_prior"]["selector_penalty"],
                row["depth_hypothesis_m"],
            ),
        )
        toss_weight_sensitivity.append(
            {
                "multiplier": multiplier,
                "selected_depth_m": None if not ranked else ranked[0]["depth_hypothesis_m"],
                "legal_candidate_count": len(ranked),
            }
        )
    contact = evidence["contact_xyz_m"]
    if np.ndim(contact) == 2:
        contact = contact[0]
    speed = evidence.get("serve_speed_witness", {})
    toss = evidence.get("toss_witness", {})
    topology = selected_topology(report)
    return {
        "key": case.key,
        "metadata": _case_metadata(case, grass_bounce_profile, surface_spec),
        "attempt_id": report["attempt_id"],
        "status": report["status"],
        "reconstructed": bool(family["reconstructed_on_family_basis"]),
        # A recovered topology may model fewer contacts than the labels carry.
        # That is a partial point and must never be read as a reconstruction of
        # the labeled one, so both counts are published beside the verdict.
        "topology": topology,
        "reconstructed_on_the_labeled_topology": bool(
            family["reconstructed_on_family_basis"] and not topology["shortened"]
        ),
        "family": family,
        "selected_or_diagnostic_depth_m": candidate["depth_hypothesis_m"],
        "selected_or_diagnostic_contact_xyz_m": contact,
        "selected_or_diagnostic_legal_checks": evidence["checks"],
        "serve_depth_bounds_m": report["configuration"]["depth_bounds_m"],
        "serve_height_bounds_m": report["configuration"]["serve_height_zero_penalty_interval_m"],
        "player_distances_m": evidence["player_distances_m"],
        "player_reach_margins_m": [1.75 - value for value in evidence["player_distances_m"]],
        "legacy_global_cap_diagnostics": evidence.get("legacy_global_cap_diagnostics"),
        "athlete_soft_priors": evidence.get("athlete_soft_priors"),
        "blocking_bounds": evidence["death_reasons"],
        "bounce_ray_errors_m": evidence["bounce_horizontal_errors_m"],
        "bounce_gate_errors_m": evidence.get(
            "bounce_gate_errors_m", evidence["bounce_horizontal_errors_m"]
        ),
        "maximum_bounce_ray_error_m": maximum_bounce,
        "maximum_bounce_gate_error_m": maximum_bounce_gate,
        "bounce_ray_margin_m": None
        if maximum_bounce_gate is None
        else 0.9144 - maximum_bounce_gate,
        "maximum_directional_window_rms_px": maximum_direction,
        "directional_window_margin_px": None
        if maximum_direction is None
        else 16.0 - maximum_direction,
        "rms_px": candidate["measurement"]["rms_px"],
        "optimizer": {
            "success": candidate["measurement"]["fit"]["success"],
            "message": candidate["measurement"]["fit"]["message"],
            "iterations": candidate["measurement"]["fit"]["iterations"],
        },
        "branch_optimizer_exits": branch_optimizer_exits(report),
        "best_branch_training_rms_px": min(
            (
                row["measurement"]["rms_px"]["training"]
                for row in (
                    report.get("refined_candidates") or report.get("coarse_candidates") or []
                )
            ),
            default=None,
        ),
        "speed_witness": speed,
        "toss_witness": {
            "status": toss.get("status"),
            "weighted_image_rms": toss.get("weighted_image_rms"),
            "selector_penalty": toss.get("selector_penalty"),
            "hard_gate": False,
        },
        "toss_cutoff_sensitivity_diagnostic_only": toss_sensitivity,
        "toss_selector_weight_sensitivity": toss_weight_sensitivity,
        "serve_prior_selected_depth_sensitivity": prior_sensitivity,
        "pictures": [str(output / case.key / "search" / name) for name in report["artifacts"]],
        "report": str(report_path),
    }


def _camera_parameter_summary(document: dict[str, Any]) -> dict[str, Any]:
    anchors = document.get("all_anchor_diagnostics", [])
    parameters = np.asarray([row["fit"]["parameters"] for row in anchors], float)
    frames = [int(row["frame"]) for row in anchors]
    anchor = document.get("anchor", {})
    active_fit = anchor.get("fit", {})
    active_frame = anchor.get("frame")
    if active_frame is None:
        active_rows = [
            row for row in document["cameras"] if row.get("source", "").endswith("ground_anchor")
        ]
        active_frame = None if not active_rows else active_rows[0]["frame"]
    references = [row.get("local_anchor_frame") for row in document["cameras"]]
    references = [value for value in references if value is not None]
    return {
        "parameter_order": [
            "rotation_vector_x",
            "rotation_vector_y",
            "rotation_vector_z",
            "translation_x",
            "translation_y",
            "translation_z",
            "log_focal_native_px",
        ],
        "anchor_frames": frames,
        "anchor_count": len(frames),
        "parameter_ranges": [] if not len(parameters) else np.ptp(parameters, axis=0).tolist(),
        "ground_fit_rms_px": [row["fit"]["native_rms_px"] for row in anchors],
        "active_reference_frame": active_frame,
        "active_reference_parameters": active_fit.get("parameters"),
        "active_reference_camera_center_m": active_fit.get("camera_center_m"),
        "active_reference_focal_native_px": active_fit.get("focal_native_px"),
        "active_reference_ground_fit_rms_px": active_fit.get("native_rms_px"),
        "transport_reference_policy": document.get("transport_reference_policy", "legacy"),
        "distinct_per_frame_references": sorted(set(references)),
        "reference_handoffs": sum(a != b for a, b in zip(references, references[1:])),
    }


def _projection_delta(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    left = {int(row["frame"]): row for row in first["cameras"] if "P" in row}
    right = {int(row["frame"]): row for row in second["cameras"] if "P" in row}
    frames = sorted(set(left) & set(right))
    xyz = np.asarray(
        [
            [x, y, z]
            for z in (0.0325, 1.5, 2.8)
            for x in (1.37, 5.485, 9.6)
            for y in (0.0, 5.485, 11.885, 18.285, 23.77)
        ]
    )
    by_height = {}
    for z in (0.0325, 1.5, 2.8):
        points = xyz[xyz[:, 2] == z]
        differences = []
        for frame in frames:
            cameras_a = np.repeat(np.asarray(left[frame]["P"])[None, :, :], len(points), axis=0)
            cameras_b = np.repeat(np.asarray(right[frame]["P"])[None, :, :], len(points), axis=0)
            differences.extend(
                np.linalg.norm(
                    camera_geometry.project(cameras_a, points)
                    - camera_geometry.project(cameras_b, points),
                    axis=1,
                ).tolist()
            )
        by_height[str(z)] = {
            "rms_px": float(np.sqrt(np.mean(np.square(differences)))),
            "max_px": float(max(differences)),
        }
    return {"shared_frames": len(frames), "fixed_court_grid_projection_delta": by_height}


def camera_diagnosis(output: Path) -> dict[str, Any]:
    root = paths.data_root() / "processed"
    sources = {
        "ao2": {
            "old": root / "pipeline_goal_20260905/s6_contiguous_owner_pt2_inputs_v1/cameras.json",
            "nearest": root / "tossprior/ao2_inputs/cameras.json",
        },
        "men2": {
            "old": root / "s6real/agent_uso2020_pt0002_inputs/cameras.json",
            "nearest": root / "tossprior/men2_inputs/cameras.json",
        },
    }
    result = {}
    for key, paths_by_arm in sources.items():
        documents = {name: json.loads(path.read_text()) for name, path in paths_by_arm.items()}
        documents["fixed_contact_reference"] = json.loads(
            (output / key / "inputs/cameras.json").read_text()
        )
        result[key] = {
            "camera_parameters": {
                name: _camera_parameter_summary(document) for name, document in documents.items()
            },
            "old_to_nearest_projection_delta": _projection_delta(
                documents["old"], documents["nearest"]
            ),
            "nearest_to_fixed_projection_delta": _projection_delta(
                documents["nearest"], documents["fixed_contact_reference"]
            ),
            "finding": (
                "Independent anchor fits change airborne projection parameters; nearest-anchor "
                "transport also changes the camera parameterization inside one continuous path. "
                "The fixed arm uses every labeled frame for qualification but one contact-near "
                "physical reference for all per-frame image registrations."
            ),
        }
    return result


def _rally_length_band(contacts: int | None) -> str:
    """Band an attempt by how many contacts its labeled topology holds."""
    if contacts is None:
        return "unknown"
    if contacts <= 1:
        return "1_serve_only"
    if contacts == 2:
        return "2_serve_plus_return"
    if contacts <= 4:
        return "3_4_short_rally"
    if contacts <= 8:
        return "5_8_medium_rally"
    return "9_plus_long_rally"


def _ending_family(kind: str | None) -> str:
    """Group the free-text ending kinds into the families the owner reads."""
    if not kind:
        return "unknown"
    text = kind.lower()
    if "fault" in text:
        return "serve_fault"
    if "net" in text:
        return "net"
    if "second_bounce" in text or "double_bounce" in text:
        return "second_bounce"
    if "winner" in text or "unreturned" in text or "missed_return" in text:
        return "unreturned_or_winner"
    if "out" in text or "wide" in text or "long" in text:
        return "out"
    return "other"


def _case_metadata(
    case: AttemptCase,
    grass_bounce_profile: str = "hold",
    surface_spec: str | None = None,
) -> dict[str, Any]:
    """Slice keys travel with every attempt row so a reader never rejoins by hand."""
    return {
        "match_id": case.match_id,
        "clip": case.clip,
        "label_file": case.label_file,
        "surface": case.surface,
        "fitted_surface": fitted_surface(case, grass_bounce_profile),
        "surface_model_spec": surface_spec,
        "gender": case.gender,
        "serve_number": case.serve_number,
        "serve_number_resolved": case.serve_number_resolved,
        "ending_kind": case.ending_kind,
        "ending_family": _ending_family(case.ending_kind),
        "contact_count": case.contact_count,
        "rally_length_band": _rally_length_band(case.contact_count),
        "lane_of_origin": case.lane,
        "dense_label_file": case.dense_label_file,
        "player_localization": case.player_localization,
        "pose_image_scale": case.pose_image_scale,
        "player_statures_m": [list(row) for row in case.player_statures_m],
        "discovery": case.discovery,
    }


def yield_by_slice(
    attempts: list[dict[str, Any]],
    cases: tuple[AttemptCase, ...],
    grass_bounce_profile: str = "hold",
) -> dict[str, Any]:
    """Reconstructed-over-denominator counts on every slice the owner asked for.

    Abstentions stay in every denominator; a held attempt is a zero, never an
    exclusion.
    """
    by_key = {case.key: case for case in cases}
    dimensions = {
        "broadcast": lambda case: case.match_id or "unknown",
        "surface": lambda case: case.surface,
        "fitted_surface": lambda case: fitted_surface(case, grass_bounce_profile),
        "gender": lambda case: case.gender or "unknown",
        "rally_length": lambda case: _rally_length_band(case.contact_count),
        "serve_number": lambda case: (
            f"serve_{case.serve_number}"
            + ("" if case.serve_number_resolved else "_unresolved_default")
        ),
        "ending_kind": lambda case: _ending_family(case.ending_kind),
        "lane_of_origin": lambda case: case.lane or "unknown",
        "status": lambda case: "n/a",
    }
    result: dict[str, Any] = {}
    for name, selector in dimensions.items():
        counts: dict[str, dict[str, Any]] = {}
        for row in attempts:
            case = by_key.get(row["key"])
            if case is None:
                continue
            label = row["status"] if name == "status" else selector(case)
            bucket = counts.setdefault(
                label, {"attempts": 0, "reconstructed": 0, "held_input_precondition": 0, "keys": []}
            )
            bucket["attempts"] += 1
            bucket["reconstructed"] += int(bool(row["reconstructed"]))
            bucket["held_input_precondition"] += int(row["status"] == "held_input_precondition")
            bucket["keys"].append(row["key"])
        for bucket in counts.values():
            bucket["yield"] = bucket["reconstructed"] / bucket["attempts"]
            bucket["keys"].sort()
        result[name] = dict(sorted(counts.items()))
    return result


def _write_html(report: dict[str, Any], output: Path) -> None:
    rows = []
    for item in report["attempts"]:
        family = item["family"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['key'])}</td>"
            f"<td>{item['reconstructed']}</td>"
            f"<td>{family['count']}</td>"
            f"<td>{html.escape(str(family['depth_y_range_m']))}</td>"
            f"<td>{family['width_m']}</td>"
            f"<td>{family['midpoint_m']}</td>"
            f"<td>{html.escape(str(item.get('input_precondition_blocker') or ''))}</td>"
            f'<td><a href="{item["key"]}/search/index.html">pictures/report</a></td>'
            "</tr>"
        )
    content = (
        '<!doctype html><meta charset="utf-8"><title>Labeled connected-family sweep</title>'
        f"<h1>{report['reconstructed_count']}/{report['attempt_denominator']} "
        f"reconstructed ({html.escape(report['cohort'])})</h1>"
        "<p>Opened human-derived development evidence; not automatic inference or independent XYZ truth.</p>"
        '<table border="1"><thead><tr><th>Attempt</th><th>Reconstructed</th><th>Family</th>'
        "<th>Range m</th><th>Width m</th><th>Midpoint m</th><th>Input blocker</th>"
        "<th>Evidence</th></tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table><p><a href="report.json">Full aggregate report</a></p>'
    )
    (output / "index.html").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--preparation-workers",
        type=int,
        default=None,
        help=(
            "attempts prepared at once; camera transport is independent per attempt, so a "
            "large cohort may use more of the machine than the eight-worker fit budget"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preparation-cache",
        type=Path,
        help="optional prior sweep root; reuse only exact label-bound contact-anchor packets",
    )
    parser.add_argument(
        "--bounce-witness-mode",
        choices=("native_ray_average", "subframe_graded_circle"),
        default="subframe_graded_circle",
    )
    parser.add_argument(
        "--athlete-prior-mode",
        choices=("global_hard_caps", "stature_pose_soft"),
        default="stature_pose_soft",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1",
    )
    parser.add_argument(
        "--serve-reach-cut",
        choices=("off", "on"),
        default="off",
        help="hard serve-contact reach cylinder cutting the contact image ray in the search",
    )
    parser.add_argument("--serve-reach-margin-statures", type=float, default=0.10)
    parser.add_argument(
        "--serve-contact-epoch",
        choices=("off", "on"),
        default="off",
        help="fit the first serve-contact epoch with its labeled bracket as a soft prior",
    )
    parser.add_argument(
        "--serve-start-prior",
        choices=("off", "on"),
        default="off",
        help="seed and softly pull serve contact from the external mixture and toss",
    )
    parser.add_argument(
        "--serve-contact-hypotheses",
        choices=("off", "on"),
        default="off",
        help="add toss/outgoing/prior/feet contact hypotheses as seed-only restarts",
    )
    parser.add_argument(
        "--serve-contact-history",
        type=Path,
        default=None,
        help="causal accepted-contact history for ranking-only same-player regularization",
    )
    parser.add_argument(
        "--serve-ending-consistency",
        choices=("off", "on"),
        default="off",
        help="gate labeled serve faults/outs on modeled and witnessed service-box calls",
    )
    parser.add_argument("--cohort", choices=sorted(COHORTS), default="labeled17")
    parser.add_argument(
        "--attempts",
        nargs="*",
        default=None,
        help="restrict a discovered cohort to these keys; the denominator follows",
    )
    parser.add_argument("--event-recovery", choices=("off", "on"), default="off")
    parser.add_argument("--max-topology-branches", type=int, default=6)
    parser.add_argument(
        "--observation-fallback",
        choices=("off", "on"),
        default="off",
        help="fail soft on the label adapter's observation preconditions",
    )
    parser.add_argument(
        "--grass-bounce-profile",
        choices=("hold", "hard", "grass", "regional", "regional_v2", "measured"),
        default="hold",
        help=(
            "the corpus has no grass profile: substitute hard, use the thin "
            "broadcast-derived grass row, hold the attempt, fit the literature's region- "
            "and day-aware kernel on the fresh-grass base (regional), fit the model-free "
            "per-broadcast kernel around the exact hard-profile prior (regional_v2), or fit the "
            "corpus-measured region map on that base (measured)"
        ),
    )
    parser.add_argument(
        "--surface-model",
        choices=SURFACE_MODEL_MODES,
        default="off",
        help=(
            "the region- and day-aware court on the measured surfaces: off leaves hard and "
            "clay exactly as they are, lines adds only the literature line-skid switch, full "
            "also adds the per-broadcast wear scalar and the literature kernel, measured "
            "replaces the kernel with the Hawk-Eye region/line/round regression"
        ),
    )
    parser.add_argument(
        "--surface-model-surfaces",
        nargs="*",
        choices=MEASURED_BOUNCE_SURFACES,
        default=list(MEASURED_BOUNCE_SURFACES),
        help=(
            "which measured surfaces --surface-model applies to; the corpus line effect is "
            "significant on clay and null on hard, so the two are separable"
        ),
    )
    parser.add_argument(
        "--surface-model-broadcasts",
        type=Path,
        default=None,
        help=(
            "a cv.experiments.connected_shooting.surface_wear_fit artifact supplying each "
            "broadcast's fitted wear scalar; broadcasts it does not name keep the prior"
        ),
    )
    parser.add_argument(
        "--dense-label-retry",
        choices=("off", "on"),
        default="off",
        help="retry an attempt held on an observation precondition with its dense revision",
    )
    parser.add_argument(
        "--dense-labels",
        choices=("off", "on"),
        default="off",
        help=(
            "fill base front abstentions from the additive every-frame dense revision; "
            "an independent second annotation, measured as its own arm"
        ),
    )
    parser.add_argument(
        "--player-ledger",
        choices=("off", "on"),
        default="off",
        help="take each contact's court position from the shipped player-state sidecar",
    )
    parser.add_argument(
        "--refine-inequalities",
        choices=("in", "out", "terminal_only"),
        default="terminal_only",
        help=(
            "experimental solver arm: keep the terminal-slack and direction-window SLSQP "
            "inequalities inside the refine solve, or take them out; the acceptance gates "
            "are unchanged either way and still decide every branch"
        ),
    )
    parser.add_argument(
        "--anchor-residuals",
        choices=("absent", "present"),
        default="present",
        help=(
            "experimental solver arm: add the supplied bounce ground-ray targets with their "
            "projection-derived sigma and a soft striker-position tie to the refine objective"
        ),
    )
    parser.add_argument(
        "--seed-restarts",
        choices=("off", "three"),
        default="three",
        help=(
            "experimental solver arm: solve every coarse depth branch from the pinhole seed, "
            "an anchored walk, and the anchored walk with flat spin, and keep the best"
        ),
    )
    parser.add_argument(
        "--reference-restart",
        choices=("off", "on"),
        default="off",
        help=(
            "experimental safety arm: retain the prior inequalities-in, anchor-free solve "
            "as one additional candidate in every depth branch"
        ),
    )
    parser.add_argument(
        "--shooting-parameterization",
        choices=("single", "shared_contacts"),
        default="single",
        help=(
            "experimental arm: replace the forward-linked solve with explicit exact shared "
            "contact states and block-sparse multiple shooting"
        ),
    )
    parser.add_argument(
        "--local-flight-refits",
        choices=("off", "on"),
        default="off",
        help=(
            "shared-contact arm only: refit one flight block without moving contacts or "
            "neighbouring flight parameters"
        ),
    )
    parser.add_argument(
        "--shared-state-second-stage-cache",
        type=Path,
        default=None,
        help=(
            "explicit promoted sweep root; refine its rank-one candidates with shared states "
            "and retain each unchanged candidate as a fallback"
        ),
    )
    parser.add_argument(
        "--bounce-bracket-frames",
        type=float,
        default=1.0,
        help="bounce timing dead zone inside the fit, native frames; 1.0 reproduces the reference",
    )
    parser.add_argument(
        "--coarse-iterations",
        type=int,
        default=150,
        help=(
            "coarse solve budget per depth branch; 150 is the promoted terminal-only arm, "
            "FIXED_CONFIGURATION['coarse_iterations'] (100) reproduces the previous reference"
        ),
    )
    args = parser.parse_args()
    if args.local_flight_refits == "on" and args.shooting_parameterization != "shared_contacts":
        raise ValueError("local flight refits require explicit shared contact states")
    if args.shared_state_second_stage_cache is not None and (
        args.shooting_parameterization != "shared_contacts" or args.event_recovery != "off"
    ):
        raise ValueError(
            "shared-state second-stage cache requires shared contacts and fixed topology"
        )
    if args.coarse_iterations < 1:
        raise ValueError("positive coarse solve budget required")
    labels_root_early = args.labels_root.resolve()
    # A machine knob, not part of the frozen scientific configuration: camera
    # transport is independent per attempt, so a large cohort may prepare more
    # attempts at once than the eight-worker fit budget.
    preparation_workers = args.preparation_workers or args.workers
    solver_arm_is_reference = (
        args.refine_inequalities == "in"
        and args.anchor_residuals == "absent"
        and args.seed_restarts == "off"
        and args.reference_restart == "off"
        and args.shooting_parameterization == "single"
        and args.local_flight_refits == "off"
        and args.serve_contact_epoch == "off"
        and args.serve_start_prior == "off"
        and args.serve_contact_hypotheses == "off"
        and args.coarse_iterations == FIXED_CONFIGURATION["coarse_iterations"]
        and args.bounce_bracket_frames == 1.0
    )
    if not 1 <= preparation_workers <= 32:
        raise ValueError("one to thirty-two preparation workers")
    cases = COHORTS[args.cohort] or discovered_cases(labels_root_early)
    if args.attempts:
        cases = tuple(case for case in cases if case.key in set(args.attempts))
        if len(cases) != len(set(args.attempts)):
            raise ValueError("every named attempt must exist in the discovered cohort")
    if args.workers != FIXED_CONFIGURATION["workers"]:
        raise ValueError("this frozen sweep requires exactly eight CPU workers")
    if any(
        case.dense_label_file is not None
        and not (labels_root_early / case.dense_label_file).is_file()
        for case in cases
    ):
        raise ValueError("a named dense revision must exist in the label root")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    labels_root = args.labels_root.resolve()
    if any(not (labels_root / case.label_file).is_file() for case in cases):
        raise ValueError("selected label revision set is incomplete")
    fitted_wear = (
        {}
        if args.surface_model_broadcasts is None
        else json.loads(args.surface_model_broadcasts.read_text())["broadcasts"]
    )
    surface_specs = {
        case.key: surface_model_spec(
            case,
            args.grass_bounce_profile,
            args.surface_model,
            fitted_wear,
            tuple(args.surface_model_surfaces),
        )
        for case in cases
    }
    configuration_path = args.output / "fixed_configuration.json"
    configuration = {
        **FIXED_CONFIGURATION,
        **({} if solver_arm_is_reference else {"coarse_iterations": args.coarse_iterations}),
        "cohort_name": args.cohort,
        "event_recovery": args.event_recovery,
        "observation_fallback": args.observation_fallback,
        "player_ledger": args.player_ledger,
        "dense_labels": args.dense_labels,
        "dense_label_retry": args.dense_label_retry,
        "grass_bounce_profile": args.grass_bounce_profile,
        "serve_reach_cut": args.serve_reach_cut,
        "serve_reach_margin_statures": args.serve_reach_margin_statures,
        **(
            {}
            if args.serve_contact_epoch == "off"
            else {"serve_contact_epoch": args.serve_contact_epoch}
        ),
        **(
            {}
            if args.serve_contact_hypotheses == "off"
            else {
                "serve_contact_hypotheses": args.serve_contact_hypotheses,
                "serve_contact_history": (
                    None
                    if args.serve_contact_history is None
                    else str(args.serve_contact_history.resolve())
                ),
                "contact_parameter_bounds_added": False,
                "contact_optimizer_inequalities_added": False,
            }
        ),
        **(
            {}
            if args.serve_start_prior == "off"
            else {
                "serve_start_prior": args.serve_start_prior,
                "depth_beam": "continuous first-contact XYZ inside the prior's three-sigma box",
            }
        ),
        **(
            {}
            if args.serve_ending_consistency == "off"
            else {"serve_ending_consistency": args.serve_ending_consistency}
        ),
        "surface_model": args.surface_model,
        "surface_model_surfaces": sorted(args.surface_model_surfaces),
        "surface_model_broadcasts": (
            None
            if args.surface_model_broadcasts is None
            else str(args.surface_model_broadcasts.resolve())
        ),
        "surface_model_specs": surface_specs,
        "max_topology_branches": args.max_topology_branches,
        "attempt_denominator": len(cases),
        "attempts": [asdict(case) for case in cases],
        # Recorded only when an explicit arm flag moved something, so a run with
        # both solver arms at their defaults writes the published reference
        # configuration unchanged.
        **(
            {}
            if solver_arm_is_reference
            else {
                "solver_experimental_arm": {
                    "refine_inequalities": args.refine_inequalities,
                    "anchor_residuals": args.anchor_residuals,
                    "seed_restarts": args.seed_restarts,
                    **(
                        {}
                        if args.reference_restart == "off"
                        else {"reference_restart": args.reference_restart}
                    ),
                    **(
                        {}
                        if args.local_flight_refits == "off"
                        else {"local_flight_refits": args.local_flight_refits}
                    ),
                    **(
                        {}
                        if args.shooting_parameterization == "single"
                        else {"shooting_parameterization": args.shooting_parameterization}
                    ),
                    **(
                        {}
                        if args.shared_state_second_stage_cache is None
                        else {
                            "shared_state_second_stage_cache": str(
                                args.shared_state_second_stage_cache
                            )
                        }
                    ),
                    "coarse_iterations": args.coarse_iterations,
                    "bounce_bracket_frames": args.bounce_bracket_frames,
                    "seeds_unchanged": args.seed_restarts == "off",
                    "depth_branches_unchanged": True,
                    "dynamics_unchanged": True,
                    "acceptance_gates_unchanged": True,
                }
            }
        ),
        "schema": (
            "connected_labeled_attempt_s6_reference_configuration_v3"
            if args.bounce_witness_mode == "subframe_graded_circle"
            and args.athlete_prior_mode == "stature_pose_soft"
            and args.event_recovery == "off"
            and args.serve_reach_cut == "off"
            and args.serve_contact_epoch == "off"
            and args.serve_start_prior == "off"
            and args.serve_contact_hypotheses == "off"
            and solver_arm_is_reference
            else "connected_labeled_attempt_explicit_arm_v3"
        ),
        "bounce_witness_mode": args.bounce_witness_mode,
        "athlete_prior_mode": args.athlete_prior_mode,
        "labels_root": str(labels_root),
        "bounce_gate_distance": (
            "raw centre-height ray distance"
            if args.bounce_witness_mode == "native_ray_average"
            else (
                "max(0, centre-height ray distance - max(0.20 m, 2 sigma)); true subframe "
                "ray where both endpoint fronts exist, documented native-ray average fallback "
                "otherwise"
            )
        ),
    }
    fixed_text = json.dumps(configuration, indent=2, allow_nan=False) + "\n"
    if configuration_path.exists() and configuration_path.read_text() != fixed_text:
        raise ValueError("existing fixed configuration differs")
    configuration_path.write_text(fixed_text)
    began = time.monotonic()
    preparation = {}
    # One attempt's camera transport is embarrassingly parallel across frames and
    # across attempts.  A cohort of dozens is far better served by one attempt per
    # worker than by one attempt at a time, and the total worker budget is the same.
    with ProcessPoolExecutor(max_workers=preparation_workers) as pool:
        prepared = {
            pool.submit(
                prepare_case,
                case,
                args.output,
                labels_root,
                args.preparation_cache,
                1 if len(cases) > preparation_workers else 8,
            ): case
            for case in cases
        }
        for future in as_completed(prepared):
            case = prepared[future]
            preparation[case.key] = future.result()
            print(json.dumps({"prepared": case.key, **preparation[case.key]}), flush=True)
    searches = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                search_case,
                case,
                args.output,
                configuration_path,
                args.bounce_witness_mode,
                args.athlete_prior_mode,
                labels_root,
                args.event_recovery,
                args.max_topology_branches,
                args.observation_fallback,
                args.player_ledger == "on",
                args.dense_labels == "on",
                args.dense_label_retry == "on",
                args.grass_bounce_profile,
                args.serve_reach_cut,
                args.serve_reach_margin_statures,
                args.serve_contact_epoch,
                args.serve_start_prior,
                args.serve_contact_hypotheses,
                args.serve_contact_history,
                args.serve_ending_consistency,
                args.refine_inequalities,
                args.anchor_residuals,
                args.seed_restarts,
                args.reference_restart,
                args.shooting_parameterization,
                args.local_flight_refits,
                args.shared_state_second_stage_cache,
                args.coarse_iterations,
                surface_specs[case.key],
                args.bounce_bracket_frames,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            searches[case.key] = future.result()
            print(json.dumps({"searched": case.key, **searches[case.key]}), flush=True)
    attempts = [
        summarize_case(case, args.output, args.grass_bounce_profile, surface_specs[case.key])
        if (args.output / case.key / "search/report.json").is_file()
        else held_case(
            case,
            args.output,
            searches[case.key],
            args.grass_bounce_profile,
            surface_specs[case.key],
        )
        for case in cases
    ]
    diagnosis = camera_diagnosis(args.output) if args.cohort in {"nine", "labeled17"} else None
    report = {
        "schema": "connected_labeled_attempt_sweep_v2",
        "status": "complete",
        "question": __doc__,
        "fixed_configuration": configuration,
        "fixed_configuration_sha256": hashlib.sha256(fixed_text.encode()).hexdigest(),
        "inputs": [provenance.file_record(labels_root / case.label_file) for case in cases],
        "code": provenance.git_record(paths.REPO_ROOT),
        "human_derived": True,
        "automatic_inference_eligible": False,
        "independent_xyz_truth_available": False,
        "cohort": args.cohort,
        "attempt_denominator": len(cases),
        "reconstructed_count": sum(row["reconstructed"] for row in attempts),
        # The like-for-like count against an arm with event recovery off: a
        # shortened topology is a partial point, never a reconstruction of the
        # labeled one.
        "reconstructed_on_the_labeled_topology_count": sum(
            row.get("reconstructed_on_the_labeled_topology", row["reconstructed"])
            for row in attempts
        ),
        "shortened_topology_count": sum(
            bool(row.get("topology", {}).get("shortened")) for row in attempts
        ),
        "preparation_workers": preparation_workers,
        "attempts": attempts,
        "yield_by_slice": yield_by_slice(attempts, cases, args.grass_bounce_profile),
        "camera_regression_diagnosis": diagnosis,
        "preparation": preparation,
        "searches": searches,
        "wall_seconds": time.monotonic() - began,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_html(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "reconstructed": report["reconstructed_count"],
                "denominator": report["attempt_denominator"],
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
