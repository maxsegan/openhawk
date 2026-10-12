"""Discover every fit-ready labeled attempt and its explicit fitted inputs.

Evaluation only.  The frozen ``nine``/``labeled17`` cohorts stay hard-coded in
``labeled_attempt_sweep``; this module answers the different question of which
*complete* attempt documents exist in a label root and what explicit inputs each
one needs.  Nothing here reads a fit result, and every derived field records how
it was derived so a wrong roster entry is visible rather than silent.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from cv.pipeline import paths, resolution as res

ATTEMPT_SCHEMA = "s6_owner_input_labels_v1"
PLAYER_ROOTS = (
    "postseg_pipeline_benchmark_969657a",
    "postseg_pipeline_benchmark_766a967",
)
PLAYER_FILE_PATTERNS = (
    "player_pose_tracked_crop_native_v1.csv",
    "player_pose_tracked_crop_native_backfill_v1.csv",
    "player_boxes_*_native_sided_v1.csv",
)
PLAYER_COLUMNS = ("x1", "y1", "x2", "y2")

# One roster row per broadcast: the two singles players in match-identifier order
# and their tour-listed statures in metres.  The twenty statures already frozen
# in ``labeled_attempt_sweep.FIXED_CONFIGURATION`` carry their published profile
# there; the remaining entries are public tour listings that this lane did not
# re-verify against a profile page, and they are marked ``unverified`` in the
# emitted manifest.  Every athlete term is soft, so a listing error moves a
# penalty rather than opening or closing a gate.
VERIFIED_STATURES = {
    "Zverev",
    "Thiem",
    "Rybakina",
    "Sabalenka",
    "Swiatek",
    "Paolini",
    "Osaka",
    "Kvitova",
    "Muchova",
    "Djokovic",
    "Alcaraz",
    "Nadal",
    "Fritz",
    "Tiafoe",
    "Garcia",
    "Moutet",
    "Schoolkate",
    "Sinner",
    "Svitolina",
    "Gauff",
}
ROSTER: dict[str, tuple[tuple[str, float], ...]] = {
    "ao2019f_w_osaka_kvitova": (("Osaka", 1.80), ("Kvitova", 1.82)),
    "ao2022f_w_barty_collins": (("Barty", 1.66), ("Collins", 1.78)),
    "ao2023f_w_sabalenka_rybakina": (("Sabalenka", 1.82), ("Rybakina", 1.84)),
    "ao2023r16_m_rublev_rune": (("Rublev", 1.88), ("Rune", 1.88)),
    "ao2026r128_m_bublik_brooksby": (("Bublik", 1.96), ("Brooksby", 1.93)),
    "atp_2022_0807_sf_299_rafael_nadal_daniil_medvedev": (("Nadal", 1.85), ("Medvedev", 1.98)),
    "atp_2024_0403_qf_297_grigor_dimitrov_carlos_alcaraz": (
        ("Dimitrov", 1.91),
        ("Alcaraz", 1.83),
    ),
    "atp_2024_520_r128_335_alexander_zverev_rafael_nadal": (("Zverev", 1.98), ("Nadal", 1.85)),
    "atp_2024_540_r16_217_taylor_fritz_alexander_zverev": (("Fritz", 1.96), ("Zverev", 1.98)),
    "atp_2025_0403_f_369_jakub_mensik_novak_djokovic": (("Mensik", 1.96), ("Djokovic", 1.88)),
    "atp_2025_520_qf_395_lorenzo_musetti_frances_tiafoe": (("Musetti", 1.85), ("Tiafoe", 1.88)),
    "atp_2025_560_r128_120_adam_walton_ugo_humbert": (("Walton", 1.83), ("Humbert", 1.85)),
    "atp_2026_580_r128_103_corentin_moutet_tristan_schoolkate": (
        ("Moutet", 1.80),
        ("Schoolkate", 1.83),
    ),
    "atpf2023sf_m_djokovic_alcaraz": (("Djokovic", 1.88), ("Alcaraz", 1.83)),
    "atpf2024rr_m_alcaraz_zverev": (("Alcaraz", 1.83), ("Zverev", 1.98)),
    "barcelona2019sf_m_thiem_nadal": (("Thiem", 1.85), ("Nadal", 1.85)),
    "beijing2024f_w_gauff_muchova": (("Gauff", 1.75), ("Muchova", 1.80)),
    "cincy2024f_m_sinner_tiafoe": (("Sinner", 1.91), ("Tiafoe", 1.88)),
    "rg2017f_m_nadal_wawrinka": (("Nadal", 1.85), ("Wawrinka", 1.83)),
    "rg2023f_w_swiatek_muchova": (("Swiatek", 1.76), ("Muchova", 1.80)),
    "rg2024f_w_swiatek_paolini": (("Swiatek", 1.76), ("Paolini", 1.63)),
    "rg2025f_m_alcaraz_sinner": (("Alcaraz", 1.83), ("Sinner", 1.91)),
    "rg2025qf_w_boisson_andreeva": (("Boisson", 1.85), ("Andreeva", 1.77)),
    "rome2026f_w_svitolina_gauff": (("Svitolina", 1.74), ("Gauff", 1.75)),
    "rotterdam2025f_m_alcaraz_de_minaur": (("Alcaraz", 1.83), ("DeMinaur", 1.83)),
    "uso2020f_m_zverev_thiem": (("Zverev", 1.98), ("Thiem", 1.85)),
    "uso2024qf_w_pegula_swiatek": (("Pegula", 1.70), ("Swiatek", 1.76)),
    "uso2024sf_m_fritz_tiafoe": (("Fritz", 1.96), ("Tiafoe", 1.88)),
    "uso2025f_m_sinner_alcaraz": (("Sinner", 1.91), ("Alcaraz", 1.83)),
    "uso2025r128_m_khachanov_basavareddy": (("Khachanov", 1.98), ("Basavareddy", 1.80)),
    "wim2019f_w_halep_williams": (("Halep", 1.68), ("Williams", 1.75)),
    "wim2023f_m_alcaraz_djokovic": (("Alcaraz", 1.83), ("Djokovic", 1.88)),
    "wim2024f_w_krejcikova_paolini": (("Krejcikova", 1.78), ("Paolini", 1.63)),
    "wim2025f_m_sinner_alcaraz": (("Sinner", 1.91), ("Alcaraz", 1.83)),
    "wim2025r32_w_sabalenka_raducanu": (("Sabalenka", 1.82), ("Raducanu", 1.75)),
    "wtaf2022f_w_garcia_sabalenka": (("Garcia", 1.77), ("Sabalenka", 1.82)),
}
# The topology sentence spells a few surnames differently from the roster key.
TOPOLOGY_ALIASES = {"DeMinaur": ("de minaur", "deminaur", "de Minaur")}

# The seventeen published attempts keep the frozen sweep's explicit configuration
# exactly, so a discovered cohort reproduces the promoted reference on them
# instead of re-deriving a serve ordinal or hitter order the reference pinned.
FROZEN_CONFIGURATION_SOURCE = "cv.experiments.connected_shooting.labeled_attempt_sweep"

SURFACES = ("hard", "clay", "grass")
# Short broadcast tags keep the emitted keys readable; they are display names,
# never a join key back into the data.
BROADCAST_TAGS = {
    "atp_2022_0807_sf_299_rafael_nadal_daniil_medvedev": "acapulco2022sf",
    "atp_2024_0403_qf_297_grigor_dimitrov_carlos_alcaraz": "miami2024qf",
    "atp_2024_520_r128_335_alexander_zverev_rafael_nadal": "clayatp2024r128",
    "atp_2024_540_r16_217_taylor_fritz_alexander_zverev": "grassatp2024r16",
    "atp_2025_0403_f_369_jakub_mensik_novak_djokovic": "miami2025f",
    "atp_2025_520_qf_395_lorenzo_musetti_frances_tiafoe": "clayatp2025qf",
    "atp_2025_560_r128_120_adam_walton_ugo_humbert": "hardatp2025r128",
    "atp_2026_580_r128_103_corentin_moutet_tristan_schoolkate": "hardatp2026r128",
}
# Lane of origin, read from the file name suffix the labeling package used.
LANE_PATTERN = re.compile(r"_(astra[a-z0-9]*|clay3labels|fit_v2)(?:_v\d+)?(?:__[a-z_0-9]+)?\.json$")


@dataclass(frozen=True)
class DiscoveredAttempt:
    """One complete labeled attempt with every explicit input it is fitted with."""

    key: str
    label_file: str
    match_id: str
    clip: str
    surface: str
    gender: str
    serve_number: int
    serve_number_resolved: bool
    pose_image_scale: float
    player_statures_m: tuple[tuple[str, float], ...]
    player_localization: str | None
    dense_label_file: str | None
    first_contact_frame: float
    ending_frame: float | None
    ending_kind: str | None
    contact_count: int
    bounce_count: int
    net_hit_count: int
    labeled_native_frames: int
    visible_native_frames: int
    lane: str
    provenance: dict[str, Any]


def _lane(name: str) -> str:
    found = LANE_PATTERN.search(name)
    if found is None:
        return "owner_seed"
    lane = found.group(1)
    return "owner_seed" if lane == "fit_v2" else lane


def _surface(document: dict[str, Any], match_id: str) -> tuple[str, str]:
    surface = document["attempt"].get("surface")
    if surface in SURFACES:
        return surface, "label_document"
    if match_id.startswith("rg") or match_id.startswith("barcelona") or match_id.startswith("rome"):
        return "clay", "broadcast_identifier"
    if match_id.startswith("wim"):
        return "grass", "broadcast_identifier"
    return "hard", "broadcast_identifier"


def _gender(document: dict[str, Any], match_id: str) -> tuple[str, str]:
    gender = document["attempt"].get("gender")
    if gender in ("men", "women"):
        return gender, "label_document"
    if "_w_" in match_id:
        return "women", "broadcast_identifier"
    return "men", "broadcast_identifier"


def _serve_number(document: dict[str, Any]) -> tuple[int, bool, str]:
    text = " ".join(
        str(document["attempt"].get(field, "")) for field in ("topology", "notes")
    ).lower()
    if "second serve" in text or "second-serve" in text:
        return 2, True, "topology_or_notes"
    if "first serve" in text or "first-serve" in text:
        return 1, True, "topology_or_notes"
    return 2, False, "unresolved_wide_second_serve_prior"


def _player_order(document: dict[str, Any], match_id: str) -> tuple[list[str], str]:
    """Order the alternating hitter cycle so the server is first."""
    roster = [name for name, _ in ROSTER[match_id]]
    topology = str(document["attempt"].get("topology") or "")
    positions: list[tuple[int, str]] = []
    for name in roster:
        keys = [name, *TOPOLOGY_ALIASES.get(name, ())]
        found = [topology.lower().find(key.lower()) for key in keys]
        found = [value for value in found if value >= 0]
        if found:
            positions.append((min(found), name))
    if positions:
        positions.sort()
        server = positions[0][1]
        return [server, *[name for name in roster if name != server]], "topology_sentence"
    return roster, "match_identifier_order_assumed"


def _covered_contacts(path: Path, clip: str, contacts: set[str]) -> int:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"side", "court_x", "court_y"}.issubset(reader.fieldnames):
            return -1
        seen = {row["frame"] for row in reader if row.get("clip") == clip}
    return len(contacts & seen)


def _player_localization(
    match_id: str, clip: str, contacts: set[str]
) -> tuple[str | None, dict[str, Any]]:
    """Pick the sided player artifact covering the most labeled contacts."""
    best: tuple[int, str] | None = None
    root = paths.data_root() / "processed"
    for benchmark in PLAYER_ROOTS:
        directory = root / benchmark / match_id
        candidates: list[Path] = []
        for pattern in PLAYER_FILE_PATTERNS:
            candidates.extend(sorted(directory.glob(pattern)))
        for candidate in candidates:
            name = candidate.name
            if not candidate.is_file():
                continue
            covered = _covered_contacts(candidate, clip, contacts)
            if covered <= 0:
                continue
            relative = f"{benchmark}/{match_id}/{name}"
            if best is None or covered > best[0]:
                best = (covered, relative)
            if covered == len(contacts):
                return relative, {
                    "contacts_covered": covered,
                    "contacts_required": len(contacts),
                    "complete": True,
                }
    if best is None:
        return None, {"contacts_covered": 0, "contacts_required": len(contacts), "complete": False}
    return best[1], {
        "contacts_covered": best[0],
        "contacts_required": len(contacts),
        "complete": False,
    }


def pose_image_scale(relative: str | None) -> tuple[float, str]:
    """Resolve the player artifact's declared native multiplier from its sidecar."""
    if relative is None:
        return 1.0, "no_player_artifact"
    path = paths.data_root() / "processed" / relative
    try:
        scale_x, scale_y = res.coordinate_scale(path, columns=PLAYER_COLUMNS)
    except Exception:  # noqa: BLE001 - an undeclared space must not be guessed
        return 1.0, "sidecar_unavailable_assumed_native"
    if abs(scale_x - scale_y) > 1e-9:
        raise ValueError(f"{path} declares a non-uniform native scale")
    return float(scale_x), "coordinate_sidecar"


def _dense_label_file(root: Path, label_file: str) -> str | None:
    """Prefer the every-frame full-dense revision, then the dense addendum."""
    stem = label_file[: -len(".json")]
    for suffix in ("_fulldense_v1", "_dense_v1"):
        for base in (stem, re.sub(r"(_v\d+)$", "", stem)):
            candidate = root / f"{base}{suffix}.json"
            if candidate.is_file():
                return candidate.name
    # The dense packages name the base export, not a later revision suffix.
    prefix = re.sub(r"_(astra[a-z0-9]*|clay3labels|fit)(_v\d+)?$", "", stem)
    for suffix in ("_fulldense_v1", "_dense_v1"):
        candidate = root / f"{prefix}{suffix}.json"
        if candidate.is_file():
            return candidate.name
    return None


def _revision_rank(name: str) -> tuple[int, int, int]:
    """Rank two documents of the same attempt; the later revision wins."""
    version = re.search(r"_v(\d+)(?:__[a-z_0-9]+)?\.json$", name)
    return (
        100 if "_fit_v2" in name else 0,
        50 if "clay3labels" in name else 10 if "astrareview2" in name else 0,
        int(version.group(1)) if version else 0,
    )


def _attempt_group(match_id: str, clip: str, first_contact_frame: float) -> tuple[str, str, int]:
    """Group revisions of one attempt; second annotations move an epoch slightly."""
    return (match_id, clip, int(round(float(first_contact_frame) / 5.0)))


def _frozen_overrides(labels_root: Path) -> dict[tuple[str, str, int], Any]:
    """Read the frozen seventeen's explicit fields, keyed by attempt rather than file.

    A later reviewed revision of one of those attempts is still the same
    attempt, so it inherits the published serve ordinal, hitter order, surface
    and player artifact instead of re-deriving them.
    """
    from cv.experiments.connected_shooting import labeled_attempt_sweep as frozen

    overrides: dict[tuple[str, str, int], Any] = {}
    for case in (*frozen.CASES, *frozen.NEW_CASES):
        path = labels_root / case.label_file
        if not path.is_file():
            continue
        document = json.loads(path.read_text())
        key = _attempt_group(
            document["match_id"],
            document["attempt"]["clip"],
            document["attempt"]["first_contact_frame"],
        )
        overrides[key] = (case, document)
    return overrides


def _frozen_pose_path(document: dict[str, Any], case: Any) -> str | None:
    """Reproduce the frozen sweep's player-artifact discovery order exactly."""
    if case.player_localization is not None:
        return case.player_localization
    root = paths.data_root() / "processed/postseg_pipeline_benchmark_969657a" / document["match_id"]
    clip = document["attempt"]["clip"]
    for name in ("player_pose_tracked_crop_native_v1.csv", "player_boxes_30_native_sided_v1.csv"):
        candidate = root / name
        if not candidate.is_file():
            continue
        with candidate.open(newline="") as handle:
            if any(row.get("clip") == clip for row in csv.DictReader(handle)):
                return f"postseg_pipeline_benchmark_969657a/{document['match_id']}/{name}"
    return None


def discover(labels_root: Path) -> list[DiscoveredAttempt]:
    """Return one entry per distinct complete labeled attempt, in a stable order."""
    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(labels_root.glob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(document, dict) or document.get("schema") != ATTEMPT_SCHEMA:
            continue
        attempt = document.get("attempt") or {}
        events = document.get("events") or {}
        if attempt.get("coverage") != "complete_attempt" or not events.get("complete"):
            continue
        if not all(
            record.get("complete") for record in document.get("ball", {}).get("records", [])
        ):
            continue
        if not all(record.get("complete") for record in document.get("court", [])):
            continue
        if document["match_id"] not in ROSTER:
            continue
        documents.append((path, document))
    groups: dict[tuple[str, str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for path, document in documents:
        attempt = document["attempt"]
        key = _attempt_group(document["match_id"], attempt["clip"], attempt["first_contact_frame"])
        groups.setdefault(key, []).append((path, document))
    frozen = _frozen_overrides(labels_root)
    discovered: list[DiscoveredAttempt] = []
    ordinals: dict[tuple[str, str], int] = {}
    for key in sorted(groups):
        path, document = max(groups[key], key=lambda item: _revision_rank(item[0].name))
        attempt = document["attempt"]
        match_id = document["match_id"]
        clip = attempt["clip"]
        records = [row for row in document["events"]["records"] if row.get("status") == "labeled"]
        contacts = [row for row in records if row["event_type"] == "contact"]
        ordinal = ordinals.get((match_id, clip), 0) + 1
        ordinals[(match_id, clip)] = ordinal
        tag = BROADCAST_TAGS.get(match_id, match_id.split("_")[0])
        surface, surface_source = _surface(document, match_id)
        gender, gender_source = _gender(document, match_id)
        serve_number, serve_resolved, serve_source = _serve_number(document)
        order, order_source = _player_order(document, match_id)
        statures = dict(ROSTER[match_id])
        contact_frames = {f"f_{int(np.ceil(float(row['frame']))):04d}.jpg" for row in contacts}
        localization, localization_receipt = _player_localization(match_id, clip, contact_frames)
        scale, scale_source = pose_image_scale(localization)
        published_pair = frozen.get(key)
        published = None if published_pair is None else published_pair[0]
        if published is not None:
            serve_number = published.serve_number
            serve_resolved = published.serve_number_resolved
            serve_source = "frozen_reference_configuration"
            order = [name for name, _ in published.player_statures_m]
            order_source = "frozen_reference_configuration"
            statures = {**statures, **dict(published.player_statures_m)}
            surface, surface_source = published.surface, "frozen_reference_configuration"
            frozen_localization = _frozen_pose_path(published_pair[1], published)
            if frozen_localization is not None:
                localization = frozen_localization
                localization_receipt = {
                    **localization_receipt,
                    "frozen_reference_override": True,
                    "frozen_reference_file": published_pair[1]["benchmark_id"],
                }
                scale, scale_source = pose_image_scale(localization)
        ball_frames = [
            row for record in document["ball"]["records"] for row in record.get("frames", [])
        ]
        ending = [row for row in records if row["event_type"] == "ending"]
        discovered.append(
            DiscoveredAttempt(
                key=f"{tag}_{clip}_a{ordinal:02d}",
                label_file=path.name,
                match_id=match_id,
                clip=clip,
                surface=surface,
                gender=gender,
                serve_number=serve_number,
                serve_number_resolved=serve_resolved,
                pose_image_scale=scale,
                # The athlete arm wants a stature for exactly the players the
                # alternating cycle names.  A one-contact attempt has only a
                # server, so naming the opponent would be an unused stature the
                # arm rejects.
                player_statures_m=tuple(
                    (name, statures[name])
                    for name in dict.fromkeys(
                        order[index % len(order)] for index in range(max(len(contacts), 1))
                    )
                ),
                player_localization=localization,
                dense_label_file=_dense_label_file(labels_root, path.name),
                first_contact_frame=float(attempt["first_contact_frame"]),
                ending_frame=None
                if attempt.get("ending_frame") is None
                else float(attempt["ending_frame"]),
                ending_kind=attempt.get("ending_kind"),
                contact_count=len(contacts),
                bounce_count=sum(row["event_type"] == "bounce" for row in records),
                net_hit_count=sum(row["event_type"] == "net_hit" for row in records),
                labeled_native_frames=len(ball_frames),
                visible_native_frames=sum(row.get("status") == "visible" for row in ball_frames),
                lane=_lane(path.name),
                provenance={
                    "surface_source": surface_source,
                    "gender_source": gender_source,
                    "serve_number_source": serve_source,
                    "player_order_source": order_source,
                    "pose_image_scale_source": scale_source,
                    "player_localization": localization_receipt,
                    "statures_unverified": sorted(
                        name for name in order if name not in VERIFIED_STATURES
                    ),
                    "superseded_revisions": sorted(
                        item[0].name for item in groups[key] if item[0] != path
                    ),
                    "ending_record_count": len(ending),
                },
            )
        )
    return discovered


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    from dataclasses import asdict

    rows = [asdict(item) for item in discover(arguments.labels_root.resolve())]
    manifest = {
        "schema": "connected_labeled_attempt_cohort_v1",
        "labels_root": str(arguments.labels_root.resolve()),
        "attempt_count": len(rows),
        "broadcast_count": len({row["match_id"] for row in rows}),
        "attempts": rows,
    }
    text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text)


if __name__ == "__main__":
    main()
