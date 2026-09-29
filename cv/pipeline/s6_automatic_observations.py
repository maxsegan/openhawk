"""Prepare shared-S6 observations from automatic artifacts, without opening labels.

``build_observations(inputs, output)`` is a preparation API, not a fitting runner.
``source_pts`` is a producer-bound FFprobe compact ``frame=pts_time`` inventory
for the processing video, not a nominal frame/fps reconstruction. The existing
player-frame extraction receipt binds native pictures and their source offset.
Unknown physical endings can produce an explicitly automatic observation scope;
they never become complete physical points. Unqualified scopes remain holds.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from functools import lru_cache
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from cv.pipeline import artifact_cache, paths, provenance, resolution
from cv.pipeline.players import source_clock_ordinals
from cv.pipeline.source_timebase import audit_timestamps, frame_pts


@dataclass(frozen=True)
class AutomaticInputs:
    """Explicit artifact identities; no ambient labels, roster or per-point policy."""

    match_id: str
    clip: str
    manifest: Path
    point_ledger: Path
    extraction_receipt: Path
    frames_directory: Path
    source_video: Path
    source_pts: Path
    ball: Path
    events: Path
    cameras: Path
    players: Path
    ancestry: tuple[Path, ...]
    optional_contact_pose: Path | None = None
    optional_contact_views: Path | None = None
    independent_event_candidates: Path | None = None


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve(record: dict, source_video: Path | None = None) -> Path:
    if record.get("path_base") == "unconfigured_external" and source_video is not None:
        declared = provenance.file_record(source_video)
        if all(record.get(k) == declared.get(k) for k in ("path_base", "path", "sha256")):
            return source_video.resolve()
        raise ValueError("external ancestry does not match the explicitly declared source video")
    roots = {
        "repository": paths.REPO_ROOT,
        "TENNIS_DATA_ROOT": paths.data_root(),
        "TENNIS_TRACKER_ROOT": paths.tracker_root(),
    }
    if record.get("path_base") not in roots:
        raise ValueError("automatic ancestry requires a configured portable path base")
    root = roots[record["path_base"]].resolve()
    result = (root / record["path"]).resolve()
    if not result.is_relative_to(root):
        raise ValueError("automatic ancestry path escapes its configured root")
    return result


def _checked_record(record: dict, source_video: Path | None = None) -> Path:
    path = _resolve(record, source_video)
    if provenance.file_sha256(path) != record.get("sha256"):
        raise ValueError(f"automatic ancestry digest changed: {path.name}")
    return path


def _automatic(document: dict, context: str) -> None:
    provenance.validate_provenance(document)
    if document["mode"] != provenance.AUTOMATIC_MODE:
        raise ValueError(f"{context} has nonautomatic ancestry")
    provenance.assert_automatic_document(document, context=context)


def _reject_assistance(document: Any, context: str) -> None:
    """Check the full tree once, then inspect additional source-origin markers."""
    provenance.assert_automatic_document(document, context=context)
    _reject_source_origin(document, context)


def _reject_source_origin(document: Any, context: str) -> None:
    if isinstance(document, dict):
        if document.get("human_derived") or document.get("human_derived_inputs"):
            raise ValueError(f"{context} contains human-derived observations")
        if document.get("annotation_origin") in {"agent", "human", "reviewed", "manual"}:
            raise ValueError(f"{context} contains nonautomatic observations")
        for value in document.values():
            if isinstance(value, (dict, list)):
                _reject_source_origin(value, context)
    elif isinstance(document, list):
        for value in document:
            if isinstance(value, (dict, list)):
                _reject_source_origin(value, context)


@lru_cache(maxsize=32)
def historical_command_closure(repository: str, commit: str, modules: tuple[str, ...]) -> dict:
    """Read immutable Git blobs using the normal receipt dependency parser."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("historical commands require an immutable source revision")

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repository, check=True, capture_output=True
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ValueError("historical command source revision or file unavailable") from error

    files = set(git("ls-tree", "-r", "--name-only", commit).decode().splitlines())

    def resolve(module):
        base = module.replace(".", "/")
        return next((name for name in (base + ".py", base + "/__init__.py") if name in files), None)

    pending = []
    for module in modules:
        path = resolve(module)
        if path is None:
            raise ValueError("historical command entrypoint is absent from source revision")
        pending.append(path)
    closure = {}
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        blob = git("show", f"{commit}:{path}")
        closure[path] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        package = ".".join(Path(path).parent.parts)
        names, required = artifact_cache.repository_import_names(blob.decode(), package)
        for name in names:
            dependency = resolve(name)
            if dependency is None and "." not in name:
                sibling = str(Path(path).parent / (name + ".py"))
                dependency = sibling if sibling in files else None
            if dependency is None and name in required:
                raise ValueError("historical source lacks declared runtime dependency")
            if dependency is not None and dependency not in closure:
                pending.append(dependency)
    return closure


def historical_multi_command_input(binding: dict, identity: dict) -> dict | None:
    config = identity.get("configuration", {})
    if config.get("receipt_timing") != "retrospective_completed_direct_commands":
        return None
    commands = config.get("executed_commands", [])
    if not commands or any(
        not isinstance(c, list) or len(c) < 3 or c[1] != "-m" or not isinstance(c[2], str)
        for c in commands
    ):
        raise ValueError("historical multi-command receipt requires explicit module commands")
    commit = config.get("source_commit", "")
    closure = historical_command_closure(
        str(paths.REPO_ROOT), commit, tuple(c[2] for c in commands)
    )
    declared = {
        r.get("path"): r for r in identity.get("inputs", []) if r.get("path_base") == "repository"
    }
    for path, expected in closure.items():
        record = declared.get(path, {})
        if record.get("kind") != "file" or any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "historical multi-command receipt has incomplete or changed code closure"
            )
    for record in identity.get("code", []):
        if (
            record.get("path_base") != "repository"
            or record.get("path") not in closure
            or any(record.get(k) != v for k, v in closure[record["path"]].items())
        ):
            raise ValueError("historical declared code is outside verified command closure")
    relative = binding.get("path")
    if relative not in closure or binding != declared.get(relative):
        return None
    _resolve(binding)
    return dict(
        path=relative,
        commit=commit,
        sha256=binding["sha256"],
        verification="git_command_closure",
        closure_files=len(closure),
    )


@lru_cache(maxsize=4096)
def _historical_blob(root: str, commit: str, relative: str) -> tuple[str, int]:
    try:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=root, check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError("historical implementation revision or path unavailable") from exc
    return hashlib.sha256(blob).hexdigest(), len(blob)


def historical_implementation_input(
    binding: dict, identity: dict, producer_source: dict | None = None
) -> dict | None:
    """Verify declared implementation inputs in their recorded clean Git revision.

    Some inference receipts additionally included code in ordinary inputs. Only
    exact members of their declared Python implementation closure can use this
    historical route. Observations and model files retain live byte validation.
    """
    if binding.get("path_base") != "repository" or not str(binding.get("path", "")).endswith(".py"):
        return None
    historical = historical_multi_command_input(binding, identity)
    if historical is not None:
        return historical
    closure = identity.get("code", [])
    if binding not in closure:
        return None
    source = identity.get("configuration", {}).get("inference_source")
    if source is not None and producer_source is not None and source != producer_source["source"]:
        raise ValueError("conflicting historical implementation source declarations")
    if source is None and producer_source is not None:
        source = producer_source["source"]
    if source is None:
        return None
    if not isinstance(source, dict):
        raise ValueError("historical implementation source declaration is invalid")
    commit = source.get("commit", "")
    if (
        source.get("dirty") is not False
        or not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", commit)
    ):
        raise ValueError("historical implementation requires a recorded clean source revision")
    relative = str(binding["path"])
    _resolve(binding)  # Retain configured-root traversal protection.
    digest, size = _historical_blob(str(paths.REPO_ROOT), commit, relative)
    if digest != binding.get("sha256") or size != binding.get("bytes"):
        raise ValueError("historical implementation content differs from receipt")
    return dict(
        path=relative,
        commit=commit,
        sha256=binding["sha256"],
        verification="git_blob",
        **(
            {"producer_source_verification": producer_source} if producer_source is not None else {}
        ),
    )


def validate_ancestry(inputs: AutomaticInputs, required: list[Path]) -> list[dict]:
    """Verify producer bindings and recursively inspect declared parent receipts.

    Provenance is an attestation by an automatic producer, never inferred from a
    filename or a caller boolean. Supplied inputs must occur in that provenance
    or in the outputs of its bound stage receipts. Historical code hashes remain
    recorded producer identity; current source edits do not rewrite old inputs.
    """
    if not inputs.ancestry:
        raise ValueError("automatic producer provenance is required")
    required_map = {p.resolve(): provenance.file_sha256(p) for p in required}
    video = provenance.file_record(inputs.source_video)
    covered: dict[Path, str] = {}
    seen: set[Path] = set()
    records = []
    automatic_roots = 0
    recovered_sources: dict[tuple, dict] = {}

    def inspect(path: Path, *, require_processing_source: bool = False) -> None:
        nonlocal automatic_roots
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        record = provenance.file_record(path, role="automatic_producer_ancestry")
        document = _read(path)
        if not isinstance(document, dict):
            raise ValueError("automatic ancestry must be a provenance document or stage receipt")
        prov = document.get("provenance", document)
        if prov.get("schema") == provenance.SCHEMA:
            _automatic(prov, path.name)
            if require_processing_source and not any(
                r.get("sha256") == video["sha256"]
                and _resolve(r, inputs.source_video) == inputs.source_video.resolve()
                for r in prov["source_videos"]
            ):
                raise ValueError("automatic producer provenance does not bind the processing video")
            for binding in prov["source_videos"]:
                _checked_record(binding, inputs.source_video)
            automatic_roots += 1
            bindings = prov["reused_artifacts"]
            for binding in bindings:
                if not binding.get("path"):
                    continue
                artifact = _checked_record(binding, inputs.source_video)
                covered[artifact] = binding["sha256"]
                if artifact.suffix == ".json":
                    nested = _read(artifact)
                    _reject_assistance(nested, artifact.name)
                    if isinstance(nested, dict) and (
                        nested.get("schema") in {provenance.SCHEMA, artifact_cache.SCHEMA}
                        or "provenance" in nested
                    ):
                        inspect(artifact)
        elif document.get("schema") == artifact_cache.SCHEMA:
            identity = document["identity"]
            provenance.assert_automatic_document(identity, context=path.name)
            unsigned = {k: v for k, v in identity.items() if k != "fingerprint"}
            if artifact_cache._digest_json(unsigned) != identity.get("fingerprint"):
                raise ValueError("automatic stage receipt fingerprint changed")
            for binding in identity.get("upstream_receipts", []):
                inspect(_checked_record(binding, inputs.source_video))
            for binding in document["outputs"]:
                if binding.get("kind") == "file":
                    artifact = _checked_record(binding, inputs.source_video)
                    covered[artifact] = binding["sha256"]
                elif binding.get("kind") == "directory":
                    directory = _resolve(binding, inputs.source_video)
                    children = [p for p in required_map if p.is_relative_to(directory)]
                    if children:
                        if artifact_cache.path_record(directory) != binding:
                            raise ValueError("automatic producer output directory changed")
                        covered.update({p: required_map[p] for p in children})
            historical_code = []
            producer_source = recovered_sources.get((record["sha256"], record["bytes"]))
            direct_source = document.get("implementation_source")
            if (
                producer_source is not None
                and direct_source is not None
                and direct_source != producer_source["source"]
            ):
                raise ValueError("conflicting recorded producer source revisions")
            if (
                producer_source is None
                and isinstance(direct_source, dict)
                and direct_source.get("dirty") is False
            ):
                producer_source = {"source": direct_source, "verification": "recorded_stage_source"}
                # New ordinary receipts bind their own source directly. Dirty
                # runs retain live byte validation, never a guessed Git state.
                for code in identity.get("code", []):
                    if historical_implementation_input(code, identity, producer_source) is None:
                        raise ValueError("recorded producer source has unsupported code closure")
            # Bind consumed files too, and inspect any explicit provenance inputs.
            for binding in identity.get("inputs", []):
                if binding.get("kind") != "file":
                    continue
                historical = historical_implementation_input(binding, identity, producer_source)
                if historical is not None:
                    historical_code.append(historical)
                    continue
                artifact = _checked_record(binding, inputs.source_video)
                if artifact.suffix == ".json":
                    nested = _read(artifact)
                    _reject_assistance(nested, artifact.name)
                    if isinstance(nested, dict) and (
                        nested.get("schema") in {provenance.SCHEMA, artifact_cache.SCHEMA}
                        or "provenance" in nested
                    ):
                        inspect(artifact)
            if historical_code:
                record["historical_implementation_verification"] = historical_code
        else:
            raise ValueError(f"unsupported automatic ancestry schema: {path.name}")
        records.append(record)

    # A caller cannot supply an unrelated automatic root beside arbitrary receipts.
    # Roots must bind every supplied stage receipt by digest in reused_artifacts.
    root_paths = []
    for path in inputs.ancestry:
        doc = _read(path)
        if isinstance(doc, dict) and doc.get("provenance", doc).get("schema") == provenance.SCHEMA:
            root_paths.append(path)
    # Collect bound source proofs throughout the graph before validating stage
    # inputs. Sibling/root ordering must not decide historical-code verification.
    scanned = set()

    def collect(path: Path) -> None:
        from cv.pipeline import s6_producer_source_recovery

        path = path.resolve()
        if path in scanned:
            return
        scanned.add(path)
        document = _read(path)
        _reject_assistance(document, path.name)
        if not isinstance(document, dict):
            return
        prov = document.get("provenance", document)
        bindings = []
        if prov.get("schema") == provenance.SCHEMA:
            _automatic(prov, path.name)
            for binding in prov["source_videos"]:
                _checked_record(binding, inputs.source_video)
            record = provenance.file_record(path, role="automatic_producer_ancestry")
            for key, source in s6_producer_source_recovery.verified_sources(
                prov, record, inputs.source_video
            ).items():
                existing = recovered_sources.get(key)
                if existing is not None and existing["source"] != source["source"]:
                    raise ValueError("conflicting recovered producer source revisions")
                recovered_sources[key] = source
            bindings = prov["reused_artifacts"]
        elif document.get("schema") == artifact_cache.SCHEMA:
            identity = document["identity"]
            bindings = [*identity.get("upstream_receipts", []), *identity.get("inputs", [])]
        for binding in bindings:
            if str(binding.get("path", "")).endswith(".json"):
                collect(_checked_record(binding, inputs.source_video))

    for path in root_paths:
        collect(path)
    for path in root_paths:
        inspect(path, require_processing_source=True)
    if automatic_roots == 0 or any(path.resolve() not in seen for path in inputs.ancestry):
        raise ValueError("unbound automatic ancestry; every receipt needs an automatic parent")
    missing = [p.name for p, digest in required_map.items() if covered.get(p) != digest]
    if missing:
        raise ValueError(f"automatic producer ancestry does not bind inputs: {missing}")
    return records


def ancestry_unchanged(records: list[dict]) -> bool:
    """Recheck file identities without confusing verification annotations with bytes."""
    for record in records:
        current = provenance.file_record(_resolve(record), role=record["role"])
        if any(record.get(key) != value for key, value in current.items()):
            return False
    return True


def _native_inventory(inputs: AutomaticInputs, fps: float, ledger: dict) -> list[dict]:
    receipt = _read(inputs.extraction_receipt)
    if receipt.get("schema") != "player_frame_extraction_v1":
        raise ValueError("original native player-frame extraction receipt required")
    source = receipt["source_identity"]
    provenance.assert_automatic_document(source, context="frame extraction")
    config = source["configuration"]
    if config.get("preserve_source_fps") is not True or config.get("native") is not True:
        raise ValueError("native source-cadence extraction required")
    if not math.isclose(float(config["fps"]), fps, rel_tol=0, abs_tol=1e-8):
        raise ValueError("native extraction cadence differs from manifest")
    video_record = provenance.file_record(inputs.source_video)
    if not any(
        binding.get("sha256") == video_record["sha256"]
        and _resolve(binding, inputs.source_video) == inputs.source_video.resolve()
        for binding in source["inputs"]
    ):
        raise ValueError("native extraction does not bind the processing video")
    unsigned = {k: v for k, v in source.items() if k != "fingerprint"}
    if artifact_cache._digest_json(unsigned) != source.get("fingerprint"):
        raise ValueError("native extraction source identity changed")
    identity = receipt["identity"]
    point = {
        "pt": int(ledger["pt"]),
        "t0": float(ledger["rally_t_start"]),
        "t1": float(ledger["rally_t_end"]),
    }
    if (
        identity.get("source_fingerprint") != source["fingerprint"]
        or identity.get("point") != point
    ):
        raise ValueError("native extraction window differs from automatic ledger")
    with inputs.source_pts.open() as handle:
        timestamps = list(frame_pts(handle))
    timebase = audit_timestamps(timestamps, fps)
    if not timebase["nominal_timeline_valid"]:
        raise ValueError("source PTS inventory has missing/irregular native timestamps")
    indices = source_clock_ordinals(receipt, timestamps, fps)
    inventory = receipt["frames"]
    expected = len(indices)
    if not indices or indices[0] < 0 or len(inventory) != expected or expected < 2:
        raise ValueError("native image count does not match the source PTS window")
    actual = sorted(inputs.frames_directory.glob("f_*.jpg"))
    if [p.name for p in actual] != [f"f_{i:04d}.jpg" for i in range(1, expected + 1)]:
        raise ValueError("native picture inventory is not the complete extracted window")
    images = []
    for index, (entry, path) in enumerate(zip(inventory, actual, strict=True)):
        record = provenance.file_record(path)
        if entry != {"name": path.name, "sha256": record["sha256"]}:
            raise ValueError("native picture bytes differ from their extraction receipt")
        if resolution.frame_size_for_path(path) != resolution.NATIVE_SIZE:
            raise ValueError("shared S6 requires declared native 1920x1080 pictures")
        images.append(
            {
                "clip": inputs.clip,
                "frame": index + 1,
                "native_pts_seconds": timestamps[indices[index]],
                "source_frame_index": indices[index] + 1,
                "source": record,
                "image_url": paths.data_relative(path),
            }
        )
    return images


def native_ball_support(
    rows: list[dict],
    image_size: resolution.FrameSize,
    *,
    jump_rejection: str = "off",
    fps: float | None = None,
    streak_centre: str = "off",
) -> tuple[list[dict], dict]:
    """Retain raw centres while refusing padding/outside-image point evidence.

    ``jump_rejection`` is default off; when on, short wrong-object islands are
    refused first (`s6_ball_jump_rejection`), label-free, and the receipt gains one
    key. Off leaves rows and receipt byte-identical to the earlier function.
    ``streak_centre`` (default off) then moves each remaining centre back along
    travel to the blur-streak middle (`s6_ball_streak_centre`), keeping the raw centre.
    """
    from cv.pipeline import s6_ball_jump_rejection, s6_ball_streak_centre

    rows, jump_receipt = s6_ball_jump_rejection.apply(rows, fps, jump_rejection)
    rows, streak_receipt = s6_ball_streak_centre.apply(rows, streak_centre)
    admitted = []
    refusals = []
    for row in rows:
        point = [row.get("x1080"), row.get("y1080")]
        if row.get("status") in {
            "visible",
            "derived_estimate",
        } and not resolution.points_inside_image(point, image_size):
            reason = "tracker_centre_outside_native_image"
            refusals.append(
                {
                    "frame": row["frame"],
                    "reason": reason,
                    "original_status": row["status"],
                    "original_xy_native": point,
                }
            )
            row = {
                **row,
                "status": "unsupported",
                "native_image_support": reason,
                "original_tracker_status": row["status"],
            }
        admitted.append(row)
    return admitted, {
        "policy": "native_image_point_support_v1",
        "image_size": {"width": image_size.width, "height": image_size.height},
        "raw_rows_retained": True,
        "clamped_or_fitted_centres": False,
        "refused_rows": len(refusals),
        "refusals": refusals,
        **({"jump_rejection": jump_receipt} if jump_receipt is not None else {}),
        **({"streak_centre": streak_receipt} if streak_receipt is not None else {}),
    }


def _active_ball_rows(observations: list[dict], events: list[dict], end: float) -> list[dict]:
    """Keep the original contact-to-end observation domain when fitting a prefix."""
    start = min(e["frame"] for e in events if e["event_type"] == "contact")
    return [r for r in observations if start <= r["frame"] <= end]


def build_observations(
    inputs: AutomaticInputs,
    output: Path,
    *,
    service_attempt_split: str = "off",
    observed_first_flight: str = "off",
    child_index: int | None = None,
    optional_contacts: str = "off",
    optional_final_contacts: str = "off",
    optional_contact_timing: str = "off",
    optional_contact_composition: str = "off",
    optional_interior_contacts: str = "off",
    independent_event_proposals: str = "off",
    optional_bounces: str = "off",
    observed_horizon_tail: str = "off",
    terminal_net_tail: str = "off",
    contact_prefix_scope: str = "off",
    contact_components: str = "off",
    leading_event_components: str = "off",
    contact_component_routing: str = "off",
    uncertain_original_occurrence: str = "off",
    automatic_abstained_occurrence: str = "off",
    automatic_event_operating_point: str | float = "off",
    automatic_ball_jump_rejection: str = "off",
    automatic_ball_streak_centre: str = "off",
    observation_partition: str = "fifth_frame_withheld",
    intrinsic_fallback_registration: str = "admit",
    near_baseline_refinement: str = "off",
    shot_homography_propagation: str = "off",
    player_body_scale: str = "off",
) -> dict:
    """Prepare one automatic attempt, retaining failures and absent optional evidence."""
    from cv.experiments.connected_shooting import auto_packet

    from cv.pipeline import s6_contact_prefix_scope as contact_prefix
    from cv.pipeline import s6_contact_components as components
    from cv.pipeline import s6_event_operating_point as operating

    contact_prefix.validate_mode(contact_prefix_scope)
    if contact_components not in components.MODES:
        raise ValueError("explicit supported contact component policy required")
    # Default off. The declared leading prefix is refused unless the unresolved
    # prefix contract can hold it and components can model the supported spans.
    leading_prefix_admitted = contact_prefix.leading_admission(
        leading_event_components, contact_prefix_scope, contact_components
    )
    # Default off. A refused prefix cut holds the attempt unless the explicit
    # routing policy hands the retained original inventory to the components.
    component_routing_admitted = contact_prefix.component_source_routing(
        contact_component_routing, contact_prefix_scope, contact_components
    )
    # Default off. The producer's own refused occurrences are retained as uncertain
    # ones only under the declared policy, which is a scope on the uncertain-occurrence
    # admission and carries that admission's two contract requirements.
    abstained_admitted = contact_prefix.abstained_occurrence_admission(
        automatic_abstained_occurrence,
        uncertain_original_occurrence,
        contact_prefix_scope,
        contact_components,
    )
    # Default off.  The consumer-side acceptance floor for the automatic event stream.
    # It is NOT an admission and carries no contract requirement: it re-reads the
    # producer's own recorded acceptance decision at a declared floor before any
    # inventory is built, so this runtime and the supplied-input preparation reach the
    # SAME document through the same helper.  Wiring it only into the preparation would
    # have repeated the `uncertain_original_occurrence` mistake -- a default that is
    # inert on the route production actually takes.
    operating_floor = operating.event_operating_point(automatic_event_operating_point)
    operating_restatement: dict = {"declared_floor": operating_floor}
    if leading_prefix_admitted and optional_bounces != "off":
        raise ValueError(
            "declared leading evidence and the optional bounce container are not combined; "
            "the bounce witness still requires an originating contact"
        )
    requested_optional = dict(
        optional_contacts=optional_contacts,
        optional_final_contacts=optional_final_contacts,
        optional_interior_contacts=optional_interior_contacts,
        independent_event_proposals=independent_event_proposals,
        optional_contact_composition=optional_contact_composition,
        optional_contact_timing=optional_contact_timing,
        optional_bounces=optional_bounces,
    )
    contact_plan, contact_bound, component_routing = None, None, None
    if optional_bounces not in {"off", "classifier", "classifier_v2", "classifier_interior"}:
        raise ValueError("explicit optional bounce policy required")
    if terminal_net_tail not in {"off", "on"}:
        raise ValueError("explicit supported terminal net tail policy required")
    if observed_horizon_tail not in {"off", "on"}:
        raise ValueError("explicit observed-horizon tail policy required")
    # Contacts and bounces may be prepared together: the bounce scope container is
    # built first, so the contact witness binds the resulting accepted events.
    if optional_bounces != "off" and observed_first_flight != "off":
        raise ValueError("optional bounces cannot combine with first-flight scope")
    if optional_contacts not in {"off", "on"}:
        raise ValueError("explicit optional contact policy required")
    if optional_final_contacts not in {"off", "on"}:
        raise ValueError("explicit optional final contact policy required")
    if optional_contact_composition not in {"off", "bounded_pairs"}:
        raise ValueError("explicit optional contact composition policy required")
    if optional_contact_composition != "off" and optional_contacts != "on":
        raise ValueError("optional contact composition requires optional contacts enabled")
    if optional_contact_timing not in {"off", "pmf_peaks"}:
        raise ValueError("explicit optional contact timing policy required")
    if optional_contact_timing != "off" and optional_contacts != "on":
        raise ValueError("optional contact timing requires optional contacts enabled")
    if optional_final_contacts == "on" and optional_contacts != "on":
        raise ValueError("optional final contacts require the optional contact policy")
    if optional_interior_contacts not in {"off", "on"}:
        raise ValueError("explicit optional interior contact policy required")
    if optional_interior_contacts == "on" and optional_contacts != "on":
        raise ValueError("optional interior contacts require the optional contact policy")
    if independent_event_proposals not in {"off", "on"}:
        raise ValueError("explicit independent event proposal policy required")
    if independent_event_proposals == "on" and (
        optional_contacts != "on" or inputs.independent_event_candidates is None
    ):
        raise ValueError(
            "independent event proposals require optional contacts and a bound sidecar"
        )
    if inputs.independent_event_candidates is not None and independent_event_proposals != "on":
        raise ValueError("independent event sidecar requires explicit policy")
    if inputs.optional_contact_pose is not None and optional_contacts != "on":
        raise ValueError("independent contact pose requires optional contacts enabled")
    if service_attempt_split not in {"off", "on"} or (
        child_index is not None and service_attempt_split != "on"
    ):
        raise ValueError("explicit service-attempt split policy required")
    if observed_first_flight not in {"off", "on"}:
        raise ValueError("explicit observed-first-flight policy required")
    if output.exists():
        raise FileExistsError(output)
    paths.data_relative(output)  # Generated inputs must have a portable runtime binding.
    if Path(inputs.match_id).name != inputs.match_id:
        raise ValueError("one safe automatic match identity required")
    if not inputs.clip.startswith("pt") or not inputs.clip[2:].isdigit():
        raise ValueError("automatic ledger clip must be pt followed by its numeric ID")
    manifest = _read(inputs.manifest)
    match = [m for m in manifest["matches"] if m["id"] == inputs.match_id]
    if len(match) != 1:
        raise ValueError("one matching automatic manifest entry required")
    match = match[0]
    fps, surface = float(match["source_fps"]), match["surface"]
    if not math.isfinite(fps) or fps <= 0 or surface not in {"hard", "clay", "grass"}:
        raise ValueError("finite native cadence and explicit surface required")
    with inputs.point_ledger.open(newline="") as handle:
        selected = [r for r in csv.DictReader(handle) if int(r["pt"]) == int(inputs.clip[2:])]
    if len(selected) != 1 or int(inputs.clip[2:]) not in match["point_ids"]:
        raise ValueError("one automatic ledger attempt required")
    ledger = selected[0]
    ball_sidecar = resolution.coordinate_manifest_path(inputs.ball)
    player_sidecar = resolution.coordinate_manifest_path(inputs.players)
    required = [
        inputs.manifest,
        inputs.point_ledger,
        inputs.extraction_receipt,
        inputs.source_pts,
        inputs.ball,
        ball_sidecar,
        inputs.events,
        inputs.cameras,
        inputs.players,
        player_sidecar,
    ]
    if inputs.optional_contact_views is not None:
        required.append(inputs.optional_contact_views)
    if inputs.independent_event_candidates is not None:
        required.append(inputs.independent_event_candidates)
    if inputs.optional_contact_pose is not None:
        from cv.pipeline.s6_optional_contacts import independent_pose_frame_receipt

        required.extend(
            [
                inputs.optional_contact_pose,
                resolution.coordinate_manifest_path(inputs.optional_contact_pose),
            ]
        )
        if pose_receipt := independent_pose_frame_receipt(inputs):
            required.append(pose_receipt)
    # The existing tracker parser follows this declared guide when diagnosing a
    # coarse lock. Bind it before that parser can read an additional CSV.
    first_source = str(_read(ball_sidecar).get("source", "")).split(" + ")[0]
    guide = Path(first_source)
    if not guide.is_absolute():
        guide = inputs.ball.parent / guide
    if guide != inputs.ball and guide.is_file() and guide.suffix == ".csv":
        required.extend([guide, resolution.coordinate_manifest_path(guide)])
    ancestry = validate_ancestry(inputs, required)
    original_records = [provenance.file_record(p, role="automatic_stage_input") for p in required]
    images = _native_inventory(inputs, fps, ledger)
    window = (1, len(images))
    declared_size = resolution.read_coordinate_manifest(inputs.ball)["image_size"]
    image_size = resolution.FrameSize(int(declared_size["width"]), int(declared_size["height"]))
    if image_size != resolution.NATIVE_SIZE:
        raise ValueError("tracker declared image size differs from verified native pictures")
    ball_rows, native_support = native_ball_support(
        auto_packet.automatic_ball_rows(inputs.ball, inputs.clip),
        image_size,
        jump_rejection=automatic_ball_jump_rejection,
        fps=fps,
        streak_centre=automatic_ball_streak_centre,
    )
    rows_by_frame = {r["frame"]: r for r in ball_rows}
    observations = []
    for image in images:
        row = rows_by_frame.get(
            image["frame"],
            {
                "frame": image["frame"],
                "status": "missing",
                "x1080": None,
                "y1080": None,
                "annotation_origin": "automatic",
                "observation_semantics": auto_packet.BALL_SEMANTICS["automatic"],
            },
        )
        observations.append(
            {
                **row,
                "native_pts_seconds": image["native_pts_seconds"],
                "source_image_sha256": image["source"]["sha256"],
            }
        )
    # Production default admits a tracked intrinsic-fallback projection. Explicit
    # "off" is the rollback and omits the admission fields. Callers that build
    # packets without naming the mode inherit admit from this argument.
    shot_ids = None
    if shot_homography_propagation == "on":
        from cv.pipeline.shot_segments import clip_propagation_shots

        shot_ids = clip_propagation_shots(inputs.frames_directory, fps)
    camera_document = auto_packet.automatic_camera_document(
        inputs.cameras,
        {"match_id": inputs.match_id, "attempt": {"clip": inputs.clip}},
        window,
        intrinsic_fallback_registration=intrinsic_fallback_registration,
        shot_homography_propagation=shot_homography_propagation,
        shot_ids=shot_ids,
    )
    near_baseline_receipt = None
    if near_baseline_refinement != "off":
        from cv.pipeline.court_near_baseline_refinement import refine_camera_document

        camera_document, near_baseline_receipt = refine_camera_document(
            camera_document,
            mode=near_baseline_refinement,
            frame_paths={int(image["frame"]): image["source"]["path"] for image in images},
        )
    players_path = inputs.players
    propagation_records: list[dict] = []
    if player_body_scale not in {"off", "camera"}:
        raise ValueError(f"unknown player_body_scale {player_body_scale!r}")
    if shot_homography_propagation == "on":
        import numpy as np

        from cv.pipeline.player_side_association import write_propagated_sided_boxes

        unsided = resolution.resolve_player_boxes(inputs.players.parent, sided=False)
        if unsided is None:
            raise ValueError("shot-homography propagation requires unsided native player boxes")
        output.mkdir(parents=True)
        players_path = write_propagated_sided_boxes(
            unsided=unsided,
            court_point=inputs.cameras.parent / "court_H_per_point.npz",
            court_frame=inputs.cameras.parent / "court_H_per_frame_v1.npz",
            fps=fps,
            clip=inputs.clip,
            output=output / "player_boxes_shot_propagated.csv",
            frames_directory=inputs.frames_directory,
            camera_projections=(
                {
                    int(camera["frame"]): np.asarray(camera["P"], dtype=float)
                    for camera in camera_document["cameras"]
                    if camera.get("supported") and camera.get("P") is not None
                }
                if player_body_scale == "camera"
                else None
            ),
        )
        # The stage executes the propagated boxes, so the producer must bind them,
        # their coordinate declaration and the upstream files they were derived from.
        propagation_records = [
            *(
                provenance.file_record(path, role="automatic_player_propagation_source")
                for path in (
                    unsided,
                    resolution.coordinate_manifest_path(unsided),
                    inputs.cameras.parent / "court_H_per_point.npz",
                    inputs.cameras.parent / "court_H_per_frame_v1.npz",
                )
                if path.is_file()
            ),
            provenance.file_record(players_path, role="automatic_shot_propagated_players"),
            provenance.file_record(
                resolution.coordinate_manifest_path(players_path),
                role="automatic_shot_propagated_player_coordinates",
            ),
        ]
    with players_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = (
            resolution.PLAYER_NATIVE_BOX_COLUMNS
            if set(resolution.PLAYER_NATIVE_BOX_COLUMNS).issubset(reader.fieldnames or ())
            else ("x0", "y0", "x1", "y1")
        )
        player_rows = [r for r in reader if r.get("clip") == inputs.clip]
    if not player_rows or any(r.get("side") not in {"near", "far"} for r in player_rows):
        raise ValueError("automatic sided player observations required")
    player_scale = auto_packet.uniform_native_scale(players_path, columns)
    emissions = _read(inputs.events)
    if isinstance(emissions, dict):
        emissions = emissions["emissions"]
    _reject_assistance(emissions, context="automatic event emissions")
    output.mkdir(parents=True, exist_ok=shot_homography_propagation == "on")

    def save(name: str, value: Any) -> Path:
        path = output / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        return path

    # Preserve original accepted and abstained emissions for diagnosis even when
    # the physical-event grammar refuses preparation. Do not turn them into fit
    # events; the shared automatic parser still decides which rows are admitted.
    joined = f"{inputs.match_id}__{inputs.clip}"
    selected_emissions = [
        row
        for row in emissions
        if row.get("match_id") == inputs.match_id
        and row.get("clip") in {inputs.clip, joined}
        and window[0] <= float(row["frame"]) <= window[1]
    ]
    child_scope = None
    service_plan = None
    parent_observations = observations
    parent_emissions = selected_emissions
    if service_attempt_split == "on":
        from cv.pipeline import service_attempt_scope

        service_plan = service_attempt_scope.plan(
            emissions=selected_emissions,
            observations=observations,
            players=players_path,
            cameras=camera_document,
            source_start=float(images[0]["native_pts_seconds"]),
            fps=fps,
            native_window=window,
            clip=inputs.clip,
        )
        # The requested ledger window can fall between native pictures. Keep it
        # as selection metadata, not the epoch of the first observed frame.
        service_plan["requested_ledger_interval_seconds"] = [
            float(ledger["rally_t_start"]),
            float(ledger["rally_t_end"]),
        ]
        service_plan["source_start_basis"] = "first_authenticated_native_picture_pts"
        plan_path = save("service_attempt_scope.json", service_plan)
        if service_plan["children"] and child_index is None:
            children = [
                build_observations(
                    inputs,
                    output / f"child_{child['child_index']:02d}",
                    service_attempt_split="on",
                    observed_first_flight=observed_first_flight,
                    optional_contacts=optional_contacts,
                    optional_final_contacts=optional_final_contacts,
                    optional_contact_timing=optional_contact_timing,
                    optional_contact_composition=optional_contact_composition,
                    optional_interior_contacts=optional_interior_contacts,
                    independent_event_proposals=independent_event_proposals,
                    optional_bounces=optional_bounces,
                    # A split service attempt is prepared by this same adapter:
                    # the shared policy must not be dropped on the child.
                    observed_horizon_tail=observed_horizon_tail,
                    terminal_net_tail=terminal_net_tail,
                    contact_prefix_scope=contact_prefix_scope,
                    contact_components=contact_components,
                    leading_event_components=leading_event_components,
                    contact_component_routing=contact_component_routing,
                    automatic_ball_jump_rejection=automatic_ball_jump_rejection,
                    automatic_ball_streak_centre=automatic_ball_streak_centre,
                    observation_partition=observation_partition,
                    child_index=child["child_index"],
                )
                for child in service_plan["children"]
            ]
            if original_records != [
                provenance.file_record(p, role="automatic_stage_input") for p in required
            ] or not ancestry_unchanged(ancestry):
                raise ValueError("automatic parent inputs changed during child preparation")
            result = {
                "schema": "s6_automatic_service_parent_preparation_v1",
                "status": "prepared_children",
                "parent_clip": inputs.clip,
                "attempt_count": len(children),
                "children": children,
                "row": None,
                "evidence": provenance.file_record(plan_path),
                "input_bindings": original_records,
                "runtime_model_calls": 0,
                "original_parent_retained": True,
            }
            save("preparation.json", result)
            return result
        if child_index is not None:
            matches = [r for r in service_plan["children"] if r["child_index"] == child_index]
            if len(matches) != 1:
                raise ValueError(
                    "child identity must be produced by the same automatic source rule"
                )
            child_scope = matches[0]
            window = tuple(child_scope["modeled_native_window"])
            observations = [r for r in observations if window[0] <= r["frame"] <= window[1]]
            selected_emissions = [
                parent_emissions[i] for i in child_scope["original_emission_indices"]
            ]
    event_path = save("automatic_events.json", selected_emissions)
    hold = child_scope.get("hold_reason") if child_scope is not None else None
    # The unresolved visible tail is only reachable through the explicit shared
    # policy. Default off, the adapter still admits supplied ground tails only.
    horizon_tail = observed_horizon_tail == "on"
    # A service-attempt child is cut at its own net cord, so its tail may open on a
    # net_hit whatever the policy says; that structural admission is unchanged. The
    # declared policy is now honoured beside it, which is what every automatic
    # `shared_s6` receipt has been asking for -- all 38 of them declare this key on
    # -- and what the supplied-input preparation has always done with it.
    net_tail = terminal_net_tail == "on"
    scope = None
    segmentation_source = {
        "kind": "automatic_point_ledger",
        "record": provenance.file_record(inputs.point_ledger),
    }
    events, ending = [], None
    try:
        # The adapter converts the original inventory; this runtime qualifies the
        # observation scope itself, so every refusal an explicit policy may own
        # reaches the same shared helper the supplied-input preparation uses.
        document = json.loads(event_path.read_text())
        document, operating_restatement = operating.restate(document, operating_floor)

        def inventory(include_abstained: bool):
            return auto_packet.automatic_event_inventory(
                document,
                inputs.match_id,
                inputs.clip,
                window,
                observation_scope=True,
                require_originating_contact=True,
                leading_physical_prefix=leading_prefix_admitted,
                include_abstained=include_abstained,
            )

        events, ending = inventory(False)
        # The scope decision is read off the ACCEPTED inventory first, because the
        # `declared_ending` mode admits abstentions only where the producer declared
        # a resolved ending. An abstention must never be what decides that.
        if abstained_admitted is not None and (
            not ending.get("unqualified_observation_scope")
            or abstained_admitted == contact_prefix.ABSTAINED_RETAINED
        ):
            events, ending = inventory(True)
        declared_ending_bound = []
        if not ending.get("unqualified_observation_scope") and any(
            event["status"] == "ambiguous" for event in events
        ):
            # A resolved automatic ending beside the producer's own refused occurrence
            # is not one resolved competitive topology: every span containing the
            # refusal is unrepresentable. Retain the original inventory instead, with
            # the declared ending kept as a bound original boundary. Identical to what
            # the supplied-input preparation does with a refused labeled occurrence.
            low, high = ending["frame_interval"]
            declared_ending_bound = [[max(float(window[0]), low), min(float(window[1]), high)]]
            ending = dict(
                frame=float(window[1]), kind="unresolved", unqualified_observation_scope=True
            )
        if ending.pop("unqualified_observation_scope", False):
            from cv.experiments.connected_shooting import observation_scope

            # Leading rows, a refused occurrence and an unresolved tail are all original
            # evidence. The shared helper owns a refusal only under the policy declared
            # for it, and the runtime still qualifies coverage and cameras before any
            # component is executed.
            scope, retained = contact_prefix.qualify_or_retain(
                events,
                window,
                event_origin=None,
                boundaries=contact_prefix.unresolved_boundaries(
                    events, boundaries=declared_ending_bound
                ),
                mode=contact_prefix_scope,
                leading_admitted=leading_prefix_admitted,
                allow_terminal_net=child_scope is not None or net_tail,
                allow_observed_horizon=horizon_tail,
                segmentation_source=segmentation_source,
            )
            ending["observation_scope"] = scope
            if retained:
                # The declared inventory is pending qualification, not a modeled
                # scope: bind the same unresolved-ending contact prefix the supplied
                # clip runtime binds, so components recover one identical source.
                source_attempt = dict(
                    point_clip=inputs.clip,
                    match_id=inputs.match_id,
                    fps=fps,
                    events=events,
                    owner_end_frame=float(ending["frame"]),
                    owner_ending_kind=ending.get("kind"),
                    owner_ball_labels=_active_ball_rows(observations, events, ending["frame"]),
                    observation_scope=scope,
                    owner_end_frame_semantics="observation_horizon_not_physical_event",
                    ending_supplied=False,
                    segmentation_source=segmentation_source,
                )
                contact_plan = contact_prefix.qualify(
                    source_attempt,
                    camera_document,
                    observation_partition=observation_partition,
                    native_window=list(window),
                    mode=contact_prefix.UNRESOLVED_ENDING,
                    observation_fallback=True,
                )
                if contact_plan["status"] != "qualified":
                    if not component_routing_admitted:
                        raise ValueError(
                            "unresolved contact prefix: "
                            + contact_plan.get("reason", contact_plan["status"])
                        )
                    # The retained original inventory stays exactly as declared and
                    # the component stage partitions it. No cut, ending, contact or
                    # native epoch is created here.
                    component_routing = components.source_routing_receipt(
                        source_attempt, contact_plan, mode=contact_component_routing
                    )
                    ending = dict(
                        frame=float(scope["observation_horizon"]),
                        kind="unresolved",
                        observation_scope=scope,
                    )
                else:
                    contact_bound = contact_prefix.bind_attempt(source_attempt, contact_plan)
                    scope = contact_bound["observation_scope"]
                    events = contact_bound["events"]
                    ending = dict(
                        frame=scope["modeled_horizon"], kind="unresolved", observation_scope=scope
                    )
                optional_contacts, optional_bounces = "off", "off"
            else:
                if contact_prefix_scope == "coverage":
                    source_attempt = dict(
                        point_clip=inputs.clip,
                        match_id=inputs.match_id,
                        fps=fps,
                        events=events,
                        owner_end_frame=float(ending["frame"]),
                        owner_ending_kind=ending.get("kind"),
                        owner_ball_labels=_active_ball_rows(observations, events, ending["frame"]),
                        observation_scope=scope,
                        owner_end_frame_semantics="observation_horizon_not_physical_event",
                        ending_supplied=False,
                        segmentation_source=segmentation_source,
                    )
                    contact_plan = contact_prefix.qualify(
                        source_attempt,
                        camera_document,
                        observation_partition=observation_partition,
                        native_window=list(window),
                        observation_fallback=True,
                    )
                    if contact_plan["status"] == "qualified":
                        contact_bound = contact_prefix.bind_attempt(source_attempt, contact_plan)
                        scope = contact_bound["observation_scope"]
                        events = contact_bound["events"]
                        ending = dict(
                            frame=scope["modeled_horizon"],
                            kind="unresolved",
                            observation_scope=scope,
                        )
                        optional_contacts, optional_bounces = "off", "off"
                if (
                    contact_bound is None
                    and scope.get("terminal_kind") == "observed_horizon"
                    and not observation_scope.supported_tail(events, observations, scope)
                ):
                    # Hold the attempt rather than raise out of preparation: an
                    # unwitnessed tail has no observed flight to model.
                    raise ValueError(
                        "unresolved terminal tail needs visible native support after the contact"
                    )
                ending["observation_scope"] = scope
        elif not ending.get("kind"):
            hold = "missing_physical_ending_kind"
    except ValueError as error:
        hold = f"automatic_event_preparation_refused: {error}"
    if contact_prefix_scope == "coverage" and hold is None and contact_plan is None:
        source_attempt = dict(
            point_clip=inputs.clip,
            match_id=inputs.match_id,
            fps=fps,
            events=events,
            owner_end_frame=float(ending["frame"]),
            owner_ending_kind=ending.get("kind"),
            owner_ball_labels=_active_ball_rows(observations, events, ending["frame"]),
        )
        contact_plan = contact_prefix.qualify(
            source_attempt,
            camera_document,
            observation_partition=observation_partition,
            native_window=list(window),
            observation_fallback=True,
        )
        if contact_plan["status"] == "qualified":
            contact_bound = contact_prefix.bind_attempt(source_attempt, contact_plan)
            scope, events = contact_bound["observation_scope"], contact_bound["events"]
            ending = dict(
                frame=scope["modeled_horizon"], kind="unresolved", observation_scope=scope
            )
            optional_contacts, optional_bounces = "off", "off"
    origins = dict.fromkeys(auto_packet.STREAMS, "automatic")
    operator = {
        "schema": "ball_observation_operator_v1",
        "kind": "nominal_center",
        "exposure_duration_frames": None,
    }
    prefix_plan = None
    if observed_first_flight == "on" and hold is not None and child_scope is None:
        from cv.pipeline import s6_first_flight_scope as prefix

        prefix_plan = prefix.prepare(
            selected_emissions, observations, window, operator, segmentation_source
        )
        prefix_plan["original_preparation_hold"] = hold
        save("first_flight_scope.json", prefix_plan)
        if prefix_plan["supported"]:
            from cv.experiments.connected_shooting import observation_scope

            scope = prefix_plan["contract"]
            events = prefix_plan["modeled_events"]
            ending = {
                "frame": scope["observation_horizon"],
                "kind": "unresolved",
                "observation_scope": scope,
            }
            hold = None
    bounce_document = None
    bounce_record = None
    if optional_bounces != "off":
        from cv.pipeline import s6_optional_bounce_scope as bounce_scope

        try:
            original, original_ending = auto_packet.automatic_event_inventory(
                selected_emissions,
                inputs.match_id,
                inputs.clip,
                window,
                observation_scope=True,
                require_originating_contact=True,
            )
            unknown_ending = original_ending.pop("unqualified_observation_scope", False)
            bounce_document = bounce_scope.build_witness(
                emissions=selected_emissions,
                events=original,
                observations=observations,
                window=window,
                match_id=inputs.match_id,
                clip=inputs.clip,
                source_events=provenance.file_record(event_path),
                segmentation_source=segmentation_source,
                mode=optional_bounces,
                # Same admission as the scope decision above: the bounce witness and
                # the observation scope must not disagree about whether this tail may
                # open on a net cord. Reachable only with `optional_bounces` on, which
                # the supplied-input swaps refuse outright, so no measured arm here.
                allow_terminal_net=child_scope is not None or net_tail,
                allow_observed_horizon=horizon_tail,
                physical_ending=None if unknown_ending else original_ending,
            )
            bounce_record = provenance.file_record(
                save("optional_bounce_witness.json", bounce_document)
            )
            # Only source event qualification may be replaced by a container.
            # Existing service-attempt holds and all non-event failures remain.
            if (
                hold is not None
                and hold.startswith("automatic_event_preparation_refused:")
                and not (child_scope is not None and child_scope.get("hold_reason"))
            ):
                scope = bounce_scope.scope_container(bounce_record, bounce_document)
                events = original
                ending = {
                    "frame": scope["observation_horizon"],
                    "kind": "unresolved",
                    "observation_scope": scope,
                }
                hold = None
        except ValueError as error:
            if hold is None:
                hold = f"automatic_optional_bounce_preparation_refused: {error}"
    evidence = {
        "schema": "s6_automatic_observed_evidence_v1",
        "match_id": inputs.match_id,
        "clip": inputs.clip,
        "native_window": list(window),
        "automatic_ledger_row": ledger,
        "source_pack": {"fps": fps, "images": images},
        "ball_observations": observations,
        "native_image_support": native_support,
        "physical_events": contact_bound["original_physical_events"]
        if contact_bound is not None
        else events,
        "original_event_emissions": selected_emissions,
        # Absent unless a floor is actually declared, so every receipt written under the
        # producer's own threshold is unchanged byte for byte.
        **(
            {"automatic_event_operating_point": operating_restatement}
            if operating_restatement.get("declared_floor") is not None
            else {}
        ),
        **({"optional_bounce_witness": bounce_record} if bounce_record else {}),
        **(
            {"near_baseline_refinement": near_baseline_receipt}
            if near_baseline_receipt is not None
            else {}
        ),
        "physical_ending": ending if hold is None and scope is None else None,
        "observation_scope": scope,
        "observation_horizon": scope["observation_horizon"] if scope else window[1],
        **({"first_flight_scope": prefix_plan} if prefix_plan is not None else {}),
        **(
            {
                "service_attempt_scope": child_scope,
                "parent_ball_observations": parent_observations,
                "parent_event_emissions": parent_emissions,
                "modeled_native_window": list(window),
                "context_only_frames": [
                    r["frame"]
                    for r in parent_observations
                    if not window[0] <= r["frame"] <= window[1]
                ],
            }
            if child_scope is not None
            else {}
        ),
        "optional_athlete_evidence": {
            "player_order": [],
            "player_statures_m": [],
            "status": "not_supplied",
        },
    }
    evidence_path = save("observed_evidence.json", evidence)
    camera_path = save("cameras.json", camera_document)
    run_provenance = provenance.build_provenance(
        root=paths.REPO_ROOT,
        mode=provenance.AUTOMATIC_MODE,
        source_videos=[provenance.file_record(inputs.source_video)],
        configuration={
            "entrypoint": "cv.pipeline.s6_automatic_observations",
            "match_id": inputs.match_id,
            "clip": inputs.clip,
            "native_window": list(window),
            "source_timestamp_field": "frame.pts_time",
            "native_image_support": native_support,
            "optional_athlete_evidence": "not_supplied",
            "origin_policy": "no accepted physical event dropped before first contact",
            **(
                {
                    "leading_event_components": leading_event_components,
                    "contact_components": contact_components,
                    "leading_prefix_admitted": leading_prefix_admitted,
                }
                if leading_prefix_admitted
                else {}
            ),
            **(
                {
                    "contact_prefix_scope": contact_prefix_scope,
                    "contact_prefix_qualification": contact_plan,
                    "requested_optional_settings": requested_optional,
                }
                if contact_prefix_scope != "off"
                else {}
            ),
            **(
                {
                    "contact_component_routing": contact_component_routing,
                    "contact_component_source_routing": component_routing,
                }
                if component_routing is not None
                else {}
            ),
            **({"optional_bounces": optional_bounces} if optional_bounces != "off" else {}),
            **({"observed_horizon_tail": "on"} if horizon_tail else {}),
            **({"terminal_net_tail": "on"} if net_tail else {}),
            **(
                {"observed_first_flight": "on", "first_flight_scope": prefix_plan}
                if observed_first_flight == "on"
                else {}
            ),
            **(
                {
                    "service_attempt_split": "on",
                    "service_attempt_scope": child_scope,
                    "service_attempt_plan": provenance.file_record(plan_path),
                }
                if service_plan is not None
                else {}
            ),
        },
        reused_artifacts=[*original_records, *ancestry, *propagation_records],
    )
    row = None
    if hold is None:
        first, last = (
            min(e["frame"] for e in events if e["event_type"] == "contact"),
            ending["frame"],
        )
        original_events = (
            contact_bound["original_physical_events"]
            if contact_bound is not None
            else (
                prefix_plan["original_physical_events"]
                if prefix_plan is not None and prefix_plan["supported"]
                else None
            )
        )
        event_rows = [
            {**e, "clip": inputs.clip}
            for e in (original_events if original_events is not None else events)
        ]
        if scope is None:
            event_rows.append(
                {
                    "event_type": "ending",
                    "clip": inputs.clip,
                    "frame": last,
                    "frame_interval": ending["frame_interval"],
                    "interval_origin": ending["interval_origin"],
                    "status": "predicted",
                    "annotation_origin": "automatic",
                    "exact_epoch_observed": False,
                    "ending_kind": ending["kind"],
                    "automatic_confidence": ending.get("automatic_confidence"),
                    "automatic_probability": ending.get("automatic_probability"),
                }
            )
        key = f"{inputs.match_id}__{inputs.clip}"
        if child_scope is not None:
            key += f"__a{child_index:02d}"
        consumer = {
            "schema": "s6_automatic_observation_document_v1",
            "benchmark_id": key,
            "match_id": inputs.match_id,
            "native_size": [1920, 1080],
            "annotation_origin": "automatic",
            "automatic_inference_eligible": True,
            "stream_origins": origins,
            "source_pack": evidence["source_pack"],
            **({"service_attempt_scope": child_scope} if child_scope is not None else {}),
            "attempt": {
                "clip": inputs.clip,
                "native_window": list(window),
                "surface": surface,
                "first_contact_frame": first,
                "ending_frame": last,
                "ending_kind": ending["kind"],
            },
            "ball_convention": auto_packet.BALL_SEMANTICS["automatic"],
            "ball": {"records": [{"clip": inputs.clip, "frames": observations}]},
            "events": {"records": event_rows},
        }
        if scope is not None:
            consumer["attempt"].pop("ending_frame")
            consumer["attempt"].update(
                observation_scope=scope, segmentation_source=segmentation_source
            )
        consumer_path = save("observations.json", consumer)
        # Scope changes model support, not the original active/context inventory.
        active = (
            contact_bound["owner_ball_labels"]
            if contact_bound is not None
            else _active_ball_rows(observations, events, last)
        )
        attempt = {
            "attempt_id": key,
            "clip": key,
            "point_clip": inputs.clip,
            "match_id": inputs.match_id,
            "fps": fps,
            "events": events,
            "first_event_frame": first,
            "owner_end_frame": last,
            "contact_count": sum(e["event_type"] == "contact" for e in events),
            "terminal_bounce_count": auto_packet.terminal_bounce_count(events, last),
            "owner_ball_labels": active,
            "ball_observations": active,
            "context_native_frames": [r["frame"] for r in observations if r not in active],
            "agent_event_rows": event_rows,
            "ball_observation_operator": operator,
            "stream_origins": origins,
            "ball_convention": auto_packet.BALL_SEMANTICS["automatic"],
            "annotation_origin": "automatic",
            "complete_native_label_inventory": False,
            "labeled_native_frames": 0,
            "visible_native_frames": sum(r["status"] == "visible" for r in active),
            "fractional_estimates": [],
            **(
                {"original_physical_events": original_events} if original_events is not None else {}
            ),
            **({"service_attempt_scope": child_scope} if child_scope is not None else {}),
        }
        if bounce_record is not None:
            attempt["optional_bounce_witness"] = bounce_record
        if scope is not None:
            attempt.update(
                observation_scope=scope,
                segmentation_source=segmentation_source,
                ending_supplied=False,
                owner_end_frame_semantics="observation_horizon_not_physical_event",
            )
            if contact_bound is not None:
                attempt.update(contact_bound)
            from cv.experiments.connected_shooting import observation_scope

            observation_scope.validate(attempt, consumer)
        packet = {
            "schema": "connected_shooting_stream_packet_v1",
            "attempts": [attempt],
            "stream_origins": origins,
            "ball_observation_operator": operator,
            "annotation_origin": "automatic",
            "human_derived": False,
            "automatic_inference_eligible": True,
            "inputs": original_records,
            "external_label_binding": {
                "record": provenance.file_record(consumer_path),
                "semantics": "automatic consumer document; legacy interface name only",
            },
        }
        if bounce_record is not None:
            packet["optional_bounce_witness"] = bounce_record
            run_provenance["reused_artifacts"].extend(
                [
                    provenance.file_record(
                        event_path, role="automatic_optional_bounce_source_events"
                    ),
                    {**bounce_record, "role": "automatic_optional_bounce_witness"},
                ]
            )
        if optional_contacts == "on":
            from cv.pipeline import s6_optional_contacts

            document = s6_optional_contacts.build_witness(
                inputs,
                attempt,
                images,
                camera_document,
                optional_final_contacts=optional_final_contacts,
                optional_contact_timing=optional_contact_timing,
                optional_contact_composition=optional_contact_composition,
                optional_interior_contacts=optional_interior_contacts,
                independent_event_proposals=independent_event_proposals,
            )
            witness_path = save("optional_contact_witness.json", document)
            packet["optional_contact_witness"] = provenance.file_record(witness_path)
            run_provenance["reused_artifacts"].append(
                provenance.file_record(witness_path, role="automatic_optional_contact_witness")
            )
        packet_path = save("packet.json", packet)
        run_provenance["reused_artifacts"].extend(
            provenance.file_record(path, role=role)
            for path, role in (
                (consumer_path, "automatic_observation_document"),
                (packet_path, "automatic_observation_packet"),
                (camera_path, "automatic_camera_document"),
            )
        )
        row = {
            "key": key,
            "surface": surface,
            "declared_flights": None,
            "observation_origin": "automatic",
            "labels": provenance.file_record(consumer_path),
            "packet": provenance.file_record(packet_path),
            "cameras": provenance.file_record(camera_path),
            "pose_csv": provenance.file_record(players_path),
            "pose_image_scale": player_scale,
            "player_order": [],
            "player_statures_m": [],
            "serve_number": 1,
            "serve_number_uncertain": True,
        }
    provenance_path = save("automatic_provenance.json", run_provenance)
    if row is not None:
        row["automatic_provenance"] = provenance.file_record(provenance_path)
    if original_records != [
        provenance.file_record(p, role="automatic_stage_input") for p in required
    ]:
        raise ValueError("automatic inputs changed during preparation")
    if not ancestry_unchanged(ancestry):
        raise ValueError("automatic ancestry changed during preparation")
    if any(
        provenance.file_record(_resolve(image["source"])) != image["source"] for image in images
    ):
        raise ValueError("native pictures changed during preparation")
    result = {
        "schema": "s6_automatic_preparation_v1",
        "status": "preparation_held" if hold else "prepared",
        "reason": hold,
        "attempt_count": 1,
        **({"service_attempt_scope": child_scope} if child_scope is not None else {}),
        **({"first_flight_scope": prefix_plan} if prefix_plan is not None else {}),
        **({"contact_prefix_scope": contact_plan} if contact_plan is not None else {}),
        "reference_flight_count": None,
        "row": row,
        "evidence": provenance.file_record(evidence_path),
        "provenance": provenance.file_record(provenance_path),
        "input_bindings": original_records,
        "runtime_model_calls": 0,
        "fitted_states_read": False,
        "stage_execution": "not_run; neutral row and optional athlete support require shared-stage integration",
    }
    save("preparation.json", result)
    return result
