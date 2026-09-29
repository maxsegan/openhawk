"""Execute one shared S6 attempt row, with component orchestration and seed-failure fallback.

The per-attempt dispatch used by the labeled harness (cv.validation.run_labeled_s6) and by
the automatic broadcast backend, so production component policy runs identically in both.
Writes ``<output>/jobs/<key>.json``, ``<output>/logs/<key>.log`` and the case under
``<output>/cases/<key>/`` (``result.json``, plus ``component_plan.json`` for component runs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from cv.pipeline import s6_labeled_stage as stage


def failure(row: dict, status: str, reason: str) -> dict:
    return {
        "key": row["key"],
        "surface": row["surface"],
        "status": status,
        "reason": reason,
        "declared_flights": row["declared_flights"],
        "verdict": {"complete_point": False, "accepted_flight_count": 0, "flight_count": 0},
    }


def execute(row: dict, output: Path, policy: dict, timeout: float) -> dict:
    component_started = (
        time.monotonic() if policy.get("contact_components", "off") != "off" else None
    )
    try:
        stage.validate_row(row)
    except (ValueError, KeyError, FileNotFoundError) as error:
        result = failure(row, "preparation_failed", str(error))
        stage.save(output / "cases" / row["key"] / "result.json", result)
        return result
    if policy.get("contact_components", "off") == "unresolved_ending":
        from cv.pipeline import s6_component_scope, s6_component_orchestration

        packet = json.loads(stage.resolve(row["packet"]).read_text())
        if not s6_component_scope.active(packet["attempts"][0]):
            try:
                return s6_component_orchestration.run_source(
                    row, output, policy, timeout, execute=execute, started=component_started
                )
            except (ValueError, KeyError, FileNotFoundError) as error:
                result = failure(row, "component_source_failed", str(error))
                stage.save(output / "cases" / row["key"] / "result.json", result)
                return result
    job = output / "jobs" / f"{row['key']}.json"
    stage.save(
        job,
        {
            "row": row,
            "policy": policy,
            **({"timeout_seconds": timeout} if component_started is not None else {}),
        },
    )
    destination = output / "cases" / row["key"]
    log = output / "logs" / f"{row['key']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [
        sys.executable,
        "-m",
        "cv.pipeline.s6_labeled_stage",
        "--job",
        str(job),
        "--output",
        str(destination),
    ]
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result = failure(row, "timeout", f"shared per-attempt budget {timeout:g}s exhausted")
        else:
            path = destination / "result.json"
            if returncode == 0 and path.is_file():
                result = json.loads(path.read_text())
            else:
                lines = log.read_text().splitlines()
                detail = lines[-1][:1000] if lines else "no diagnostic output"
                result = failure(row, "execution_failed", f"worker exit {returncode}: {detail}")
    result["wall_seconds"] = time.monotonic() - started
    if component_started is not None:
        result["execution_timeout_seconds"] = timeout
    from cv.experiments.connected_shooting import candidate_attempts as attempts

    if attempts.should_fallback_to_components(result.get("reason", ""), policy):
        from cv.pipeline import s6_component_orchestration, s6_component_scope
        from cv.pipeline import s6_contact_components as components
        from cv.pipeline import s6_contact_prefix_scope as prefix

        packet = json.loads(stage.resolve(row["packet"]).read_text())
        if not s6_component_scope.active(packet["attempts"][0]):
            stash = output / "cases" / f"{row['key']}__whole_point_seed_failure"
            if destination.exists():
                if stash.exists():
                    raise ValueError("seed-failure stash already present")
                destination.rename(stash)
            remaining = max(1.0, timeout - (time.monotonic() - started))
            fallback_policy = dict(policy)
            fallback_policy["contact_components"] = components.MODE
            if fallback_policy.get("contact_prefix_scope", "off") == "off":
                fallback_policy["contact_prefix_scope"] = prefix.UNRESOLVED_ENDING
            try:
                result = s6_component_orchestration.run_source(
                    row,
                    output,
                    fallback_policy,
                    remaining,
                    execute=execute,
                    started=started if component_started is None else component_started,
                    source_packet=components.as_unresolved_source(packet),
                    seed_failure=str(result.get("reason") or "no completed candidate"),
                    whole_point=result,
                )
            except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
                if stash.exists() and not destination.exists():
                    stash.rename(destination)
                result["seed_failure_fallback"] = {
                    "status": "unavailable",
                    "reason": str(error),
                }
    stage.save(destination / "result.json", result)
    return result
