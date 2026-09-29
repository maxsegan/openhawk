"""Experimental correction: serve covariance governs serve reversals only.

The original report described this restriction as 'for a serve'. The production
scorer currently applies it to every rally flight. Clone only this conditional
for a bounded fixed-vector comparison; do not mutate the shared scorer or any
measurement, witness, physical trajectory, label, epoch or numerical threshold.
Run --composition REVIEWED_REPORT --output ARTIFACT_DIRECTORY.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import inspect
import json
from pathlib import Path
import traceback

import numpy as np

from cv.experiments.connected_shooting import full_native_continuation as full
from cv.experiments.connected_shooting import labeled_context_census as census
from cv.pipeline import paths, provenance


def make_local_scorer():
    """Clone one explicit experimental branch, leaving original module intact."""
    source = inspect.getsource(full.profile.acceptance.score)
    needle = "and serve_witness_covariance_resolved\n"
    if source.count(needle) != 1:
        raise ValueError("shared scorer changed; re-review the single scope patch")
    source = source.replace(
        needle, 'and (row.get("role") != "serve" or serve_witness_covariance_resolved)\n'
    )
    namespace = dict(vars(full.profile.acceptance))
    exec(compile(source, "<explicit_labeled_serve_scope_experiment>", "exec"), namespace)
    return namespace["score"], source


LOCAL_SCORE, EXPERIMENTAL_SOURCE = make_local_scorer()


def run_case(job):
    key = job["item"]["key"]
    result = dict(key=key, status="pending")
    try:
        ctx, p, _, _, search, threshold, duration, loaded = full.load_case(job)
        derived = job.get("derived")
        if derived:
            p = np.asarray(derived["parameters"])
            epochs = derived.get("contact_frames", derived.get("original_contact_frames"))
            if epochs is not None and float(epochs[0]) != ctx["scene"].contact_frames[0]:
                ctx = full.profile.profile_context(ctx, float(epochs[0]))
            if "terminal_rebound_frames" in derived:
                ctx, receipt = census.extended_context(
                    ctx,
                    json.loads(loaded["arguments"].labels.read_text()),
                    json.loads(loaded["arguments"].cameras.read_text()),
                    search,
                    last_context_frame=derived["scene"]["contact_frames"][-1],
                )
                if ctx is None:
                    raise ValueError("reviewed terminal context no longer available")
                assert ctx["scene"].contact_frames.tolist() == derived["scene"]["contact_frames"]
        before, measurement = census.score(
            ctx, p, threshold, duration, job["item"]["metadata"]["ending_kind"]
        )
        expected = derived["verdict"] if derived else job["original_verdict"]
        assert [f["checks"] for f in before["flights"]] == [
            f["checks"] for f in expected["flights"]
        ], "reviewed fixed-vector gates did not reproduce"
        assert before["accepted_flight_count"] == job["reviewed"]["after"]["accepted_flights"]
        after, newmeasurement = census.score(
            ctx, p, threshold, duration, job["item"]["metadata"]["ending_kind"], scorer=LOCAL_SCORE
        )
        assert full.profile.jsonable(measurement) == full.profile.jsonable(newmeasurement)
        changes = []
        for left, right in zip(before["flights"], after["flights"], strict=True):
            altered = [k for k in left["checks"] if left["checks"][k] != right["checks"][k]]
            assert not altered or altered == ["bounce_rays_agree"]
            if altered:
                assert right["role"] != "serve"
                changes.append(
                    dict(
                        flight_index=right["flight_index"],
                        before_accepted=left["accepted"],
                        after_accepted=right["accepted"],
                        before=left,
                        after=right,
                    )
                )
        result.update(
            status="measured",
            before=before,
            after=after,
            changes=changes,
            parameters=p,
            measurement=measurement,
            sources=loaded["sources"],
            derived_vector=bool(derived),
            exact_measurement_unchanged=True,
            all_non_witness_gates_equal=True,
        )
    except Exception as exc:
        result.update(
            status="execution_failure",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    out = Path(job["output"]) / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "scope.json").write_text(json.dumps(full.profile.jsonable(result), indent=2) + "\n")
    return {
        k: v
        for k, v in full.profile.jsonable(result).items()
        if k not in ["measurement", "parameters"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.composition.read_text())
    basepath = paths.data_root() / "processed/ownerfix/adopted_perflight3/report.json"
    oldpath = (
        paths.data_root()
        / "processed/localrepair20260910/net_transition_matched_v1/composed89/report.json"
    )
    base = json.loads(basepath.read_text())
    older = json.loads(oldpath.read_text())
    items = {x["key"]: x for x in base["attempts"]}
    derived = {x["key"]: x for x in [*older["replays"], *prior["replays"]]}
    jobs = []
    for row in prior["attempts"]:
        item = items[row["key"]]
        if not item["fitted"]:
            continue
        pf = json.loads(Path(item["document"]).read_text())
        saved = next(x for x in pf["rungs"] if x["rung"] == prior["rung"])
        jobs.append(
            dict(
                item=item,
                reviewed=row,
                original_verdict=saved["verdict"],
                derived=derived.get(row["key"]),
                rung=prior["rung"],
                output=str(args.output),
            )
        )
    (args.output / "experimental_score_snapshot.py").write_text(EXPERIMENTAL_SOURCE)
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(run_case, jobs):
            results.append(row)
            if row.get("changes") or row["status"] != "measured":
                print(
                    row["key"],
                    row["status"],
                    row.get("before", {}).get("accepted_flight_count"),
                    "->",
                    row.get("after", {}).get("accepted_flight_count"),
                    row.get("error", ""),
                    flush=True,
                )
            (args.output / "report.json").write_text(
                json.dumps(dict(status="running", results=results), indent=2) + "\n"
            )
    attempts = deepcopy(prior["attempts"])
    lookup = {x["key"]: x for x in results}
    for row in attempts:
        result = lookup.get(row["key"], {})
        if result.get("status") == "measured":
            v = result["after"]
            row["after"].update(
                accepted_flights=v["accepted_flight_count"],
                complete_point=v["complete_point"],
                partial_point=v["partial_point"],
                gaps=v["gaps"],
                failure_counts=v["failure_counts"],
            )
    report = dict(
        status="complete_native_change_review_pending",
        automatic_inference_eligible=False,
        promoted=False,
        scope="Opened labeled development; one declared serve-rule scope correction. No parameter, label, event, physical law, witness or numeric threshold changes.",
        results=results,
        attempts=attempts,
        source=[
            provenance.file_record(p)
            for p in [
                args.composition,
                oldpath,
                basepath,
                Path(__file__),
                Path(census.__file__),
                Path(full.profile.acceptance.__file__),
                args.output / "experimental_score_snapshot.py",
            ]
        ],
        summary=dict(
            attempt_denominator=len(attempts),
            fitted_replays=len(results),
            no_fit_unchanged=len(attempts) - len(results),
            before_complete=41,
            before_flights=292,
            complete_points=sum(x["after"]["complete_point"] for x in attempts),
            accepted_flights=sum(x["after"]["accepted_flights"] for x in attempts),
            flight_denominator=sum(x["labeled_flights"] or 0 for x in attempts),
        ),
        execution_failures=[x["key"] for x in results if x["status"] != "measured"],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
