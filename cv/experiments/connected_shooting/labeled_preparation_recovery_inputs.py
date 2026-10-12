"""Explicit preparation controls using existing seed and player recovery code.

Run ``python -m ...labeled_preparation_recovery_inputs --help``. All arguments
after ``--`` are ordinary whole-point-search arguments. No labels, solver bounds,
native observations, fit objective, or acceptance thresholds are changed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import postbounce_initialization
from cv.pipeline import provenance


@contextmanager
def recovery_scope(*, launch_feasibility: bool, contact_player_feet: bool, receipts: list):
    """Install explicit process-local adapters and always restore original callables."""
    original_seed = postbounce_initialization.seed
    original_players = whole.alternating_player_states
    original_feet = whole.toss_witness.server_feet
    associated = []

    def seed(*args, **kwargs):
        kwargs["recover_launch_feasibility"] = True
        result = original_seed(*args, **kwargs)
        receipts.append({"adapter": "existing_bounded_postbounce_seed", "evidence": result[1]})
        return result

    def players(*args, **kwargs):
        result = original_players(*args, **kwargs)
        associated[:] = result
        return result

    def feet(*args, **kwargs):
        try:
            return original_feet(*args, **kwargs)
        except ValueError as error:
            if not associated or "server feet lack a short pre-contact automatic track" not in str(
                error
            ):
                raise
            result = whole.contact_player_feet_fallback(associated[0], error)
            # This is the existing associated contact box, not a newly measured
            # pre-contact track. Preserve that distinction explicitly.
            receipt = {
                "adapter": "existing_contact_player_state_for_optional_feet",
                "precontact_track_available": False,
                "source_player": associated[0],
                "result": result,
            }
            receipts.append(receipt)
            if kwargs.get("fallback_receipt") is not None:
                kwargs["fallback_receipt"].append(receipt)
            return result

    try:
        if launch_feasibility:
            postbounce_initialization.seed = seed
        if contact_player_feet:
            whole.alternating_player_states = players
            whole.toss_witness.server_feet = feet
        yield
    finally:
        postbounce_initialization.seed = original_seed
        whole.alternating_player_states = original_players
        whole.toss_witness.server_feet = original_feet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-feasibility", action="store_true")
    parser.add_argument("--contact-player-feet", action="store_true")
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if not (args.launch_feasibility or args.contact_player_feet):
        parser.error("at least one explicit recovery control required")
    output = Path(remaining[remaining.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "human_derived": True,
        "automatic_inference_eligible": False,
        "producer": provenance.file_record(Path(__file__)),
        "dependencies": [
            provenance.file_record(Path(module.__file__))
            for module in (whole, postbounce_initialization, whole.toss_witness)
        ],
        "arguments": remaining,
        "controls": vars(args),
        "applications": [],
        "status": "running",
    }
    path = output.parent / "input_recovery_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    old = sys.argv
    try:
        with recovery_scope(**vars(args), receipts=manifest["applications"]):
            sys.argv = [__file__, *remaining]
            whole.main()
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        sys.argv = old
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
