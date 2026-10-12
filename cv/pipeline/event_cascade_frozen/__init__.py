"""Frozen evidence renderer and prompts of the FREEZE v2 event cascade.

Verbatim copies, so the product path does not import experiment directories or
scripts that live under the data root. Only the bare-name imports were made
package-relative; no line of logic changed. Sources:

- ``fairlib``, ``render_evidence``, ``run_label``, ``guided_procedure``:
  ``cv/experiments/labeller_fair_evidence`` (Qwen ``guided_nothink`` and the hosted
  ``zero`` profiles, the N-shot demonstrations, the 5x5 native-crop grids).
- ``or_call``: ``cv/experiments/vlm_labeller_axis/or_call.py`` (OpenRouter URL and the
  key read in-process from the local ``.env``; the key is never written anywhere).
- ``trace``, ``trace_from_track``: ``$TENNIS_DATA_ROOT/processed/vlm_event_benchmark_v1/scripts``.

``test_event_cascade_frozen.py`` checks the copies against their sources where those
still exist, and that a rendered candidate is byte-identical to the recorded evidence.
"""
