import json
import os
import subprocess
import sys

import pytest

from cv.pipeline import gpu_lease


def _status(temp=40, used=0, total=98000, util=0, throttle=0):
    return {
        "temp_c": temp,
        "util_percent": util,
        "mem_used_mib": used,
        "mem_total_mib": total,
        "sm_mhz": 2000,
        "throttle": throttle,
    }


def _spec(tmp_path, **extra):
    return {
        "dir": str(tmp_path / "leases"),
        "gpus": {"1": 6, "2": 6},
        "stage_units": {"tracking": 3},
        "default_units": 2,
        "hot_c": 83,
        "min_free_gb_per_unit": 15,
        **extra,
    }


def test_leases_fill_the_least_loaded_gpu_and_stop_at_capacity(tmp_path):
    spec = _spec(tmp_path)
    probe = lambda: {1: _status(), 2: _status()}  # noqa: E731
    first, _ = gpu_lease.try_acquire(spec, "tracking", probe)
    second, _ = gpu_lease.try_acquire(spec, "tracking", probe)
    assert {first.gpu, second.gpu} == {"1", "2"} and first.units == 3
    third, _ = gpu_lease.try_acquire(spec, "player_detection", probe)
    fourth, _ = gpu_lease.try_acquire(spec, "player_detection", probe)
    assert {third.gpu, fourth.gpu} == {"1", "2"} and third.units == 2
    # Each card now holds 5 of 6 units: another tracking or light lease does not fit.
    none, reason = gpu_lease.try_acquire(spec, "event_inference", probe)
    assert none is None and reason == "no_free_units"
    first.release()
    again, _ = gpu_lease.try_acquire(spec, "tracking", probe)
    assert again.gpu == first.gpu
    for lease in (second, third, fourth, again):
        lease.release()


def test_hot_or_full_gpus_are_avoided_and_hot_takes_one_only_when_idle(tmp_path):
    spec = _spec(tmp_path)
    hot = lambda: {1: _status(temp=88), 2: _status(used=90000)}  # noqa: E731
    lease, _ = gpu_lease.try_acquire(spec, "tracking", hot)
    assert lease.gpu == "1"  # GPU 2 lacks 45 GB free; hot GPU 1 holds none of ours yet
    none, reason = gpu_lease.try_acquire(spec, "player_detection", hot)
    assert none is None and reason == "gpu_hot_or_full"
    lease.release()
    throttled = lambda: {1: _status(throttle=0x20), 2: _status(util=50)}  # noqa: E731
    lease, _ = gpu_lease.try_acquire(spec, "tracking", throttled)
    assert lease.gpu == "2"
    lease.release()


def test_small_cards_run_one_stage_at_a_time(tmp_path):
    spec = _spec(tmp_path, gpus={"0": 3}, min_free_gb_per_unit=0)
    probe = lambda: {}  # noqa: E731
    light, _ = gpu_lease.try_acquire(spec, "player_detection", probe)
    assert light.units == 2
    blocked, _ = gpu_lease.try_acquire(spec, "event_inference", probe)
    assert blocked is None
    light.release()
    tracking, _ = gpu_lease.try_acquire(spec, "tracking", probe)
    assert tracking.units == 3
    tracking.release()


def test_leasing_is_off_without_the_spec_and_not_taken_twice(tmp_path, monkeypatch):
    monkeypatch.delenv(gpu_lease.ENV_SPEC, raising=False)
    with gpu_lease.lease("tracking") as override:
        assert override == {}
    monkeypatch.setenv(gpu_lease.ENV_SPEC, json.dumps(_spec(tmp_path)))
    monkeypatch.setenv(gpu_lease.ENV_LOG, str(tmp_path / "leases.jsonl"))
    monkeypatch.delenv(gpu_lease.ENV_HELD, raising=False)
    monkeypatch.setattr(gpu_lease, "probe_gpus", lambda: {})
    with gpu_lease.environment("tracking"):
        gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        assert os.environ[gpu_lease.ENV_HELD] == gpu
        with gpu_lease.lease("event_inference") as nested:
            assert nested == {}  # the holder's children use its lease
        out = gpu_lease.run(
            [sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
            stage="event_inference",
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert out == gpu
    assert gpu_lease.ENV_HELD not in os.environ
    record = json.loads((tmp_path / "leases.jsonl").read_text().splitlines()[0])
    assert record["stage"] == "tracking" and record["gpu"] == gpu and record["units"] == 3


def test_a_dead_holder_frees_its_lease_and_other_processes_wait(tmp_path):
    spec = _spec(tmp_path, gpus={"1": 3})
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json, sys, time\n"
            "from cv.pipeline import gpu_lease\n"
            f"lease, _ = gpu_lease.try_acquire(json.loads({json.dumps(json.dumps(spec))}), "
            "'tracking', lambda: {})\n"
            "print(lease.gpu, flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "1"
        blocked, reason = gpu_lease.try_acquire(spec, "tracking", lambda: {})
        assert blocked is None and reason == "no_free_units"
    finally:
        holder.kill()
        holder.wait()
    lease, _ = gpu_lease.try_acquire(spec, "tracking", lambda: {})
    assert lease is not None
    lease.release()


def test_acquire_waits_and_records_why(tmp_path):
    spec = _spec(tmp_path, gpus={"1": 3})
    held, _ = gpu_lease.try_acquire(spec, "tracking", lambda: {})
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        held.release()

    lease, wait = gpu_lease.acquire(spec, "tracking", probe=lambda: {}, sleep=sleep)
    assert lease.gpu == "1" and wait["blocks"] == {"no_free_units": 1} and sleeps == [10.0]
    lease.release()


@pytest.mark.parametrize("stage", ["tracking", "player_boxes", "event_inference"])
def test_broadcast_runner_leases_gpu_stages_only(stage):
    from cv.pipeline import broadcast_runner

    assert stage in broadcast_runner.GPU_STAGES
    assert "postseg_base" not in broadcast_runner.GPU_STAGES
    assert "point_gates" not in broadcast_runner.GPU_STAGES


def test_stage_memory_needs_hold_a_stage_until_the_card_has_room(tmp_path):
    spec = _spec(
        tmp_path,
        min_free_gb_per_unit=0,
        stage_free_gb={"tracking": 26.0, "player_detection": 7.5},
        default_free_gb=10.0,
        reserve_gb=2.0,
    )
    busy = lambda: {1: _status(used=80000, total=98000), 2: _status(used=90000)}  # noqa: E731
    none, reason = gpu_lease.try_acquire(spec, "tracking", busy)  # 17.6 GB free < 26
    assert none is None and reason == "gpu_hot_or_full"
    lease, _ = gpu_lease.try_acquire(spec, "player_detection", busy)
    assert lease.gpu == "1"
    lease.release()
    # A 10 GB card: the need is capped at 8 GB (its size less the reserve).
    small = lambda: {1: _status(used=2000, total=10240), 2: _status(used=9000, total=10240)}  # noqa: E731
    lease, _ = gpu_lease.try_acquire(spec, "tracking", small)
    assert lease.gpu == "1"
    lease.release()
    assert gpu_lease.stage_need_mib({}, "tracking", 10240) == 0.0


def test_court_geometry_leases_only_with_the_court_gpu_package(monkeypatch):
    from cv.pipeline import canonical_runner as cr

    ran = []
    monkeypatch.setattr(cr.gpu_lease, "run", lambda command, **kw: ran.append(("lease", kw)))
    monkeypatch.setattr(cr, "_run", lambda command: ran.append(("plain", None)))
    for key in ("TENNIS_COURT_SIFT", "TENNIS_COURT_SUPPORT"):
        monkeypatch.delenv(key, raising=False)
    cr._run_stage("court_geometry", ["x"])
    monkeypatch.setenv("TENNIS_COURT_SIFT", "cuda")
    cr._run_stage("court_geometry", ["x"])
    cr._run_stage("court_geometry", ["x", "--jobs", "8"])
    cr._run_stage("camera_calibration", ["x"])
    assert ran == [
        ("plain", None),
        ("lease", {"stage": "court_geometry", "need_gb": 20.0, "check": True}),
        ("lease", {"stage": "court_geometry", "need_gb": 40.0, "check": True}),
        ("plain", None),
    ]


def test_a_caller_need_holds_the_lease_on_every_box(tmp_path):
    # The local box's spec has no stage_free_gb; court geometry's own need still applies.
    spec = _spec(tmp_path, min_free_gb_per_unit=0, reserve_gb=2.0)
    busy = lambda: {1: _status(used=60000), 2: _status(used=70000)}  # noqa: E731 (~37, ~27 GB)
    none, reason = gpu_lease.try_acquire(spec, "court_geometry", busy, need_gb=40.0)
    assert none is None and reason == "gpu_hot_or_full"
    lease, _ = gpu_lease.try_acquire(spec, "court_geometry", busy, need_gb=30.0)
    assert lease.gpu == "1"
    lease.release()
    # Capped at the card less its reserve: a 32 GB card waits until it is all but empty.
    assert gpu_lease.stage_need_mib(spec, "court_geometry", 32607, 40.0) == 32607 - 2048
    assert gpu_lease.stage_need_mib(spec, "court_geometry", 98000) == 0.0


def test_a_closed_cohort_gate_holds_the_lease_until_it_opens(tmp_path):
    gate = tmp_path / "leases" / "cohort-x-2.open"
    spec = _spec(tmp_path, gate=str(gate))
    probe = lambda: {1: _status(), 2: _status()}  # noqa: E731
    assert gpu_lease.try_acquire(spec, "tracking", probe) == (None, "cohort_wait")
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.touch()
    held, reason = gpu_lease.try_acquire(spec, "tracking", probe)
    assert reason == "ok"
    held.release()
