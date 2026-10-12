"""The pose step's GPU guard (fix-list 27): box-wide lock, free-memory wait, OOM retry."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from argparse import Namespace

import pytest

from cv.pipeline import player_pose_3d, pose_gpu

OOM_TEXT = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.97 GiB.\n"


def config(tmp_path, **overrides) -> dict:
    return {
        "lock_dir": str(tmp_path / "box_slots" / "local-test"),
        "free_gb": 22.0,
        "free_wait_s": 1800.0,
        "attempts": 3,
        "backoff_s": 0.0,
        **overrides,
    }


def card(free_gb: float, total_gb: float = 95.0) -> dict:
    total = int(total_gb * 1024)
    return {0: {"mem_total_mib": total, "mem_used_mib": total - int(free_gb * 1024)}}


def test_two_concurrent_pose_runs_serialize_on_the_box_lock(tmp_path):
    active, peak, lock = [0], [0], threading.Lock()

    def attempt():
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        return 0, ""

    results = []

    def one():
        results.append(
            pose_gpu.guarded_run(attempt, gpu=0, config=config(tmp_path), probe=lambda: card(40))
        )

    threads = [threading.Thread(target=one) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 1
    assert [code for code, _, _ in results] == [0, 0]
    waits = sorted(attempts[0]["lock_wait_s"] for _, attempts, _ in results)
    assert waits[0] < 0.15 and waits[1] >= 0.15  # the second waited for the first
    assert all(attempts[0]["lock"].endswith("local-test/pose.lock") for _, attempts, _ in results)


def test_oom_on_first_attempt_is_retried_and_recorded(tmp_path):
    outcomes = iter([(1, "Traceback\n" + OOM_TEXT), (0, "done\n")])
    sleeps = []
    record = tmp_path / "logs" / "pose_attempts.jsonl"
    code, attempts, used = pose_gpu.guarded_run(
        lambda: next(outcomes),
        gpu=0,
        config=config(tmp_path, backoff_s=30.0),
        record=record,
        probe=lambda: card(40),
        sleep=sleeps.append,
    )
    assert code == 0 and len(attempts) == 2
    assert attempts[0]["oom"] and attempts[0]["backoff_s"] == 30.0
    assert not attempts[1]["oom"] and attempts[1]["returncode"] == 0
    assert sleeps == [30.0]
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    assert [line["attempt"] for line in lines] == [1, 2]
    assert used["attempts"] == 3


def test_non_oom_failure_is_not_retried(tmp_path):
    calls = []

    def attempt():
        calls.append(1)
        return 1, "ValueError: no staged documents\n"

    code, attempts, _ = pose_gpu.guarded_run(
        attempt, gpu=0, config=config(tmp_path), probe=lambda: card(40), sleep=lambda s: None
    )
    assert code == 1 and len(calls) == 1 and not attempts[0]["oom"]


def test_exhausted_oom_retries_fail_loudly(tmp_path):
    record = tmp_path / "attempts.jsonl"
    with pytest.raises(pose_gpu.PoseOutOfMemory, match="CUDA out of memory on all 2 attempts"):
        pose_gpu.guarded_run(
            lambda: (1, OOM_TEXT),
            gpu=0,
            config=config(tmp_path, attempts=2, backoff_s=5.0),
            record=record,
            probe=lambda: card(40),
            sleep=lambda s: None,
        )
    assert len(record.read_text().splitlines()) == 2


def test_free_memory_wait_is_honoured(tmp_path):
    frees = iter([5.0, 10.0, 30.0])
    sleeps, started = [], []

    def attempt():
        started.append(len(sleeps))
        return 0, ""

    _, attempts, _ = pose_gpu.guarded_run(
        attempt,
        gpu=0,
        config=config(tmp_path),
        probe=lambda: card(next(frees)),
        sleep=sleeps.append,
    )
    assert sleeps == [pose_gpu.POLL_SECONDS] * 2 and started == [2]
    assert attempts[0]["free_check"] == "ok" and attempts[0]["free_polls"] == 2


def test_free_memory_wait_is_bounded_and_recorded():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    wait = pose_gpu.wait_for_free(
        0, 22.0, 60.0, probe=lambda: card(5), sleep=sleep, clock=lambda: now[0]
    )
    assert wait["free_check"] == "timeout" and wait["free_wait_s"] >= 60.0
    assert pose_gpu.wait_for_free(0, 22.0, 60.0, probe=dict)["free_check"] == "no_probe"
    assert pose_gpu.wait_for_free(None, 22.0, 60.0)["free_check"] == "off"


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    for key in (
        pose_gpu.ENV_LOCK_DIR,
        pose_gpu.ENV_FREE_GB,
        pose_gpu.ENV_ATTEMPTS,
        pose_gpu.ENV_BACKOFF_S,
        pose_gpu.ENV_FREE_WAIT_S,
    ):
        monkeypatch.delenv(key, raising=False)
    default = pose_gpu.settings()
    assert default["lock_dir"].startswith(str(tmp_path / "processed" / "box_slots" / "local-"))
    assert (default["free_gb"], default["attempts"]) == (22.0, 3)
    monkeypatch.setenv(pose_gpu.ENV_LOCK_DIR, "off")
    monkeypatch.setenv(pose_gpu.ENV_ATTEMPTS, "5")
    assert pose_gpu.settings()["lock_dir"] is None and pose_gpu.settings()["attempts"] == 5


def test_product_pose_step_retries_oom(tmp_path, monkeypatch):
    from cv.pipeline import gpu_lease, product_runner

    monkeypatch.setenv(player_pose_3d.ENV, "on")
    monkeypatch.setenv(pose_gpu.ENV_LOCK_DIR, str(tmp_path / "locks"))
    monkeypatch.setenv(pose_gpu.ENV_BACKOFF_S, "0")
    monkeypatch.delenv(gpu_lease.ENV_SPEC, raising=False)
    monkeypatch.setattr(pose_gpu, "probe_gpus", lambda: card(40))
    (tmp_path / "portal" / "3d" / "data").mkdir(parents=True)
    (tmp_path / "upstream" / "m" / "audit_frames_native_1080").mkdir(parents=True)
    calls = []

    def fake_run(command, *, stage, need_gb=None, **kwargs):
        calls.append((stage, need_gb, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        if len(calls) == 1:
            kwargs["stdout"].write(OOM_TEXT)
            return subprocess.CompletedProcess(command, 1)
        pose = tmp_path / "pose"
        pose.mkdir(exist_ok=True)
        (pose / player_pose_3d.RECEIPT_NAME).write_text(json.dumps({"counts": {"rows": 2}}))
        (pose / player_pose_3d.NPZ_NAME).write_bytes(b"x")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gpu_lease, "run", fake_run)
    args = Namespace(out=tmp_path, match_id="m", gpu=0, gpu_lease=None)
    detail = product_runner.step_pose(args)
    assert detail["status"] == "done" and detail["counts"] == {"rows": 2}
    assert [a["oom"] for a in detail["attempts"]] == [True, False]
    assert calls == [("player_pose", None, "0")] * 2
    assert len((tmp_path / "logs" / "pose_attempts.jsonl").read_text().splitlines()) == 2

    # Every attempt out of memory: the step fails instead of publishing without pose.
    (tmp_path / "pose" / player_pose_3d.RECEIPT_NAME).unlink()
    monkeypatch.setattr(
        gpu_lease,
        "run",
        lambda command, **kw: (
            kw["stdout"].write(OOM_TEXT),
            subprocess.CompletedProcess(command, 1),
        )[1],
    )
    with pytest.raises(pose_gpu.PoseOutOfMemory):
        product_runner.step_pose(args)
