import os
import signal
import subprocess
import sys

import pytest

from cv.pipeline import s6_stage_forkserver as forkserver


@pytest.fixture(autouse=True)
def _servers():
    yield
    forkserver.shutdown()


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def test_child_runs_a_module_as_main_with_its_args_and_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("")
    process = forkserver.StageProcess(
        ["-n", "1", "-r", "1", "print('hello from timeit')"], log=log, env=_env(), module="timeit"
    )
    assert process.wait(30) == 0
    assert "hello from timeit" in log.read_text()


def test_exit_code_and_traceback_reach_the_caller(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("")
    process = forkserver.StageProcess(["--no-such-flag"], log=log, env=_env())
    assert process.wait(60) == 2  # argparse error, as `python -m` would exit
    assert "usage" in log.read_text()


def test_timeout_and_group_kill(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("")
    process = forkserver.StageProcess(
        ["-n", "1", "-r", "1", "import time; time.sleep(60)"], log=log, env=_env(), module="timeit"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(0.5)
    assert os.getpgid(process.pid) == process.pid  # its own session, as start_new_session
    os.killpg(process.pid, signal.SIGKILL)
    assert process.wait(30) == -signal.SIGKILL


def test_popen_uses_subprocess_when_off_and_for_other_commands(tmp_path, monkeypatch):
    log = tmp_path / "run.log"
    monkeypatch.setenv(forkserver.SWITCH, "on")
    with log.open("w") as stream:
        process = forkserver.popen(
            [sys.executable, "-c", "print(1)"], stream=stream, log=log, env=_env()
        )
    assert isinstance(process, subprocess.Popen) and process.wait() == 0
    monkeypatch.setenv(forkserver.SWITCH, "off")
    command = [sys.executable, "-m", forkserver.MODULE, "--help"]
    with log.open("w") as stream:
        process = forkserver.popen(command, stream=stream, log=log, env=_env())
    assert isinstance(process, subprocess.Popen) and process.wait() == 0
    monkeypatch.setenv(forkserver.SWITCH, "on")
    with log.open("w") as stream:
        process = forkserver.popen(command, stream=stream, log=log, env=_env())
    assert isinstance(process, forkserver.StageProcess) and process.wait(60) == 0
    assert "--job" in log.read_text()


def test_preload_never_loads_an_implementation_module():
    code = (
        "import importlib, sys\n"
        "from cv.pipeline import s6_stage_forkserver as f\n"
        "for m in f.PRELOAD: importlib.import_module(m)\n"
        "print([m for m in f.identity_modules() if m != 'cv.pipeline.s6_stage_forkserver'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=_env(), capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_children_cpu_reaches_the_caller_after_shutdown(tmp_path):
    import resource

    log = tmp_path / "run.log"
    log.write_text("")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    burn = "x = 0\nfor i in range(3_000_000): x += i"
    process = forkserver.StageProcess(
        ["-n", "1", "-r", "1", burn], log=log, env=_env(), module="timeit"
    )
    assert process.wait(60) == 0
    forkserver.shutdown()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    assert after.ru_utime - before.ru_utime > 0.05
