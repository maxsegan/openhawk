import os
import subprocess
import sys

import pytest

from cv.pipeline import math_threads


def test_unset_switch_changes_nothing():
    assert math_threads.threads({}) is None
    assert math_threads.environment({}) == {}
    assert math_threads.environment({"TENNIS_MATH_THREADS": "off"}) == {}


def test_switch_sets_every_pool_and_records_itself():
    env = math_threads.environment({"TENNIS_MATH_THREADS": "2"})
    assert env["TENNIS_MATH_THREADS"] == "2"
    assert {env[name] for name in math_threads.VARIABLES} == {"2"}


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_bad_values_are_refused(value):
    with pytest.raises(ValueError):
        math_threads.threads({"TENNIS_MATH_THREADS": value})


def test_apply_limits_this_process_and_its_children():
    code = (
        "import numpy, subprocess, sys, threadpoolctl\n"
        "from cv.pipeline import math_threads\n"
        "math_threads.apply()\n"
        "print(sorted({i['num_threads'] for i in threadpoolctl.threadpool_info()}))\n"
        "print(subprocess.run([sys.executable, '-c', 'import os; "
        'print(os.environ["OPENBLAS_NUM_THREADS"])\'], capture_output=True, text=True).stdout)'
    )
    env = {k: v for k, v in os.environ.items() if k not in math_threads.VARIABLES}
    env["TENNIS_MATH_THREADS"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["[1]", "1"]
