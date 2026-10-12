"""Shared GPU fit server: many CPU fit processes, one GPU context, merged launches.

Question: when many component/attempt processes each need small batches of S6 solves, does one
server per GPU that merges their concurrent requests into single kernel launches keep the card
full (solves/s near the single-process saturation plateau)?

    python -m cv.experiments.gpu_fit.server --socket /tmp/s6_gpu_fit.sock [--device N] [--window-ms 3]

Clients: set ``TENNIS_S6_GPU_FIT_SERVER=/tmp/s6_gpu_fit.sock`` and ``gpu_fit.solve`` sends its
launch there instead of opening its own CUDA context. Each request carries its own problems,
problem index, starts and options; the server groups queued requests with equal options, packs
their problems into one launch, and returns each client exactly its own rows (in order).
A failed launch returns the error to every request in it. Research prototype: no auth beyond
the socket's filesystem permissions.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from multiprocessing.connection import Client, Listener

import numpy as np

from cv.experiments.gpu_fit import gpu_fit

AUTHKEY = b"s6-gpu-fit"
MERGE_ENV = "TENNIS_S6_GPU_FIT_SERVER_MERGE"
STREAMS_ENV = "TENNIS_S6_GPU_FIT_SERVER_STREAMS"


class Stats:
    """Server counters, logged as one JSON line every ``--stats-seconds`` (deltas per interval).

    Per kind (``solve``/``evaluate``): requests, items (seeds or points), launches, the requests
    merged per launch, queue wait (enqueue to launch start) and per device the launch-busy
    seconds; plus the queue length and the server's own CPU use (a server near one core is
    bound by its Python loop, not by the cards)."""

    KINDS = ("solve", "evaluate")

    def __init__(self, devices):
        self.lock = threading.Lock()
        self.devices = list(devices)
        self.started = time.time()
        self.total = self._zero()
        self.interval = self._zero()

    def _zero(self) -> dict:
        row = {
            kind: {"requests": 0, "items": 0, "launches": 0, "wait_s": 0.0, "wait_max_s": 0.0}
            for kind in self.KINDS
        }
        row["busy_s"] = {str(device): 0.0 for device in self.devices}
        row["errors"] = 0
        return row

    def launch(self, kind, device, group, items, started, finished, failed=False):
        with self.lock:
            for row in (self.total, self.interval):
                entry = row[kind]
                entry["requests"] += len(group)
                entry["items"] += items
                entry["launches"] += 1
                for item in group:
                    wait = started - item[2]
                    entry["wait_s"] += wait
                    entry["wait_max_s"] = max(entry["wait_max_s"], wait)
                row["busy_s"][str(device)] += finished - started
                row["errors"] += int(failed)

    def snapshot(self, seconds: float, queued: int, cpu_s: float) -> dict:
        with self.lock:
            row, self.interval = self.interval, self._zero()
        out = {"at": round(time.time(), 1), "interval_s": round(seconds, 1), "queued": queued}
        for kind in self.KINDS:
            entry = row[kind]
            n, launches = entry["requests"], entry["launches"]
            out[kind] = {
                "requests_per_s": round(n / seconds, 1),
                "items_per_s": round(entry["items"] / seconds, 1),
                "launches": launches,
                "requests_per_launch": round(n / launches, 2) if launches else None,
                "items_per_launch": round(entry["items"] / launches, 1) if launches else None,
                "wait_mean_ms": round(1e3 * entry["wait_s"] / n, 2) if n else None,
                "wait_max_ms": round(1e3 * entry["wait_max_s"], 1),
            }
        out["busy_fraction"] = {k: round(v / seconds, 3) for k, v in row["busy_s"].items()}
        out["server_cpu_fraction"] = round(cpu_s / seconds, 3)
        out["errors"] = row["errors"]
        return out


def _log_stats(stats: Stats, requests: queue.Queue, seconds: float) -> None:
    last, last_cpu = time.perf_counter(), time.process_time()
    while True:
        time.sleep(seconds)
        now, cpu = time.perf_counter(), time.process_time()
        row = stats.snapshot(now - last, requests.qsize(), cpu - last_cpu)
        last, last_cpu = now, cpu
        print("stats " + json.dumps(row), flush=True)


def _serve_client(connection, requests: queue.Queue):
    try:
        while True:
            message = connection.recv()
            reply = queue.Queue(maxsize=1)
            requests.put((message, reply, time.perf_counter()))
            connection.send(reply.get())
    except (EOFError, OSError):
        pass
    finally:
        connection.close()


def _kind(item) -> str:
    return "evaluate" if item[0].get("kind") == "evaluate" else "solve"


def _launch(group, device, stats):
    """One kernel launch for requests with identical options."""
    problems, index, starts, spans = [], [], [], []
    for message, _, _ in group:
        offset = len(problems)
        problems.extend(message["problems"])
        index.extend(int(i) + offset for i in message["index"])
        starts.extend(message["seeds"])
        spans.append(len(message["index"]))
    options = group[0][0]["options"]
    started = time.perf_counter()
    try:
        results, kernel_ms, total_ms = gpu_fit.solve_local(
            np.array(problems), np.asarray(index, np.int32), starts, device=device, **options
        )
    except Exception as error:  # every request in the launch sees the failure
        stats.launch("solve", device, group, len(index), started, time.perf_counter(), True)
        for _, reply, _ in group:
            reply.put({"error": repr(error)})
        return
    stats.launch("solve", device, group, len(index), started, time.perf_counter())
    cursor = 0
    for (_, reply, _), span in zip(group, spans, strict=True):
        reply.put(
            {
                "results": results[cursor : cursor + span],
                "kernel_ms": kernel_ms,
                "total_ms": total_ms,
                "merged_requests": len(group),
                "merged_solves": len(index),
            }
        )
        cursor += span


def _evaluate(group, device, stats):
    """Evaluation requests of one precision. One request: exactly ``gpu_fit.evaluate_local``
    as the client would run it; several (``--merge-evaluate``): one ``evaluate_many`` launch,
    which returns each request the same rows bit for bit."""
    messages = [message for message, _, _ in group]
    points = sum(len(message["qs"]) for message in messages)
    started = time.perf_counter()
    try:
        if len(group) == 1:
            message = messages[0]
            answers = [
                gpu_fit.evaluate_local(
                    message["problem"],
                    message["qs"],
                    device=device,
                    precision=message["precision"],
                    raw=message["raw"],
                )
            ]
        else:
            answers = gpu_fit.evaluate_many(
                [(m["problem"], m["qs"], m["raw"]) for m in messages],
                device=device,
                precision=messages[0]["precision"],
            )
    except Exception as error:
        stats.launch("evaluate", device, group, points, started, time.perf_counter(), True)
        for _, reply, _ in group:
            reply.put({"error": repr(error)})
        return
    stats.launch("evaluate", device, group, points, started, time.perf_counter())
    for (_, reply, _), answer in zip(group, answers, strict=True):
        reply.put({"evaluation": answer, "merged_requests": len(group)})


def _dispatch(
    requests: queue.Queue, device: int, window_ms: float, max_solves: int, merge: bool, stats
):
    """One card's loop: take queued requests, merge them, launch, reply.

    ``merge`` off is the original server: an evaluation runs alone as soon as it is taken, and
    solves arriving inside the window share a launch. On: everything queued when the card frees
    or arriving inside a solve window is taken; evaluations of one precision go in one launch,
    ahead of the window's solves.
    """
    try:
        gpu_fit.kernel_info(device)  # create this card's context and load the kernel once
    except Exception as error:  # the other cards' dispatchers keep serving the queue
        print(f"WARNING device {device} not serving: {error!r}", flush=True)
        return
    while True:
        first = requests.get()
        if _kind(first) == "evaluate" and not merge:
            _evaluate([first], device, stats)
            continue
        pending = [first]
        waiting = _kind(first) == "solve"
        deadline = time.perf_counter() + window_ms / 1e3
        while sum(len(m["index"]) for m, _, _ in pending if "index" in m) < max_solves:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 and not merge:
                break  # the original server: nothing past the solve window
            try:
                if waiting and remaining > 0:
                    item = requests.get(timeout=remaining)
                else:
                    item = requests.get_nowait()
            except queue.Empty:
                break
            if _kind(item) == "evaluate" and not merge:
                _evaluate([item], device, stats)
            else:
                pending.append(item)
        evaluations: dict = {}
        groups: dict = {}
        for item in pending:
            if _kind(item) == "evaluate":
                evaluations.setdefault(item[0]["precision"], []).append(item)
            else:
                key = tuple(sorted(item[0]["options"].items()))
                groups.setdefault(key, []).append(item)
        for group in evaluations.values():
            _evaluate(group, device, stats)
        for group in groups.values():
            _launch(group, device, stats)


def serve(
    socket_path: str,
    devices,
    window_ms: float,
    max_solves: int,
    merge_evaluate: bool = False,
    stats_seconds: float = 30.0,
    streams: int = 1,
):
    """Serve ``devices``: ``streams`` dispatcher threads per card, all taking from one queue.

    Each dispatcher launches on its own thread's stream, so with ``streams`` > 1 a card runs
    several launches at once (a launch lasts as long as its slowest problem and leaves most of
    the card idle). 1 is the original one-launch-at-a-time server."""
    devices = [devices] if isinstance(devices, int) else list(devices)
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    listener = Listener(socket_path, family="AF_UNIX", authkey=AUTHKEY)
    requests: queue.Queue = queue.Queue()
    stats = Stats(devices)
    gpu_fit.kernel_info(devices[0])  # fail before serving when the kernel cannot load
    for slot in range(max(1, streams)):
        for device in devices:
            threading.Thread(
                target=_dispatch,
                args=(requests, device, window_ms, max_solves, merge_evaluate, stats),
                name=f"dispatch-{device}-{slot}",
                daemon=True,
            ).start()
    if stats_seconds > 0:
        threading.Thread(
            target=_log_stats, args=(stats, requests, stats_seconds), daemon=True
        ).start()

    def accept():
        while True:
            connection = listener.accept()
            threading.Thread(target=_serve_client, args=(connection, requests), daemon=True).start()

    print(
        f"serving {socket_path} on devices {','.join(map(str, devices))} "
        f"(merge_evaluate={'on' if merge_evaluate else 'off'}, streams={streams}, "
        f"window_ms={window_ms})",
        flush=True,
    )
    accept()


_CLIENT = None
_CLIENT_LOCK = threading.Lock()  # one connection per process; a request is a send and its reply


class ServerLost(ConnectionError):
    """The fit server closed the connection mid-request (it died or was stopped).

    An ``OSError``, as an unreachable server's ``FileNotFoundError`` is: the hooks answer the
    request on the CPU and record the fallback. The dead connection is dropped, so the next
    request connects afresh (to a restarted server, or fails fast and falls back again)."""


def _exchange(socket_path: str, message: dict) -> dict:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = Client(socket_path, family="AF_UNIX", authkey=AUTHKEY)
        try:
            _CLIENT.send(message)
            answer = _CLIENT.recv()
        except (EOFError, OSError) as error:  # EOF, reset, broken pipe: the server is gone
            try:
                _CLIENT.close()
            except OSError:
                pass
            _CLIENT = None
            raise ServerLost(f"fit server connection lost: {type(error).__name__}") from error
    if "error" in answer:
        raise RuntimeError(f"GPU fit server: {answer['error']}")
    return answer


def request(socket_path: str, problems, index, seeds, options: dict):
    """Client side: one launch request; returns the same tuple as ``gpu_fit.solve``."""
    answer = _exchange(
        socket_path,
        {
            "problems": list(np.atleast_1d(problems)),
            "index": np.asarray(index, np.int32).tolist(),
            "seeds": [np.asarray(s, float) for s in seeds],
            "options": options,
        },
    )
    return answer["results"], answer["kernel_ms"], answer["total_ms"]


def evaluate_request(socket_path: str, problem, qs, precision: int, raw: bool):
    """Client side: ``gpu_fit.evaluate`` on the server's context; the same tuple, bit for bit."""
    message = {
        "kind": "evaluate",
        "problem": np.atleast_1d(problem),
        "qs": np.atleast_2d(np.asarray(qs, float)),
        "precision": precision,
        "raw": raw,
    }
    return _exchange(socket_path, message)["evaluation"]


def _devices(text: str | None) -> list[int]:
    if text is None:
        return [gpu_fit.default_device()]
    return [int(part) for part in str(text).split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="card, or comma list of cards (one dispatcher each); default TENNIS_S6_GPU_FIT_DEVICE",
    )
    parser.add_argument("--window-ms", type=float, default=3.0)
    parser.add_argument("--max-solves", type=int, default=20480)
    parser.add_argument(
        "--merge-evaluate",
        choices=("on", "off"),
        default=os.environ.get(MERGE_ENV, "off"),
        help=f"merge queued evaluations into one launch (default {MERGE_ENV} or off)",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=int(os.environ.get(STREAMS_ENV, "1")),
        help=f"concurrent launches per card (default {STREAMS_ENV} or 1)",
    )
    parser.add_argument("--stats-seconds", type=float, default=30.0, help="0: no stats lines")
    args = parser.parse_args()
    serve(
        args.socket,
        _devices(args.device),
        args.window_ms,
        args.max_solves,
        merge_evaluate=args.merge_evaluate == "on",
        stats_seconds=args.stats_seconds,
        streams=args.streams,
    )


if __name__ == "__main__":
    main()
