"""Evidence pictures and model calls for the event cascade.

Each candidate is rendered once from native 1080 frames and the automatic ball
track (two 5x5 grids: clean and tracker trail) with the frozen fair-evidence
renderer. Qwen runs on local vLLM servers (no dollars, GPU-seconds recorded);
Gemini Flash and Opus go through OpenRouter with the key read in-process by
the fair-evidence client. Every reply is saved with model, tokens and dollars,
and every paid call is appended to the run's spend ledger, which a hard cap
checks before each call. A reply that already parses is never bought twice.

Prompts, demonstrations and the renderer are the frozen ones in
``cv/pipeline/event_cascade_frozen`` (copied from ``cv/experiments/labeller_fair_evidence``) (profile ``guided_nothink`` for Qwen,
``zero`` for the hosted tiers). With ``opus_evidence=1280`` the Opus tier gets
both grids resized to 1280 px (INTER_AREA), ported from
``cascade_cost/opus_resolution._resized``.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from cv.pipeline.event_cascade import TIER_BY_NAME, opus_side, reply_is_good, reply_profile

PRESENTATION = "trail"
QWEN_PER_PORT = 2
HOSTED_WORKERS = 4
SPEND_LOCK = threading.Lock()


def _fair():
    """The frozen fair-evidence renderer and prompt modules (``event_cascade_frozen``)."""
    from cv.pipeline.event_cascade_frozen import render_evidence, run_label

    return render_evidence, run_label


def reply_path(root: Path, tier: str, draw: int, row: dict, chosen: dict) -> Path:
    return (
        root
        / tier
        / reply_profile(tier, chosen)
        / f"draw{draw}"
        / row["clip"]
        / f"{row['id']}.json"
    )


def render(row: dict, evidence: Path) -> dict:
    """Render one candidate, or reuse a manifest entry whose pictures still exist."""
    render_evidence, _ = _fair()
    target = evidence / row["clip"] / "manifest.json"
    if target.is_file():
        saved = json.loads(target.read_text())
        previous = [item for item in saved["nominations"] if item["id"] == row["id"]]
        if previous and all(Path(image["path"]).is_file() for image in previous[0]["images"]):
            return previous[0]
    folder = Path(row["frame_dir"])
    track = Path(row["track_csv"])
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    if not track.is_file():
        raise FileNotFoundError(track)
    points = render_evidence.tft.load_track(track, row["point"])
    nomination = {
        "clip": row["clip"],
        "id": row["id"],
        "nominated_frame": row["nominated_frame"],
        "nominated_type": row["event_type"],
        "layout": "one",
        "frames_per_image": 1,
    }
    record = render_evidence.render_nomination(
        nomination, points, folder, evidence / row["clip"], layout="one", frames_per_image=1
    )
    record["layout"] = "one"
    record["frames_per_image"] = 1
    saved = (
        json.loads(target.read_text())
        if target.is_file()
        else {"clip": row["clip"], "nominations": []}
    )
    saved["nominations"] = [item for item in saved["nominations"] if item["id"] != row["id"]] + [
        record
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(saved))
    return record


def resized(nom: dict, side: int | None, evidence: Path) -> dict:
    """Both grids at ``side`` px (saving B). ``None`` keeps the full-size grids."""
    nom = json.loads(json.dumps(nom))
    nom["layout"] = "one"
    nom["frames_per_image"] = 1
    if side is None:
        return nom
    import cv2

    for image in nom["images"]:
        source = Path(image["path"])
        dest = evidence.parent / f"s{side}" / nom["clip"] / source.parent.name / source.name
        if not dest.is_file():
            picture = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if picture is None:
                raise FileNotFoundError(source)
            small = cv2.resize(picture, (side, side), interpolation=cv2.INTER_AREA)
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image["path"] = str(dest)
        image["width"] = image["height"] = side
    return nom


def spent(ledger: Path) -> float:
    if not ledger.is_file():
        return 0.0
    return sum(
        float(json.loads(line).get("cost") or 0)
        for line in ledger.read_text().splitlines()
        if line.strip()
    )


def _saved(
    row: dict, tier: str, draw: int, cap: int, last: dict, verdict, started: float, usd: float
) -> dict:
    spec = TIER_BY_NAME[tier]
    usage = last.get("usage") or {}
    return {
        "clip": row["clip"],
        "id": row["id"],
        "attempt": row["attempt"],
        "event_index": row["event_index"],
        "event_type": row["event_type"],
        "frame": row["frame"],
        "nominated_frame": row["nominated_frame"],
        "band": row["band"],
        "model": spec.model,
        "tier": tier,
        "draw": draw,
        "profile": spec.profile,
        "presentation": PRESENTATION,
        "layout": "one",
        "finish_reason": last.get("finish_reason"),
        "max_tokens": cap,
        "verdict": verdict,
        "raw": last.get("raw") or "",
        "usage": usage,
        "error": last.get("error"),
        "usd": usd,
        "dt": round(time.time() - started, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def call(
    row: dict,
    nom: dict,
    tier: str,
    draw: int,
    *,
    root: Path,
    chosen: dict,
    ledger: Path,
    cap_usd: float,
    base_url: str | None = None,
    demos: list[dict] | None = None,
) -> str:
    """One draw of one tier. Qwen needs ``base_url``; hosted tiers use OpenRouter."""
    _, run_label = _fair()
    out = reply_path(root, tier, draw, row, chosen)
    if reply_is_good(out):
        return f"skip {row['id']} {tier} draw {draw}"
    spec = TIER_BY_NAME[tier]
    hosted = tier != "qwen"
    if hosted and base_url is not None:
        raise ValueError("hosted tiers go through OpenRouter")
    if not hosted and base_url is None:
        raise ValueError("Qwen needs a local server")
    if tier == "opus":
        nom = resized(nom, opus_side(chosen), root / "evidence" / "one")
    parts = run_label.build_parts(nom, PRESENTATION, spec.profile, demos or [])
    if hosted:
        image_extra, body_extra = run_label.provider_kwargs(spec.model)
    else:
        image_extra, body_extra = None, {"chat_template_kwargs": {"enable_thinking": False}}
    last: dict = {}
    for cap in spec.max_tokens:
        if hosted:
            with SPEND_LOCK:
                if spent(ledger) + 0.08 > cap_usd:
                    raise SystemExit(
                        f"cascade spend cap ${cap_usd} reached at ${spent(ledger):.4f}"
                    )
        started = time.time()
        try:
            last = _with_402_retry(
                lambda cap=cap: run_label.openrouter(
                    spec.model, parts, cap, spec.reasoning, image_extra, body_extra, base_url
                )
            )
        except Exception as error:  # noqa: BLE001 - recorded, then the next cap is tried
            detail = ""
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail = error.read().decode(errors="replace")[:400]
                except Exception:  # noqa: BLE001
                    detail = ""
            last = {
                "raw": "",
                "finish_reason": "error",
                "usage": {},
                "error": f"{type(error).__name__}: {error} {detail}",
            }
            time.sleep(5)
            continue
        verdict = run_label.parse_verdict(last.get("raw") or "")
        usage = last.get("usage") or {}
        usd = float(usage.get("cost") or 0) if hosted else 0.0
        saved = _saved(row, tier, draw, cap, last, verdict, started, usd)
        out.parent.mkdir(parents=True, exist_ok=True)
        if hosted:
            with SPEND_LOCK, ledger.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "cost": usd,
                            "model": spec.model,
                            "tier": tier,
                            "candidate": row["id"],
                            "draw": draw,
                            "prompt": usage.get("prompt_tokens"),
                            "completion": usage.get("completion_tokens"),
                            "evidence_px": opus_side(chosen) if tier == "opus" else None,
                        }
                    )
                    + "\n"
                )
        if verdict is None or last.get("finish_reason") in {"length", "max_tokens"}:
            out.with_suffix(".truncated.json").write_text(json.dumps(saved))
            continue
        out.write_text(json.dumps(saved))
        return f"ok {row['id']} {tier} draw {draw} {verdict.get('event_type')} ${usd:.4f} dt={saved['dt']}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".fail.json").write_text(
        json.dumps(
            {
                "id": row["id"],
                "tier": tier,
                "draw": draw,
                "last": {"error": last.get("error"), "finish_reason": last.get("finish_reason")},
            }
        )
    )
    return f"FAIL {row['id']} {tier} draw {draw} {last.get('error') or last.get('finish_reason')}"


def _with_402_retry(fn, attempts: int = 3):
    """Pause and retry an HTTP 402; other errors propagate."""
    for attempt in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as error:
            if error.code != 402 or attempt + 1 == attempts:
                raise
            time.sleep(30 if attempt == 0 else 60)
    raise RuntimeError("retry fell through")


def qwen_demos() -> list[dict]:
    _, run_label = _fair()
    return run_label.demo_items(TIER_BY_NAME["qwen"].profile)


def wait_ready(ports: tuple[int, ...], timeout_s: float = 900) -> None:
    deadline = time.time() + timeout_s
    pending = set(ports)
    while pending and time.time() < deadline:
        for port in list(pending):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5
                ) as response:
                    if response.status == 200:
                        pending.discard(port)
            except Exception:  # noqa: BLE001 - not up yet
                pass
        if pending:
            time.sleep(5)
    if pending:
        raise SystemExit(f"vLLM not ready on {sorted(pending)}")


QWEN_WEIGHTS = "models/vlm_frontier/Qwen3.8-27B-FP8"
QWEN_SERVED_NAME = "Qwen/Qwen3.8-27B-FP8"
VLLM = "vllm"


def vllm_command(
    weights: str, served_name: str, port: int, *, max_model_len: int, memory_fraction: float
) -> list[str]:
    from cv.pipeline.paths import data_root

    return [
        VLLM,
        "serve",
        str(data_root() / weights),
        "--served-model-name",
        served_name,
        "--tensor-parallel-size",
        "1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(memory_fraction),
    ]


@contextlib.contextmanager
def vllm_servers(commands: dict[int, list[str]], log_dir: Path):
    """One local vLLM server per GPU (``{gpu: command}``), stopped by pid on exit."""
    import os
    import subprocess

    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        for gpu, command in commands.items():
            log = (log_dir / f"vllm_gpu{gpu}.log").open("a")
            processes.append(
                subprocess.Popen(
                    command,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
        (log_dir / "pids.txt").write_text("".join(f"{p.pid}\n" for p in processes))
        yield processes
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()


def qwen_servers(gpus: tuple[int, ...], ports: tuple[int, ...], log_dir: Path):
    """The FREEZE v2 Qwen servers, one per GPU (flags of ``cascade_mainline/live.sh``)."""
    if len(gpus) != len(ports):
        raise ValueError("one port per Qwen GPU")
    return vllm_servers(
        {
            gpu: vllm_command(
                QWEN_WEIGHTS, QWEN_SERVED_NAME, port, max_model_len=131072, memory_fraction=0.90
            )
            for gpu, port in zip(gpus, ports, strict=True)
        },
        log_dir,
    )
