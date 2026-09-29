"""Resumable one-shot labeller. Skips a reply that already parses.

One nomination per call. OpenRouter models get reasoning effort ``low`` unless
``--reasoning none``. Astra uses ``codex exec -m gpt-6-astra -s read-only``
with API keys removed from the environment. A truncated reply is re-run at a
higher cap; the body is never edited.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


from .guided_procedure import procedure_for  # noqa: E402
from .fairlib import art_root, task_preamble, target_text  # noqa: E402
from .or_call import URL, key  # noqa: E402

CAP_LADDER = (8192, 16384, 65536)
DEMO_PROFILES = ("reasoned", "guided", "guided_nothink")
# Thinking off. guided_nothink keeps the N-shot prefix. The other two drop it:
# nodemo is the same procedure with no demonstrations, reject asks for no_event first.
NOTHINK_PROFILES = ("guided_nothink", "guided_nothink_nodemo", "guided_reject_nothink")
PROFILE_CHOICES = ("reasoned", "zero", "guided", "guided_nothink",
                   "guided_nothink_nodemo", "guided_reject_nothink")
# Addendum 2 (2026-09-23): phase C's cumulative meter stops at $190.
# screen.py that is already running imported 170 and is on the local model only.
HARD_STOP = 190.0
LOCK = threading.Lock()
SPEND = art_root() / "spend.jsonl"


def parse_verdict(text: str) -> dict | None:
    best = None
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "event_type" in obj:
                        best = obj
                    break
    return best


def spent() -> float:
    if not SPEND.exists():
        return 0.0
    total = 0.0
    for line in SPEND.read_text().splitlines():
        if not line.strip():
            continue
        total += float(json.loads(line).get("cost") or 0)
    return total


def log_spend(row: dict) -> None:
    with LOCK:
        with SPEND.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def manifest_rows(layout: str, frames_per_image: int, clips: list[str]) -> list[dict]:
    tag = f"per{frames_per_image}" if layout == "several" else "pergrid"
    root = art_root() / "evidence" / layout / tag
    rows = []
    for clip in clips:
        path = root / clip / "manifest.json"
        if not path.is_file():
            raise SystemExit(f"missing evidence manifest {path}")
        doc = json.loads(path.read_text())
        for nom in doc["nominations"]:
            nom["layout"] = layout
            nom["frames_per_image"] = frames_per_image
            rows.append(nom)
    return rows


def ordered_images(nom: dict, presentation: str) -> list[dict]:
    kinds = ["clean"] if presentation == "clean" else ["clean", "trail"]
    chosen = []
    for kind in kinds:
        for image in nom["images"]:
            if image["kind"] == kind:
                chosen.append(image)
    if not chosen:
        raise RuntimeError(f"no images for {nom['clip']} {nom['id']} {presentation}")
    return chosen


def build_parts(nom: dict, presentation: str, profile: str, demos: list[dict]) -> list[dict]:
    """Cache order: static instructions and N-shot images, then this candidate.

    Provider prefix caches only match a shared head. The candidate id, frame
    list and candidate crops are therefore after every demonstration image.
    """
    procedure = procedure_for(profile).strip()
    parts: list[dict] = [{
        "type": "text",
        "text": task_preamble(presentation=presentation).rstrip() + "\n\n"
        + (procedure + "\n\n" if procedure else ""),
    }]
    image_index = 1
    if profile in DEMO_PROFILES and demos:
        parts.append({
            "type": "text",
            "text": "LABELED DEMONSTRATIONS follow, then the unlabeled candidate.\n",
        })
        pending = "demonstration"
        for item in demos:
            if item["kind"] == "text":
                pending = item["text"]
                parts.append({"type": "text", "text": item["text"] + "\n"})
            else:
                label = (item.get("label") or pending)[:180]
                parts.append({"type": "text", "text": f"Image {image_index}: {label}\n"})
                parts.append({"type": "image", "path": item["path"]})
                image_index += 1
    images = ordered_images(nom, presentation)
    parts.append({
        "type": "text",
        "text": "UNLABELED CANDIDATE images, after the demonstrations:\n",
    })
    for image in images:
        frames = ",".join(str(frame) for frame in image["frames"])
        hint = "FALLIBLE TRACKER HINT" if image["kind"] == "trail" else "CLEAN"
        parts.append({"type": "text", "text": f"Image {image_index}: {hint} frames {frames}\n"})
        parts.append({"type": "image", "path": image["path"]})
        image_index += 1
    parts.append({"type": "text", "text": "\n" + target_text(nom)})
    return parts


def parts_text(parts: list[dict]) -> str:
    return "".join(part["text"] for part in parts if part["type"] == "text")


def parts_paths(parts: list[dict]) -> list[str]:
    return [part["path"] for part in parts if part["type"] == "image"]


def demo_items(profile: str) -> list[dict]:
    if profile not in DEMO_PROFILES:
        return []
    path = art_root() / "demos" / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"reasoned profile needs {path}; run demos.py")
    items = json.loads(path.read_text())["items"]
    # Attach a stable label to each image from the preceding text line.
    label = "demonstration"
    out = []
    for item in items:
        if item["kind"] == "text":
            label = item["text"][:180]
            out.append(item)
        else:
            out.append({**item, "label": label})
    return out


def _data_url(path: str) -> str:
    blob = Path(path).read_bytes()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(blob).decode()


def openrouter_content(parts: list[dict], image_extra: dict | None) -> list[dict]:
    """Interleave text and images. A single text blob before every image puts
    the candidate sentence in front of the shared demonstration images."""
    content = []
    for part in parts:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
            continue
        image_url = {"url": _data_url(part["path"])}
        if image_extra:
            image_url.update(image_extra)
        content.append({"type": "image_url", "image_url": image_url})
    return content


def openrouter(model: str, parts: list[dict], max_tokens: int,
               reasoning: str | None, image_extra: dict | None,
               body_extra: dict | None = None, base_url: str | None = None) -> dict:
    content = openrouter_content(parts, image_extra)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "usage": {"include": True},
    }
    if reasoning:
        body["reasoning"] = {"effort": reasoning}
    if body_extra:
        body.update(body_extra)
    headers = {"Content-Type": "application/json"}
    if base_url:
        url = base_url.rstrip("/") + "/chat/completions"
        headers["Authorization"] = "Bearer EMPTY"
        body.pop("usage", None)
    else:
        url = URL
        headers["Authorization"] = f"Bearer {key()}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=900) as response:
        payload = json.loads(response.read().decode())
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "raw": message.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") or {},
        "error": payload.get("error"),
    }


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SESSION_ID = re.compile(
    r"session id:\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


def session_ids_from_stdout(raw: str) -> list[str]:
    """Codex prints ``session id:`` with ANSI bold around the label when stderr is a tty.

    The calibration picker missed those ids and then stored whichever session
    file was newest, which is a different call once four workers overlap.
    """
    return SESSION_ID.findall(ANSI.sub("", raw or ""))


def _last_token_usage(text: str) -> dict:
    """The last total_token_usage object in a codex session. Nested objects are legal."""
    decoder = json.JSONDecoder()
    needle = '"total_token_usage":'
    last: dict = {}
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return last
        try:
            obj, _ = decoder.raw_decode(text[index + len(needle):])
        except json.JSONDecodeError:
            start = index + len(needle)
            continue
        if isinstance(obj, dict):
            last = obj
        start = index + len(needle)


def codex_usage_for(raw: str, root: Path | None = None) -> dict:
    """Tokens for this call, from the session id printed in its own stdout.

    ``input_tokens`` is the whole prompt. ``cached_input_tokens`` is the subset
    read from cache, so uncached = input - cached - cache_write. Output already
    includes reasoning: the session's total_tokens equals input + output.
    """
    found = list(dict.fromkeys(session_ids_from_stdout(raw)))
    if len(found) != 1:
        return {"session_ids": found} if found else {}
    session_id = found[0]
    root = root or (Path.home() / ".codex" / "sessions")
    files = sorted(root.glob(f"*/*/*/*{session_id}*"))
    usage: dict = {"session_id": session_id}
    if len(files) != 1:
        usage["session_files"] = [str(path) for path in files]
        return usage
    path = files[0]
    usage.update(_last_token_usage(path.read_text(errors="replace")))
    usage["session"] = str(path)
    if "input_tokens" in usage:
        usage.setdefault("prompt_tokens", int(usage["input_tokens"]))
        cached = int(usage.get("cached_input_tokens") or 0)
        written = int(usage.get("cache_write_input_tokens") or 0)
        usage["uncached_input_tokens"] = int(usage["input_tokens"]) - cached - written
    if "output_tokens" in usage:
        usage.setdefault("completion_tokens", int(usage["output_tokens"]))
    return usage


def cached_tokens(usage: dict) -> int | None:
    """OpenRouter reports ``prompt_tokens_details.cached_tokens``. Codex reports
    ``cached_input_tokens``. Both are the prefix (or image) cache hit."""
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return int(details["cached_tokens"])
    if usage.get("cached_input_tokens") is not None:
        return int(usage["cached_input_tokens"])
    return None


def codex_call(model: str, parts: list[dict]) -> dict:
    env = os.environ.copy()
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        env.pop(name, None)
    cmd = [
        "codex", "exec", "--skip-git-repo-check", "-s", "read-only",
        "-m", model.split("/")[-1], "-C", "/tmp",
    ]
    # Demonstration images first, candidate frames last. Codex has no interleaved
    # parts; -i order is the image order the prompt's "Image N" lines describe.
    for path in parts_paths(parts):
        cmd.extend(["-i", path])
    cmd.extend(["--", parts_text(parts)])
    proc = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, env=env, cwd="/tmp",
        capture_output=True, text=True, timeout=2400, check=False,
    )
    raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return {
        "raw": raw,
        "finish_reason": "codex_exit_" + str(proc.returncode),
        "usage": codex_usage_for(raw),
        "error": None if proc.returncode == 0 else f"exit {proc.returncode}",
    }


def already_good(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    doc = json.loads(path.read_text())
    if doc.get("finish_reason") in {"length", "max_tokens"}:
        return False
    return isinstance(doc.get("verdict"), dict) and "event_type" in doc["verdict"]


def provider_kwargs(model: str) -> tuple[dict | None, dict | None]:
    """Image settings the token probe showed OpenRouter actually honors."""
    if model.startswith("google/"):
        # ULTRA_HIGH is rejected. HIGH is the highest accepted value and matches
        # the default token count; LOW is what drops it, so the field is live.
        return None, {"media_resolution": "MEDIA_RESOLUTION_HIGH"}
    if model.startswith("openai/"):
        return {"detail": "high"}, None
    return None, None


def one(model: str, transport: str, nom: dict, presentation: str, profile: str,
        draw: int, out: Path, reasoning: str | None, image_extra: dict | None,
        body_extra: dict | None, demos: list[dict], base_url: str | None = None,
        api_model: str | None = None, cap_ladder: tuple[int, ...] | None = None) -> str:
    if already_good(out):
        return f"skip {nom['clip']}/{nom['id']}"
    # Local vLLM is electricity, not the OpenRouter meter.
    if transport == "openrouter" and base_url is None and spent() >= HARD_STOP:
        return f"STOP budget {spent():.2f}"
    parts = build_parts(nom, presentation, profile, demos if profile in DEMO_PROFILES else [])
    paths = parts_paths(parts)
    sent_model = api_model or model
    last = {}
    for cap in (cap_ladder or CAP_LADDER):
        started = time.time()
        try:
            if transport == "codex":
                last = codex_call(model, parts)
            else:
                last = openrouter(
                    sent_model, parts, cap, reasoning, image_extra, body_extra, base_url,
                )
        except Exception as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode(errors="replace")[:400]
                if reasoning and exc.code == 400 and "reasoning" in detail.lower():
                    reasoning = None
                    continue
            last = {"raw": "", "finish_reason": "error", "usage": {},
                    "error": f"{type(exc).__name__}: {exc} {detail}"}
            time.sleep(5)
            continue
        verdict = parse_verdict(last.get("raw") or "")
        reason = last.get("finish_reason")
        usage = last.get("usage") or {}
        cost = float(usage.get("cost") or 0)
        row = {
            "clip": nom["clip"], "id": nom["id"], "model": model, "draw": draw,
            "profile": profile, "presentation": presentation,
            "layout": nom["layout"], "finish_reason": reason,
            "max_tokens": cap, "verdict": verdict, "raw": last.get("raw") or "",
            "usage": usage, "error": last.get("error"),
            "n_images": len(paths), "dt": round(time.time() - started, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": cached_tokens(usage),
            "api_model": sent_model,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        spend_row = {
            "cost": cost, "model": sent_model, "out": str(out),
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
            "cached_tokens": cached_tokens(usage),
        }
        if verdict is None or reason in {"length", "max_tokens"}:
            out.with_suffix(".truncated.json").write_text(json.dumps(row))
            log_spend({**spend_row, "truncated": True})
            continue
        out.write_text(json.dumps(row))
        log_spend({**spend_row, "truncated": False})
        return (f"ok {nom['clip']}/{nom['id']} finish={reason} "
                f"prompt={usage.get('prompt_tokens')} cost={cost}")
    return f"FAIL {nom['clip']}/{nom['id']} {last.get('error') or last.get('finish_reason')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--transport", choices=("openrouter", "codex", "local"), default="openrouter")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base, for --transport local")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--cap-ladder", default=None, help="comma-separated max_tokens retries, e.g. 4096,8192")
    parser.add_argument("--layout", choices=("several", "one"), required=True)
    parser.add_argument("--frames-per-image", type=int, default=1)
    parser.add_argument("--presentation", choices=("clean", "trail"), required=True)
    parser.add_argument("--draw", type=int, default=1)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--limit", type=int, default=0, help="label only the first N nominations")
    parser.add_argument("--api-model", default=None,
                        help="provider id when it differs from --model; the directory stays --model")
    args = parser.parse_args()
    if args.transport == "codex":
        args.workers = min(args.workers, 4)
        args.reasoning = "none"
    if args.transport == "local" and not args.base_url:
        raise SystemExit("--transport local needs --base-url")
    image_extra, body_extra = provider_kwargs(args.model)
    if args.profile in NOTHINK_PROFILES:
        # vLLM chat template switch for Qwen3-family models: answer without a <think> block.
        body_extra = {**(body_extra or {}), "chat_template_kwargs": {"enable_thinking": False}}
    ladder = tuple(int(x) for x in args.cap_ladder.split(",")) if args.cap_ladder else None
    demos = demo_items(args.profile)
    rows = manifest_rows(args.layout, args.frames_per_image, args.clips)
    if args.limit:
        rows = rows[:args.limit]
    slug = args.model.replace("/", "_")
    out_root = (art_root() / "labels" / slug / args.profile / args.layout /
                args.presentation / f"draw{args.draw}")
    print(f"{args.model} {args.profile} {args.layout} {args.presentation} "
          f"draw {args.draw}: {len(rows)} nominations, spent ${spent():.4f}", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    reasoning = None if args.reasoning == "none" else args.reasoning
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for nom in rows:
            out = out_root / nom["clip"] / f"{nom['id']}.json"
            futures.append(pool.submit(
                one, args.model, args.transport, nom, args.presentation, args.profile,
                args.draw, out, reasoning, image_extra, body_extra, demos, args.base_url,
                args.api_model, ladder,
            ))
        for future in futures:
            line = future.result()
            print(line, flush=True)
            if line.startswith("STOP"):
                break
    print(f"spent ${spent():.4f}", flush=True)


if __name__ == "__main__":
    main()
