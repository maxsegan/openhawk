#!/usr/bin/env python3
"""Move ignored datasets into shared storage and link them into a checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

DATA_KINDS = ("raw", "processed")
ROOT_MODEL_FILES = (
    "yolo26m-pose.pt",
    "yolo26m.pt",
    "yolov8m-pose.pt",
    "yolov8m.pt",
)
ENV_NAME = "TENNIS_DATA_ROOT"


def repository_root(path: Path | None = None) -> Path:
    candidate = (path or Path(__file__).resolve().parents[1]).resolve()
    result = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def dotenv_value(repo: Path, name: str) -> str | None:
    env_path = repo / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")
    return None


def resolve_shared_root(repo: Path, argument: Path | None) -> Path:
    configured = argument or os.environ.get(ENV_NAME) or dotenv_value(repo, ENV_NAME)
    if not configured:
        raise SystemExit(f"Set {ENV_NAME} in .env or pass --shared-root before migrating data.")
    return Path(configured).expanduser().resolve()


def tracked_layout(repo: Path) -> tuple[set[str], set[str]]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--", "data/raw", "data/processed"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_files = {line for line in result.stdout.splitlines() if line}
    tracked_directories = {"data/raw", "data/processed"}
    for tracked_file in tracked_files:
        parent = Path(tracked_file).parent
        while parent.parts and parent != Path("data"):
            tracked_directories.add(parent.as_posix())
            parent = parent.parent
    return tracked_files, tracked_directories


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_match(first: Path, second: Path) -> bool:
    first_stat = first.stat()
    second_stat = second.stat()
    return first_stat.st_size == second_stat.st_size and file_digest(first) == file_digest(second)


def remove_empty_parents(path: Path, stop: Path) -> None:
    parent = path.parent
    while parent != stop and parent.is_dir():
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def merge_ignored_path(
    source: Path,
    target: Path,
    repo_relative: str,
    tracked_files: set[str],
    tracked_directories: set[str],
    local_root: Path,
) -> tuple[int, int]:
    if source.is_symlink():
        source.unlink()
        remove_empty_parents(source, local_root)
        return 0, 1

    if source.is_dir():
        if repo_relative not in tracked_directories and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            return 1, 0

        moved = removed = 0
        target.mkdir(parents=True, exist_ok=True)
        for child in list(source.iterdir()):
            child_relative = f"{repo_relative}/{child.name}"
            child_moved, child_removed = merge_ignored_path(
                child,
                target / child.name,
                child_relative,
                tracked_files,
                tracked_directories,
                local_root,
            )
            moved += child_moved
            removed += child_removed
        if source != local_root:
            try:
                source.rmdir()
            except OSError:
                pass
        return moved, removed

    if repo_relative in tracked_files:
        return 0, 0

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not files_match(source, target):
            raise RuntimeError(f"Conflicting local and shared files: {source} and {target}")
        source.unlink()
        remove_empty_parents(source, local_root)
        return 0, 1

    source.replace(target)
    remove_empty_parents(source, local_root)
    return 1, 0


def migrate(repo: Path, shared_root: Path) -> None:
    tracked_files, tracked_directories = tracked_layout(repo)
    shared_root.mkdir(parents=True, exist_ok=True)
    moved = removed = 0

    for kind in DATA_KINDS:
        local_root = repo / "data" / kind
        shared_kind = shared_root / kind
        shared_kind.mkdir(parents=True, exist_ok=True)
        if not local_root.exists():
            continue
        for child in list(local_root.iterdir()):
            repo_relative = f"data/{kind}/{child.name}"
            child_moved, child_removed = merge_ignored_path(
                child,
                shared_kind / child.name,
                repo_relative,
                tracked_files,
                tracked_directories,
                local_root,
            )
            moved += child_moved
            removed += child_removed

    model_root = shared_root / "models" / "pipeline"
    model_root.mkdir(parents=True, exist_ok=True)
    for name in ROOT_MODEL_FILES:
        source = repo / name
        target = model_root / name
        if source.is_symlink() or not source.exists():
            continue
        if target.exists():
            if not files_match(source, target):
                raise RuntimeError(f"Conflicting local and shared model files: {source} and {target}")
            source.unlink()
            removed += 1
        else:
            source.replace(target)
            moved += 1

    print(f"Migrated {moved} paths; removed {removed} duplicate/local links.")


def create_link(local_path: Path, shared_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(shared_path, local_path.parent)
    local_path.symlink_to(relative_target, target_is_directory=shared_path.is_dir())


def link_shared_path(
    shared_path: Path,
    local_path: Path,
    repo_relative: str,
    tracked_files: set[str],
    tracked_directories: set[str],
) -> int:
    if shared_path.is_dir() and repo_relative in tracked_directories:
        local_path.mkdir(parents=True, exist_ok=True)
        linked = 0
        for child in shared_path.iterdir():
            linked += link_shared_path(
                child,
                local_path / child.name,
                f"{repo_relative}/{child.name}",
                tracked_files,
                tracked_directories,
            )
        return linked

    if repo_relative in tracked_files:
        return 0

    if local_path.is_symlink():
        if local_path.resolve() == shared_path.resolve():
            return 0
        local_path.unlink()
    elif local_path.exists():
        if local_path.is_dir() and not any(local_path.iterdir()):
            local_path.rmdir()
        else:
            raise RuntimeError(
                f"Local path blocks shared data link: {local_path}. Run migrate first."
            )

    create_link(local_path, shared_path)
    return 1


def link(repo: Path, shared_root: Path) -> None:
    tracked_files, tracked_directories = tracked_layout(repo)
    linked = 0
    for kind in DATA_KINDS:
        shared_kind = shared_root / kind
        if not shared_kind.is_dir():
            continue
        local_root = repo / "data" / kind
        local_root.mkdir(parents=True, exist_ok=True)
        for child in shared_kind.iterdir():
            linked += link_shared_path(
                child,
                local_root / child.name,
                f"data/{kind}/{child.name}",
                tracked_files,
                tracked_directories,
            )

    model_root = shared_root / "models" / "pipeline"
    for name in ROOT_MODEL_FILES:
        shared_model = model_root / name
        if not shared_model.is_file():
            continue
        local_model = repo / name
        if local_model.is_symlink():
            if local_model.resolve() == shared_model.resolve():
                continue
            local_model.unlink()
        elif local_model.exists():
            if not files_match(local_model, shared_model):
                raise RuntimeError(
                    f"Local model blocks shared model link: {local_model}. Run migrate first."
                )
            local_model.unlink()
        create_link(local_model, shared_model)
        linked += 1
    print(f"Created {linked} shared-data links in {repo}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move gitignored raw/processed data into one shared tree and overlay "
            "it into a repository checkout."
        )
    )
    parser.add_argument("command", choices=("migrate", "link", "setup"))
    parser.add_argument("--repo", type=Path, help="Checkout to operate on")
    parser.add_argument("--shared-root", type=Path)
    args = parser.parse_args()

    repo = repository_root(args.repo)
    shared_root = resolve_shared_root(repo, args.shared_root)
    if shared_root == repo or repo in shared_root.parents:
        raise SystemExit("Shared data root must live outside the repository.")

    if args.command in {"migrate", "setup"}:
        migrate(repo, shared_root)
    if args.command in {"link", "setup"}:
        link(repo, shared_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
