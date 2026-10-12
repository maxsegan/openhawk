"""Per-stage runtime and compute manifests required by ``AGENT_DIRECTIVE.md``."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from .provenance import (
        AUTOMATIC_MODE,
        build_provenance,
        file_record,
        portable_configuration,
    )
except ImportError:  # pragma: no cover - direct script imports
    from provenance import AUTOMATIC_MODE, build_provenance, file_record, portable_configuration


class StageRun:
    def __init__(
        self,
        out_dir: str,
        stage: str,
        config,
        gpu_devices: list[int] | None = None,
        seed: int = 0,
        *,
        mode: str = AUTOMATIC_MODE,
        source_videos: list[str] | None = None,
        models: list[dict] | None = None,
        reused_artifacts: list[str] | None = None,
        fallbacks: list[dict] | None = None,
        human_inputs: list[dict] | None = None,
        reviewed_inputs: list[dict] | None = None,
        manual_overrides: list[dict] | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.stage = stage
        raw_config = asdict(config) if is_dataclass(config) else dict(vars(config))
        # argparse subcommands attach their dispatch function to the namespace. Runtime-only
        # callables are neither configuration nor JSON serializable.
        self.config = {key: value for key, value in raw_config.items() if not callable(value)}
        self.gpu_devices = gpu_devices or []
        self.seed = seed
        self.mode = mode
        self.source_videos = source_videos or self._configured_files("video")
        self.models = models or self._configured_models()
        self.reused_artifacts = reused_artifacts or self._configured_artifacts()
        self.fallbacks = fallbacks or []
        self.human_inputs = human_inputs or []
        self.reviewed_inputs = reviewed_inputs or []
        self.manual_overrides = manual_overrides or []
        self.started_wall = time.time()
        self.started_iso = datetime.now(timezone.utc).isoformat()
        self._provenance()

    def _configured_files(self, marker: str) -> list[str]:
        output = []
        for key, value in self.config.items():
            if marker not in key.lower() or not isinstance(value, (str, os.PathLike)):
                continue
            path = Path(value).expanduser()
            if path.is_file():
                output.append(str(path))
        return output

    def _configured_models(self) -> list[dict]:
        output = []
        for key, value in self.config.items():
            lowered = key.lower()
            if not any(marker in lowered for marker in ("model", "weight", "checkpoint")):
                continue
            if isinstance(value, (str, os.PathLike)) and Path(value).expanduser().is_file():
                output.append(file_record(value, role=key))
            elif value not in (None, "", [], {}):
                output.append({"name": str(value), "role": key})
        return output

    def _configured_artifacts(self) -> list[str]:
        output = []
        for key, value in self.config.items():
            lowered = key.lower()
            if not any(marker in lowered for marker in ("manifest", "input", "track", "camera")):
                continue
            if isinstance(value, (str, os.PathLike)) and Path(value).expanduser().is_file():
                output.append(str(value))
        return output

    def _provenance(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return build_provenance(
            root=root,
            mode=self.mode,
            source_videos=[file_record(path, role="stage_source") for path in self.source_videos],
            models=self.models,
            configuration=self.config,
            reused_artifacts=[
                file_record(path, role="stage_input") for path in self.reused_artifacts
            ],
            fallbacks=self.fallbacks,
            human_inputs=self.human_inputs,
            reviewed_inputs=self.reviewed_inputs,
            manual_overrides=self.manual_overrides,
        )

    def _git_hash(self) -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def _gpu_models(self) -> list[str]:
        if not self.gpu_devices:
            return []
        try:
            lines = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            models = {int(line.split(",", 1)[0]): line.split(",", 1)[1].strip() for line in lines}
            return [models.get(i, "unknown") for i in self.gpu_devices]
        except (OSError, subprocess.SubprocessError, ValueError):
            return ["unknown" for _ in self.gpu_devices]

    def finish(self, status: str = "ok", outputs: dict | None = None) -> str:
        ended = time.time()
        wall_seconds = ended - self.started_wall
        manifest = {
            "stage": self.stage,
            "status": status,
            "started_at": self.started_iso,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(wall_seconds, 3),
            "gpu_devices": self.gpu_devices,
            "gpu_models": self._gpu_models(),
            "gpu_hours": round(wall_seconds * len(self.gpu_devices) / 3600.0, 6),
            "seed": self.seed,
            "git_hash": self._git_hash(),
            "config": portable_configuration(self.config),
            "provenance": self._provenance(),
            "outputs": portable_configuration(outputs or {}),
        }
        manifest_dir = os.path.join(self.out_dir, "run_manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(manifest_dir, f"{self.stage}_{stamp}.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(
            f"manifest: {path} | wall={wall_seconds:.1f}s | gpu_hours={manifest['gpu_hours']:.6f}"
        )
        return path
