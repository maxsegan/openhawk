"""Pixel-and-audio S5 event model over native crops.

Three branches are fused late:

* a 3D ResNet-18 (``torchvision.models.video.r3d_18``) over the 16-frame
  ``192x108`` native crop sequence,
* a 2D ResNet-18 over the ``384x216`` native court-context crop,
* a small CNN over the 0.5 s log-mel spectrogram,

feeding three heads: a four-way event type (``none``/``contact``/``bounce``/
``net_hit``), a sub-frame time offset, and a native ``x``/``y`` offset from the
crop centre.  Weights come from the local torch hub cache only; nothing is
downloaded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cv.pipeline.event_crops import (
    CONTEXT_HEIGHT,
    CONTEXT_WIDTH,
    MEL_BANDS,
    MEL_FRAMES,
    SEQUENCE_LENGTH,
    TIGHT_HEIGHT,
    TIGHT_WIDTH,
    memmap_member,
)
from cv.pipeline import event_time_distribution as timing
from cv.pipeline.automatic_ball_track import (
    CURRENT_EVENT_TRACK,
    LEGACY_EVENT_TRACK,
    validate_event_track,
)
from cv.pipeline.event_time_distribution import TIME_BINS, TIME_RADIUS
from cv.pipeline.provenance import file_record, file_sha256

CLASSES = ("none", "contact", "bounce", "net_hit")
# Abstention operating points, fitted on the development broadcasts only and
# applied unchanged to the held-out ones.  A runtime emission is kept when its
# decoder path marginal is at or above the chosen threshold.  ``VERIFIED`` names
# the artifact the pair came from so the numbers can be re-derived.
ABSTENTION_CALIBRATION = {
    "schema": "event_abstention_complete_point_calibration_v1",
    "source": (
        "$TENNIS_DATA_ROOT/processed/wk3_eventthreshold/calibration/development_selection.json"
    ),
    "fitted_on": (
        "32 development broadcasts, 1176 truth events and 118 event points, tolerance 3 frames"
    ),
    "selection_rule": (
        "maximum development complete-event points subject to no more than one "
        "false positive per 100 emitted physical events; ties prefer more true "
        "positives, then higher precision, then a higher threshold"
    ),
    "selected_development_result": {
        "threshold": 0.9521754400734156,
        "true_positives": 1031,
        "false_positives": 10,
        "false_negatives": 145,
        "precision": 0.9903938520653218,
        "complete_event_points": 44,
        "event_points": 118,
    },
    "operating_points": {
        # Largest recall with no observed false positive on development.
        "zero_false_positives": 0.999962197168566,
        # Largest recall with false positives under 2% of emissions.
        "bounded_false_positives": 0.9898053678698258,
        # Maximum complete-event points with <=1 false positive per 100 emissions.
        "complete_points_one_fp_per_100": 0.9521754400734156,
    },
    # The decoder grammar the threshold is now paired with.  The shipped
    # ``NET_LINE = 0.5`` was in the wrong units; the default is now a per-point
    # line derived from the net's own pixels, promoted on the held-out slice
    # under the rule ``cv.validation.score_event_cohort.HELD_OUT_FLOOR_RULE``.
    # The threshold itself did not move.  See WK3_REPORT.md of package
    # ``netline``.
    "decoder_grammar": {
        "arm": "derived_dead_band_net_span",
        "net_line": "per point, the net's own top edge in the feature frame",
        "side_dead_band": "per point, the net's own projected thickness",
        "source": (
            "$TENNIS_DATA_ROOT/processed/wk3_netline/calibration/development_calibration.json"
        ),
        "selection_rule": (
            "maximum development complete-event points at the shipped threshold among the "
            "arms whose net line is derived from the court geometry; the binding line is the "
            "held-out slice, not the development false-positive budget, because the "
            "development broadcasts are inside the checkpoint's training set"
        ),
        "held_out_promotion": {
            "floor": 0.99,
            "shipped": {
                "true_positives": 150,
                "false_positives": 1,
                "false_negatives": 89,
                "precision": 0.9933774834437086,
            },
            "promoted": {
                "true_positives": 164,
                "false_positives": 1,
                "false_negatives": 75,
                "precision": 0.9939393939393939,
            },
            "held_out_scope": "8 broadcasts, 239 truth events, 28 event points, tolerance 3",
        },
        "development_at_the_shipped_threshold": {
            "shipped": {
                "complete_event_points": 44,
                "true_positives": 1031,
                "false_positives": 10,
                "predictions": 1041,
            },
            "promoted": {
                "complete_event_points": 55,
                "true_positives": 1067,
                "false_positives": 19,
                "predictions": 1086,
            },
            "note": (
                "the promoted arm's development false-positive rate is 1.75%, over the 1% "
                "budget the previous rule used; reading the nine extra emissions in native "
                "crops, five are real ball events the truth does not cover, one cannot be "
                "typed from one frame and three are wrong"
            ),
        },
    },
    # The frame ``court_y`` is in.  Until 2026-09-04 the event feature builder
    # read the ball track's legacy 960x540 columns and pushed them through a
    # native homography, so ``court_y`` was not a court fraction and the derived
    # net line above was derived inside that defective frame.  The builder now
    # reads the columns the coordinate sidecar declares; the net's ground line
    # lands on 0.5 exactly and the derived line follows the corrected frame.
    # The threshold did not move and the checkpoint did not change.  See
    # WK3_REPORT.md of package ``eventnative``.
    "feature_coordinates": {
        "contract": (
            "the event feature builder reads the ball track through the coordinate "
            "sidecar's declared column map; court_x/court_y are true court fractions"
        ),
        "source": (
            "$TENNIS_DATA_ROOT/processed/wk3_eventnative/calibration/development_calibration.json"
        ),
        "arm": "corrected_features",
        "held_out_promotion": {
            "floor": 0.99,
            "shipped": {
                "true_positives": 164,
                "false_positives": 1,
                "false_negatives": 75,
                "precision": 0.9939393939393939,
            },
            "promoted": {
                "true_positives": 166,
                "false_positives": 1,
                "false_negatives": 73,
                "precision": 0.9940119760479041,
            },
            "held_out_scope": "8 broadcasts, 239 truth events, 28 event points, tolerance 3",
        },
        "development_at_the_shipped_threshold": {
            "shipped": {
                "complete_event_points": 55,
                "true_positives": 1067,
                "false_positives": 19,
                "predictions": 1086,
            },
            "promoted": {
                "complete_event_points": 55,
                "true_positives": 1067,
                "false_positives": 19,
                "predictions": 1086,
            },
            "note": "development is an exact tie; the whole difference is on the held-out slice",
        },
        "retrain": (
            "the network reads only pixels and audio, so the corrected channels are not "
            "model inputs and the training crop dataset is bit-identical; a checkpoint "
            "retrained under the docs/wk1/events_pass3.md protocol scored 161/1 held-out "
            "against the shipped checkpoint's 166/1 on the corrected features and was not "
            "promoted"
        ),
    },
}
ABSTENTION_OPERATING_POINTS = ABSTENTION_CALIBRATION["operating_points"]
CALIBRATED_CHECKPOINT_SHA256 = "728757639ffa71ddd62f09768ea662f2e51c1905bebebd889efe173c3ffa8c69"
CALIBRATED_RUNTIME_SETTINGS = {
    "net_line": "derived",
    "side_dead_band": "derived",
    "correct_frames": True,
    "emission_mode": "best_path",
    "terminal_second_bounce": True,
    "alternate_crops": False,
    "supported_impulse_events": False,
    "court_geometry": "point_static",
    "court_frame_missing": "hold",
}
DEFAULT_OPERATING_POINT = "complete_points_one_fp_per_100"
# ``--net-line derived`` builds the per-point net line from the run's own camera
# artifacts; ``legacy`` restores the 0.5 that was in the wrong units.
DERIVED_NET_LINE = "derived"
LEGACY_NET_LINE_CHOICE = "legacy"
DEFAULT_NET_LINE = DERIVED_NET_LINE
DEFAULT_SIDE_DEAD_BAND = DERIVED_NET_LINE
HUB_CHECKPOINTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
R3D_CHECKPOINT = HUB_CHECKPOINTS / "r3d_18-b3b3357e.pth"
RESNET_CHECKPOINT = HUB_CHECKPOINTS / "resnet18-f37072fd.pth"
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEL_MEAN = -6.0
MEL_STD = 3.0
# The time head is a softmax over integer frame offsets inside the crop window.
# Its grid lives in ``event_time_distribution`` so a consumer of a saved
# prediction artifact can check the exact bins without importing torch.
TIME_SIGMA = 0.8
# Where the time softmax reads its evidence.  "fused" is the pass-2 head: one
# linear layer on the pooled clip embedding, by which point ``r3d_18`` has
# collapsed the 16 input frames to two temporal positions.  The pyramid heads
# tap the trunk where the temporal stride is still 1 ("pyramid1", layers 1-3) or
# 2 ("pyramid2", layers 2-3) and run a small 1-D convolution over time, so a bin
# is predicted from the frames around it rather than from a clip summary.
TIME_HEADS = ("fused", "pyramid1", "pyramid2")
# (layer index, channels, temporal stride relative to the input clip).
TRUNK_TAPS = {
    "pyramid1": ((1, 64, 1), (2, 128, 2), (3, 256, 4)),
    "pyramid2": ((2, 128, 2), (3, 256, 4)),
}


# --------------------------------------------------------------------------- #
# shards
# --------------------------------------------------------------------------- #
class CropStore:
    """Memory-mapped access to the crop shards of a whole cohort.

    The maps are opened lazily and are dropped when the store is pickled, so a
    dataloader worker re-opens them itself instead of receiving tens of
    gigabytes down a pipe.
    """

    def __init__(self, directory: Path, manifest: dict) -> None:
        self.directory = Path(directory)
        self.manifest = manifest
        self.shard_paths: list[Path] = []
        shard_index: dict[str, int] = {}
        for position, shard in enumerate(manifest["shards"]):
            path = Path(shard["path"])
            if not path.exists():
                path = self.directory / Path(shard["path"]).name
            self.shard_paths.append(path)
            shard_index[shard["broadcast"]] = position
        rows = manifest["rows_index"]
        self.shard_of_row = np.asarray(
            [shard_index[row["broadcast"]] for row in rows], dtype=np.int32
        )
        counters: dict[int, int] = {}
        offsets = np.zeros(len(rows), dtype=np.int64)
        for position, row in enumerate(rows):
            shard = shard_index[row["broadcast"]]
            offsets[position] = counters.get(shard, 0)
            counters[shard] = offsets[position] + 1
        self.offset_in_shard = offsets
        self._maps: dict[tuple[str, int], np.ndarray] = {}

    @classmethod
    def open(cls, directory: Path) -> "CropStore":
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        return cls(Path(directory), manifest)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_maps"] = {}
        state["manifest"] = {
            key: value for key, value in self.manifest.items() if key != "rows_index"
        }
        return state

    def member(self, kind: str, shard: int) -> np.ndarray:
        key = (kind, shard)
        if key not in self._maps:
            self._maps[key] = memmap_member(self.shard_paths[shard], kind)
        return self._maps[key]

    def _column(self, name: str, dtype) -> np.ndarray:
        rows = self.manifest["rows_index"]
        return np.asarray([row[name] for row in rows], dtype=dtype)

    def broadcasts(self) -> np.ndarray:
        return self._column("broadcast", None)

    def clips(self) -> np.ndarray:
        return self._column("clip", None)

    def frames(self) -> np.ndarray:
        return self._column("frame", np.int32)

    def court_y(self) -> np.ndarray:
        return np.asarray(
            [
                math.nan if row["court_y"] is None else row["court_y"]
                for row in self.manifest["rows_index"]
            ],
            dtype=np.float32,
        )

    def court_x(self) -> np.ndarray:
        return np.asarray(
            [
                math.nan if row["court_x"] is None else row["court_x"]
                for row in self.manifest["rows_index"]
            ],
            dtype=np.float32,
        )

    def court_transport_records(self) -> list[dict | None]:
        return [row.get("court_geometry") for row in self.manifest["rows_index"]]

    def _gather(self, kind: str, dtype, width: int | None = None) -> np.ndarray:
        shape = (len(self.shard_of_row),) if width is None else (len(self.shard_of_row), width)
        output = np.zeros(shape, dtype=dtype)
        for shard in range(len(self.shard_paths)):
            rows = np.flatnonzero(self.shard_of_row == shard)
            if not rows.size:
                continue
            values = np.asarray(self.member(kind, shard))
            output[rows] = values[self.offset_in_shard[rows]]
        return output

    def centres(self) -> np.ndarray:
        return self._gather("centres", np.float32, 2)

    def track_observed(self) -> np.ndarray:
        return self._gather("track_observed", bool)

    def proposal_sources(self) -> np.ndarray:
        return np.asarray(
            [str(row.get("proposal_source", "track")) for row in self.manifest["rows_index"]]
        )


class MergedCropStore:
    """Several crop stores read as one row space.

    The base store centres every crop on the composed track; an alternate store
    holds a second picture of the frames the track cannot centre.  The decoder
    has to see both on one lattice -- two proposals of the same frame compete
    for the same node -- so the row spaces are concatenated in store order.
    """

    def __init__(self, stores: list[CropStore]) -> None:
        if not stores:
            raise ValueError("a merged crop store needs at least one store")
        self.stores = list(stores)
        self.manifest = {
            "shards": [shard for store in stores for shard in store.manifest["shards"]],
            "merged_rows": [len(store.shard_of_row) for store in stores],
        }

    def _concatenate(self, name: str) -> np.ndarray:
        return np.concatenate([getattr(store, name)() for store in self.stores])

    def broadcasts(self) -> np.ndarray:
        return self._concatenate("broadcasts")

    def clips(self) -> np.ndarray:
        return self._concatenate("clips")

    def frames(self) -> np.ndarray:
        return self._concatenate("frames")

    def court_y(self) -> np.ndarray:
        return self._concatenate("court_y")

    def court_x(self) -> np.ndarray:
        return self._concatenate("court_x")

    def court_transport_records(self) -> list[dict | None]:
        return [record for store in self.stores for record in store.court_transport_records()]

    def centres(self) -> np.ndarray:
        return np.concatenate([store.centres() for store in self.stores])

    def track_observed(self) -> np.ndarray:
        return self._concatenate("track_observed")

    def proposal_sources(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(
                    [
                        str(row.get("proposal_source", "track"))
                        for row in store.manifest["rows_index"]
                    ]
                )
                for store in self.stores
            ]
        )


class EventCropDataset(Dataset):
    """Serve (tight, context, mel, targets) for a list of manifest row indices."""

    def __init__(
        self,
        store: CropStore,
        indices: np.ndarray,
        targets: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
        xy: np.ndarray | None = None,
        xy_mask: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        augment: bool = False,
        time_offsets: np.ndarray | None = None,
        time_mask: np.ndarray | None = None,
        translation_store=None,
        translation_radius: int = 0,
        translation_seed: int = 20260901,
    ) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = targets
        self.offsets = offsets
        self.xy = xy
        self.xy_mask = xy_mask
        self.weights = weights
        self.augment = augment
        self.time_offsets = time_offsets
        self.time_mask = time_mask
        self.translation_store = translation_store
        self.translation_radius = translation_radius
        self.translation_seed = translation_seed
        if translation_radius and translation_store is None:
            raise ValueError("native translation requires a bound expanded RGB cache")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        row = int(self.indices[position])
        shard = int(self.store.shard_of_row[row])
        offset = int(self.store.offset_in_shard[row])
        tight = np.asarray(self.store.member("tight", shard)[offset])
        context = np.asarray(self.store.member("context", shard)[offset])
        shift = (0, 0)
        if self.translation_store is not None:
            tight, context, shift = self.translation_store.sample(
                shard,
                offset,
                position,
                radius=self.translation_radius,
                seed=self.translation_seed,
            )
        mel = np.asarray(self.store.member("mel", shard)[offset], dtype=np.float32)
        flip = self.augment and bool(np.random.rand() < 0.5)
        if flip:
            tight = tight[:, :, ::-1]
            context = context[:, ::-1]
        tight = torch.from_numpy(np.ascontiguousarray(tight[..., ::-1]))
        context = torch.from_numpy(np.ascontiguousarray(context[..., ::-1]))
        item = {
            "tight": tight,
            "context": context,
            "mel": torch.from_numpy(mel),
            "row": row,
        }
        if self.targets is not None:
            item["target"] = int(self.targets[row])
            item["offset"] = float(self.offsets[row])
            xy = np.asarray(self.xy[row], dtype=np.float32)
            if shift != (0, 0):
                from cv.pipeline.event_crop_translation import shifted_xy

                xy = shifted_xy(xy, shift)
            if flip:
                xy = np.asarray([-xy[0], xy[1]], dtype=np.float32)
            item["xy"] = torch.from_numpy(xy)
            item["xy_mask"] = float(self.xy_mask[row])
            item["weight"] = float(self.weights[row])
            item["time_target"] = (
                0.0 if self.time_offsets is None else float(self.time_offsets[row])
            )
            item["time_mask"] = 0.0 if self.time_mask is None else float(self.time_mask[row])
        return item


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _video_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models.video import r3d_18

    model = r3d_18(weights=None)
    if pretrained and R3D_CHECKPOINT.exists():
        model.load_state_dict(torch.load(R3D_CHECKPOINT, map_location="cpu"))
    width = model.fc.in_features
    model.fc = nn.Identity()
    return model, width


def _image_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    if pretrained and RESNET_CHECKPOINT.exists():
        model.load_state_dict(torch.load(RESNET_CHECKPOINT, map_location="cpu"))
    width = model.fc.in_features
    model.fc = nn.Identity()
    return model, width


class AudioBranch(nn.Module):
    """Small CNN over the log-mel spectrogram."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.width = width

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.body(mel.unsqueeze(1)).flatten(1)


class TemporalTimeHead(nn.Module):
    """Per-frame time softmax over the trunk's fine temporal feature maps.

    Each tapped map is averaged over space, brought back to the clip's own
    ``SEQUENCE_LENGTH`` frames by repeating along time, concatenated with the
    fused clip/audio embedding broadcast over time, and passed through a small
    1-D convolution.  The output is one logit per input frame, so frame ``i``
    is offset ``i - TIME_RADIUS``; the ``+TIME_RADIUS`` bin of the grid has no
    frame of its own and repeats its neighbour.
    """

    def __init__(
        self, taps: tuple[tuple[int, int, int], ...], context: int, width: int = 128
    ) -> None:
        super().__init__()
        self.layers = tuple(tap[0] for tap in taps)
        self.strides = tuple(tap[2] for tap in taps)
        channels = sum(tap[1] for tap in taps) + context
        self.body = nn.Sequential(
            nn.Conv1d(channels, width, 3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, 1, 1),
        )

    def forward(self, maps: dict[int, torch.Tensor], context: torch.Tensor) -> torch.Tensor:
        series = []
        length = 0
        for layer, stride in zip(self.layers, self.strides):
            pooled = maps[layer].mean(dim=(3, 4))
            if stride > 1:
                pooled = pooled.repeat_interleave(stride, dim=2)
            length = max(length, pooled.shape[2])
            series.append(pooled)
        series = [row[:, :, :length] for row in series]
        series.append(context.unsqueeze(2).expand(-1, -1, length))
        frames = self.body(torch.cat(series, dim=1)).squeeze(1)
        if frames.shape[1] < TIME_BINS:
            frames = nn.functional.pad(
                frames.unsqueeze(1), (0, TIME_BINS - frames.shape[1]), mode="replicate"
            ).squeeze(1)
        return frames[:, :TIME_BINS]


class EventVideoNet(nn.Module):
    """Late-fused pixel-and-audio event classifier with regression heads."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        time_head: str = "fused",
    ) -> None:
        super().__init__()
        if time_head not in TIME_HEADS:
            raise ValueError(f"unknown time head: {time_head}")
        self.video, video_width = _video_backbone(pretrained)
        self.image, image_width = _image_backbone(pretrained)
        self.audio = AudioBranch()
        width = video_width + image_width + self.audio.width
        self.fusion = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(width, 256), nn.ReLU(inplace=True)
        )
        self.type_head = nn.Linear(256, len(CLASSES))
        self.offset_head = nn.Linear(256, 1)
        self.xy_head = nn.Linear(256, 2)
        self.time_head_kind = time_head
        self.time_head = (
            nn.Linear(256, TIME_BINS)
            if time_head == "fused"
            else TemporalTimeHead(TRUNK_TAPS[time_head], 256)
        )
        self.register_buffer(
            "time_grid",
            torch.arange(-TIME_RADIUS, TIME_RADIUS + 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "video_mean", torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "video_std", torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def trunk(self, video: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run ``r3d_18`` block by block and keep the intermediate maps.

        This is ``VideoResNet.forward`` written out, so the pooled embedding is
        the same tensor the pass-2 model saw; only the intermediate maps are new.
        """

        maps: dict[int, torch.Tensor] = {}
        hidden = self.video.stem(video)
        for index in (1, 2, 3, 4):
            hidden = getattr(self.video, f"layer{index}")(hidden)
            maps[index] = hidden
        return self.video.avgpool(hidden).flatten(1), maps

    def forward(
        self, tight: torch.Tensor, context: torch.Tensor, mel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        video = tight.permute(0, 4, 1, 2, 3).float().div_(255.0)
        video = (video - self.video_mean) / self.video_std
        image = context.permute(0, 3, 1, 2).float().div_(255.0)
        image = (image - self.image_mean) / self.image_std
        audio = (mel.float() - MEL_MEAN) / MEL_STD
        clip, maps = self.trunk(video)
        features = torch.cat((clip, self.image(image), self.audio(audio)), dim=1)
        hidden = self.fusion(features)
        time_logits = (
            self.time_head(hidden)
            if self.time_head_kind == "fused"
            else self.time_head(maps, hidden)
        )
        time_probabilities = torch.softmax(time_logits.float(), dim=1)
        return {
            "logits": self.type_head(hidden),
            "offset": self.offset_head(hidden).squeeze(1),
            "xy": self.xy_head(hidden),
            "time_logits": time_logits,
            "time_offset": time_probabilities @ self.time_grid,
            "time_confidence": time_probabilities.max(dim=1).values,
        }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def time_target(offsets: torch.Tensor, sigma: float = TIME_SIGMA) -> torch.Tensor:
    """Return a Gaussian soft target over the integer offset bins.

    A hard one-hot target throws away the sub-frame part of a half-frame truth
    label; a narrow Gaussian keeps it, and its expectation is the sub-frame time
    the emission needs.
    """

    grid = torch.arange(-TIME_RADIUS, TIME_RADIUS + 1, dtype=torch.float32, device=offsets.device)
    weights = torch.exp(-0.5 * ((grid[None, :] - offsets[:, None]) / sigma) ** 2)
    return weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 96
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    negative_ratio: float = 4.0
    workers: int = 16
    seed: int = 20260901
    pretrained: bool = True
    offset_weight: float = 1.0
    xy_weight: float = 0.5
    time_weight: float = 1.0
    # Multiplies the per-row weight of each class in the type loss.  net_hit is
    # 3.8% of the labelled events and the first pass never emitted one.
    class_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # One of TIME_HEADS; see the constant for what each reads.
    time_head: str = "fused"
    # A cache binding both RGB streams to their original native exposures.
    # Zero radius runs the same cache without augmentation for the paired control.
    translation_cache: str | None = None
    translation_radius: int = 0


def sample_epoch(
    positives: np.ndarray,
    negatives: np.ndarray,
    ratio: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Return one epoch's row indices: every positive plus sampled negatives."""

    count = min(len(negatives), int(round(ratio * max(len(positives), 1))))
    drawn = generator.choice(negatives, size=count, replace=False) if count else negatives[:0]
    epoch = np.concatenate((positives, drawn))
    generator.shuffle(epoch)
    return epoch


def train_model(
    store: CropStore,
    train_positive: np.ndarray,
    train_negative: np.ndarray,
    targets: np.ndarray,
    offsets: np.ndarray,
    xy: np.ndarray,
    xy_mask: np.ndarray,
    weights: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    time_offsets: np.ndarray | None = None,
    time_mask: np.ndarray | None = None,
) -> tuple[EventVideoNet, dict]:
    torch.manual_seed(config.seed)
    translation_store = None
    if config.translation_cache is not None:
        from cv.pipeline.event_crop_translation import NativeTranslationStore

        translation_store = NativeTranslationStore(Path(config.translation_cache), store)
    elif config.translation_radius:
        raise ValueError("native translation training requires its prepared RGB cache")
    generator = np.random.default_rng(config.seed)
    # Shapes are fixed for the whole benchmark, so let cuDNN pick its algorithms
    # once instead of re-deciding on every batch.
    torch.backends.cudnn.benchmark = True
    model = EventVideoNet(pretrained=config.pretrained, time_head=config.time_head).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps = max(
        1,
        config.epochs
        * math.ceil(
            (
                len(train_positive)
                + min(len(train_negative), int(config.negative_ratio * len(train_positive)))
            )
            / config.batch_size
        ),
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=steps, pct_start=0.25
    )
    history = []
    step = 0
    epochs = [
        sample_epoch(train_positive, train_negative, config.negative_ratio, generator)
        for _ in range(config.epochs)
    ]
    boundaries = np.cumsum([len(rows) for rows in epochs])
    # One dataloader for the whole fold: re-creating it per epoch would pay the
    # worker start-up cost 32 times over the benchmark.
    loader = DataLoader(
        EventCropDataset(
            store,
            np.concatenate(epochs),
            targets,
            offsets,
            xy,
            xy_mask,
            weights,
            augment=True,
            time_offsets=time_offsets,
            time_mask=time_mask,
            translation_store=translation_store,
            translation_radius=config.translation_radius,
            translation_seed=config.seed,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=4 if config.workers else None,
    )
    class_weight = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
    model.train()
    epoch = 0
    total = 0.0
    seen = 0
    consumed = 0
    for batch in loader:
        tight = batch["tight"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        mel = batch["mel"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(tight, context, mel)
            classification = nn.functional.cross_entropy(
                output["logits"].float(), target, reduction="none"
            )
            weight = weight * class_weight[target]
            loss = (classification * weight).sum() / weight.sum().clamp(min=1e-6)
            event = target > 0
            if event.any():
                loss = loss + config.offset_weight * nn.functional.smooth_l1_loss(
                    output["offset"].float()[event],
                    batch["offset"].to(device).float()[event],
                    beta=0.2,
                )
            mask = batch["xy_mask"].to(device).float() > 0
            if mask.any():
                loss = loss + config.xy_weight * nn.functional.smooth_l1_loss(
                    output["xy"].float()[mask],
                    batch["xy"].to(device).float()[mask],
                    beta=0.2,
                )
            time_mask_batch = batch["time_mask"].to(device).float() > 0
            if config.time_weight and time_mask_batch.any():
                soft = time_target(batch["time_target"].to(device).float()[time_mask_batch])
                log_probabilities = nn.functional.log_softmax(
                    output["time_logits"].float()[time_mask_batch], dim=1
                )
                loss = loss + config.time_weight * (-(soft * log_probabilities).sum(dim=1).mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step < steps - 1:
            schedule.step()
        step += 1
        total += float(loss.detach()) * len(target)
        seen += len(target)
        consumed += len(target)
        while epoch < len(boundaries) and consumed >= boundaries[epoch]:
            history.append(
                {"epoch": epoch, "rows": int(len(epochs[epoch])), "loss": total / max(seen, 1)}
            )
            epoch += 1
            total = 0.0
            seen = 0
    if seen:
        history.append({"epoch": epoch, "rows": seen, "loss": total / seen})
    return model, {"history": history, "steps": step}


@torch.no_grad()
def predict(
    model: EventVideoNet,
    store: CropStore,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 96,
    workers: int = 8,
) -> dict[str, np.ndarray]:
    model.eval()
    loader = DataLoader(
        EventCropDataset(store, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    probabilities = np.zeros((len(indices), len(CLASSES)), dtype=np.float32)
    offsets = np.zeros(len(indices), dtype=np.float32)
    xy = np.zeros((len(indices), 2), dtype=np.float32)
    time_offsets = np.zeros(len(indices), dtype=np.float32)
    time_confidence = np.zeros(len(indices), dtype=np.float32)
    # The whole local timing softmax, not only its mean and its max.  The grid
    # is the model's own buffer, repeated per row so the array stays aligned
    # with every other prediction column.
    time_pmf = np.zeros((len(indices), TIME_BINS), dtype=np.float32)
    cursor = 0
    for batch in loader:
        tight = batch["tight"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        mel = batch["mel"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(tight, context, mel)
        size = len(batch["row"])
        probabilities[cursor : cursor + size] = (
            torch.softmax(output["logits"].float(), dim=1).cpu().numpy()
        )
        offsets[cursor : cursor + size] = output["offset"].float().cpu().numpy()
        xy[cursor : cursor + size] = output["xy"].float().cpu().numpy()
        time_offsets[cursor : cursor + size] = output["time_offset"].float().cpu().numpy()
        time_confidence[cursor : cursor + size] = output["time_confidence"].float().cpu().numpy()
        time_pmf[cursor : cursor + size] = (
            torch.softmax(output["time_logits"].float(), dim=1).cpu().numpy()
        )
        cursor += size
    grid = model.time_grid.detach().float().cpu().numpy()
    if not np.array_equal(grid, timing.expected_grid()):
        raise ValueError("model time grid does not match the shared offset contract")
    prediction = {
        "probabilities": probabilities,
        "offset": offsets,
        "xy": xy,
        "time_offset": time_offsets,
        "time_confidence": time_confidence,
        timing.PMF_KEY: time_pmf,
        timing.GRID_KEY: np.tile(grid, (len(indices), 1)),
    }
    timing.arrays(prediction, rows=len(indices))
    return prediction


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #
def load_checkpoint(path: Path, device: torch.device) -> EventVideoNet:
    """Load a trained model without touching the network."""

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(bundle.get("classes", ())) != CLASSES:
        raise ValueError(f"checkpoint class order does not match runtime: {bundle.get('classes')}")
    model = EventVideoNet(pretrained=False, time_head=bundle.get("time_head", "fused"))
    model.load_state_dict(bundle["state_dict"])
    return model.to(device).eval()


def abstention_threshold(
    operating_point: str | None = DEFAULT_OPERATING_POINT,
    marginal_threshold: float | None = None,
) -> tuple[float, str]:
    """Return the runtime path-marginal threshold and where it came from.

    An explicit ``marginal_threshold`` wins; otherwise the named calibrated
    operating point is used, and ``None`` falls back to the uncalibrated 0.5 the
    first two passes shipped.
    """

    if marginal_threshold is not None:
        threshold = float(marginal_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("abstention threshold must be a finite probability in [0, 1]")
        return threshold, "explicit --marginal-threshold"
    if operating_point is None:
        return 0.5, "uncalibrated default"
    if operating_point not in ABSTENTION_OPERATING_POINTS:
        raise ValueError(f"unknown abstention operating point: {operating_point}")
    return (
        float(ABSTENTION_OPERATING_POINTS[operating_point]),
        f"{operating_point} calibrated on {ABSTENTION_CALIBRATION['fitted_on']}",
    )


def runtime_calibration_identity(
    checkpoint: Path,
    operating_point: str | None,
    marginal_threshold: float | None,
    settings: dict,
) -> dict:
    """Bind a named calibration to model bytes and disclose experimental settings.

    A different model requires an explicit uncalibrated threshold. Explicit
    decoder/crop experiments may reuse a reference threshold, but never its
    historical precision claim. Code revision remains a separate provenance axis.
    """
    model = file_record(checkpoint, role="event_model_checkpoint")
    named = operating_point is not None and marginal_threshold is None
    if named and model["sha256"] != CALIBRATED_CHECKPOINT_SHA256:
        raise ValueError(
            "checkpoint does not match the named event calibration; use its recorded model "
            "or an explicit --marginal-threshold for an uncalibrated experiment"
        )
    settings = {"court_geometry": "point_static", "court_frame_missing": "hold"} | settings
    deviations = {
        key: {"reference": expected, "requested": settings.get(key)}
        for key, expected in CALIBRATED_RUNTIME_SETTINGS.items()
        if settings.get(key) != expected
    }
    if settings.get("native_ground_evidence", False):
        deviations["native_ground_evidence"] = {"reference": False, "requested": True}
    if settings.get("native_net_evidence", False):
        deviations["native_net_evidence"] = {"reference": False, "requested": True}
    if settings.get("native_streak_fallback", False):
        deviations["native_streak_fallback"] = {"reference": False, "requested": True}
    return {
        "schema": "event_runtime_calibration_identity_v1",
        "checkpoint": model,
        "reference_checkpoint_sha256": CALIBRATED_CHECKPOINT_SHA256,
        "named_operating_point": operating_point if named else None,
        "status": (
            "reference_model_and_settings" if named and not deviations else "uncalibrated_variant"
        ),
        "setting_deviations": deviations,
        "note": "Historical calibration compatibility is not validation on this source video.",
    }


def crop_frame_rates(shards: list[dict]) -> dict[str, float]:
    """Require consistent declared source cadence across base/alternate crop shards."""
    rates: dict[str, float] = {}
    for shard in shards:
        try:
            broadcast = str(shard["broadcast"])
            rate = float(shard["fps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("missing or invalid crop frame rate") from error
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("crop frame rate must be positive and finite")
        if broadcast in rates and not math.isclose(rates[broadcast], rate, rel_tol=1e-12):
            raise ValueError(f"inconsistent crop frame rates for {broadcast}")
        rates[broadcast] = rate
    return rates


def _point_key(row: dict) -> str:
    clip = str(row["clip"])
    if "__" in clip:
        return clip
    return f"{row['match_id']}__{clip}"


OUTSIDE_PLAY_SCOPE_REASON = "outside_retained_play_interval"
UNSUPPORTED_CAMERA_VIEW_REASON = "unsupported_camera_view"


def _covered(spans: list, emission_frame: float) -> bool:
    return any(float(low) <= emission_frame <= float(high) for low, high in spans)


def _play_scope_failures(point_row: dict, emission_frame: float) -> list[str]:
    """Why a scoped point may not emit at this frame; empty when it may.

    A point row that declares ``play_scope_native_interval`` must also declare both
    the retained play span union it certifies and the reliable per-frame camera
    support it was certified against, so neither the holes a broadcast cut leaves
    inside the trim nor the frames with no registered play view are silently
    admitted.  The two are reported separately: a live-play span can still contain a
    close-up, and only the individual rows there are refused.  Rows of an unscoped
    point are unaffected.
    """

    if "play_scope_native_interval" not in point_row:
        return []
    spans = point_row.get("play_scope_native_spans")
    supported = point_row.get("play_scope_camera_supported_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("a scoped point gate row must carry its retained play spans")
    if not isinstance(supported, list) or not supported:
        raise ValueError("a scoped point gate row must carry its camera support spans")
    return [
        *([] if _covered(spans, emission_frame) else [OUTSIDE_PLAY_SCOPE_REASON]),
        *([] if _covered(supported, emission_frame) else [UNSUPPORTED_CAMERA_VIEW_REASON]),
    ]


def apply_arc_emission_gate(
    rows: list[dict],
    *,
    tracking_gate: dict | None = None,
    point_gate: dict | None = None,
) -> list[dict]:
    """Hold event claims whose emitted frame is outside every retained motion arc.

    Threshold abstention and upstream gating are separate decisions.  Every row
    preserves ``model_abstain`` so changing tracking scope cannot masquerade as a
    model-calibration change.  When gate artifacts are supplied, a missing point
    fails closed.

    A retained point that carries a ``play_scope_native_interval`` holds the rows
    emitted outside its retained play spans with an explicit
    ``outside_retained_play_interval`` reason and lets the rows inside keep their own
    model decision.  Rows of a point without that field behave exactly as before.
    """

    tracking = {
        f"{row['match_id']}__{row['clip']}": row for row in (tracking_gate or {}).get("rows", [])
    }
    validity = {
        f"{row['match_id']}__{row['clip']}": row for row in (point_gate or {}).get("rows", [])
    }
    output = []
    for source in rows:
        row = dict(source)
        key = _point_key(row)
        model_abstain = bool(row.get("abstain"))
        track_row = tracking.get(key)
        point_row = validity.get(key)
        emission_frame = float(row["frame"])

        matching_arcs = []
        retained_arcs = []
        if track_row is not None:
            matching_arcs = [
                arc
                for arc in track_row.get("arcs", [])
                if float(arc["start_frame"]) <= emission_frame <= float(arc["end_frame"])
            ]
            retained_arcs = [arc for arc in matching_arcs if arc.get("decision") == "retain"]
        tracking_hold = tracking_gate is not None and not retained_arcs
        play_scope_failures = (
            _play_scope_failures(point_row, emission_frame) if point_row is not None else []
        )
        point_hold = point_gate is not None and (
            point_row is None
            or point_row.get("decision") != "retain"
            or bool(play_scope_failures)
        )
        gate_held = tracking_hold or point_hold

        point_reasons = (
            list(point_row.get("reasons", []))
            if point_row is not None
            else (["missing_point_validity"] if point_gate is not None else [])
        )
        point_reasons.extend(play_scope_failures)
        if tracking_hold:
            point_reasons.append(
                "missing_tracking_arc" if track_row is None else "tracking_arc_abstained"
            )
        row.update(
            {
                "model_abstain": model_abstain,
                "abstain": model_abstain or gate_held,
                "gate_held": gate_held,
                "point_gate_verdict": (
                    point_row.get("decision")
                    if point_row is not None
                    else "hold"
                    if point_gate is not None
                    else "unavailable"
                ),
                "point_gate_failure_reasons": point_reasons,
                "tracking_arc_gate": {
                    "available": track_row is not None,
                    "emission_frame": emission_frame,
                    "decision": (
                        "retain"
                        if retained_arcs
                        else "hold"
                        if tracking_gate is not None
                        else "unavailable"
                    ),
                    "matching_arc_ids": [arc.get("arc_id") for arc in matching_arcs],
                    "retained_arc_ids": [arc.get("arc_id") for arc in retained_arcs],
                },
            }
        )
        output.append(row)
    return output


def resolve_grammar(root: Path, net_line, side_dead_band, event_track=LEGACY_EVENT_TRACK):
    """Return the decoder grammar and a record of where its net line came from.

    ``net_line="derived"`` -- the default -- pushes the net's own pixels through
    the event feature builder's own projection for every point of ``root``, so
    the side-alternation penalty reads a line the court geometry fixes instead of
    a constant.  ``"legacy"`` restores the shipped 0.5, and a float is used as
    given.  ``side_dead_band="derived"`` uses each point's own net thickness.

    ``event_track`` selects which ball track's declared pixel space the derived
    line is expressed in, so the grammar and the features it scores agree.
    """

    from cv.pipeline.event_grammar_decoder import LEGACY_NET_LINE, GrammarConfig
    from cv.pipeline.event_model_v2_features import derive_net_lines

    table = None
    if net_line == DERIVED_NET_LINE or side_dead_band == DERIVED_NET_LINE:
        table = derive_net_lines(Path(root), event_track=event_track)
    if net_line == DERIVED_NET_LINE:
        if not table:
            raise ValueError(
                f"no point of {root} has a camera bundle that can place the net; "
                f"pass --net-line legacy or an explicit value"
            )
        lines = {clip: float(item["net_line"]) for clip, item in table.items()}
        resolved_line = lines
        default = float(np.median(list(lines.values())))
        source = {
            "kind": DERIVED_NET_LINE,
            "points": len(lines),
            "fallback_net_line": default,
            "per_point_source": {
                name: sum(1 for item in table.values() if item["source"] == name)
                for name in sorted({item["source"] for item in table.values()})
            },
        }
    elif net_line in (None, LEGACY_NET_LINE_CHOICE):
        resolved_line = LEGACY_NET_LINE
        default = LEGACY_NET_LINE
        source = {"kind": LEGACY_NET_LINE_CHOICE, "net_line": LEGACY_NET_LINE}
    else:
        resolved_line = float(net_line)
        default = float(net_line)
        source = {"kind": "explicit", "net_line": float(net_line)}
    if side_dead_band == DERIVED_NET_LINE:
        resolved_band = (
            {clip: abs(float(item["net_span"])) for clip, item in (table or {}).items()}
            if table
            else 0.0
        )
        source["side_dead_band"] = DERIVED_NET_LINE
    else:
        resolved_band = float(side_dead_band or 0.0)
        source["side_dead_band"] = resolved_band
    return (
        GrammarConfig(
            net_line=resolved_line,
            side_dead_band=resolved_band,
            net_line_default=default,
        ),
        source,
    )


def verify_crop_manifest_coordinates(
    root: Path, manifest: dict, event_track: str = LEGACY_EVENT_TRACK
) -> dict:
    """Refuse a reused crop manifest whose ``court_y`` predates the column fix.

    ``court_x``/``court_y`` are carried in the crop manifest, not recomputed at
    decode time, so reusing a manifest cut before 2026-09-04 on a cohort whose
    coordinate sidecar declares native columns feeds the decoder the defective
    half-resolution projection that ``docs/wk1/net_line.md`` traced.  The fixed
    builder records the resolved column space in the feature manifest; a manifest
    that lacks it and comes from a cohort with a declared native column map is
    stale and the run stops rather than decoding against the wrong frame.
    """

    from cv.pipeline.automatic_ball_track import track_name as event_track_name
    from cv.pipeline.event_model_v2_features import (
        NATIVE_COLUMNS_KEY,
        track_pixel_space,
    )

    # The check must open the track the run actually reads; validating the
    # legacy artifact's pixel space would say nothing about a selected one.
    selected = event_track_name(event_track)
    recorded = (manifest.get("feature_manifest") or {}).get("track_pixel_space")
    active = Path(root) / "active_play_v1.json"
    native: list[str] = []
    if active.exists():
        for broadcast in sorted({key.split("/", 1)[0] for key in json.loads(active.read_text())}):
            track = Path(root) / broadcast / selected
            if track.exists() and track_pixel_space(track).space == NATIVE_COLUMNS_KEY:
                native.append(broadcast)
    if native and not recorded:
        raise ValueError(
            f"the crop manifest was built before the track column fix and its court_x/court_y "
            f"are the half-resolution projection ({len(native)} broadcasts of {root} declare "
            f"native track columns); rebuild it with cv.pipeline.event_crops.build_dataset or "
            f"pass a corrected manifest"
        )
    return {
        "native_column_broadcasts": len(native),
        "manifest_records_column_space": bool(recorded),
    }


def run(
    root: Path,
    checkpoint: Path,
    output: Path,
    crops: Path,
    *,
    device: torch.device | None = None,
    jobs: int = 8,
    workers: int = 8,
    operating_point: str | None = DEFAULT_OPERATING_POINT,
    marginal_threshold: float | None = None,
    reuse_crops: bool = True,
    correct_frames: bool = True,
    tracking_gate: Path | None = None,
    point_gate: Path | None = None,
    alternate_crops: Path | None = None,
    net_line: float | str | None = DEFAULT_NET_LINE,
    side_dead_band: float | str = DEFAULT_SIDE_DEAD_BAND,
    emission_mode: str = "best_path",
    terminal_second_bounce: bool = True,
    supported_impulse_events: bool = False,
    native_wing_sampling: str = "consecutive",
    native_streak_fallback: bool = False,
    model_accepted_tracking_held_events: bool = False,
    native_net_evidence: bool = False,
    native_ground_evidence: bool = False,
    predictions_output: Path | None = None,
    court_geometry: str = "point_static",
    court_frame_missing: str = "hold",
    event_track: str = LEGACY_EVENT_TRACK,
    live_shot_camera: bool = False,
) -> dict:
    """Score a runner's artifacts and write ``event_model_v3``-shaped emissions.

    Everything on this path is label-free: the crops come from the ball track and
    the active-play spans, the model is a frozen checkpoint, and the decoder is
    the point grammar.  The emission rows carry native ``x``/``y``, a sub-frame
    time and an abstention, so they drop into the same consumers.
    """

    from cv.pipeline.event_crops import ensure_runtime_dataset
    from cv.pipeline.event_model_v2_features import validate_court_transport

    validate_court_transport(court_geometry, court_frame_missing)
    validate_event_track(event_track)
    transport_args = (
        {"court_geometry": court_geometry, "court_frame_missing": court_frame_missing}
        if court_geometry != "point_static"
        else {}
    )
    # As with the transport arguments, the selected track enters the runtime
    # identity only when it is not the default, so a default run's calibration
    # identity keeps its shape and any selection is its own cache entry.
    track_args = {"event_track": event_track} if event_track != LEGACY_EVENT_TRACK else {}
    from cv.pipeline.event_grammar_decoder import decode, emission_rows, point_end_rows
    from cv.pipeline.event_impulse_support import (
        TrackSnapshot,
        recover_events,
        sampling_configuration,
        streak_fallback_configuration,
    )

    sampling_configuration(native_wing_sampling)
    if native_wing_sampling != "consecutive" and not supported_impulse_events:
        raise ValueError("native wing sampling requires supported impulse events")
    streak_configuration = streak_fallback_configuration(
        native_streak_fallback, supported_impulse_events=supported_impulse_events
    )
    if supported_impulse_events and (tracking_gate is None or point_gate is None):
        raise ValueError("supported impulse events require explicit tracking and point gates")
    if model_accepted_tracking_held_events and (tracking_gate is None or point_gate is None):
        raise ValueError(
            "model-accepted tracking-held admission requires explicit tracking and point gates"
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold, threshold_source = abstention_threshold(operating_point, marginal_threshold)
    calibration_identity = runtime_calibration_identity(
        checkpoint,
        operating_point,
        marginal_threshold,
        {
            "net_line": net_line,
            "side_dead_band": side_dead_band,
            "correct_frames": correct_frames,
            "emission_mode": emission_mode,
            "terminal_second_bounce": terminal_second_bounce,
            "alternate_crops": alternate_crops is not None,
            "supported_impulse_events": supported_impulse_events,
            **(
                {"native_wing_sampling": native_wing_sampling}
                if native_wing_sampling != "consecutive"
                else {}
            ),
            **({"native_streak_fallback": True} if native_streak_fallback else {}),
            **({"native_net_evidence": True} if native_net_evidence else {}),
            **({"native_ground_evidence": True} if native_ground_evidence else {}),
            **({"live_shot_camera": True} if live_shot_camera else {}),
            **transport_args,
            **track_args,
        },
    )
    crop_cache = ensure_runtime_dataset(
        root,
        crops,
        jobs,
        reuse=reuse_crops,
        live_shot_camera=live_shot_camera,
        **transport_args,
        **track_args,
    )
    stores = [CropStore.open(crops)]
    coordinate_check = verify_crop_manifest_coordinates(root, stores[0].manifest, event_track)
    alternate_crop_cache = None
    if alternate_crops is not None:
        alternate_crop_cache = ensure_runtime_dataset(
            root,
            alternate_crops,
            jobs,
            reuse=reuse_crops,
            alternate=True,
            live_shot_camera=live_shot_camera,
            **transport_args,
            **track_args,
        )
        stores.append(CropStore.open(alternate_crops))
    fps = crop_frame_rates([shard for one in stores for shard in one.manifest["shards"]])
    model = load_checkpoint(checkpoint, device)
    predictions = [
        predict(model, one, np.arange(len(one.shard_of_row)), device, workers=workers)
        for one in stores
    ]
    prediction = {
        name: np.concatenate([one[name] for one in predictions]) for name in predictions[0]
    }
    store = stores[0] if len(stores) == 1 else MergedCropStore(stores)
    rows = np.arange(len(store.clips()))
    # Persist the classifier row space consumed by event_paths.  Keeping this
    # beside every default emission receipt avoids a second GPU replay and,
    # more importantly, binds alternative-path decoding to the exact pixels
    # and model outputs used for the firm stream.
    prediction_artifact = predictions_output or output.with_suffix(".predictions.npz")
    prediction_artifact.parent.mkdir(parents=True, exist_ok=True)
    sources = (
        store.proposal_sources()
        if hasattr(store, "proposal_sources")
        else np.asarray(["track"] * len(rows))
    )
    persisted_prediction = {
        **prediction,
        "clips": store.clips(),
        "broadcasts": store.broadcasts(),
        "frames": store.frames(),
        "court_y": store.court_y(),
        "centres": store.centres(),
        "track_observed": store.track_observed(),
        "source": sources,
    }
    if any(len(values) != len(rows) for values in persisted_prediction.values()):
        raise ValueError("classifier prediction arrays are not row-aligned")
    np.savez(prediction_artifact, **persisted_prediction)
    grammar, net_line_source = resolve_grammar(root, net_line, side_dead_band, event_track)
    grammar = replace(grammar, terminal_second_bounce=terminal_second_bounce)
    events = decode(
        prediction["probabilities"],
        store.clips(),
        store.frames(),
        store.court_y(),
        config=grammar,
        emission_mode=emission_mode,
    )
    emissions = emission_rows(
        events,
        store,
        prediction["probabilities"],
        prediction["offset"],
        prediction["xy"],
        store.centres(),
        fps,
        threshold,
        time_offset=prediction["time_offset"],
        time_distribution=timing.arrays(prediction, rows=len(store.frames())),
        correct_frames=correct_frames,
    )
    net_evidence_record = None
    ground_evidence_record = None
    if native_ground_evidence:
        from cv.pipeline.event_ground_evidence import (
            CONFIG as GROUND_CONFIG,
            GroundSnapshot,
            condition_emissions as ground_condition,
        )
        from cv.pipeline.event_net_evidence import CONFIG as NET_CONFIG, NetSnapshot

        ground_snapshot = GroundSnapshot.load(root, sorted(set(store.broadcasts().tolist())))
        net_snapshot = (
            NetSnapshot.load(root, sorted(set(store.broadcasts().tolist())))
            if native_net_evidence
            else None
        )
        emissions, ground_audit = ground_condition(
            persisted_prediction,
            store,
            grammar,
            fps,
            threshold,
            emission_mode=emission_mode,
            correct_frames=correct_frames,
            baseline=emissions,
            snapshot=ground_snapshot,
            net_snapshot=net_snapshot,
        )
        evidence_path = output.with_suffix(".ground_evidence.json")
        evidence_path.write_text(
            json.dumps(
                {
                    "configuration": GROUND_CONFIG,
                    "common_net_configuration": NET_CONFIG if native_net_evidence else None,
                    "nodes": ground_audit,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        ground_evidence_record = dict(
            configuration=GROUND_CONFIG,
            inputs=ground_snapshot.track.records
            + (net_snapshot.track.records if net_snapshot else []),
            artifact=file_record(evidence_path, role="native_ground_decoder_evidence"),
            supported_existing_bounce_nodes=sum(
                r["supported"] and r["event_type"] == "bounce" for r in ground_audit
            ),
            network_probabilities_changed=False,
            original_decoder_decisions_preserved=True,
            common_net_configuration=NET_CONFIG if native_net_evidence else None,
        )
        ground_snapshot.track.assert_unchanged()
        if net_snapshot is not None:
            net_snapshot.track.assert_unchanged()
    elif native_net_evidence:
        from cv.pipeline.event_net_evidence import (
            CONFIG as NET_CONFIG,
            NetSnapshot,
            condition_emissions,
        )

        net_snapshot = NetSnapshot.load(root, sorted(set(store.broadcasts().tolist())))
        emissions, net_audit = condition_emissions(
            persisted_prediction,
            store,
            grammar,
            fps,
            threshold,
            emission_mode=emission_mode,
            correct_frames=correct_frames,
            baseline=emissions,
            snapshot=net_snapshot,
        )
        evidence_path = output.with_suffix(".net_evidence.json")
        evidence_path.write_text(
            json.dumps(
                {"configuration": NET_CONFIG, "nodes": net_audit},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        net_evidence_record = {
            "configuration": NET_CONFIG,
            "inputs": net_snapshot.track.records,
            "artifact": file_record(evidence_path, role="native_net_decoder_evidence"),
            "supported_existing_net_nodes": sum(row["supported"] for row in net_audit),
            "network_probabilities_changed": False,
            "original_decoder_decisions_preserved": True,
        }
        net_snapshot.track.assert_unchanged()
    impulse_snapshot = (
        TrackSnapshot.load(
            root,
            sorted({row["match_id"] for row in emissions}),
            extra_paths=(tracking_gate, point_gate),
        )
        if supported_impulse_events
        else None
    )
    tracking_gate_payload = json.loads(tracking_gate.read_text()) if tracking_gate else None
    point_gate_payload = json.loads(point_gate.read_text()) if point_gate else None
    emissions = apply_arc_emission_gate(
        emissions,
        tracking_gate=tracking_gate_payload,
        point_gate=point_gate_payload,
    )
    streak_support_record = None
    if impulse_snapshot is not None:
        pictures = None
        if streak_configuration is not None:
            from cv.pipeline.event_native_pictures import NativePictureSnapshot

            # Bound onto the run's existing track snapshot, so every consumed native
            # picture and cadence document lands in the ordinary impulse-support inputs
            # below and is re-checked by the snapshot's own assert_unchanged.
            pictures = NativePictureSnapshot.bind(
                root, sorted({row["match_id"] for row in emissions}), impulse_snapshot
            )
        emissions = recover_events(
            emissions,
            tracking_gate_payload,
            impulse_snapshot,
            native_wing_sampling=native_wing_sampling,
            native_streak_fallback=native_streak_fallback,
            pictures=pictures,
        )
        if pictures is not None:
            support_path = output.with_suffix(".streak_support.json")
            support_path.parent.mkdir(parents=True, exist_ok=True)
            support_path.write_text(
                json.dumps(
                    pictures.support_document(streak_configuration),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            reports = [
                row.get("impulse_support", {}).get("native_streak_fallback", {})
                for row in emissions
            ]
            streak_support_record = {
                "configuration": streak_configuration,
                "artifact": file_record(support_path, role="native_streak_fallback_support"),
                "pictures_consumed": len(pictures.bound_pictures),
                "refusal_counts": pictures.refusal_counts(),
                # Evidence that was looked for and was not there: a hash cannot bind it,
                # so a resume checks these paths are still absent before reusing a hold.
                "expected_absent_inputs": len(pictures.expected_absent),
                "samples_consulted": sum(r.get("measured_support_consulted", 0) for r in reports),
                "samples_supported": sum(r.get("measured_support_passed", 0) for r in reports),
                "qualified_wings": sum(
                    r.get("consulted") is True and r.get("qualified") is True for r in reports
                ),
                "certifies_physical_ending": False,
            }
    admission_record = None
    if model_accepted_tracking_held_events:
        from cv.pipeline.event_model_admission import (
            admission_record as model_admission_record,
            admit_model_accepted_events,
        )

        # Applied after any impulse recovery, so a row an observed impulse already
        # supported is no longer held and is never admitted twice, and before the
        # ending rule, which reads the degraded identity block these rows carry.
        emissions = admit_model_accepted_events(emissions, tracking_gate_payload)
        admission_record = model_admission_record(emissions, enabled=True)
    emissions.extend(point_end_rows(emissions))
    if file_sha256(checkpoint) != calibration_identity["checkpoint"]["sha256"]:
        raise ValueError("event checkpoint changed during inference")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(emissions, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "automatic_event_video_model_run_v1",
        "root": str(Path(root).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "calibration_identity": calibration_identity,
        "device": str(device),
        "crops": str(Path(crops).resolve()),
        "alternate_crops": str(Path(alternate_crops).resolve()) if alternate_crops else None,
        "crop_cache": crop_cache,
        "alternate_crop_cache": alternate_crop_cache,
        "rows_scored": int(len(rows)),
        "classifier_predictions": {
            **file_record(prediction_artifact, role="event_classifier_predictions"),
            "layout": sorted(persisted_prediction),
            "rows": int(len(rows)),
            "local_time_distribution": {
                "schema": timing.SCHEMA,
                "status": timing.status(persisted_prediction),
                "arrays": list(timing.KEYS),
                "bins": int(TIME_BINS),
                "offset_reference": timing.OFFSET_REFERENCE,
                "calibrated": False,
            },
        },
        "emissions": len(emissions),
        "abstained_emissions": sum(row["abstain"] is True for row in emissions),
        "model_abstained_emissions": sum(row.get("model_abstain") is True for row in emissions),
        "gate_held_emissions": sum(row.get("gate_held") is True for row in emissions),
        "point_end_emissions": sum(row["event_type"] == "point_end" for row in emissions),
        "marginal_threshold": threshold,
        "abstention": {
            "operating_point": operating_point if marginal_threshold is None else None,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "calibration_scope": (
                "historical_reference_model_and_settings"
                if calibration_identity["status"] == "reference_model_and_settings"
                else "reference_only_not_calibrated_for_this_variant"
            ),
            "calibration": ABSTENTION_CALIBRATION,
        },
        "decoder": "event_grammar_decoder linear-chain Viterbi with sum-product marginals",
        "grammar": grammar.as_dict(),
        "net_line_source": net_line_source,
        "crop_manifest_coordinates": coordinate_check,
        "emission_mode": emission_mode,
        "frame_correction": correct_frames,
        "point_grammar": "absent; see the point_grammar block on every emission",
        "labels_or_reviewed_inputs": [],
        "tracking_arc_gate": str(tracking_gate.resolve()) if tracking_gate else None,
        "point_validity_gate": str(point_gate.resolve()) if point_gate else None,
        "impulse_event_support": {
            "enabled": supported_impulse_events,
            "configuration": sampling_configuration(native_wing_sampling)
            if supported_impulse_events
            else None,
            "inputs": impulse_snapshot.records if impulse_snapshot is not None else [],
            "recovered_events": sum(
                row.get("impulse_support", {}).get("supported") is True for row in emissions
            ),
            "scope": "event claims only; track frames and terminal evidence are not restored",
        },
    }
    if admission_record is not None:
        manifest["model_accepted_tracking_held_events"] = admission_record
    if event_track != LEGACY_EVENT_TRACK:
        # Bind the selected track and its declared guide in the run manifest, so
        # the emissions name the exact ball evidence they were decoded from.
        manifest["ball_track"] = (
            (stores[0].manifest.get("feature_manifest") or {}).get("ball_track")
            or stores[0].manifest.get("ball_track")
            or {"event_track": event_track}
        )
    if streak_support_record is not None:
        manifest["native_streak_fallback"] = streak_support_record
    if ground_evidence_record is not None:
        manifest["native_ground_evidence"] = ground_evidence_record
    if net_evidence_record is not None:
        manifest["native_net_evidence"] = net_evidence_record
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def build_parser():
    """The runtime command line, split out so the flags can be asserted."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="optional NPZ path; defaults beside --output as <stem>.predictions.npz",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--court-geometry", choices=("point_static", "reliable_per_frame"), default="point_static"
    )
    parser.add_argument("--court-frame-missing", choices=("hold", "point_static"), default="hold")
    parser.add_argument(
        "--event-track",
        choices=(LEGACY_EVENT_TRACK, CURRENT_EVENT_TRACK),
        default=LEGACY_EVENT_TRACK,
        help="which composed ball track supplies the crop centres, the numeric "
        "observations and the derived net line; the default is unchanged",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        type=torch.device,
        help="explicit inference device, e.g. cpu or cuda:1 (logical visible-device index)",
    )
    parser.add_argument(
        "--operating-point",
        choices=(*sorted(ABSTENTION_OPERATING_POINTS), "none"),
        default=DEFAULT_OPERATING_POINT,
        help="calibrated abstention threshold to apply; 'none' keeps the "
        "uncalibrated 0.5 the first two passes shipped",
    )
    parser.add_argument(
        "--net-line",
        default=DEFAULT_NET_LINE,
        help=(
            "'derived' (default) builds the court_y that separates the two halves per point "
            "from the net's own pixels; 'legacy' restores the 0.5 that was in the wrong units "
            "and called 98.2%% of rows far; a float is used as given"
        ),
    )
    parser.add_argument(
        "--side-dead-band",
        default=DEFAULT_SIDE_DEAD_BAND,
        help=(
            "'derived' (default) uses each point's own net thickness as the half-width around "
            "the net line in which the side is treated as unknown; a float is used as given"
        ),
    )
    parser.add_argument(
        "--emission-mode",
        choices=("best_path", "path_marginal", "nbest_union"),
        default="best_path",
        help="which lattice nodes leave the decoder",
    )
    parser.add_argument(
        "--marginal-threshold",
        type=float,
        help="override the calibrated threshold with an explicit path marginal",
    )
    parser.add_argument(
        "--alternate-crops",
        type=Path,
        help="directory of alternate proposal crops; scored with the same "
        "checkpoint and decoded on one lattice with the base rows",
    )
    parser.add_argument("--rebuild-crops", action="store_true")
    parser.set_defaults(terminal_second_bounce=True)
    parser.add_argument(
        "--terminal-second-bounce",
        dest="terminal_second_bounce",
        action="store_true",
        help="enable the default general second-bounce grammar (compatibility spelling)",
    )
    parser.add_argument(
        "--no-second-bounce",
        dest="terminal_second_bounce",
        action="store_false",
        help="disable the general second-bounce grammar for an explicit historical replay",
    )
    parser.add_argument("--no-frame-correction", action="store_true")
    parser.add_argument(
        "--native-wing-sampling",
        choices=("consecutive", "available_native"),
        default="consecutive",
        help="use original measured wing timestamps; requires supported impulse events",
    )
    parser.add_argument(
        "--supported-impulse-events",
        action="store_true",
        help="optional two-wing support for tracking-held contacts/bounces; not a calibrated default",
    )
    parser.add_argument(
        "--native-streak-fallback",
        action="store_true",
        help="let one wing satisfy the existing localization allowance using streak "
        "extents measured in the original pictures; requires supported impulse events",
    )
    parser.add_argument(
        "--model-accepted-tracking-held-events",
        action="store_true",
        help="optional admission of model-accepted contacts/bounces whose only remaining "
        "veto is tracking-arc fit quality; the hold is kept as evidence and the row "
        "cannot certify an ending; default disabled",
    )
    parser.add_argument(
        "--native-ground-evidence",
        action="store_true",
        help="optional uncalibrated native-motion/ground evidence before decoding",
    )
    parser.add_argument(
        "--native-net-evidence",
        action="store_true",
        help="optional uncalibrated native-motion/net-envelope evidence before decoding",
    )
    parser.add_argument(
        "--tracking-gate",
        type=Path,
        help="per-arc untouched_tracking_point_gate_v1.json; missing points fail closed",
    )
    parser.add_argument(
        "--point-gate",
        type=Path,
        help="camera/timing point_validity_gate_v1.json; missing points fail closed",
    )
    parser.add_argument(
        "--live-shot-camera",
        action="store_true",
        help="scope a clip held only for no_play_camera_shot to its wide shot so the "
        "event model runs there; default disabled, same switch as the S6 release",
    )
    return parser


def _side_argument(value):
    """Keep ``derived``/``legacy`` as names and read anything else as a float."""

    if value in (None, DERIVED_NET_LINE, LEGACY_NET_LINE_CHOICE):
        return value
    return float(value)


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            run(
                args.root,
                args.checkpoint,
                args.output,
                args.crops,
                device=args.device,
                jobs=args.jobs,
                workers=args.workers,
                operating_point=None if args.operating_point == "none" else args.operating_point,
                marginal_threshold=args.marginal_threshold,
                reuse_crops=not args.rebuild_crops,
                correct_frames=not args.no_frame_correction,
                net_line=_side_argument(args.net_line),
                side_dead_band=_side_argument(args.side_dead_band),
                emission_mode=args.emission_mode,
                terminal_second_bounce=args.terminal_second_bounce,
                supported_impulse_events=args.supported_impulse_events,
                native_wing_sampling=args.native_wing_sampling,
                native_streak_fallback=args.native_streak_fallback,
                model_accepted_tracking_held_events=args.model_accepted_tracking_held_events,
                native_net_evidence=args.native_net_evidence,
                native_ground_evidence=args.native_ground_evidence,
                tracking_gate=args.tracking_gate,
                point_gate=args.point_gate,
                alternate_crops=args.alternate_crops,
                predictions_output=args.predictions_output,
                court_geometry=args.court_geometry,
                court_frame_missing=args.court_frame_missing,
                event_track=args.event_track,
                live_shot_camera=args.live_shot_camera,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ABSTENTION_CALIBRATION",
    "DEFAULT_NET_LINE",
    "DEFAULT_SIDE_DEAD_BAND",
    "DERIVED_NET_LINE",
    "LEGACY_NET_LINE_CHOICE",
    "resolve_grammar",
    "ABSTENTION_OPERATING_POINTS",
    "CLASSES",
    "DEFAULT_OPERATING_POINT",
    "abstention_threshold",
    "OUTSIDE_PLAY_SCOPE_REASON",
    "UNSUPPORTED_CAMERA_VIEW_REASON",
    "apply_arc_emission_gate",
    "build_parser",
    "load_checkpoint",
    "run",
    "CONTEXT_HEIGHT",
    "CONTEXT_WIDTH",
    "CropStore",
    "MergedCropStore",
    "EventCropDataset",
    "EventVideoNet",
    "MEL_BANDS",
    "MEL_FRAMES",
    "SEQUENCE_LENGTH",
    "TIGHT_HEIGHT",
    "TIGHT_WIDTH",
    "TIME_BINS",
    "TIME_HEADS",
    "TIME_RADIUS",
    "TRUNK_TAPS",
    "TemporalTimeHead",
    "TrainConfig",
    "predict",
    "time_target",
    "sample_epoch",
    "train_model",
]
