# Models and weights

No weights are stored in this repository. The authors' trained weights are published as assets of
the GitHub Release `weights-v1` together with a model card (`MODEL_CARD.md`), `SHA256SUMS` and
`install_weights.sh`, which downloads each file, checks its SHA-256 and puts it where the code
reads it:

```bash
export TENNIS_DATA_ROOT=/path/to/tennis-data
curl -fsSLO https://github.com/maxsegan/openhawk/releases/download/weights-v1/install_weights.sh
bash install_weights.sh
```

| asset (release `weights-v1`) | bytes | sha256 |
|---|---|---|
| `event_video_model.pt` | 179,124,659 | `fc26570b9dc505afeb7616d461a160a6dac15ab7c65322e70f5e44816d698124` |
| `wasb_native_crop_ep3.pth.tar` | 6,109,005 | `ff315dd956eb0bd0ec6e53175d973c328e24b52f7f336287390b2e3023ee3124` |
| `tracknetv2_native_crop_ep3.pth.tar` | 45,412,482 | `a8d2c5be599aee92c974e6b6f412f00893f5dab78e0262f2540d8014e3f6a9a6` |
| `view_head_v1.json` | 58,516 | `5eeb17107a4c39756e6f3ce32805de5cce07bbdfdf3c43232f9dd4ed2fe28c5f` |
| `serve_heads_v1.pkl` | 908,861 | `b2f8b2be4f8a57e3dc8422960a2b27aa8b6ddc6cc3958f34f6ac7e4585d675e5` |
| `serve_location_prior_v1.json` | 1,236,030 | `9251d6a79c529e583a2d68906fe6a4c23fd5e225991a40a58784be4f6cd1a420` |

**Built with DINOv3.** The Stage-1 view head is trained on DINOv3 ViT-S/16 features; the release
ships a copy of the DINOv3 License (`DINOv3_LICENSE.md`) with it.

Checkpoints and `.pkl` files are pickles, so verify the hash before loading. Third-party weights
are obtained from their own distributors (table below). The code resolves paths relative to
`TENNIS_DATA_ROOT` unless noted.

| stage | model | where the code looks | how to obtain | licence |
|---|---|---|---|---|
| S4 ball | WASB-SBDT and TrackNetV2 tennis weights (`wasb_tennis_best.pth.tar`, `tracknetv2_tennis_best.pth.tar`) and the WASB-SBDT source | `$TENNIS_TRACKER_ROOT` (default `~/.cache/tennis-trackers/WASB-SBDT`) / `pretrained_weights/` | clone github.com/nttcom/WASB-SBDT and download its tennis weights (see its MODEL_ZOO) | MIT (NTT Communications) |
| S4 ball | fine-tuned native-crop refinement weights (`{wasb,tracknetv2}_native_crop_ep3.pth.tar`) | `models/pipeline/ball_finetune_v1/` | release `weights-v1` | Apache-2.0; derived from the MIT WASB-SBDT weights |
| S3 players | Ultralytics YOLOv8m (`yolov8m.pt`) | `models/pipeline/yolov8m.pt` | Ultralytics release assets | AGPL-3.0 (Ultralytics) |
| S3 pose | Ultralytics YOLO26m-pose (`yolo26m-pose.pt`) | `models/pipeline/yolo26m-pose.pt` | Ultralytics release assets | AGPL-3.0 (Ultralytics) |
| S1 points | view head and serve heads (`view_head_v1.json`, `serve_heads_v1.pkl`) | `models/pipeline/s1_stage_v1/` | release `weights-v1` | Apache-2.0; the view head is built with DINOv3 and also subject to the DINOv3 License (shipped in the release); it needs the DINOv3 ViT-S/16 backbone via timm |
| S1 score | Qwen3.8-27B-FP8 scoreboard reader, served locally with vLLM | `models/vlm_frontier/Qwen3.8-27B-FP8` | Hugging Face | see the model card (`cv/pipeline/vlm_models.json` records licences) |
| S5 events | event video model (`event_video_model.pt`: 3D ResNet-18 over native crops, with audio and track features) and its calibrated operating threshold | path in `cv/pipeline/product_runner.py` (`EVENT_MODEL`) | release `weights-v1` | Apache-2.0; initialised from torchvision Kinetics/ImageNet weights (see the model card) |
| S5 events | torchvision Kinetics-400 `r3d_18` and ImageNet `resnet18` initialisations | `~/.cache/torch/hub/checkpoints/` | downloaded by torchvision | torchvision code BSD-3-Clause; check the pretrained weights' terms |
| S6 | serve-location prior (`serve_location_prior_v1.json`) | `processed/serve_prior/` | release `weights-v1` (per-player and global priors, fitted by the authors from the public Hawk-Eye CourtVision corpus) | Apache-2.0; underlying data not redistributed |
| rollback only | Gemini Flash and Claude Opus via OpenRouter (paid event cascade) | `OPENROUTER_API_KEY` | provider accounts | provider terms |

Everything else the fitter uses is in the repository: the production policy
(`cv/pipeline/product_s6_policy.json`), gates (`cv/pipeline/default_gates.json`), the bounce law
(`physics/bounce_law_hawkeye_holdout.json` plus `physics/bounce_reference.py`), and public player
heights (`cv/pipeline/player_biometrics.json`, from ATP/WTA profiles).

The serve-location prior is fitted from the same public Hawk-Eye CourtVision corpus as the bounce
law. The release keeps its per-player and global priors and drops the per-match priors, which are
keyed by corpus match ids and never apply to a new video. The physics, fitting and scoring code
and the unit tests run without any weights.
