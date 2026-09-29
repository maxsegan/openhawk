# Models and weights

No weights are distributed in this repository. The code resolves paths relative to
`TENNIS_DATA_ROOT` unless noted.

| stage | model | where the code looks | how to obtain | licence |
|---|---|---|---|---|
| S4 ball | WASB-SBDT and TrackNetV2 tennis weights (`wasb_tennis_best.pth.tar`, `tracknetv2_tennis_best.pth.tar`) and the WASB-SBDT source | `$TENNIS_TRACKER_ROOT` (default `~/.cache/tennis-trackers/WASB-SBDT`) / `pretrained_weights/` | clone github.com/nttcom/WASB-SBDT and download its tennis weights (see its MODEL_ZOO) | MIT (NTT Communications) |
| S4 ball | fine-tuned native-crop refinement weights (`{wasb,tracknetv2}_native_crop_ep3.pth.tar`) | `models/pipeline/ball_finetune_v1/` | trained by the authors on development broadcasts; on request | authors' |
| S3 players | Ultralytics YOLOv8m (`yolov8m.pt`) | `models/pipeline/yolov8m.pt` | Ultralytics release assets | AGPL-3.0 (Ultralytics) |
| S3 pose | Ultralytics YOLO26m-pose (`yolo26m-pose.pt`) | `models/pipeline/yolo26m-pose.pt` | Ultralytics release assets | AGPL-3.0 (Ultralytics) |
| S1 points | view head and serve heads (`view_head_v1.json`, `serve_heads_v1.pkl`) | `models/pipeline/s1_stage_v1/` | trained by the authors; on request | authors' |
| S1 score | Qwen3.8-27B-FP8 scoreboard reader, served locally with vLLM | `models/vlm_frontier/Qwen3.8-27B-FP8` | Hugging Face | see the model card (`cv/pipeline/vlm_models.json` records licences) |
| S5 events | event video model (`event_video_model.pt`: 3D ResNet-18 over native crops, with audio and track features) and its calibrated operating threshold | path in `cv/pipeline/product_runner.py` (`EVENT_MODEL`) | trained by the authors on development labels; on request | authors' |
| S5 events | torchvision Kinetics-400 `r3d_18` and ImageNet `resnet18` initialisations | `~/.cache/torch/hub/checkpoints/` | downloaded by torchvision | torchvision code BSD-3-Clause; check the pretrained weights' terms |
| S6 | serve-location prior (`serve_location_prior_v1.json`) | `processed/serve_prior/` | fitted by the authors from automatic development outputs; on request | authors' |
| rollback only | Gemini Flash and Claude Opus via OpenRouter (paid event cascade) | `OPENROUTER_API_KEY` | provider accounts | provider terms |

Everything else the fitter uses is in the repository: the production policy
(`cv/pipeline/product_s6_policy.json`), gates (`cv/pipeline/default_gates.json`), the bounce law
(`physics/bounce_law_hawkeye_holdout.json` plus `physics/bounce_reference.py`), and public player
heights (`cv/pipeline/player_biometrics.json`, from ATP/WTA profiles).

Without the authors' trained weights the upstream stages S1 and S5 cannot run as measured; the
physics, fitting and scoring code and the unit tests run without any weights.
