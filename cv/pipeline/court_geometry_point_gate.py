"""Gate per-point court geometry before court-aware tracking or event inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cv.pipeline.court_geometry_gate import assess_court_homography  # noqa: E402

def build_report(manifest: dict, output_root: Path) -> dict:
    rows = []
    for match in manifest["matches"]:
        match_id = match["id"]
        match_root = output_root / match_id
        calibration = np.load(match_root / "court_H_per_point.npz")
        homographies = {
            int(point): homography
            for point, homography in zip(
                calibration["pts"],
                calibration["H"],
            )
        }
        point_ids = match.get(
            "point_ids",
            range(1, int(manifest["points_per_match"]) + 1),
        )
        for point in point_ids:
            clip = f"pt{point:04d}"
            frame_path = next(
                (match_root / "audit_frames_native_1080" / clip).glob("f_*.jpg")
            )
            image = cv2.imread(str(frame_path))
            if image is None:
                raise FileNotFoundError(frame_path)
            height, width = image.shape[:2]
            homography = homographies.get(point)
            assessment = (
                assess_court_homography(homography, width, height)
                if homography is not None and np.isfinite(homography).all()
                else {"valid": False, "reasons": ["missing_automatic_homography"]}
            )
            rows.append(
                {
                    "point": f"{match_id}/{clip}",
                    "match_id": match_id,
                    "clip": clip,
                    "image_width": width,
                    "image_height": height,
                    **assessment,
                }
            )
    return {
        "schema": "court_geometry_point_gate_v1",
        "labels_loaded": False,
        "points": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "invalid": sum(not row["valid"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    report = build_report(json.loads(args.manifest.read_text()), args.out)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"court geometry valid {report['valid']}/{report['points']} "
        f"-> {args.output}"
    )
    for row in report["rows"]:
        if not row["valid"]:
            print(f"  {row['point']}: {','.join(row['reasons'])}")


if __name__ == "__main__":
    main()
