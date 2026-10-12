"""Tests for export_point_3d.py — the portal 3D exporter.

Runs under the system interpreter (stdlib only). Uses tiny synthetic CSV/gz fixtures so it
does not depend on the large refined artifacts, plus one optional smoke test against the
real dev9 artifacts when they are present on the box.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import export_point_3d as ex  # noqa: E402


BALL_HEADER = (
    "match,clip,frame,x,y,z,vx,vy,vz,spin_x,spin_y,spin_z,confidence,"
    "ci95_x_m,ci95_y_m,ci95_z_m,velocity_ci95_ms,spin_ci95_rad_s,segment,fit_rms_px,source"
)
CONTACT_HEADER = (
    "match,clip,contact_index,frame,frame_detector,frame_delta,frame_lo,frame_hi,t_lo_s,t_hi_s,"
    "timing_source,timing_residual_px,side,phase,status,x,y,z,ci95_x_m,ci95_y_m,ci95_z_m,"
    "vx_in,vy_in,vz_in,speed_in,vx_out,vy_out,vz_out,speed_out,junction_gap_pass1_m,"
    "junction_gap_pass2_m,racket_normal_x,racket_normal_y,racket_normal_z,racket_normal_speed_ms,"
    "racket_face_yaw_deg,racket_face_pitch_deg,velocity_ci95_ms,racket_normal_ci95_deg,"
    "racket_speed_ci95_ms,rms_in_px,rms_out_px,uncertainty_method"
)
BOUNCE_HEADER = "match,clip,segment,frame,x,y,z,vx_in,vy_in,vz_in,vx_out,vy_out,vz_out,regime,fit_rms_px"


def write_fixture(base, stem, clip="pt0001"):
    ball = "\n".join([
        BALL_HEADER,
        f"rg2025f,{clip},100,5.0,10.0,1.5,3.0,20.0,-2.0,100.0,50.0,0.0,0.9,0.2,0.2,0.2,1.0,150.0,1,2.0,{stem}",
        f"rg2025f,{clip},101,5.1,11.0,1.6,3.0,20.0,-2.0,100.0,50.0,0.0,0.9,0.2,0.2,0.2,1.0,150.0,1,2.0,{stem}",
        # gap at 102-104 (missing frames -> viewer renders as a break)
        f"rg2025f,{clip},105,5.5,13.0,1.2,3.0,20.0,-4.0,80.0,40.0,0.0,0.05,3.5,3.5,3.5,6.0,900.0,2,12.0,{stem}",
    ])
    contacts = "\n".join([
        CONTACT_HEADER,
        # a fit rally contact just before frame 100 with an outgoing arc
        f"rg2025f,{clip},0,99.5,99.5,0.0,97.5,101.5,1.95,2.03,detector,,near,rally,fit,"
        f"4.9,9.5,1.0,0.2,0.2,0.2,,,,,3.0,20.0,-2.0,20.3,,,,,,28.0,10.0,1.1,,,3.0,heuristic",
        # an unsupported (dead) contact
        f"rg2025f,{clip},1,120,120,0.0,118,122,2.4,2.44,detector,,unknown,serve,"
        f"unsupported_dead_or_same_side,3.0,24.0,2.6,0.2,0.2,0.2,,,,,,,,,,,,,,,,,0.5,9.0,heuristic",
    ])
    bounces = "\n".join([
        BOUNCE_HEADER,
        f"rg2025f,{clip},2,110,5.2,12.5,0.03,3.0,15.0,-11.0,3.0,14.0,9.0,grip,5.0",
        # out-of-court bounce
        f"rg2025f,{clip},3,130,-1.0,25.0,0.03,3.0,15.0,-11.0,3.0,14.0,9.0,slide,7.0",
    ])
    quality = {
        "version": stem, "accepted": False,
        "gate": {"rms_median_px_max": 3.0},
        "clips": [{"clip": clip, "contacts": 2, "obs": 3, "accepted": False,
                   "junction_median_pass1": float("nan")}],
        "tier1_labels_consumed": False,
    }
    with gzip.open(os.path.join(base, f"{stem}.csv.gz"), "wt") as h:
        h.write(ball + "\n")
    with open(os.path.join(base, f"{stem}_contacts.csv"), "w") as h:
        h.write(contacts + "\n")
    with open(os.path.join(base, f"{stem}_bounces.csv"), "w") as h:
        h.write(bounces + "\n")
    with open(os.path.join(base, f"{stem}_quality.json"), "w") as h:
        json.dump(quality, h)


class SyntheticExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = os.path.join(self.tmp, "data", "processed", "rg2025f")
        os.makedirs(self.base)
        write_fixture(self.base, "unit_pt0001_i1")
        self._orig_repo = ex.REPO
        ex.REPO = self.tmp

    def tearDown(self):
        ex.REPO = self._orig_repo

    def _doc(self):
        rows_ball = ex._read_csv(os.path.join(self.base, "unit_pt0001_i1.csv.gz"))
        rows_c = ex._read_csv(os.path.join(self.base, "unit_pt0001_i1_contacts.csv"))
        rows_b = ex._read_csv(os.path.join(self.base, "unit_pt0001_i1_bounces.csv"))
        with open(os.path.join(self.base, "unit_pt0001_i1_quality.json")) as h:
            q = json.load(h, parse_constant=lambda _c: None)
        return ex.export_clip("rg2025f", "pt0001", "unit_pt0001_i1", rows_ball, rows_c, rows_b, q, None)

    def test_frames_and_gap(self):
        doc = self._doc()
        f = doc["frames"]
        self.assertEqual(f["frame"], [100, 101, 105])  # gap 102-104 preserved as missing
        self.assertEqual(doc["counts"]["frames"], 3)
        # speed = norm(3,20,2) ~ 20.32
        self.assertAlmostEqual(f["speed"][0], 20.322, places=2)
        # low-confidence flagging: frame 105 has conf 0.05 and rms 12 -> low_conf
        self.assertEqual(f["method"], ["fit", "fit", "low_conf"])

    def test_spin_present_and_mag(self):
        doc = self._doc()
        self.assertTrue(doc["spin_present"])
        # spin mag frame100 = norm(100,50,0)=111.8
        self.assertAlmostEqual(doc["frames"]["spin_mag"][0], 111.8, places=1)

    def test_contacts(self):
        doc = self._doc()
        self.assertEqual(len(doc["contacts"]), 2)
        c0 = doc["contacts"][0]
        self.assertEqual(c0["status"], "fit")
        self.assertTrue(c0["fit"])
        self.assertEqual(c0["side"], "near")
        self.assertEqual(c0["phase"], "rally")
        self.assertAlmostEqual(c0["z"], 1.0)
        self.assertEqual(c0["frame_lo"], 97.5)
        self.assertEqual(c0["frame_hi"], 101.5)
        self.assertAlmostEqual(c0["speed_out"], 20.3)
        # spin looked up from the outgoing arc (frame 100)
        self.assertAlmostEqual(c0["spin_out"]["mag"], 111.8, places=1)
        self.assertEqual(c0["spin_out"]["frame"], 100)
        # racket proxy carried
        self.assertAlmostEqual(c0["racket_normal_speed"], 28.0)
        c1 = doc["contacts"][1]
        self.assertFalse(c1["fit"])
        self.assertEqual(c1["status"], "unsupported_dead_or_same_side")

    def test_bounces_in_court(self):
        doc = self._doc()
        self.assertEqual(len(doc["bounces"]), 2)
        self.assertTrue(doc["bounces"][0]["in_court"])
        self.assertFalse(doc["bounces"][1]["in_court"])  # x=-1 out
        self.assertEqual(doc["counts"]["bounces_in_court"], 1)
        self.assertEqual(doc["bounces"][0]["regime"], "grip")

    def test_court_and_meta(self):
        doc = self._doc()
        self.assertEqual(doc["court"]["net_center_h"], 0.914)
        self.assertEqual(doc["court"]["net_post_h"], 1.07)
        self.assertAlmostEqual(doc["court"]["width"], 10.97)
        self.assertAlmostEqual(doc["court"]["length"], 23.77)
        self.assertEqual(doc["point"], 1)
        self.assertEqual(doc["fps"], 50.0)

    def test_quality_nan_scrubbed(self):
        doc = self._doc()
        # NaN in the quality clip stats must be None (valid JSON), not NaN
        s = json.dumps(doc)  # would raise if NaN present with allow_nan=False? json allows NaN by default
        self.assertNotIn("NaN", s)
        self.assertFalse(doc["quality"]["accepted"])

    def test_json_strictly_valid(self):
        doc = self._doc()
        # strict: reject NaN/Infinity so browsers' JSON.parse succeeds
        s = json.dumps(doc, allow_nan=False)
        json.loads(s)

    def test_frame_range(self):
        doc = self._doc()
        # spans ball 100-105, contacts 99.5 (->100) & 120, bounces 110 & 130
        self.assertEqual(doc["frame_range"], [100, 130])


class RealArtifactSmoke(unittest.TestCase):
    """Optional: exercise the real dev9 pt0021 prune_loop artifact if present."""

    def test_pt0021_if_present(self):
        base = os.path.join(ex.REPO, "data", "processed", "rg2025f")
        stem = "prune_loop_pt0021_i1"
        if not os.path.exists(os.path.join(base, f"{stem}.csv.gz")):
            self.skipTest("real dev9 artifact not present")
        paths = ex._artifact_paths(base, stem)
        ball = ex._read_csv(paths["ball"])
        contacts = ex._read_csv(paths["contacts"])
        bounces = ex._read_csv(paths["bounces"])
        with open(paths["quality"]) as h:
            q = json.load(h, parse_constant=lambda _c: None)
        doc = ex.export_clip("rg2025f", "pt0021", stem, ball, contacts, bounces, q, None)
        self.assertGreater(doc["counts"]["frames"], 0)
        # strictly valid JSON (no NaN leaks from real data)
        json.loads(json.dumps(doc, allow_nan=False))
        # player positions: v3 feet-on-ground when present, else ledger/boxes fallback
        self.assertIn(doc["players"]["source"],
                      ("player_court_v3", "player_boxes_v2", "player_ledger", "none"))


if __name__ == "__main__":
    unittest.main()
