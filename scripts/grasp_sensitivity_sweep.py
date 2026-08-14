#!/usr/bin/env python3
"""GraspScope sensitivity sweeps.

Two robustness studies beyond the main 12-class / YOLO-World-s sweep:

A. **Detector scale** (s vs. l). Rerun the perception audit + closed-loop sweep
   with ``yolov8l-world.pt`` on the same 12-class corpus and vocabulary. Asks:
   does the qualitative law (coverage cliff + gate) hold when detection quality
   improves, and how does the gate value move?

B. **Vocabulary size** (12 vs. 8). Rebuild the synthetic corpus with an 8-class
   deployment vocabulary (bottle, cup, bowl, book, banana, apple, orange,
   carrot), rerun the audit + closed-loop sweep with the s detector. Asks: how
   much does a smaller vocabulary cost in required coverage?

Outputs:
    data/grasp_detector_scale/frontier.json   s vs l frontier comparison
    data/grasp_vocab_size/frontier.json       12 vs 8 vocab comparison
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = _WORKSPACE / "scripts"


def run(*args: str) -> None:
    print(f"$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=_WORKSPACE, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-alpha", type=int, default=150)
    ap.add_argument("--skip-detector", action="store_true")
    ap.add_argument("--skip-vocab", action="store_true")
    ap.add_argument("--gpu-model-dir", default="", help="dir with yolov8*.pt weights")
    args = ap.parse_args()

    weights = args.gpu_model_dir

    # ---------- A. detector scale: yolov8l on the existing 12-class corpus ----------
    if not args.skip_detector:
        l_model = f"{weights}/yolov8l-world.pt" if weights else "yolov8l-world.pt"
        print("\n=== [A] detector scale: YOLO-World l on 12-class corpus ===")
        run(
            sys.executable, str(SCRIPTS / "grasp_audit_perception.py"),
            "--synth-coco", "data/grasp_synth/coco_flat/annotations.json",
            "--synth-imgs", "data/grasp_synth/images",
            "--real-coco", "data/grasp_real/coco/annotations.json",
            "--real-imgs", "data/coco/val2017",
            "--out", "data/grasp_gate_l",
            "--model", l_model,
            "--conf", "0.15", "--imgsz", "640",
        )
        run(
            sys.executable, str(SCRIPTS / "grasp_run_closedloop.py"),
            "--profiles", "data/grasp_gate_l/profiles.json",
            "--manifest", "data/grasp_synth/synthetic_corpus.json",
            "--images", "data/grasp_synth/images",
            "--out", "data/grasp_detector_scale",
            "--max-fail-rate", "0.25",
            "--n-scenes", str(args.per_alpha),
            "--seeds", "5",
        )

    # ---------- B. vocabulary size: 8-class corpus, s detector ----------
    if not args.skip_vocab:
        v8 = ["bottle", "cup", "bowl", "book", "banana", "apple", "orange", "carrot"]
        s_model = f"{weights}/yolov8s-world.pt" if weights else "yolov8s-world.pt"
        print("\n=== [B] vocabulary size: 8-class corpus, YOLO-World s ===")
        # rebuild the corpus with the 8-class vocab
        run(
            sys.executable, str(SCRIPTS / "grasp_build_corpus.py"),
            "--synth-out", "data/grasp_synth_v8",
            "--real-out", "data/grasp_real_v8",
            "--per-alpha", str(args.per_alpha),
            "--max-real", "149",
            "--vocab-size", "8",
            "--seed", "0",
        )
        run(
            sys.executable, str(SCRIPTS / "grasp_audit_perception.py"),
            "--synth-coco", "data/grasp_synth_v8/coco_flat/annotations.json",
            "--synth-imgs", "data/grasp_synth_v8/images",
            "--real-coco", "data/grasp_real_v8/coco/annotations.json",
            "--real-imgs", "data/coco/val2017",
            "--out", "data/grasp_gate_v8",
            "--model", s_model,
            "--conf", "0.15", "--imgsz", "640",
            "--vocab-size", "8",
        )
        run(
            sys.executable, str(SCRIPTS / "grasp_run_closedloop.py"),
            "--profiles", "data/grasp_gate_v8/profiles.json",
            "--manifest", "data/grasp_synth_v8/synthetic_corpus.json",
            "--images", "data/grasp_synth_v8/images",
            "--out", "data/grasp_vocab_size",
            "--max-fail-rate", "0.25",
            "--n-scenes", str(args.per_alpha),
            "--seeds", "5",
        )

    # ---------- summarize ----------
    print("\n=== summary ===")
    # main run
    main_fr = json.loads((_WORKSPACE / "data" / "grasp_closedloop" / "frontier.json").read_text())
    print(f"[main] gate coverage_min = {main_fr['gate'].get('coverage_min')}")

    det_fr = _WORKSPACE / "data" / "grasp_detector_scale" / "frontier.json"
    if det_fr.is_file():
        d = json.loads(det_fr.read_text())
        print(f"[detector=l] gate coverage_min = {d['gate'].get('coverage_min')}")

    v8_fr = _WORKSPACE / "data" / "grasp_vocab_size" / "frontier.json"
    if v8_fr.is_file():
        d = json.loads(v8_fr.read_text())
        print(f"[vocab=8] gate coverage_min = {d['gate'].get('coverage_min')}")

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
