#!/usr/bin/env python3
"""Build the full OpenVocab-GraspGate corpus:

1. synthetic shelf sweep across the alpha grid (controlled coverage);
2. real COCO scenes annotated for the grasp catalogue;
3. COCO-format exports so YoloWorldAdapter + existing metrics run directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE / "src"))

from opengate.graspgate import scenes as gs

ANN = r"d:\ccfa\opengate\data\coco\annotations\instances_val2017.json"
IMGS = r"d:\ccfa\opengate\data\coco\val2017"


def main() -> int:
    p = argparse.ArgumentParser(description="Build GraspGate corpus (synth + real)")
    p.add_argument("--synth-out", default="data/grasp_synth")
    p.add_argument("--real-out", default="data/grasp_real")
    p.add_argument("--per-alpha", type=int, default=150)
    p.add_argument("--max-real", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bank = gs.CropBank(ANN, IMGS)
    print(f"[corpus] crop bank: {len(bank.classes())} classes")

    # ---- 1. synthetic sweep ----
    synth_out = Path(args.synth_out)
    print(f"[corpus] emitting synthetic sweep -> {synth_out}")
    man = gs.emit_sweep(
        bank, synth_out, scenes_per_alpha=args.per_alpha, n_objects=9, seed=args.seed
    )
    print(f"[corpus] synthetic scenes: {len(man['all_scenes'])}")

    # ---- 2. real pack ----
    real_out = Path(args.real_out)
    real = gs.load_coco_scenes(
        ANN, IMGS, max_scenes=args.max_real, min_objs=1, seed=args.seed
    )
    print(f"[corpus] real scenes: {len(real)}")

    # ---- 3. COCO exports ----
    # synthetic: reconstruct Scene objects from manifest, export COCO flat
    synth_flat: list[gs.Scene] = []
    for i, rec in enumerate(man["all_scenes"]):
        objs = [
            gs.SceneObject(
                cls=g["cls"], xyxy=g["xyxy"], in_vocab=True, oov=False, source="synthetic"
            )
            for g in rec["gt"]
        ]
        synth_flat.append(
            gs.Scene(
                scene_id=rec["scene_id"],
                image_path=str((synth_out / rec["image_path"]).resolve()),
                objects=objs,
                image_wh=[960, 540],
                family="synthetic",
                alpha=rec["alpha"],
            )
        )
    gs.scenes_to_coco(synth_flat, synth_out / "coco_flat")
    print(f"[corpus] synthetic coco export: {len(synth_flat)} scenes")

    gs.scenes_to_coco(real, real_out / "coco")
    print(f"[corpus] real coco export: {len(real)} scenes")

    # ---- 4. real corpus manifest ----
    real_man = {
        "family": "real",
        "scenes": [
            {
                "scene_id": s.scene_id,
                "image_path": str(Path(s.image_path).name),
                "gt": s.gt_boxes(),
                "n_objects": len(s.objects),
                "n_in_vocab": sum(1 for o in s.objects if o.in_vocab),
                "n_oov": sum(1 for o in s.objects if o.oov),
            }
            for s in real
        ],
    }
    (real_out / "real_corpus.json").write_text(
        json.dumps(real_man, indent=2), encoding="utf-8"
    )

    # ---- 5. summary ----
    gt_cnt: Counter[str] = Counter()
    for s in man["all_scenes"]:
        for g in s["gt"]:
            gt_cnt[g["cls"]] += 1
    summary = {
        "synthetic": {
            "n_scenes": len(man["all_scenes"]),
            "alpha_grid": man["alpha_grid"],
            "scenes_per_alpha": args.per_alpha,
            "image_dir": str((synth_out / "images").resolve()),
            "coco_ann": str((synth_out / "coco_flat" / "annotations.json").resolve()),
            "in_vocab_gt_by_class": dict(gt_cnt),
        },
        "real": {
            "n_scenes": len(real),
            "image_dir": str(IMGS),
            "coco_ann": str((real_out / "coco" / "annotations.json").resolve()),
            "n_in_vocab_total": sum(1 for s in real for o in s.objects if o.in_vocab),
            "n_oov_total": sum(1 for s in real for o in s.objects if o.oov),
        },
    }
    (synth_out / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[corpus] summary -> {synth_out / 'corpus_summary.json'}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
