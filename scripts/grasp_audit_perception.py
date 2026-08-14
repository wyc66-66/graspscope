#!/usr/bin/env python3
"""Perception audit for the GraspScope corpus.

Runs YOLO-World (CC3M open-vocab weights) over:

- the synthetic shelf sweep (alpha x 150 scenes)  -- controlled coverage;
- the real COCO grasp pack                          -- honest real baseline.

For each alpha tier it computes per-class recall and OOV-FP at several
confidence thresholds and emits a :class:`PerceptionErrorProfile` (the exact
structure consumed by the closed-loop grasp simulator). The detector always
runs with the *deployment vocabulary* V; GT objects are in-vocabulary only.

Outputs:
    data/grasp_gate/<tier>/profile.json        PerceptionErrorProfile per tier
    data/grasp_gate/profiles.json              all profiles + alpha axis
    data/grasp_gate/per_class_recall.csv       per-class recall summary
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE / "src"))

from graspscope.closedloop.error_profile import ClassProfile, PerceptionErrorProfile
from graspscope.grasp import scenes as gs

CONF_GRID = [0.15, 0.25, 0.4, 0.5, 0.7]

MATCH_IOU = 0.5


def _load_coco(coco_ann: Path) -> tuple[list[dict], dict[int, dict], dict[int, str]]:
    data = json.loads(coco_ann.read_text(encoding="utf-8"))
    cat_name = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    imgs = {int(i["id"]): i for i in data.get("images", [])}
    return data.get("annotations", []), imgs, cat_name


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def run_audit(
    coco_ann: Path,
    images_dir: Path,
    vocab: list[str],
    *,
    model: str = "yolov8s-world.pt",
    conf: float = 0.15,
    imgsz: int = 640,
    tier: str = "real",
    max_images: int = 10000,
) -> dict:
    """Run YOLO-World over a COCO pack; return per-class tp/gt and OOV-FP sweep."""
    from graspscope.adapters.yolo_world import YoloWorldAdapter

    anns, imgs, cat_name = _load_coco(coco_ann)
    vset = set(vocab)

    adapter = YoloWorldAdapter({"model": model, "conf": conf, "imgsz": imgsz, "cache_dir": None})
    from graspscope.schema import Sample

    samples = []
    img_order = sorted(imgs.values(), key=lambda i: int(i["id"]))
    for img in img_order[:max_images]:
        path = images_dir / img["file_name"]
        if not path.is_file():
            continue
        samples.append(
            Sample(
                sample_id=str(int(img["id"])),
                image_path=str(path.resolve()),
                view_id="default",
                image_wh=[int(img.get("width", 0)), int(img.get("height", 0))],
            )
        )

    if not samples:
        return {"n_images": 0, "n_gt": 0, "n_tp": 0, "tp_by_class": {}, "gt_by_class": {}, "oov_fp_by_conf": {}}

    preds = adapter.predict(samples, list(vocab))

    # index anns by image
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in anns:
        anns_by_img[int(a["image_id"])].append(a)

    tp_by_class: dict[str, int] = defaultdict(int)
    loc_by_class: dict[str, int] = defaultdict(int)
    gt_by_class: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_images = 0
    n_gt = 0
    n_tp = 0
    n_loc = 0

    for sample, pred in zip(samples, preds):
        img_id = int(sample.sample_id)
        gt_anns = anns_by_img.get(img_id, [])
        gt_boxes = []
        for a in gt_anns:
            name = cat_name.get(int(a["category_id"]), "")
            if name not in vset:
                continue
            x, y, w, h = a["bbox"]
            gt_boxes.append(([x, y, x + w, y + h], name))
        if not gt_boxes and not pred.boxes:
            continue
        n_images += 1
        # phase 1: greedy localization match (any predicted class, IoU only)
        # phase 2: among localized GT, class correctness
        used_preds = [False] * len(pred.boxes)
        matched = [False] * len(gt_boxes)
        pred_choice = [None] * len(gt_boxes)
        # strict TP first (prefer correct-class matches)
        for pi, b in enumerate(pred.boxes):
            if b.cls not in vset:
                continue
            best_i, best_iou = -1, 0.0
            for gi, (gb, gname) in enumerate(gt_boxes):
                if matched[gi] or gname != b.cls:
                    continue
                iou = _iou(b.xyxy, gb)
                if iou > best_iou:
                    best_iou, best_i = iou, gi
            if best_i >= 0 and best_iou >= MATCH_IOU:
                matched[best_i] = True
                used_preds[pi] = True
                pred_choice[best_i] = b.cls
                tp_by_class[b.cls] += 1
                n_tp += 1
        # localization: any still-unmatched GT covered by an unused pred
        for gi, (gb, gname) in enumerate(gt_boxes):
            if matched[gi]:
                continue
            best_p, best_iou = -1, 0.0
            for pi, b in enumerate(pred.boxes):
                if used_preds[pi]:
                    continue
                iou = _iou(b.xyxy, gb)
                if iou > best_iou:
                    best_iou, best_p = iou, pi
            if best_p >= 0 and best_iou >= MATCH_IOU:
                matched[gi] = True
                used_preds[best_p] = True
                pred_choice[gi] = pred.boxes[best_p].cls
                confusion[gname][pred.boxes[best_p].cls] += 1
                loc_by_class[gname] += 1
                n_loc += 1
        for gi, (gb, gname) in enumerate(gt_boxes):
            gt_by_class[gname] += 1
            n_gt += 1
            if matched[gi]:
                n_loc += 0  # already counted
        # count localization for strictly-matched GT too
    # loc_total: for strict TP rows we count them as localized
    for cls in vset:
        loc_by_class[cls] += tp_by_class.get(cls, 0)

    # OOV-FP sweep: predictions whose class is NOT in V (B0 full-vocab effect)
    # measured as a fraction of high-score detections.
    oov_fp_by_conf: dict[str, float] = {}
    for c in CONF_GRID:
        high_total = 0
        high_oov = 0
        for pred in preds:
            hi = [b for b in pred.boxes if b.score >= c]
            high_total += len(hi)
            high_oov += sum(1 for b in hi if b.cls.lower() not in vset)
        oov_fp_by_conf[f"{c:.2f}"] = float(high_oov / high_total) if high_total else 0.0
    oov_fp_by_conf["default"] = oov_fp_by_conf.get("0.50", 0.0)

    return {
        "tier": tier,
        "n_images": n_images,
        "n_gt": n_gt,
        "n_tp": n_tp,
        "n_loc": n_loc,
        "tp_by_class": dict(tp_by_class),
        "loc_by_class": dict(loc_by_class),
        "gt_by_class": dict(gt_by_class),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "oov_fp_by_conf": oov_fp_by_conf,
    }


def build_profile(result: dict, tier: str, vocab: list[str], alpha: float | None = None) -> PerceptionErrorProfile:
    classes: dict[str, ClassProfile] = {}
    for cls in vocab:
        gt = int(result["gt_by_class"].get(cls, 0))
        tp = int(result["tp_by_class"].get(cls, 0))
        loc = int(result["loc_by_class"].get(cls, 0))
        classes[cls] = ClassProfile(
            cls=cls,
            recall=(tp / gt) if gt else 0.0,
            n_gt=gt,
            n_tp=tp,
            loc_recall=(loc / gt) if gt else 0.0,
            confusion=dict(result["confusion"].get(cls, {})),
        )
    return PerceptionErrorProfile(
        name=f"grasp_{tier}",
        classes=classes,
        oov_fp_by_conf=dict(result["oov_fp_by_conf"]),
        conf_default=0.5,
        sources=[f"grasp_audit:{tier}"],
        injection_source="2d",
        fusion_strategy="2d",
        meta={
            "tier": tier,
            "alpha": alpha,
            "grasp_vocab": list(vocab),
            "model": "yolov8l-world-cc3m.pt",
            "n_images": result["n_images"],
            "n_gt": result["n_gt"],
            "n_tp": result["n_tp"],
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-coco", default="data/grasp_synth/coco_flat/annotations.json")
    ap.add_argument("--synth-imgs", default="data/grasp_synth/images")
    ap.add_argument("--real-coco", default="data/grasp_real/coco/annotations.json")
    ap.add_argument("--real-imgs", default="data/coco/val2017")
    ap.add_argument("--out", default="data/grasp_gate")
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--vocab-size", type=int, default=12, choices=[8, 12],
                    help="deployment vocabulary size")
    ap.add_argument("--per-alpha-subsample", type=int, default=0,
                    help="cap scenes per alpha (0 = all) for smoke tests")
    args = ap.parse_args()

    vocab = gs.DEPLOY_VOCAB if args.vocab_size >= 12 else gs.DEPLOY_VOCAB[:8]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, dict] = {}

    # ---- synthetic tiers: one profile per alpha ----
    # the synthetic COCO flat export mixes all alphas; the GT does not carry
    # alpha. To get per-alpha recall we must run per-alpha COCO subsets. The
    # flat export drops alpha; reconstruct from the corpus manifest instead.
    synth_man = json.loads(
        (Path(args.synth_coco).parent.parent / "synthetic_corpus.json").read_text(encoding="utf-8")
    )
    synth_img_dir = Path(args.synth_imgs)

    # group scenes by alpha, rebuild per-alpha coco on the fly
    from collections import OrderedDict

    by_alpha: dict[str, list[dict]] = OrderedDict()
    for s in synth_man["all_scenes"]:
        by_alpha.setdefault(f"{s['alpha']:.1f}", []).append(s)
    if args.per_alpha_subsample:
        for k in by_alpha:
            by_alpha[k] = by_alpha[k][: args.per_alpha_subsample]

    # write per-alpha COCO subset (images rel to synth_img_dir)
    for alpha_str, scenes in by_alpha.items():
        cats = [
            {"id": i + 1, "name": name, "supercategory": "grasp"}
            for i, name in enumerate(gs.GRASP_PRODUCTS)
        ]
        name_to_id = {c["name"]: int(c["id"]) for c in cats}
        images, anns = [], []
        ann_id = 1
        for si, rec in enumerate(scenes):
            images.append(
                {
                    "id": si,
                    "file_name": Path(rec["image_path"]).name,
                    "width": 960,
                    "height": 540,
                }
            )
            for g in rec["gt"]:
                x1, y1, x2, y2 = g["xyxy"]
                anns.append(
                    {
                        "id": ann_id,
                        "image_id": si,
                        "category_id": name_to_id[g["cls"]],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        coco = {
            "info": {"description": "grasp synth per-alpha"},
            "licenses": [],
            "categories": cats,
            "images": images,
            "annotations": anns,
        }
        tier_coco = out / f"alpha_{alpha_str}" / "coco.json"
        tier_coco.parent.mkdir(parents=True, exist_ok=True)
        tier_coco.write_text(json.dumps(coco), encoding="utf-8")

        print(f"[audit] running alpha={alpha_str} ({len(scenes)} scenes)...")
        res = run_audit(
            tier_coco,
            synth_img_dir,
            vocab,
            model=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            tier=f"alpha_{alpha_str}",
        )
        prof = build_profile(res, f"alpha_{alpha_str}", vocab, alpha=float(alpha_str))
        prof_path = out / f"alpha_{alpha_str}" / "profile.json"
        prof.save(prof_path)
        rec_vals = [p.recall for p in prof.classes.values() if p.n_gt > 0]
        loc_vals = [p.loc_recall for p in prof.classes.values() if p.loc_recall is not None and p.n_gt > 0]
        profiles[f"alpha_{alpha_str}"] = {
            "alpha": float(alpha_str),
            "n_images": res["n_images"],
            "n_gt": res["n_gt"],
            "n_tp": res["n_tp"],
            "n_loc": res["n_loc"],
            "recall_mean": sum(rec_vals) / len(rec_vals) if rec_vals else 0.0,
            "loc_recall_mean": sum(loc_vals) / len(loc_vals) if loc_vals else 0.0,
            "oov_fp_at_0.5": res["oov_fp_by_conf"].get("0.50", 0.0),
            "per_class_recall": {
                c: round(prof.classes[c].recall, 4) for c in vocab
            },
            "per_class_loc_recall": {
                c: round(prof.classes[c].loc_recall or 0.0, 4) for c in vocab
            },
            "confusion": {
                c: prof.classes[c].confusion for c in vocab if prof.classes[c].confusion
            },
        }
        print(
            f"  -> n_img={res['n_images']} loc_recall={profiles[f'alpha_{alpha_str}']['loc_recall_mean']:.3f} "
            f"recall={profiles[f'alpha_{alpha_str}']['recall_mean']:.3f} "
            f"oov_fp@0.5={profiles[f'alpha_{alpha_str}']['oov_fp_at_0.5']:.3f}"
        )

    # ---- real tier ----
    real_coco = Path(args.real_coco)
    real_imgs = Path(args.real_imgs)
    print("[audit] running real pack ...")
    res = run_audit(real_coco, real_imgs, vocab, model=args.model, conf=args.conf, imgsz=args.imgsz, tier="real")
    prof = build_profile(res, "real", vocab, alpha=None)
    (out / "real" / "profile.json").parent.mkdir(parents=True, exist_ok=True)
    prof.save(out / "real" / "profile.json")
    rec_vals = [p.recall for p in prof.classes.values() if p.n_gt > 0]
    loc_vals = [p.loc_recall for p in prof.classes.values() if p.loc_recall is not None and p.n_gt > 0]
    profiles["real"] = {
        "alpha": None,
        "n_images": res["n_images"],
        "n_gt": res["n_gt"],
        "n_tp": res["n_tp"],
        "n_loc": res["n_loc"],
        "recall_mean": sum(rec_vals) / len(rec_vals) if rec_vals else 0.0,
        "loc_recall_mean": sum(loc_vals) / len(loc_vals) if loc_vals else 0.0,
        "oov_fp_at_0.5": res["oov_fp_by_conf"].get("0.50", 0.0),
        "per_class_recall": {c: round(prof.classes[c].recall, 4) for c in vocab},
        "per_class_loc_recall": {c: round(prof.classes[c].loc_recall or 0.0, 4) for c in vocab},
        "confusion": {
            c: prof.classes[c].confusion for c in vocab if prof.classes[c].confusion
        },
    }
    print(
        f"  -> n_img={res['n_images']} loc_recall={profiles['real']['loc_recall_mean']:.3f} "
        f"recall={profiles['real']['recall_mean']:.3f} "
        f"oov_fp@0.5={profiles['real']['oov_fp_at_0.5']:.3f}"
    )

    # ---- save combined profiles ----
    (out / "profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    # per-class recall CSV
    import csv

    with (out / "per_class_recall.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tier", "alpha"] + vocab)
        for tier, p in profiles.items():
            w.writerow([tier, p.get("alpha"), *[round(p["per_class_recall"].get(c, 0.0), 4) for c in vocab]])
    print(f"[audit] profiles -> {out / 'profiles.json'}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
