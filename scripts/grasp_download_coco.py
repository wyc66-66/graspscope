#!/usr/bin/env python3
"""Download COCO val2017 images biased toward grasp-product classes.

Scores every val2017 image by the number of grasp-catalogue GT boxes it
carries (with DEPLOY_VOCAB classes weighted more), downloads the top-scoring
``--n`` images into ``data/coco/grasp_val/``, and writes a filtered COCO
annotation file so the grasp audit has a product-rich real-image pool.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "coco" / "annotations" / "instances_val2017.json"
IMG_BASE = "http://images.cocodataset.org/val2017/"
OUT_IMG = ROOT / "data" / "grasp_val"
OUT_ANN = ROOT / "data" / "coco" / "annotations" / "instances_val2017_grasp.json"

DEPLOY_VOCAB = {
    "bottle",
    "cup",
    "bowl",
    "book",
    "banana",
    "apple",
    "orange",
    "carrot",
    "cake",
    "donut",
    "sports ball",
    "vase",
}
GRASP_PRODUCTS = DEPLOY_VOCAB | {
    "wine glass",
    "broccoli",
    "sandwich",
    "pizza",
    "knife",
    "fork",
    "spoon",
    "remote",
    "cell phone",
    "keyboard",
    "laptop",
    "teddy bear",
}


def _download(url: str, dest: Path, timeout: int = 90) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GraspScope/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        return dest.stat().st_size > 0
    except Exception:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Download COCO images rich in grasp products")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    data = json.loads(ANN.read_text(encoding="utf-8"))
    cat_name = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    id_to_img = {int(i["id"]): i for i in data.get("images", [])}

    score: Counter[int] = Counter()
    per_img_gt: dict[int, list[dict]] = {}
    for a in data.get("annotations", []):
        img_id = int(a["image_id"])
        name = cat_name.get(int(a.get("category_id", -1)), "")
        if name not in GRASP_PRODUCTS:
            continue
        w = 2.0 if name in DEPLOY_VOCAB else 1.0
        score[img_id] += w
        per_img_gt.setdefault(img_id, []).append({**a, "class": name})

    cand = sorted(score.items(), key=lambda kv: -kv[1])[: int(args.n * 3)]
    rng = random.Random(args.seed)
    rng.shuffle(cand)

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict] = []
    kept_gt: dict[int, list[dict]] = {}
    for img_id, sc in cand:
        if len(downloaded) >= args.n:
            break
        img = id_to_img.get(img_id)
        if img is None:
            continue
        dest = OUT_IMG / img["file_name"]
        if _download(IMG_BASE + img["file_name"], dest):
            downloaded.append({**img, "score": float(sc)})
            kept_gt[img_id] = per_img_gt[img_id]

    # filtered COCO annotation file
    idset = {int(d["id"]) for d in downloaded}
    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": [i for i in data.get("images", []) if int(i["id"]) in idset],
        "annotations": [
            a for a in data.get("annotations", []) if int(a["image_id"]) in idset
        ],
    }
    OUT_ANN.write_text(json.dumps(out), encoding="utf-8")
    print(f"downloaded {len(downloaded)} grasp-rich images -> {OUT_IMG}")
    print(f"wrote filtered anns -> {OUT_ANN}")
    # quick class coverage report
    cnt: Counter[str] = Counter()
    for ann in out["annotations"]:
        cnt[cat_name.get(int(ann["category_id"]), "?")] += 1
    for name, n in cnt.most_common():
        print(f"{n:5d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
