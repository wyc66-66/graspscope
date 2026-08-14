#!/usr/bin/env python3
"""Download COCO val2017 images for classes with zero crops in the crop bank.

Scans the existing crop bank for classes below a per-class crop target and
downloads images carrying those classes (threaded, with retry), so the synthetic
compositor has enough crops for every product class.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE / "src"))

from opengate.graspgate import scenes as gs

ANN = _WORKSPACE / "data" / "coco" / "annotations" / "instances_val2017.json"
IMG_BASE = "http://images.cocodataset.org/val2017/"
OUT_IMG = _WORKSPACE / "data" / "coco" / "val2017"


def _download_one(img_id: int, file_name: str, dest: Path, timeout: int = 90) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                IMG_BASE + file_name, headers={"User-Agent": "OpenGateGraspGate/0.1"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
            if dest.stat().st_size > 0:
                return True
        except Exception:  # noqa: BLE001
            dest.unlink(missing_ok=True)
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Top up crop bank for zero-coverage classes")
    p.add_argument("--per-class-target", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-images", type=int, default=80)
    args = p.parse_args()

    data = json.loads(ANN.read_text(encoding="utf-8"))
    cat_name = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    id_to_img = {int(i["id"]): i for i in data.get("images", [])}

    bank = gs.CropBank(ANN, OUT_IMG)
    want_classes = gs.GRASP_PRODUCTS

    # classes needing crops
    need: set[str] = set()
    for c in want_classes:
        if bank.count(c) < args.per_class_target:
            need.add(c)
    print("classes needing top-up:", sorted(need))

    # candidate images per needed class
    candidates: list[tuple[int, str]] = []
    seen: set[int] = set()
    for a in data.get("annotations", []):
        name = cat_name.get(int(a.get("category_id", -1)), "")
        if name not in need:
            continue
        img_id = int(a["image_id"])
        if img_id in seen:
            continue
        seen.add(img_id)
        img = id_to_img.get(img_id)
        if img is None:
            continue
        if (OUT_IMG / img["file_name"]).is_file():
            continue
        candidates.append((img_id, img["file_name"]))
        if len(candidates) >= args.max_images:
            break
    print(f"candidate images: {len(candidates)}")

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_download_one, img_id, fname, OUT_IMG / fname): fname
            for img_id, fname in candidates
        }
        for i, fut in enumerate(as_completed(futs), 1):
            fname = futs[fut]
            try:
                if fut.result():
                    ok += 1
            except Exception:  # noqa: BLE001
                pass
            if i % 10 == 0:
                print(f"  progress {i}/{len(candidates)} ok={ok}")

    # refresh crop bank stats
    bank2 = gs.CropBank(ANN, OUT_IMG)
    print("\npost top-up crop counts:")
    for c in sorted(want_classes):
        print(f"  {bank2.count(c):4d}  {c}")
    print(f"done: downloaded {ok} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
