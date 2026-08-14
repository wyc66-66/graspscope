"""GraspScope scene data pipeline.

Builds the two scene families used by the grasp deployability audit:

1. **Real scenes** -- COCO val2017 images annotated for the *grasp vocabulary*
   (desktop / retail-style objects: bottle, cup, bowl, book, banana, ...). These
   give an honest open-vocabulary detection baseline on real imagery.

2. **Synthetic scenes** -- procedurally composited shelves: foreground object
   crops cut from COCO are pasted onto a plain shelf background with controlled
   occlusion, scale and lighting. The compositor *controls* the deployment axis
   the real world cannot:

   - vocabulary coverage :math:`\\alpha` -- the fraction of shelf products whose
     class is covered by the deployment vocabulary ``V``. Reducing ``alpha``
     means more *out-of-vocabulary* (OOV) products on the shelf: real products
     the detector has no word for. This is exactly the retail reality Galbot
     faces when a new SKU arrives before its name is added to ``V``.

   GT is produced by construction (zero annotation cost), so coverage sweeps
   are exact rather than sampled.

The alpha axis fully determines the OOV fraction: a scene with coverage alpha
has ``1 - alpha`` of its products out of vocabulary. No separate OOV dial is
needed (they are the same physical quantity).

Everything is emitted in COCO-format JSON plus an GraspScope manifest so the
existing :mod:`graspscope.data` loaders and ``YoloWorldAdapter`` can consume it
without modification.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Full grasp product catalogue (COCO classes) -- every class exists in COCO GT.
# The *deployment vocabulary* V is a subset; products outside V are OOV.
# ---------------------------------------------------------------------------
GRASP_PRODUCTS = [
    "bottle",
    "cup",
    "wine glass",
    "bowl",
    "book",
    "banana",
    "apple",
    "orange",
    "carrot",
    "broccoli",
    "cake",
    "donut",
    "sandwich",
    "pizza",
    "knife",
    "fork",
    "spoon",
    "sports ball",
    "remote",
    "cell phone",
    "keyboard",
    "laptop",
    "teddy bear",
    "vase",
]

# Deployment vocabulary V: a fixed retail-style subset (mirrors Galbot SKU
# categories). Products outside V are OOV -- present on the shelf, no GT, no
# word in the detector prompt.
DEPLOY_VOCAB = [
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
]

OOV_PRODUCTS = [c for c in GRASP_PRODUCTS if c not in DEPLOY_VOCAB]

# Scene size used for all synthetic scenes (W x H). Kept smaller than the
# camera-native 1280x720 so objects occupy more relative area at a given
# inference imgsz (YOLO-World detects them more reliably), while still
# resembling a shelf photo.
SYNTH_SIZE = (960, 540)

_SHELF_PALETTES = [
    ((185, 178, 166), (140, 132, 120)),
    ((200, 196, 186), (150, 145, 132)),
    ((176, 170, 158), (128, 120, 108)),
]


@dataclass
class SceneObject:
    """A single object placed in a (real or synthetic) scene."""

    cls: str
    xyxy: list[float]
    in_vocab: bool = True
    oov: bool = False  # out-of-vocabulary (not covered by deployment vocab V)
    source: str = "synthetic"  # synthetic | real
    crop_key: str | None = None  # synthetic crop identity (seed)


@dataclass
class Scene:
    """One annotated scene + its GraspScope-format representation."""

    scene_id: str
    image_path: str
    objects: list[SceneObject]
    image_wh: list[int]
    family: str  # real | synthetic
    alpha: float | None = None  # vocabulary coverage (synthetic only)
    meta: dict[str, Any] = field(default_factory=dict)

    def gt_boxes(self) -> list[dict[str, Any]]:
        out = []
        for o in self.objects:
            if o.oov:
                continue  # OOV products carry no GT (unknown to deployment)
            x1, y1, x2, y2 = o.xyxy
            out.append(
                {
                    "xyxy": [x1, y1, x2, y2],
                    "cls": o.cls,
                    "score": 1.0,
                    "oov": False,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Real-scene loader (COCO val2017)
# ---------------------------------------------------------------------------
def load_coco_scenes(
    coco_ann: str | Path,
    images_dir: str | Path,
    *,
    vocab: list[str] | None = None,
    max_scenes: int = 200,
    min_objs: int = 1,
    seed: int = 0,
) -> list[Scene]:
    """Load COCO val2017 images restricted to grasp-catalogue GT.

    A scene (image) is kept if it has at least ``min_objs`` GT boxes whose
    category is in the grasp catalogue. Non-catalogue COCO classes are dropped
    (they are background for the grasp audit).
    """
    products = set(vocab or GRASP_PRODUCTS)
    data = json.loads(Path(coco_ann).read_text(encoding="utf-8"))
    cat_name = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    imgs_dir = Path(images_dir)

    by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in data.get("annotations", []):
        by_img[int(a["image_id"])].append(a)

    id_to_img = {int(i["id"]): i for i in data.get("images", [])}

    rng = random.Random(seed)
    scenes: list[Scene] = []
    for img_id in sorted(id_to_img):
        if len(scenes) >= max_scenes:
            break
        img = id_to_img[img_id]
        anns = by_img.get(img_id, [])
        objs: list[SceneObject] = []
        for a in anns:
            name = cat_name.get(int(a.get("category_id", -1)), "")
            if name not in products:
                continue
            x, y, w, h = (float(v) for v in a["bbox"])
            in_v = name in (vocab or DEPLOY_VOCAB)
            objs.append(
                SceneObject(
                    cls=name,
                    xyxy=[x, y, x + w, y + h],
                    in_vocab=in_v,
                    oov=not in_v,
                    source="real",
                )
            )
        if len(objs) < min_objs:
            continue
        fname = img["file_name"]
        path = imgs_dir / fname
        if not path.is_file():
            continue
        scenes.append(
            Scene(
                scene_id=f"real_{img_id}",
                image_path=str(path.resolve()),
                objects=objs,
                image_wh=[int(img.get("width", 0)), int(img.get("height", 0))],
                family="real",
                alpha=None,
                meta={"coco_id": int(img_id), "file_name": fname},
            )
        )
    rng.shuffle(scenes)
    return scenes


# ---------------------------------------------------------------------------
# Synthetic compositor
# ---------------------------------------------------------------------------
class CropBank:
    """Foreground object crops cut from COCO, cached by class."""

    def __init__(
        self,
        coco_ann: str | Path,
        images_dir: str | Path,
        classes: list[str] | None = None,
        *,
        pad_ratio: float = 0.12,
        min_size: int = 12,
    ):
        self.classes_all = classes or GRASP_PRODUCTS
        self.cset = set(self.classes_all)
        self.pad_ratio = pad_ratio
        data = json.loads(Path(coco_ann).read_text(encoding="utf-8"))
        self.cat_name = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
        self.imgs_dir = Path(images_dir)
        id_to_img = {int(i["id"]): i for i in data.get("images", [])}

        self._crops: dict[str, list[Any]] = defaultdict(list)
        for a in data.get("annotations", []):
            name = self.cat_name.get(int(a.get("category_id", -1)), "")
            if name not in self.cset:
                continue
            img_id = int(a["image_id"])
            img = id_to_img.get(img_id)
            if img is None:
                continue
            path = self.imgs_dir / img["file_name"]
            if not path.is_file():
                continue
            try:
                src = Image.open(path).convert("RGB")
            except OSError:
                continue
            x, y, w, h = (float(v) for v in a["bbox"])
            if w < min_size or h < min_size:
                continue
            px, py = self.pad_ratio * w, self.pad_ratio * h
            x1, y1 = max(0, int(x - px)), max(0, int(y - py))
            x2 = min(src.width, int(x + w + px))
            y2 = min(src.height, int(y + h + py))
            crop = src.crop((x1, y1, x2, y2))
            if crop.width < min_size or crop.height < min_size:
                continue
            self._crops[name].append(crop)

    def classes(self) -> list[str]:
        return sorted(self._crops)

    def count(self, cls: str) -> int:
        return len(self._crops.get(cls, []))

    def available(self, cls: str) -> bool:
        return len(self._crops.get(cls, [])) > 0

    def sample(self, cls: str, rng: random.Random) -> Image.Image:
        pool = self._crops.get(cls)
        if not pool:
            raise KeyError(f"no crops for class {cls!r}")
        return rng.choice(pool)


def _rounded_rect_soft(
    img: Image.Image,
    radius: int,
) -> tuple[Image.Image, Image.Image]:
    """Return (image, soft-alpha-mask) with rounded feather edges."""
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=radius, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, radius // 2)))
    return img, mask


def make_shelf_background(
    width: int,
    height: int,
    *,
    rng: random.Random,
    shelves: int = 3,
) -> Image.Image:
    """Programmatic grocery-shelf background (no external assets).

    Includes a shelf grid, shelf-edge lip, and a soft horizontal gradient so
    the detector sees plausible depth cues rather than a flat color plane.
    """
    shelf_col, gap_col = rng.choice(_SHELF_PALETTES)
    img = Image.new("RGB", (width, height), shelf_col)
    d = ImageDraw.Draw(img)
    sh = height / (shelves + 0.6)

    for i in range(shelves + 1):
        y = int(sh * i)
        lip_h = int(sh * 0.10)
        d.rectangle([0, y, width, y + lip_h], fill=gap_col)
        # shelf-edge lip highlight
        d.rectangle([0, y + lip_h, width, y + lip_h + 3], fill=(230, 226, 216))
        # front face shading under each shelf
        d.rectangle([0, y + lip_h + 3, width, y + int(sh * 0.16)], fill=tuple(
            int(c * 0.94) for c in gap_col
        ))
    # vertical panel gaps (columns)
    for cx in range(int(width * 0.08), width, int(width * 0.18)):
        d.rectangle([cx, 0, cx + 5, height], fill=tuple(int(c * 0.9) for c in gap_col))

    # soft vertical shading
    grad = Image.new("L", (1, height), 0)
    for y in range(height):
        grad.putpixel((0, y), int(16 * math.sin(math.pi * y / height)))
    shade = grad.resize((width, height))
    dark = Image.new("RGB", (width, height), (10, 8, 6))
    img = Image.composite(img, dark, shade.point(lambda v: 255 - v))

    # sparse shelf-grid holes for texture (like pegboard)
    for y in range(0, height, 14):
        for x in range(rng.randint(0, 7), width, 16):
            if rng.random() < 0.30:
                d.ellipse([x, y, x + 3, y + 3], fill=tuple(int(c * 0.82) for c in shelf_col))
    return img


def compose_synthetic_scene(
    bank: CropBank,
    *,
    scene_id: str,
    alpha: float,
    rng: random.Random,
    size: tuple[int, int] = SYNTH_SIZE,
    n_objects: int = 9,
    out_path: str | Path | None = None,
    deploy_vocab: list[str] | None = None,
) -> Scene:
    """Compose one synthetic shelf scene with controlled coverage alpha.

    Placement: products are placed along shelf rows with random scale / small
    rotation; a fraction overlap their predecessor (occlusion). Each product is
    drawn from the full catalogue. The first ``k = round(alpha * n_objects)``
    products are in-vocabulary (GT boxes emitted); the remaining ``n-k`` are
    OOV products (real appearance, no GT) -- coverage alpha is realized exactly
    by construction.

    ``deploy_vocab`` overrides the deployment vocabulary (default 12 classes);
    used for the vocabulary-size sensitivity study.
    """
    vocab = deploy_vocab or DEPLOY_VOCAB
    oov_cat = [c for c in GRASP_PRODUCTS if c not in vocab]
    width, height = size
    canvas = make_shelf_background(width, height, rng=rng)
    rows = 3
    row_h = height / (rows + 0.6)

    n_inv = max(0, min(n_objects, round(alpha * n_objects)))
    n_oov = n_objects - n_inv

    # choose k distinct-ish classes: sample with replacement but avoid repeats
    # where possible for visual diversity.
    def _sample_class(in_vocab: bool, used: set[str], rng: random.Random) -> str:
        pool = [c for c in (vocab if in_vocab else oov_cat) if bank.available(c)]
        fresh = [c for c in pool if c not in used]
        if fresh:
            return rng.choice(fresh)
        if pool:
            return rng.choice(pool)
        raise ValueError(f"no {('in-vocab' if in_vocab else 'OOV')} crops available")

    placed: list[SceneObject] = []
    used: set[str] = set()
    row_idx = 0
    row_x = 0
    max_row_w = int(width * 0.88)

    for i in range(n_objects):
        in_vocab = i < n_inv
        cls = _sample_class(in_vocab, used, rng)
        used.add(cls)
        crop = bank.sample(cls, rng)

        target_w = rng.uniform(110, 180)
        ratio = crop.width / max(1, crop.height)
        target_h = min(target_w / ratio, row_h * 0.78)
        target_w = target_h * ratio
        target_w = min(target_w, max_row_w * 0.55)
        crop = crop.resize((max(4, int(target_w)), max(4, int(target_h))), Image.LANCZOS)

        ang = rng.uniform(-14, 14)
        crop = crop.rotate(ang, expand=True, resample=Image.BICUBIC)

        crop, mask = _rounded_rect_soft(crop, radius=int(max(2, min(crop.width, crop.height) * 0.06)))
        crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.82, 1.12))
        crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.9, 1.1))

        y_center = int(row_h * (row_idx + 0.72))
        if row_x + crop.width > max_row_w:
            row_idx += 1
            row_x = int(width * 0.06)
            y_center = int(row_h * (row_idx + 0.72))
        if row_idx >= rows:
            row_idx = rows - 1
            y_center = int(row_h * (rows - 0.42))

        if placed and rng.random() < 0.30:
            prev = placed[-1]
            px2 = prev.xyxy[2]
            if px2 < width * 0.7:
                row_x = max(row_x, int(px2) - int(crop.width * rng.uniform(0.06, 0.18)))

        x1 = max(0, int(row_x))
        y1 = max(0, int(y_center - crop.height / 2))
        y1 = min(y1, height - crop.height)
        canvas.paste(crop, (int(x1), int(y1)), mask)

        sh = Image.new("RGBA", (int(crop.width * 0.8), 14), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.ellipse([0, 0, sh.width, sh.height], fill=(0, 0, 0, 60))
        sh = sh.filter(ImageFilter.GaussianBlur(3))
        canvas.paste(sh, (int(x1 + crop.width * 0.1), int(y1 + crop.height - 4)), sh)

        placed.append(
            SceneObject(
                cls=cls,
                xyxy=[float(x1), float(y1), float(x1 + crop.width), float(y1 + crop.height)],
                in_vocab=in_vocab,
                oov=not in_vocab,
                source="synthetic",
            )
        )
        row_x = x1 + crop.width + rng.uniform(14, 40)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out, quality=95)

    return Scene(
        scene_id=scene_id,
        image_path=str(Path(out_path).resolve()) if out_path else "",
        objects=placed,
        image_wh=[width, height],
        family="synthetic",
        alpha=round(alpha, 3),
        meta={"n_objects": n_objects, "n_oov": n_oov},
    )


# ---------------------------------------------------------------------------
# Sweep driver: emit a corpus of scenes across the alpha axis
# ---------------------------------------------------------------------------
ALPHA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


def emit_sweep(
    bank: CropBank,
    out_dir: str | Path,
    *,
    scenes_per_alpha: int = 120,
    alphas: list[float] = ALPHA_GRID,
    n_objects: int = 9,
    seed: int = 0,
    deploy_vocab: list[str] | None = None,
) -> dict[str, Any]:
    """Generate synthetic scenes across the alpha grid.

    Returns a corpus manifest:
        {
          "family": "synthetic",
          "alpha_grid": [...],
          "scenes_per_alpha": N,
          "cells": {"alpha=0.2": [relative paths...], ...},
          "all_scenes": [ {scene_id, image_path, alpha, gt} ... ]
        }
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest: dict[str, Any] = {
        "family": "synthetic",
        "alpha_grid": alphas,
        "scenes_per_alpha": scenes_per_alpha,
        "cells": {},
        "all_scenes": [],
    }
    if deploy_vocab is not None:
        manifest["deploy_vocab"] = list(deploy_vocab)
    idx = 0
    for alpha in alphas:
        cell_key = f"alpha={alpha}"
        files: list[str] = []
        for _ in range(scenes_per_alpha):
            idx += 1
            sid = f"synth_{idx:05d}_a{int(alpha * 10):02d}"
            img_rel = f"images/{sid}.jpg"
            img_path = out / img_rel
            scene = compose_synthetic_scene(
                bank,
                scene_id=sid,
                alpha=alpha,
                rng=rng,
                n_objects=n_objects,
                out_path=img_path,
                deploy_vocab=deploy_vocab,
            )
            manifest["all_scenes"].append(
                {
                    "scene_id": sid,
                    "image_path": img_rel,
                    "alpha": scene.alpha,
                    "n_objects": scene.meta["n_objects"],
                    "n_oov": scene.meta["n_oov"],
                    "gt": scene.gt_boxes(),
                    # OOV products present on the shelf (no GT) so the closed
                    # loop can inject realistic phantoms from real OOV classes.
                    "oov_objects": [
                        {"cls": o.cls, "xyxy": o.xyxy}
                        for o in scene.objects
                        if o.oov
                    ],
                }
            )
            files.append(img_rel)
        manifest["cells"][cell_key] = files
    (out / "synthetic_corpus.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# COCO-format export (so YoloWorldAdapter / existing metrics run directly)
# ---------------------------------------------------------------------------
def scenes_to_coco(
    scenes: list[Scene],
    out_dir: str | Path,
    *,
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize scenes to COCO-format annotations (images + gt boxes)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cats = categories or [
        {"id": i + 1, "name": name, "supercategory": "grasp"}
        for i, name in enumerate(GRASP_PRODUCTS)
    ]
    name_to_id = {c["name"]: int(c["id"]) for c in cats}
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    ann_id = 1
    for si, scene in enumerate(scenes):
        img_id = si
        images.append(
            {
                "id": img_id,
                "file_name": Path(scene.image_path).name,
                "width": scene.image_wh[0],
                "height": scene.image_wh[1],
            }
        )
        for o in scene.objects:
            if o.oov:
                continue
            x1, y1, x2, y2 = o.xyxy
            anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_id[o.cls],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "info": {"description": "GraspScope scenes"},
        "licenses": [],
        "categories": cats,
        "images": images,
        "annotations": anns,
    }
    (out / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    return coco
