"""Disk cache for adapter predictions keyed by image+vocab+model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from opengate.schema import Prediction, Sample
from opengate.schema.models import prediction_from_dict


def _key(sample_id: str, image_path: str, vocab: list[str], model_key: str) -> str:
    payload = json.dumps(
        {"sid": sample_id, "img": image_path, "vocab": vocab, "model": model_key},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def cache_dir(config: dict[str, Any]) -> Path | None:
    raw = config.get("cache_dir")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cached(
    cache: Path,
    sample: Sample,
    vocab: list[str],
    model_key: str,
) -> Prediction | None:
    name = _key(sample.sample_id, sample.image_path, vocab, model_key) + ".json"
    path = cache / name
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return prediction_from_dict(data)


def store_cached(
    cache: Path,
    sample: Sample,
    vocab: list[str],
    model_key: str,
    pred: Prediction,
) -> None:
    name = _key(sample.sample_id, sample.image_path, vocab, model_key) + ".json"
    path = cache / name
    payload = {
        "sample_id": pred.sample_id,
        "boxes": [
            {"xyxy": list(b.xyxy), "cls": b.cls, "score": b.score} for b in pred.boxes
        ],
        "meta": pred.meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
