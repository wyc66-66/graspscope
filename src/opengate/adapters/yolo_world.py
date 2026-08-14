from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from opengate.adapters.base import Adapter, register_adapter
from opengate.adapters.cache import cache_dir, load_cached, store_cached
from opengate.adapters.weights import resolve_weights
from opengate.errors import MissingFileError
from opengate.schema import Box, Prediction, Sample


@register_adapter
class YoloWorldAdapter(Adapter):
    """Ultralytics YOLO-World open-vocabulary detector."""

    name = "yolo_world"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._model = None
        self._weights_key = ""

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "yolo_world adapter requires optional deps: pip install 'opengate[l1]'"
            ) from e
        weights = resolve_weights(self.config, "yolov8s-world.pt")
        self._weights_key = weights
        self._model = YOLO(weights)
        return self._model

    def predict(self, samples: list[Sample], vocab: list[str]) -> list[Prediction]:
        model = self._load()
        conf = float(self.config.get("conf", 0.25))
        cache = cache_dir(self.config)
        model_key = f"yolo_world:{self._weights_key}:conf={conf}"
        if vocab:
            # ultralytics keeps the CLIP text encoder on the device where the
            # model was first loaded; repeated set_classes calls can leave the
            # token_embedding on CPU. Force the text model to the same device.
            try:
                model.to("cuda" if model.device.type == "cuda" else "cpu")
            except Exception:  # noqa: BLE001
                pass
            model.set_classes(list(vocab))
        out: list[Prediction] = []
        for s in samples:
            if cache:
                hit = load_cached(cache, s, vocab, model_key)
                if hit is not None:
                    out.append(hit)
                    continue
            path = Path(s.image_path)
            if not path.is_file():
                # try under OPENGATE_DATA_ROOT
                root = os.environ.get("OPENGATE_DATA_ROOT")
                if root:
                    alt = Path(root) / path.name
                    if alt.is_file():
                        path = alt
            if not path.is_file():
                raise MissingFileError(f"Image not found for yolo_world: {s.image_path}")
            results = model.predict(
                str(path),
                conf=conf,
                imgsz=int(self.config.get("imgsz", 640)),
                verbose=False,
            )
            boxes: list[Box] = []
            if results:
                r0 = results[0]
                names = r0.names or {}
                if r0.boxes is not None:
                    for b in r0.boxes:
                        xyxy = b.xyxy[0].tolist()
                        cls_id = int(b.cls[0].item())
                        score = float(b.conf[0].item())
                        cls_name = str(
                            names.get(
                                cls_id,
                                vocab[cls_id] if cls_id < len(vocab) else cls_id,
                            )
                        )
                        boxes.append(Box(xyxy=xyxy, cls=cls_name, score=score))
            pred = Prediction(sample_id=s.sample_id, boxes=boxes)
            if cache:
                store_cached(cache, s, vocab, model_key, pred)
            out.append(pred)
        return out
