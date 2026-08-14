from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from graspscope.adapters.base import Adapter, register_adapter
from graspscope.adapters.cache import cache_dir, load_cached, store_cached
from graspscope.adapters.weights import resolve_weights
from graspscope.errors import MissingFileError
from graspscope.schema import Box, Prediction, Sample


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
                "yolo_world adapter requires optional deps: pip install 'graspscope[l1]'"
            ) from e
        weights = resolve_weights(self.config, "yolov8s-world.pt")
        self._weights_key = weights
        self._model = YOLO(weights)
        # Move to GPU when available; ultralytics otherwise defaults to CPU and
        # per-image predict calls drop to ~300ms each even on a gaming GPU.
        try:
            import torch

            if torch.cuda.is_available():
                self._model.to("cuda")
        except Exception:  # noqa: BLE001
            pass
        return self._model

    def predict(self, samples: list[Sample], vocab: list[str]) -> list[Prediction]:
        model = self._load()
        conf = float(self.config.get("conf", 0.25))
        cache = cache_dir(self.config)
        model_key = f"yolo_world:{self._weights_key}:conf={conf}"
        if vocab:
            # ultralytics keeps the CLIP text encoder on the device where the
            # model was first loaded; repeated set_classes calls can leave the
            # token_embedding on CPU. Force the text model back onto the same
            # device as the backbone.
            try:
                dev = str(getattr(model, "device", ""))
                model.to("cuda" if "cuda" in dev else "cpu")
            except Exception:  # noqa: BLE001
                pass
            model.set_classes(list(vocab))

        # Resolve every sample to a real file first (cheap, cacheable).
        paths: list[Path] = []
        missing: list[str] = []
        for s in samples:
            path = Path(s.image_path)
            if not path.is_file():
                # try under GRASPSCOPE_DATA_ROOT
                root = os.environ.get("GRASPSCOPE_DATA_ROOT")
                if root:
                    alt = Path(root) / path.name
                    if alt.is_file():
                        path = alt
            if not path.is_file():
                missing.append(s.image_path)
            paths.append(path)
        if missing:
            raise MissingFileError(f"Images not found for yolo_world: {missing[:5]} ...")

        # Predict in batches so ultralytics can pipeline the GPU, instead of
        # one python-level call per image. Cached samples are excluded from
        # the batch and merged back in order.
        out: list[Prediction | None] = [None] * len(samples)
        to_run = []
        for i, s in enumerate(samples):
            if cache:
                hit = load_cached(cache, s, vocab, model_key)
                if hit is not None:
                    out[i] = hit
                    continue
            to_run.append((i, s))

        results_map: dict[int, Prediction] = {}
        for start in range(0, len(to_run), self._batch_size()):
            chunk = to_run[start : start + self._batch_size()]
            results = model.predict(
                [str(paths[i]) for i, _ in chunk],
                conf=conf,
                imgsz=int(self.config.get("imgsz", 640)),
                verbose=False,
            )
            for (i, s), r0 in zip(chunk, results):
                boxes: list[Box] = []
                if r0.boxes is not None:
                    names = r0.names or {}
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
                results_map[i] = pred

        return [p if p is not None else results_map[i] for i, p in enumerate(out)]

    def _batch_size(self) -> int:
        return int(self.config.get("batch", 64))
