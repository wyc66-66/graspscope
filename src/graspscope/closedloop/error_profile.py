"""Perception error profiles: how a real perception stack fails.

A :class:`PerceptionErrorProfile` is a compact, versioned description of a
deployment perception stack's failure modes, derived from real GPU measurements:

- **per-class recall** (known classes detected / GT) — drives *misses*;
- **OOV-FP** at several confidence thresholds (out-of-vocabulary detections as
  a fraction of all high-confidence detections) — drives *phantoms*;
- **label confusion** probabilities (optional) — drives *mislabels*.

Profiles come from the open-vocabulary 2D detection audit on the target scenes
(real COCO frames + synthetic shelf tiers). This module is pure Python so it is
CI-safe.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any


@dataclass
class ClassProfile:
    """Per-class perception quality in deployment."""

    cls: str
    recall: float
    precision: float = 1.0
    n_gt: int = 0
    n_tp: int = 0
    # Recall from a secondary source (e.g. a 3D detector), when the profile was
    # fused from multiple measurement channels. Kept separate from ``recall``
    # (which comes from the primary 2D open-vocabulary audit) for honest fusion.
    recall_3d: float | None = None
    # localization recall (IoU match regardless of predicted class) drives
    # "grasp the object at the wrong identity"; per-class label confusion counts
    # {predicted_class: n} for GT objects whose box was localized but classified
    # incorrectly.
    loc_recall: float | None = None
    confusion: dict[str, int] = field(default_factory=dict)


@dataclass
class PerceptionErrorProfile:
    """Versioned error profile consumed by the closed-loop engine."""

    name: str
    classes: dict[str, ClassProfile]
    oov_fp_by_conf: dict[str, float] = field(default_factory=dict)
    conf_default: float = 0.5
    sources: list[str] = field(default_factory=list)
    generated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: dict[str, Any] = field(default_factory=dict)
    # Which measurement(s) drive the injected miss/phantom rates:
    # "2d" (primary open-vocabulary detection audit), "3d" (secondary channel,
    # e.g. a 3D detector), or "fused" (both merged by the fusion strategy below).
    injection_source: str = "2d"
    # Fusion strategy when injection_source == "fused": "min" (use the lower of
    # recall / recall_3d per class — conservative), "max", or "2d" / "3d"
    # (prefer one channel). Applies to per-class miss rates.
    fusion_strategy: str = "min"

    def miss_rate(self, cls: str) -> float:
        """Probability that an in-vocabulary object of *cls* is not perceived."""
        prof = self.classes.get(cls)
        if prof is None:
            return 1.0  # unknown class -> always missed (not in vocabulary)
        return max(0.0, 1.0 - self._effective_recall(prof))

    def _effective_recall(self, prof: ClassProfile) -> float:
        """Resolve the recall used for injection under the fusion strategy."""
        r2d = prof.recall
        r3d = prof.recall_3d
        if self.injection_source == "3d":
            return r3d if r3d is not None else r2d
        if self.injection_source == "2d":
            return r2d
        # fused
        if r3d is None:
            return r2d
        if self.fusion_strategy == "max":
            return max(r2d, r3d)
        if self.fusion_strategy == "3d":
            return r3d
        return min(r2d, r3d)  # default conservative "min"

    def oov_fp(self, conf: float | None = None) -> float:
        """Out-of-vocab false-positive rate at a confidence threshold.

        Uses the exact measured value when ``conf`` is one of the measured
        thresholds, otherwise **linearly interpolates** between the two
        bracketing measured thresholds (OOV-FP is monotonically decreasing in
        confidence for a well-calibrated detector, so interpolation is
        well-behaved). Falls back to the default when no measured threshold
        exists.
        """
        c = f"{conf:.2f}" if conf is not None else f"{self.conf_default:.2f}"
        if c in self.oov_fp_by_conf:
            return float(self.oov_fp_by_conf[c])
        if conf is None:
            return float(self.oov_fp_by_conf.get("default", 0.0))
        measured = sorted(
            (float(k), float(v))
            for k, v in self.oov_fp_by_conf.items()
            if k != "default" and _is_float(k)
        )
        if not measured:
            return float(self.oov_fp_by_conf.get("default", 0.0))
        if conf <= measured[0][0]:
            return measured[0][1]
        if conf >= measured[-1][0]:
            return measured[-1][1]
        for (ca, va), (cb, vb) in pairwise(measured):
            if ca <= conf <= cb:
                t = (conf - ca) / (cb - ca)
                return float(va * (1 - t) + vb * t)
        return float(self.oov_fp_by_conf.get("default", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerceptionErrorProfile:
        classes = {}
        for k, v in d.get("classes", {}).items():
            if isinstance(v, dict):
                # tolerate missing new fields (older JSON profiles)
                v.setdefault("recall_3d", None)
                v.setdefault("loc_recall", None)
                v.setdefault("confusion", {})
                classes[k] = ClassProfile(**v)
            else:
                classes[k] = v
        return cls(
            name=str(d.get("name", "unnamed")),
            classes=classes,
            oov_fp_by_conf={str(k): float(v) for k, v in d.get("oov_fp_by_conf", {}).items()},
            conf_default=float(d.get("conf_default", 0.5)),
            sources=[str(x) for x in d.get("sources", [])],
            generated=str(d.get("generated", "")),
            meta=dict(d.get("meta") or {}),
            injection_source=str(d.get("injection_source", "2d")),
            fusion_strategy=str(d.get("fusion_strategy", "min")),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PerceptionErrorProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
