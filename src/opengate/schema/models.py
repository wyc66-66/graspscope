from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Box:
    """Axis-aligned box in xyxy pixel coordinates."""

    xyxy: list[float]
    cls: str
    score: float = 1.0

    def to_xywh(self) -> list[float]:
        x1, y1, x2, y2 = self.xyxy
        return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


@dataclass
class Sample:
    sample_id: str
    image_path: str
    view_id: str = "default"
    gt_boxes: list[Box] = field(default_factory=list)
    image_wh: list[int] | None = None  # [W, H] optional for QC
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    episode_id: str
    sample_ids: list[str]
    vocab: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    sample_id: str
    boxes: list[Box] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QCIssue:
    sample_id: str
    rule: str
    message: str
    severity: str = "warn"  # warn | error


@dataclass
class GateResult:
    passed: bool
    reason: str
    thresholds: dict[str, float] = field(default_factory=dict)
    observed: dict[str, float] = field(default_factory=dict)
    layer: str = "l1"


@dataclass
class InstructionItem:
    item_id: str
    image_path: str
    instruction: str
    accepted_answers: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportBundle:
    run_id: str
    adapter: str
    layers: list[str]
    metrics: dict[str, Any]
    gate: list[GateResult]
    qc_issues: list[QCIssue] = field(default_factory=list)
    per_episode: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def box_from_dict(d: dict[str, Any]) -> Box:
    return Box(
        xyxy=[float(x) for x in d["xyxy"]],
        cls=str(d["cls"]),
        score=float(d.get("score", 1.0)),
    )


def sample_from_dict(d: dict[str, Any]) -> Sample:
    return Sample(
        sample_id=str(d["sample_id"]),
        image_path=str(d["image_path"]),
        view_id=str(d.get("view_id", "default")),
        gt_boxes=[box_from_dict(b) for b in d.get("gt_boxes", [])],
        image_wh=[int(x) for x in d["image_wh"]] if d.get("image_wh") else None,
        meta=dict(d.get("meta") or {}),
    )


def episode_from_dict(d: dict[str, Any]) -> Episode:
    return Episode(
        episode_id=str(d["episode_id"]),
        sample_ids=[str(x) for x in d["sample_ids"]],
        vocab=[str(x) for x in d["vocab"]],
        meta=dict(d.get("meta") or {}),
    )


def prediction_from_dict(d: dict[str, Any]) -> Prediction:
    return Prediction(
        sample_id=str(d["sample_id"]),
        boxes=[box_from_dict(b) for b in d.get("boxes", [])],
        meta=dict(d.get("meta") or {}),
    )
