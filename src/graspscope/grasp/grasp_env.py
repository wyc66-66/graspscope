"""Closed-loop grasping simulator for GraspScope.

Turns a scene of products (in-vocabulary + OOV) plus a perception error profile
into per-grasp outcomes, closing the loop from **perception failures** to
**grasp failures**:

- **miss**            : an in-vocabulary product the perception stack fails to
                        localize -> no candidate -> grasp attempt targets
                        nothing / the wrong slot -> "empty grasp" (抓空).
- **label confusion** : a product is localized but assigned a wrong in-vocab
                        class -> robot picks the wrong object -> "wrong object"
                        (抓错).
- **OOV phantom**     : an out-of-vocabulary product is detected as some known
                        class (open-vocab detector has no word for it, so it
                        snaps to the closest prompt) -> "wrong object".
- **execution failure**: even with correct perception the physical grasp can
                        drop; anchored to published gripper success (~0.9-0.95,
                        e.g. GraspVLA reports ~90-95% in its real trials).

The simulator is deliberately *relative* rather than a physics engine: it
quantifies how perception degradation propagates into grasp reliability, with
the execution layer calibrated to public numbers. The interface is decoupled
so the execution layer can later be swapped for a full simulator.

``run_episode`` mirrors the grasp closed-loop run: given a scenario
(scene + tier) it returns a binary/typed outcome, and the safety-frontier
module (:mod:`graspscope.closedloop.frontier`) aggregates many episodes into the
coverage->reliability frontier and safety cliff.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graspscope.closedloop.error_profile import PerceptionErrorProfile


@dataclass
class GraspProduct:
    """A product placed on the shelf (ground truth)."""

    product_id: str
    cls: str
    xyxy: list[float]  # pixel box in the scene image
    in_vocab: bool = True


@dataclass
class GraspScenario:
    """One grasp episode scenario (a scene at a given coverage tier)."""

    scenario_id: str
    tier: str
    coverage: float  # alpha: fraction of products in-vocabulary
    products: list[GraspProduct]
    image_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def in_vocab_products(self) -> list[GraspProduct]:
        return [p for p in self.products if p.in_vocab]


@dataclass
class GraspOutcome:
    """Result of one grasp episode."""

    scenario_id: str
    tier: str
    coverage: float
    success: bool
    failure_type: str  # success | empty_grasp | wrong_object | drop
    perceived: list[dict[str, Any]]  # products the perception stack "saw"
    target_cls: str | None = None  # class the robot intended to grasp
    n_attempts: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "tier": self.tier,
            "coverage": self.coverage,
            "success": self.success,
            "failure_type": self.failure_type,
            "target_cls": self.target_cls,
            "n_attempts": self.n_attempts,
            "n_perceived": len(self.perceived),
        }


class GraspEnv:
    """Closed-loop grasp simulator parameterized by a perception profile.

    :param profile: per-tier perception error profile (recall / loc_recall /
        oov_fp). For tiers below the real alpha grid, profiles are provided per
        tier by the caller (one profile per alpha sweep point).
    :param exec_success: execution-layer success probability (published anchor,
        e.g. 0.92).
    :param phantom_rate: probability per OOV product that it is hallucinated as
        a known class (drives wrong-object at low alpha).
    :param seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        profile: PerceptionErrorProfile,
        *,
        exec_success: float = 0.92,
        phantom_rate: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.profile = profile
        self.exec_success = exec_success
        self.phantom_rate = phantom_rate
        self._rng = random.Random(seed)

    # -- scene construction -------------------------------------------------
    @staticmethod
    def scene_from_manifest_rec(
        rec: dict[str, Any],
        image_dir: str | Path,
        *,
        tier: str,
    ) -> GraspScenario:
        """Build a scenario from one synthetic-corpus manifest record.

        The manifest record already encodes the full product set at the
        scenario's coverage: ``gt`` are in-vocabulary products (with boxes),
        ``oov_objects`` are out-of-vocabulary products on the shelf.
        """
        products: list[GraspProduct] = []
        for i, g in enumerate(rec.get("gt", [])):
            products.append(
                GraspProduct(
                    product_id=f"{rec['scene_id']}_v{i}",
                    cls=g["cls"],
                    xyxy=g["xyxy"],
                    in_vocab=True,
                )
            )
        for j, o in enumerate(rec.get("oov_objects", [])):
            products.append(
                GraspProduct(
                    product_id=f"{rec['scene_id']}_oov{j}",
                    cls=o["cls"],
                    xyxy=o.get("xyxy", [0.0, 0.0, 0.0, 0.0]),
                    in_vocab=False,
                )
            )
        return GraspScenario(
            scenario_id=rec["scene_id"],
            tier=tier,
            coverage=float(rec["alpha"]),
            products=products,
            image_path=str(Path(image_dir) / Path(rec["image_path"]).name),
            meta={"n_objects": rec.get("n_objects", 0), "n_oov": rec.get("n_oov", 0)},
        )

    @staticmethod
    def scene_from_coco(
        coco: dict[str, Any],
        image_dir: str | Path,
        *,
        image_id: int,
        coverage: float,
        tier: str,
        oov_classes: list[str],
        rng: random.Random,
    ) -> GraspScenario:
        """Build a scenario from one COCO image + its annotations.

        In-vocabulary products come from the COCO GT annotations. OOV products
        are synthesized: for each in-vocab product present, with probability
        ``1 - coverage`` we *add* an OOV product (a product not in V) so the
        scene's effective coverage equals ``coverage``.
        """
        images = {int(i["id"]): i for i in coco["images"]}
        cats = {int(c["id"]): c["name"] for c in coco["categories"]}
        anns = [
            a for a in coco["annotations"] if int(a["image_id"]) == image_id
        ]
        img = images.get(image_id)
        if img is None:
            raise KeyError(f"image {image_id} not in coco")
        products: list[GraspProduct] = []
        for i, a in enumerate(anns):
            name = cats.get(int(a["category_id"]), "?")
            x, y, w, h = (float(v) for v in a["bbox"])
            products.append(
                GraspProduct(
                    product_id=f"{image_id}_p{i}",
                    cls=name,
                    xyxy=[x, y, x + w, y + h],
                    in_vocab=True,
                )
            )
        # add OOV products to hit the requested coverage
        n_in = len(products)
        if n_in > 0:
            n_oov = max(0, round(n_in * (1.0 - coverage)))
            for i in range(n_oov):
                cls = rng.choice(oov_classes)
                products.append(
                    GraspProduct(
                        product_id=f"{image_id}_oov{i}",
                        cls=cls,
                        xyxy=[0, 0, 0, 0],
                        in_vocab=False,
                    )
                )
        return GraspScenario(
            scenario_id=f"{tier}_{image_id}",
            tier=tier,
            coverage=coverage,
            products=products,
            image_path=str(Path(image_dir) / img["file_name"]),
        )

    # -- perception injection ------------------------------------------------
    def _perceive(self, scenario: GraspScenario) -> list[dict[str, Any]]:
        """Corrupt ground truth per the profile -> perceived products.

        Returns a list of dicts ``{product_id, cls, in_vocab, source}`` where
        source is 'true' (localized correctly), 'confused' (wrong label) or
        'phantom' (OOV hallucinated as known).
        """
        perceived: list[dict[str, Any]] = []
        for p in scenario.products:
            if not p.in_vocab:
                # OOV product: no word in V -> phantom if it fires
                if self._rng.random() < self.phantom_rate:
                    # hallucinate as a random in-vocab class
                    fake = self._rng.choice(list(self.profile.classes))
                    perceived.append(
                        {
                            "product_id": p.product_id,
                            "cls": fake,
                            "in_vocab": False,
                            "source": "phantom",
                            "gt_cls": p.cls,
                        }
                    )
                continue
            # in-vocab product
            prof = self.profile.classes.get(p.cls)
            if prof is None:
                continue
            loc = prof.loc_recall if prof.loc_recall is not None else prof.recall
            if self._rng.random() > loc:
                continue  # missed -> no candidate for this product
            # localized; was it labeled correctly? Use the measured confusion
            # distribution when available (which classes it gets mixed into),
            # otherwise fall back to uniform in-vocab confusion.
            cond_confuse = prof.recall / max(1e-9, loc) if loc > 0 else 1.0
            if prof.recall > 0 and self._rng.random() >= (prof.recall / max(1e-9, loc)):
                conf_map = prof.confusion
                if conf_map:
                    fake = self._rng.choices(
                        list(conf_map), weights=list(conf_map.values())
                    )[0]
                else:
                    fake = self._rng.choice(
                        [c for c in self.profile.classes if c != p.cls]
                    )
                perceived.append(
                    {
                        "product_id": p.product_id,
                        "cls": fake,
                        "in_vocab": True,
                        "source": "confused",
                        "gt_cls": p.cls,
                    }
                )
            else:
                perceived.append(
                    {
                        "product_id": p.product_id,
                        "cls": p.cls,
                        "in_vocab": True,
                        "source": "true",
                        "gt_cls": p.cls,
                    }
                )
        return perceived

    # -- grasping ------------------------------------------------------------
    def _grasp(self, scenario: GraspScenario, perceived: list[dict[str, Any]]) -> GraspOutcome:
        """Pick an in-vocab perceived product and try to grasp it."""
        candidates = [p for p in perceived if p["in_vocab"]]
        if not candidates:
            return GraspOutcome(
                scenario_id=scenario.scenario_id,
                tier=scenario.tier,
                coverage=scenario.coverage,
                success=False,
                failure_type="empty_grasp",
                perceived=perceived,
                target_cls=None,
            )
        # prefer a correctly-labeled product; else take any candidate
        target = next((p for p in candidates if p["source"] == "true"), candidates[0])
        # execution layer: even correct perception can fail physically; allow
        # one retry (mirrors real picking loops that re-grasp after a drop).
        for attempt in range(2):
            if self._rng.random() > self.exec_success:
                continue  # dropped; retry
            # did we actually grasp the intended object?
            correct = target["source"] == "true"
            if not correct:
                return GraspOutcome(
                    scenario_id=scenario.scenario_id,
                    tier=scenario.tier,
                    coverage=scenario.coverage,
                    success=False,
                    failure_type="wrong_object",
                    perceived=perceived,
                    target_cls=target["cls"],
                    n_attempts=attempt + 1,
                )
            return GraspOutcome(
                scenario_id=scenario.scenario_id,
                tier=scenario.tier,
                coverage=scenario.coverage,
                success=True,
                failure_type="success",
                perceived=perceived,
                target_cls=target["cls"],
                n_attempts=attempt + 1,
            )
        # both attempts dropped
        return GraspOutcome(
            scenario_id=scenario.scenario_id,
            tier=scenario.tier,
            coverage=scenario.coverage,
            success=False,
            failure_type="drop",
            perceived=perceived,
            target_cls=target["cls"],
            n_attempts=2,
        )

    def run_episode(self, scenario: GraspScenario) -> GraspOutcome:
        """One closed-loop grasp episode: perceive -> grasp -> outcome."""
        perceived = self._perceive(scenario)
        return self._grasp(scenario, perceived)

    # -- episode batch ---------------------------------------------------------
    def run_corpus(
        self,
        scenarios: list[GraspScenario],
    ) -> list[GraspOutcome]:
        """Run every scenario once; outcomes feed the frontier aggregator."""
        return [self.run_episode(s) for s in scenarios]

    # -- scenario factory for alpha sweep -------------------------------------
    @classmethod
    def build_sweep_scenarios(
        cls,
        manifest: dict[str, Any],
        image_dir: str | Path,
        *,
        alphas: list[float],
        n_scenes: int = 150,
        seed: int = 0,
    ) -> list[GraspScenario]:
        """Build one scenario set per alpha tier from the synthetic manifest.

        Each tier uses the scenes emitted at that exact alpha, so coverage is
        realized by construction (products are already split into in-vocab /
        OOV at the scene level). This makes the coverage sweep exact.
        """
        rng = random.Random(seed)
        scenarios: list[GraspScenario] = []
        by_alpha: dict[str, list[dict[str, Any]]] = {}
        for rec in manifest["all_scenes"]:
            by_alpha.setdefault(f"{rec['alpha']:.1f}", []).append(rec)
        for alpha in alphas:
            tier = f"alpha_{alpha:.1f}"
            recs = by_alpha.get(f"{alpha:.1f}", [])[:n_scenes]
            for rec in recs:
                scenarios.append(
                    cls.scene_from_manifest_rec(rec, image_dir, tier=tier)
                )
        return scenarios
