"""Tests for the GraspScope closed-loop grasp simulator."""
from __future__ import annotations

import pytest

from graspscope.closedloop.error_profile import ClassProfile, PerceptionErrorProfile
from graspscope.grasp.grasp_env import (
    GraspEnv,
    GraspOutcome,
    GraspProduct,
    GraspScenario,
)


def make_profile(recall=0.9, loc_recall=0.95, oov_fp=0.2) -> PerceptionErrorProfile:
    return PerceptionErrorProfile(
        name="test",
        classes={
            "cup": ClassProfile(cls="cup", recall=recall, loc_recall=loc_recall),
            "box": ClassProfile(cls="box", recall=recall, loc_recall=loc_recall),
        },
        oov_fp_by_conf={"0.50": oov_fp},
        conf_default=0.5,
    )


def make_scenario(coverage=1.0, tier="t1", n_in_vocab=3, n_oov=0) -> GraspScenario:
    products = [
        GraspProduct(product_id=f"v{i}", cls="cup", xyxy=[i, 0, i + 1, 1], in_vocab=True)
        for i in range(n_in_vocab)
    ]
    products += [
        GraspProduct(product_id=f"o{i}", cls="water_bottle", xyxy=[0, 0, 1, 1], in_vocab=False)
        for i in range(n_oov)
    ]
    return GraspScenario(scenario_id="s", tier=tier, coverage=coverage, products=products)


class TestSceneBuilders:
    def test_manifest_rec_builder(self):
        rec = {
            "scene_id": "sc1",
            "alpha": 0.8,
            "image_path": "sc1.png",
            "n_objects": 2,
            "n_oov": 1,
            "gt": [{"cls": "cup", "xyxy": [1, 2, 3, 4]}],
            "oov_objects": [{"cls": "water_bottle", "xyxy": [5, 6, 7, 8]}],
        }
        s = GraspEnv.scene_from_manifest_rec(rec, image_dir="imgs", tier="a0.8")
        assert s.coverage == 0.8
        assert len(s.products) == 2
        assert len(s.in_vocab_products) == 1
        assert s.image_path.endswith("sc1.png")

    def test_coco_builder_missing_image_raises(self):
        with pytest.raises(KeyError):
            GraspEnv.scene_from_coco(
                {"images": [], "annotations": [], "categories": []},
                image_dir=".",
                image_id=1,
                coverage=1.0,
                tier="t",
                oov_classes=["x"],
                rng=__import__("random").Random(0),
            )

    def test_coco_builder_adds_oov(self):
        coco = {
            "images": [{"id": 1, "file_name": "a.png"}],
            "categories": [{"id": 1, "name": "cup"}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]}],
        }
        s = GraspEnv.scene_from_coco(
            coco,
            image_dir=".",
            image_id=1,
            coverage=0.2,
            tier="a0.2",
            oov_classes=["bottle"],
            rng=__import__("random").Random(0),
        )
        assert len(s.in_vocab_products) == 1
        # coverage=0.2 with 1 in-vocab product -> round(1*0.8) = 1 OOV added
        assert len(s.products) == 2


class TestGraspOutcomes:
    def test_empty_grasp_when_all_missed(self):
        # recall=0 -> nothing perceived -> empty grasp
        env = GraspEnv(make_profile(recall=0.0, loc_recall=0.0), seed=0)
        out = env.run_episode(make_scenario())
        assert out.success is False
        assert out.failure_type == "empty_grasp"

    def test_success_when_perfect(self):
        env = GraspEnv(make_profile(recall=1.0, loc_recall=1.0), exec_success=1.0, seed=0)
        out = env.run_episode(make_scenario())
        assert out.success is True
        assert out.failure_type == "success"

    def test_drop_with_low_exec(self):
        env = GraspEnv(make_profile(recall=1.0, loc_recall=1.0), exec_success=0.0, seed=0)
        out = env.run_episode(make_scenario())
        assert out.success is False
        assert out.failure_type == "drop"

    def test_phantom_yields_wrong_object(self):
        # all in-vocab products missed, but an OOV phantom fires as a known class
        profile = PerceptionErrorProfile(
            name="t",
            classes={"cup": ClassProfile(cls="cup", recall=0.0, loc_recall=0.0)},
        )
        env = GraspEnv(profile, phantom_rate=1.0, exec_success=1.0, seed=1)
        # OOV product will be hallucinated as "cup"; no true in-vocab candidate
        out = env.run_episode(make_scenario(n_in_vocab=1, n_oov=1))
        # with phantom fired there is a candidate but it is OOV -> wrong object
        if any(p["in_vocab"] for p in out.perceived):
            assert out.failure_type == "wrong_object"
        else:
            assert out.failure_type == "empty_grasp"

    def test_outcome_dict_shape(self):
        env = GraspEnv(make_profile(), seed=0)
        out = env.run_episode(make_scenario())
        d = out.to_dict()
        assert d["scenario_id"] == "s"
        assert d["tier"] == "t1"
        assert isinstance(d["success"], bool)
        assert d["n_attempts"] >= 1


class TestDeterminism:
    def test_same_seed_same_outcome(self):
        env1 = GraspEnv(make_profile(recall=0.5), seed=42)
        env2 = GraspEnv(make_profile(recall=0.5), seed=42)
        o1 = env1.run_episode(make_scenario())
        o2 = env2.run_episode(make_scenario())
        assert o1.success == o2.success
        assert o1.failure_type == o2.failure_type

    def test_corpus_runs(self):
        env = GraspEnv(make_profile(), seed=0)
        scenarios = [make_scenario(coverage=1.0, tier="t", n_in_vocab=2) for _ in range(20)]
        outcomes = env.run_corpus(scenarios)
        assert len(outcomes) == 20
        assert all(isinstance(o, GraspOutcome) for o in outcomes)
