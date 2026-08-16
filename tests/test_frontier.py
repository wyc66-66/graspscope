"""Tests for GraspScope closed-loop safety statistics (frontier, cliff, gates)."""
from __future__ import annotations

import pytest

from graspscope.closedloop.frontier import (
    SafetyFrontier,
    _is_monotonic_nondecreasing,
    aggregate_tier,
    benjamini_hochberg,
    build_frontier,
    detect_cliff,
    fisher_exact_p,
    gate_rule_from_frontier,
    load_scenarios,
    two_proportion_test,
    wilson_ci,
)
from graspscope.closedloop.error_profile import ClassProfile, PerceptionErrorProfile


def make_point(tier, coverage, n, collision, intervention=0.0, failed=0.0):
    from graspscope.closedloop.frontier import FrontierPoint

    return FrontierPoint(
        tier=tier,
        coverage=coverage,
        n_scenarios=n,
        collision_rate=collision,
        intervention_rate=intervention,
        failure_rate=failed,
    )


# --------------------------------------------------------------------------- #
# Wilson interval
# --------------------------------------------------------------------------- #
class TestWilsonCI:
    def test_p0_bounds(self):
        lo, hi = wilson_ci(75, 0)
        assert lo >= 0.0
        assert hi < 0.1  # 0/75 upper bound stays small

    def test_p1_bounds(self):
        lo, hi = wilson_ci(75, 75)
        assert lo > 0.9
        assert hi <= 1.0

    def test_midpoint_contains_p(self):
        lo, hi = wilson_ci(100, 50)
        assert lo <= 0.5 <= hi

    def test_zero_n(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_monotone_in_n(self):
        # more evidence tightens the interval
        _, hi_small = wilson_ci(10, 5)
        _, hi_large = wilson_ci(1000, 500)
        assert hi_large < hi_small


# --------------------------------------------------------------------------- #
# Cliff detection
# --------------------------------------------------------------------------- #
class TestDetectCliff:
    def test_sharp_cliff_found(self):
        points = [
            make_point("a1.0", 1.0, 75, 0.02),
            make_point("a0.8", 0.8, 75, 0.03),
            make_point("a0.6", 0.6, 75, 0.04),
            make_point("a0.4", 0.4, 75, 0.55),
            make_point("a0.2", 0.2, 75, 0.80),
        ]
        cliff = detect_cliff(points)
        # max-separation split sits between a0.6 and a0.4; cliff tier is the
        # last point on the safe side (a0.6), i.e. safety collapses below it.
        assert cliff["cliff_tier"] == "a0.6"
        assert cliff["cliff_coverage"] == pytest.approx(0.6)
        assert cliff["cliff_separation"] > 1.0
        assert cliff["monotonic"] is True

    def test_flat_curve_no_cliff(self):
        points = [make_point(f"a{i}", i / 10, 75, 0.5) for i in range(10, 0, -1)]
        cliff = detect_cliff(points)
        assert cliff["cliff_separation"] < 1.0

    def test_requires_min_three_tiers(self):
        points = [make_point("a1.0", 1.0, 75, 0.1), make_point("a0.5", 0.5, 75, 0.9)]
        cliff = detect_cliff(points)
        assert cliff["cliff_tier"] is None

    def test_monotonicity_detects_nonmonotone(self):
        vals = [0.02, 0.05, 0.03, 0.6, 0.8]
        assert not _is_monotonic_nondecreasing(vals)
        assert _is_monotonic_nondecreasing([0.02, 0.03, 0.6, 0.8])


# --------------------------------------------------------------------------- #
# Frontier aggregation + bootstrap
# --------------------------------------------------------------------------- #
def _scenarios() -> list[dict]:
    rows = []
    for tier, cov, coll, n in [
        ("a1.0", 1.0, 0.02, 75),
        ("a0.8", 0.8, 0.03, 75),
        ("a0.6", 0.6, 0.04, 75),
        ("a0.4", 0.4, 0.55, 75),
        ("a0.2", 0.2, 0.80, 75),
    ]:
        for _ in range(n):
            rows.append(
                {
                    "tier": tier,
                    "coverage": cov,
                    "collision": 1.0 if cov <= 0.4 else coll,
                    "intervention": 0.0,
                    "failed": 0.0,
                }
            )
    return rows


class TestBuildFrontier:
    def test_aggregates_and_finds_cliff(self):
        front = build_frontier(_scenarios(), n_bootstrap=50)
        assert len(front.points) == 5
        assert front.cliff_coverage == pytest.approx(0.6)
        assert front.monotonic is True
        assert front.cliff_coverage_ci is not None
        lo, hi = front.cliff_coverage_ci
        assert lo <= 0.6 <= hi

    def test_curve_shape(self):
        front = build_frontier(_scenarios(), n_bootstrap=50)
        curve = front.curve()
        assert [p["coverage"] for p in curve] == [1.0, 0.8, 0.6, 0.4, 0.2]
        assert all("collision_rate_ci_lo" in p for p in curve)

    def test_to_dict_roundtrip(self):
        front = build_frontier(_scenarios(), n_bootstrap=50)
        d = front.to_dict()
        assert d["method"] == "max-two-sample-separation"
        assert len(d["curve"]) == 5


class TestAggregateTier:
    def test_perfect_tier(self):
        rows = [{"tier": "t", "coverage": 1.0, "collision": 0.0, "intervention": 0.0, "failed": 0.0} for _ in range(100)]
        pt = aggregate_tier(rows, "t")
        assert pt.collision_rate == 0.0
        assert pt.n_scenarios == 100
        assert pt.collision_ci is not None

    def test_empty_tier_returns_zero(self):
        pt = aggregate_tier([], "missing")
        assert pt.n_scenarios == 0
        assert pt.collision_rate == 0.0


# --------------------------------------------------------------------------- #
# Gate rule
# --------------------------------------------------------------------------- #
class TestGateRule:
    def test_rule_derived_at_threshold(self):
        front = build_frontier(_scenarios(), n_bootstrap=20)
        rule = gate_rule_from_frontier(front, max_rate=0.10)
        assert rule["found"] is True
        assert rule["coverage_min"] >= 0.6

    def test_no_rule_when_never_safe(self):
        bad = [
            make_point("a1.0", 1.0, 75, 0.9),
            make_point("a0.5", 0.5, 75, 0.9),
            make_point("a0.2", 0.2, 75, 0.9),
        ]
        front = SafetyFrontier(points=bad, metric="collision_rate")
        rule = gate_rule_from_frontier(front, max_rate=0.10)
        assert rule["found"] is False


# --------------------------------------------------------------------------- #
# Two-proportion statistics
# --------------------------------------------------------------------------- #
class TestTwoProportion:
    def test_fisher_extreme(self):
        # 0/75 vs 60/75: astronomically small p
        p = fisher_exact_p(75, 0, 75, 60)
        assert p < 1e-10

    def test_fisher_null(self):
        p = fisher_exact_p(50, 25, 50, 25)
        assert p > 0.05

    def test_two_proportion_test(self):
        r = two_proportion_test(75, 3, 75, 55)
        assert r["p_fisher"] < 0.01
        assert r["p_ztest"] < 0.01
        assert abs(r["z"]) > 3.0  # sign is directional; magnitude is the signal

    def test_benjamini_hochberg(self):
        raw = [0.001, 0.01, 0.05, 0.5, 0.6]
        q = benjamini_hochberg(raw)
        assert len(q) == 5
        assert all(0.0 <= x <= 1.0 for x in q)
        # BH q-values are never below their raw p, and remain in order
        assert q[0] == pytest.approx(5 / 1 * 0.001)  # strongest p -> m/p * p
        assert all(q[i] <= q[i + 1] for i in range(4))
        assert max(q) <= 1.0


# --------------------------------------------------------------------------- #
# Error profiles
# --------------------------------------------------------------------------- #
class TestErrorProfile:
    def _profile(self):
        return PerceptionErrorProfile(
            name="p",
            classes={
                "cup": ClassProfile(cls="cup", recall=0.9, loc_recall=0.95),
                "box": ClassProfile(cls="box", recall=0.7),
            },
            oov_fp_by_conf={"0.30": 0.5, "0.50": 0.2, "0.80": 0.05},
            conf_default=0.5,
        )

    def test_miss_rate(self):
        p = self._profile()
        assert p.miss_rate("cup") == pytest.approx(0.1)
        assert p.miss_rate("box") == pytest.approx(0.3)
        assert p.miss_rate("unknown_class") == 1.0  # not in vocab

    def test_oov_fp_exact_threshold(self):
        p = self._profile()
        assert p.oov_fp(0.5) == pytest.approx(0.2)

    def test_oov_fp_interpolates(self):
        p = self._profile()
        assert p.oov_fp(0.40) == pytest.approx(0.35)  # midpoint 0.5<->0.2

    def test_oov_fp_clamps_outside_range(self):
        p = self._profile()
        assert p.oov_fp(0.10) == pytest.approx(0.5)
        assert p.oov_fp(0.95) == pytest.approx(0.05)

    def test_fusion_min_is_conservative(self):
        p = PerceptionErrorProfile(
            name="fused",
            classes={"x": ClassProfile(cls="x", recall=0.9, recall_3d=0.5)},
            injection_source="fused",
            fusion_strategy="min",
        )
        assert p.miss_rate("x") == pytest.approx(0.5)

    def test_fusion_max(self):
        p = PerceptionErrorProfile(
            name="fused",
            classes={"x": ClassProfile(cls="x", recall=0.9, recall_3d=0.5)},
            injection_source="fused",
            fusion_strategy="max",
        )
        assert p.miss_rate("x") == pytest.approx(0.1)

    def test_roundtrip_save_load(self, tmp_path):
        p = self._profile()
        path = tmp_path / "profile.json"
        p.save(path)
        q = PerceptionErrorProfile.load(path)
        assert q.name == p.name
        assert q.miss_rate("cup") == pytest.approx(0.1)
        assert q.oov_fp(0.40) == pytest.approx(0.35)

    def test_from_dict_tolerates_missing_new_fields(self):
        d = {
            "name": "legacy",
            "classes": {"cup": {"cls": "cup", "recall": 0.8}},
            "oov_fp_by_conf": {},
        }
        p = PerceptionErrorProfile.from_dict(d)
        assert p.miss_rate("cup") == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Scenario loading
# --------------------------------------------------------------------------- #
class TestLoadScenarios:
    def test_list_json(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text('[{"tier":"a","collision":0}]', encoding="utf-8")
        assert load_scenarios(f) == [{"tier": "a", "collision": 0}]

    def test_wrapped_json(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text('{"scenarios":[{"tier":"a"}]}', encoding="utf-8")
        assert load_scenarios(f) == [{"tier": "a"}]
