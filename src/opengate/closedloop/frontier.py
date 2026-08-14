"""Safety-frontier analysis for closed-loop perception degradation.

Aggregates per-scenario closed-loop metrics into a **safety frontier**: for
each perception degradation tier (vocabulary coverage / error mode), it reports
collision rate, intervention rate, failure rate and mean PDMS score, and
detects the **safety cliff** — the coverage level below which closed-loop
safety collapses.

Cliff detection uses a pure-Python change-point estimator (max two-sample
separation over candidate coverage thresholds), so it is dependency-free and
CI-safe.

This module has no nuPlan dependency: it consumes the plain-JSON per-scenario
metrics produced by ``wsl2/nuplan_closedloop/run_sweep.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

# Safety metrics available in a nuPlan closed-loop run (binary per scenario).
SAFETY_METRICS = ("collision", "intervention", "failed")


@dataclass
class FrontierPoint:
    """Aggregated closed-loop safety at one perception tier."""

    tier: str
    coverage: float  # aggregate perception coverage in [0, 1]
    n_scenarios: int
    collision_rate: float
    intervention_rate: float
    failure_rate: float
    pdms_mean: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Wilson 95% CIs for the binary rates (lower, upper) — None before compute.
    collision_ci: tuple[float, float] | None = None
    intervention_ci: tuple[float, float] | None = None
    failure_ci: tuple[float, float] | None = None

    def with_cis(self) -> FrontierPoint:
        """Return a copy with Wilson 95% CIs computed for the binary rates."""
        coll_lo, coll_hi = wilson_ci(self.n_scenarios, round(self.collision_rate * self.n_scenarios))
        int_lo, int_hi = wilson_ci(self.n_scenarios, round(self.intervention_rate * self.n_scenarios))
        fail_lo, fail_hi = wilson_ci(self.n_scenarios, round(self.failure_rate * self.n_scenarios))
        return FrontierPoint(
            tier=self.tier,
            coverage=self.coverage,
            n_scenarios=self.n_scenarios,
            collision_rate=self.collision_rate,
            intervention_rate=self.intervention_rate,
            failure_rate=self.failure_rate,
            pdms_mean=self.pdms_mean,
            meta=dict(self.meta),
            collision_ci=(coll_lo, coll_hi),
            intervention_ci=(int_lo, int_hi),
            failure_ci=(fail_lo, fail_hi),
        )


@dataclass
class SafetyFrontier:
    """Ordered safety frontier + cliff detection result."""

    points: list[FrontierPoint]
    metric: str = "collision_rate"
    cliff_tier: str | None = None
    cliff_coverage: float | None = None
    cliff_separation: float = 0.0
    monotonic: bool = False
    method: str = "max-two-sample-separation"
    # Bootstrap 95% CI around the cliff location and separation.
    cliff_coverage_ci: tuple[float, float] | None = None
    cliff_separation_ci: tuple[float, float] | None = None
    n_bootstrap: int = 0

    def sorted_points(self) -> list[FrontierPoint]:
        return sorted(self.points, key=lambda p: p.coverage, reverse=True)

    def curve(self) -> list[dict[str, float]]:
        """Points as {coverage, <metric>, <metric>_ci_lo, <metric>_ci_hi}."""
        out = []
        for p in self.sorted_points():
            d: dict[str, Any] = {"coverage": p.coverage}
            for m in ("collision_rate", "intervention_rate", "failure_rate"):
                d[m] = getattr(p, m)
                ci = getattr(p, f"{m[:-5]}_ci", None)
                if ci is not None:
                    d[f"{m}_ci_lo"], d[f"{m}_ci_hi"] = ci
            if p.pdms_mean is not None:
                d["pdms_mean"] = p.pdms_mean
            out.append(d)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "method": self.method,
            "monotonic": self.monotonic,
            "cliff_tier": self.cliff_tier,
            "cliff_coverage": self.cliff_coverage,
            "cliff_separation": self.cliff_separation,
            "cliff_coverage_ci": list(self.cliff_coverage_ci) if self.cliff_coverage_ci else None,
            "cliff_separation_ci": list(self.cliff_separation_ci) if self.cliff_separation_ci else None,
            "n_bootstrap": self.n_bootstrap,
            "curve": self.curve(),
        }


def _safe_rate(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (k successes / n).

    Returns (lower, upper) 95% confidence bounds. Uses the Wilson interval so
    it behaves well at p=0 and p=1 (no degenerate zero-width interval).
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * sqrt(max(0.0, (p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _bootstrap_separation(
    rows_by_tier: dict[str, list[float]],
    coverages: list[float],
    metric_values: dict[str, float],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Bootstrap the cliff coverage and separation over per-scenario resamples.

    For each bootstrap draw we resample each tier's binary outcomes (with
    replacement), re-aggregate rates, run max-two-sample separation, and record
    the chosen (coverage, separation). Returns (coverages_boot, seps_boot).
    """
    rng = np.random.default_rng(seed)
    tiers = list(rows_by_tier)
    # tiers sorted by coverage descending (matching detect_cliff ordering)
    tiers_sorted = sorted(tiers, key=lambda t: coverages[t], reverse=True)
    coverages_sorted = [coverages[t] for t in tiers_sorted]
    n_per_tier = [len(rows_by_tier[t]) for t in tiers_sorted]

    boot_coverages: list[float] = []
    boot_seps: list[float] = []
    for _ in range(n_boot):
        boot_rates: dict[str, float] = {}
        for t, n in zip(tiers_sorted, n_per_tier):
            vals = rows_by_tier[t]
            if n == 0:
                boot_rates[t] = metric_values[t]
                continue
            resampled = rng.choice(vals, size=n, replace=True)
            boot_rates[t] = float(np.mean(resampled))
        values = [boot_rates[t] for t in tiers_sorted]
        best_i, best_sep = -1, -1.0
        for i in range(1, len(values)):
            left = values[:i]
            right = values[i:]
            mu_l, mu_r = float(np.mean(left)), float(np.mean(right))
            std_pooled = float(np.sqrt((np.var(left) + np.var(right)) / 2.0)) or 1e-9
            sep = abs(mu_r - mu_l) / std_pooled
            if sep > best_sep:
                best_sep = sep
                best_i = i
        if best_i > 0:
            boot_coverages.append(coverages_sorted[best_i - 1])
            boot_seps.append(best_sep)
    return boot_coverages, boot_seps


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def aggregate_tier(scenarios: list[dict[str, Any]], tier: str) -> FrontierPoint:
    """Aggregate per-scenario dicts for a single tier into a FrontierPoint."""
    rows = [s for s in scenarios if s.get("tier") == tier]
    coverage = float(rows[0].get("coverage", 1.0)) if rows else 0.0
    collision = [float(r.get("collision", 0)) for r in rows]
    intervention = [float(r.get("intervention", 0)) for r in rows]
    failed = [float(r.get("failed", 0)) for r in rows]
    pdms = [float(r["pdm_score"]) for r in rows if r.get("pdm_score") is not None]
    return FrontierPoint(
        tier=tier,
        coverage=coverage,
        n_scenarios=len(rows),
        collision_rate=_safe_rate(collision),
        intervention_rate=_safe_rate(intervention),
        failure_rate=_safe_rate(failed),
        pdms_mean=float(np.mean(pdms)) if pdms else None,
    ).with_cis()


def _is_monotonic_nondecreasing(vals: list[float]) -> bool:
    """Safety degradation is non-increasing safety as coverage decreases.

    We require the *risk* metric to be non-decreasing as coverage drops
    (frontier sorted by coverage descending).
    """
    from itertools import pairwise

    return all(b >= a - 1e-9 for a, b in pairwise(vals))


def detect_cliff(points: list[FrontierPoint], metric: str = "collision_rate") -> dict[str, Any]:
    """Find the coverage threshold maximizing two-sample separation of *metric*.

    Returns a dict with the cliff location, separation score, and monotonicity.
    """
    ordered = sorted(points, key=lambda p: p.coverage, reverse=True)
    if len(ordered) < 3:
        return {"cliff_tier": None, "cliff_coverage": None, "cliff_separation": 0.0,
                "monotonic": False}
    coverages = [p.coverage for p in ordered]
    values = [float(getattr(p, metric)) for p in ordered]

    best: tuple[int, float] = (-1, -1.0)
    for i in range(1, len(ordered)):
        left = values[:i]
        right = values[i:]
        # two-sample separation normalized by pooled std (t-like statistic)
        mu_l, mu_r = float(np.mean(left)), float(np.mean(right))
        std_pooled = float(np.sqrt((np.var(left) + np.var(right)) / 2.0)) or 1e-9
        sep = abs(mu_r - mu_l) / std_pooled
        if sep > best[1]:
            best = (i, sep)

    i, sep = best
    return {
        "cliff_tier": ordered[i - 1].tier if i > 0 else None,
        "cliff_coverage": coverages[i - 1] if i > 0 else None,
        "cliff_separation": round(sep, 3),
        "monotonic": bool(_is_monotonic_nondecreasing(values)),
    }


def build_frontier(
    scenarios: list[dict[str, Any]],
    metric: str = "collision_rate",
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
) -> SafetyFrontier:
    """Aggregate per-scenario metrics into a SafetyFrontier with cliff detection.

    Each tier's binary outcomes are bootstrapped (``n_bootstrap`` draws) to give
    a 95% CI on the cliff location and separation.
    """
    tiers: list[str] = []
    for s in scenarios:
        t = s.get("tier")
        if t is not None and t not in tiers:
            tiers.append(t)
    points = [aggregate_tier(scenarios, t) for t in tiers]
    cliff = detect_cliff(points, metric)

    # Bootstrap CI around the cliff estimate (per-scenario outcome resampling).
    metric_field = metric  # "collision_rate" -> rows carry "collision"
    row_key = {
        "collision_rate": "collision",
        "intervention_rate": "intervention",
        "failure_rate": "failed",
    }.get(metric, "collision")
    rows_by_tier: dict[str, list[float]] = {
        t: [float(r.get(row_key, 0)) for r in scenarios if r.get("tier") == t]
        for t in tiers
    }
    coverages = {p.tier: p.coverage for p in points}
    metric_values = {p.tier: float(getattr(p, metric_field)) for p in points}

    cliff_coverage_ci: tuple[float, float] | None = None
    cliff_separation_ci: tuple[float, float] | None = None
    n_used = 0
    if len(tiers) >= 3:
        boot_coverages, boot_seps = _bootstrap_separation(
            rows_by_tier, coverages, metric_values, n_boot=n_bootstrap, seed=bootstrap_seed
        )
        n_used = len(boot_coverages)
        if boot_coverages:
            cliff_coverage_ci = (
                round(_percentile(boot_coverages, 2.5), 3),
                round(_percentile(boot_coverages, 97.5), 3),
            )
            cliff_separation_ci = (
                round(_percentile(boot_seps, 2.5), 3),
                round(_percentile(boot_seps, 97.5), 3),
            )

    return SafetyFrontier(
        points=points,
        metric=metric,
        cliff_tier=cliff["cliff_tier"],
        cliff_coverage=cliff["cliff_coverage"],
        cliff_separation=cliff["cliff_separation"],
        monotonic=cliff["monotonic"],
        cliff_coverage_ci=cliff_coverage_ci,
        cliff_separation_ci=cliff_separation_ci,
        n_bootstrap=n_used,
    )


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """Load per-scenario metrics JSON produced by the nuPlan sweep."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("scenarios", [])
    return [dict(r) for r in rows]


def gate_rule_from_frontier(
    frontier: SafetyFrontier,
    max_rate: float = 0.05,
    safety_metric: str = "collision_rate",
) -> dict[str, Any]:
    """Derive a deployment gate: coverage threshold that keeps risk <= max_rate.

    Returns a dict describing the rule, e.g.:
        {"coverage_min": 0.7, "collision_rate_at_rule": 0.03,
         "evidence": "..."}
    """
    ordered = frontier.sorted_points()
    ok = [p for p in ordered if getattr(p, safety_metric) <= max_rate]
    if not ok:
        return {"coverage_min": None, "found": False}
    best = ok[0]  # highest coverage among those meeting the bound
    return {
        "coverage_min": round(best.coverage, 3),
        "found": True,
        "tier": best.tier,
        "collision_rate_at_rule": round(getattr(best, safety_metric), 4),
        "n_scenarios": best.n_scenarios,
    }


def _log_factorial(k: int) -> float:
    """Natural log of k! (k >= 0), exact for k < 2**26."""
    if k < 2:
        return 0.0
    # log-gamma is exact enough and avoids overflow for the sizes we use.
    return float(np.log(np.arange(2, k + 1, dtype=np.float64)).sum())


def fisher_exact_p(n1: int, k1: int, n2: int, k2: int) -> float:
    """Two-tailed Fisher exact test p-value for two binomial proportions.

    Builds the exact hypergeometric distribution for the 2x2 table
    [[k1, n1-k1], [k2, n2-k2]] and sums the probability of all tables at
    least as extreme as the observed one (two-sided). Pure Python + numpy,
    so it runs in CI without scipy.
    """
    a, b, c, d = k1, n1 - k1, k2, n2 - k2
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    if min(r1, r2, c1, c2) < 0:
        return 1.0
    # Hypergeometric probability of a table with top-left = x:
    # C(r1,x) C(r2,c1-x) / C(n,c1)
    def _log_hyper(x: int) -> float:
        if x < 0 or x > min(r1, c1) or c1 - x < 0 or c1 - x > r2:
            return -float("inf")
        return _log_factorial(r1) + _log_factorial(r2) + _log_factorial(c1) + _log_factorial(n - c1) - (
            _log_factorial(x) + _log_factorial(r1 - x) + _log_factorial(c1 - x) + _log_factorial(r2 - (c1 - x)) + _log_factorial(n)
        )

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    probs = {x: _log_hyper(x) for x in range(lo, hi + 1)}
    obs = probs.get(a, -float("inf"))
    # Two-sided: sum all tables with probability <= observed probability.
    return float(sum(np.exp(v) for v in probs.values() if v <= obs + 1e-12))


def two_proportion_test(n1: int, k1: int, n2: int, k2: int) -> dict[str, float]:
    """Two-proportion significance tests between two binomial rates.

    Returns Fisher exact two-sided p-value and a normal-approximation
    z-test p-value for H0: p1 == p2. Both are computed in pure Python/numpy.
    """
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    # pooled two-proportion z-test
    pooled = (k1 + k2) / (n1 + n2) if n1 + n2 else 0.0
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)) if n1 and n2 else 0.0
    z = (p1 - p2) / se if se > 0 else 0.0
    # two-sided p from the standard normal tail
    from math import erf, sqrt

    pz = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return {
        "p_fisher": float(fisher_exact_p(n1, k1, n2, k2)),
        "p_ztest": float(pz),
        "z": float(z),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Given raw (two-sided) p-values, returns the q-values that control the
    false-discovery rate at level alpha (report q <= 0.05 as FDR-significant).
    Pure Python + numpy, CI-safe.

    For m tests with sorted p-values p_(1) <= ... <= p_(m), the largest k with
    p_(k) <= (k/m) * alpha is the rejection set; the q-value of the i-th
    smallest p is min over j>=i of (m/j) * p_(j), capped at 1.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)  # ascending
    ranked = np.asarray(p_values, dtype=np.float64)[order]
    # naive BH q-values: q_(i) = (m / (i+1)) * p_(i), then enforce monotonicity
    # by taking the running minimum from the largest p down.
    q = np.clip((m / np.arange(1, m + 1)) * ranked, 0.0, 1.0)
    for i in range(m - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    out = np.empty(m, dtype=np.float64)
    out[order] = q
    return [float(x) for x in out]
