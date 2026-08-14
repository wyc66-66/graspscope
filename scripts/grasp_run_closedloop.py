#!/usr/bin/env python3
"""Closed-loop grasp sweep: coverage alpha -> grasp reliability frontier.

Reads the per-alpha perception profiles (produced by grasp_audit_perception.py)
and the synthetic corpus manifest, runs the closed-loop grasp simulator over
every scene, and aggregates into a safety frontier (reliability vs alpha) with:

- Wilson 95% CIs on success/failure rates;
- max-two-sample-separation cliff detection (which alpha does reliability
  collapse at?);
- Fisher-exact + BH-FDR significance between adjacent tiers;
- a deployment gate: the minimum coverage (vocabulary coverage) required to
  keep grasp failure <= ``--max-fail-rate``.

Also computes the same frontier for the *real* scene pack (coverage=1.0) as an
honest real-world anchor.

Outputs (data/grasp_closedloop/):
    scenarios.json          per-scenario grasps (for the replay UI)
    frontier.json           SafetyFrontier (cliff + CIs)
    gate.json               deployment gate rule
    summary.md              human-readable summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE / "src"))

from graspscope.closedloop.error_profile import PerceptionErrorProfile
from graspscope.closedloop.frontier import (
    benjamini_hochberg,
    build_frontier,
    two_proportion_test,
    wilson_ci,
)
from graspscope.grasp.grasp_env import GraspEnv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", default="data/grasp_gate/profiles.json")
    ap.add_argument("--manifest", default="data/grasp_synth/synthetic_corpus.json")
    ap.add_argument("--images", default="data/grasp_synth/images")
    ap.add_argument("--out", default="data/grasp_closedloop")
    ap.add_argument("--exec-success", type=float, default=0.92)
    ap.add_argument("--phantom-rate", type=float, default=0.35)
    ap.add_argument("--max-fail-rate", type=float, default=0.15)
    ap.add_argument("--n-scenes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of RNG seeds to average over (smooths stochasticity)")
    args = ap.parse_args()

    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    profiles_by_alpha: dict[str, PerceptionErrorProfile] = {}
    profiles_dir = Path(args.profiles).parent
    for a in alphas:
        key = f"alpha_{a:.1f}"
        p = profiles.get(key)
        if p is None:
            print(f"WARN: no profile for {key}; skipping")
            continue
        # profiles live alongside --profiles (one dir per tier), not hardcoded
        prof_path = profiles_dir / key / "profile.json"
        profiles_by_alpha[key] = PerceptionErrorProfile.load(prof_path)

    scenarios = GraspEnv.build_sweep_scenarios(
        manifest, args.images, alphas=alphas, n_scenes=args.n_scenes, seed=args.seed
    )
    print(f"[closedloop] {len(scenarios)} scenarios across {len(alphas)} tiers x {args.seeds} seeds")

    # ---- run closed loop over seeds (scene set is fixed; perception RNG varies) ----
    # Start from the alpha=1.0 profile of the *same* sweep so the initial env
    # state matches the profiles being swept (not the main experiment's).
    env = GraspEnv(
        PerceptionErrorProfile.load(
            profiles_dir / "alpha_1.0" / "profile.json"
        ),
        exec_success=args.exec_success,
        phantom_rate=args.phantom_rate,
        seed=args.seed,
    )
    outcomes = []
    by_tier: dict[str, list] = {}
    for s in range(args.seeds):
        env.seed = args.seed + s
        env._rng = __import__("random").Random(args.seed + s)
        for sc in scenarios:
            env.profile = profiles_by_alpha[sc.tier]
            oc = env.run_episode(sc)
            oc.meta["seed"] = s  # type: ignore[attr-defined]
            outcomes.append(oc)
            by_tier.setdefault(sc.tier, []).append(oc)

    # ---- aggregate ----
    per_scenario: list[dict] = []
    for oc in outcomes:
        per_scenario.append(oc.to_dict())
    with (out / "scenarios.json").open("w", encoding="utf-8") as f:
        json.dump(per_scenario, f, indent=2)

    # build rows for frontier: each scenario -> success / failure typed
    rows = []
    for oc in outcomes:
        rows.append(
            {
                "tier": oc.tier,
                "coverage": oc.coverage,
                "success": 1 if oc.success else 0,
                "failed": 0 if oc.success else 1,
                "collision": 0,  # grasp domain: no collision channel
                "intervention": 0,
                "failure_type": oc.failure_type,
            }
        )

    # ---- frontier on failure rate (or success rate) ----
    metric = "failure_rate"
    frontier = build_frontier(rows, metric=metric, n_bootstrap=1000, bootstrap_seed=args.seed)

    # failure decomposition by tier
    from collections import Counter

    decomp: dict[str, dict[str, float]] = {}
    for tier, ocs in by_tier.items():
        cnt = Counter(oc.failure_type for oc in ocs)
        n = len(ocs)
        decomp[tier] = {
            k: round(v / n, 4) for k, v in sorted(cnt.items())
        }

    # ---- significance: pairwise adjacent tiers (Fisher + BH) ----
    ordered_tiers = [f"alpha_{a:.1f}" for a in alphas]
    adj_pairs: list[dict] = []
    raw_p: list[float] = []
    for i in range(len(ordered_tiers) - 1):
        t1, t2 = ordered_tiers[i], ordered_tiers[i + 1]
        r1 = [r for r in rows if r["tier"] == t1]
        r2 = [r for r in rows if r["tier"] == t2]
        k1 = sum(r["failed"] for r in r1)
        k2 = sum(r["failed"] for r in r2)
        if not r1 or not r2:
            continue
        test = two_proportion_test(len(r1), k1, len(r2), k2)
        raw_p.append(test["p_fisher"])
        adj_pairs.append(
            {
                "from": t1,
                "to": t2,
                "fail_rate_from": round(k1 / len(r1), 4),
                "fail_rate_to": round(k2 / len(r2), 4),
                "p_fisher": round(test["p_fisher"], 5),
                "p_ztest": round(test["p_ztest"], 5),
            }
        )
    qvals = benjamini_hochberg(raw_p)
    for pair, q in zip(adj_pairs, qvals):
        pair["q_fdr"] = round(q, 5)

    # ---- deployment gate: interpolated coverage for failure-rate bound ----
    # Fit a monotone piecewise-linear curve through (coverage, failure_rate)
    # and find the smallest coverage where failure_rate <= max_fail_rate.
    pts = sorted(frontier.sorted_points(), key=lambda p: p.coverage)  # asc
    covs = [p.coverage for p in pts]
    fails = [p.failure_rate for p in pts]
    target = args.max_fail_rate
    if fails[0] <= target:
        # even the worst coverage (alpha=0.2) meets the bound -> no restriction
        gate = {"coverage_min": None, "found": False,
                "reason": "all tiers meet failure-rate bound", "max_fail_rate": target}
    else:
        gate = {"coverage_min": None, "found": False, "max_fail_rate": target}
        for i in range(1, len(covs)):
            c0, c1 = covs[i - 1], covs[i]
            f0, f1 = fails[i - 1], fails[i]
            if f1 <= target <= f0:
                frac = (f0 - target) / max(1e-9, (f0 - f1))
                alpha_star = c0 + frac * (c1 - c0)
                gate = {
                    "coverage_min": round(alpha_star, 3),
                    "found": True,
                    "interp_from": round(f0, 3),
                    "interp_to": round(f1, 3),
                    "tier_below": pts[i - 1].tier,
                    "tier_above": pts[i].tier,
                    "max_fail_rate": target,
                }
                break
    gate["cliff_tier"] = frontier.cliff_tier

    # ---- real pack anchor ----
    # locate the real profile + coco pack relative to the sweep's own outputs:
    # the real profile lives next to the tier profiles (profiles_dir/real/),
    # and the real coco pack lives next to the tier profile dir.
    real_profile_path = profiles_dir / "real" / "profile.json"
    real_anchor: dict[str, float] | None = None
    if real_profile_path.is_file():
        real_prof = PerceptionErrorProfile.load(real_profile_path)
        env.profile = real_prof
        # real images aren't in the synth manifest; use real coco pack instead.
        # Prefer the sweep-specific real pack (data/grasp_real_v8) and fall back
        # to the main real pack.
        real_coco_candidates = [
            _WORKSPACE / "data" / "grasp_real_v8" / "coco" / "annotations.json",
            _WORKSPACE / "data" / "grasp_real" / "coco" / "annotations.json",
        ]
        real_coco_path = next((p for p in real_coco_candidates if p.is_file()), None)
        real_imgs = _WORKSPACE / "data" / "coco" / "val2017"
        rng_scenarios = []
        if real_coco_path is not None:
            real_coco = json.loads(real_coco_path.read_text(encoding="utf-8"))
            images = sorted(int(i["id"]) for i in real_coco["images"])[:50]
            for img_id in images:
                rng_scenarios.append(
                    GraspEnv.scene_from_coco(
                        real_coco,
                        real_imgs,
                        image_id=img_id,
                        coverage=1.0,
                        tier="real",
                        oov_classes=[],
                        rng=__import__("random").Random(args.seed + 1),
                    )
                )
            real_outcomes = env.run_corpus(rng_scenarios)
            n_ok = sum(1 for oc in real_outcomes if oc.success)
            n = len(real_outcomes)
            lo, hi = wilson_ci(n, n_ok)
            real_anchor = {
                "n": n,
                "success_rate": round(n_ok / n, 4),
                "success_ci": [round(lo, 4), round(hi, 4)],
                "failure_rate": round(1 - n_ok / n, 4),
            }
        print(f"[closedloop] real anchor: {real_anchor}")

    # ---- write artifacts ----
    # scene previews: representative images per tier from the corpus manifest.
    # scenarios.json holds grasp results but not image paths; the manifest does.
    previews: list[dict] = []
    cases: list[dict] = []
    try:
        cells: dict[str, list] = manifest.get("cells") or {}
        img_index: dict[str, str] = {}
        for alpha, paths in cells.items():
            cov = float(alpha.replace("alpha=", ""))
            for p in paths:
                img_index[Path(p).stem] = f"alpha_{cov:g}"
        for alpha, paths in cells.items():
            cov = float(alpha.replace("alpha=", ""))
            for p in paths[:2]:
                img = Path(p).name
                previews.append(
                    {
                        "scene_id": Path(p).stem,
                        "tier": f"alpha_{cov:g}",
                        "coverage": cov,
                        "image_url": "/api/graspscope/scene?f=" + img,
                        "success": True,
                    }
                )
        # representative grasp outcomes: 2 per tier (success + worst failure)
        scen = json.loads((out / "scenarios.json").read_text(encoding="utf-8"))
        by_tier: dict[str, list] = {}
        for sc in scen:
            by_tier.setdefault(sc.get("tier", ""), []).append(sc)
        for tier in ["alpha_0.2", "alpha_0.4", "alpha_0.6", "alpha_0.8", "alpha_1.0"]:
            recs = by_tier.get(tier) or []
            if not recs:
                continue
            ok = [r for r in recs if r.get("success")]
            fail = [r for r in recs if not r.get("success")]
            picks = ([ok[0]] if ok else []) + ([fail[0]] if fail else [])
            for r in picks:
                sid = r.get("scenario_id", "?")
                ftype = r.get("failure_type", "success")
                cases.append(
                    {
                        "scene_id": sid,
                        "tier": r.get("tier"),
                        "coverage": r.get("coverage"),
                        "success": r.get("success"),
                        "failure_type": ftype,
                        "target_cls": r.get("target_cls"),
                        "n_attempts": r.get("n_attempts"),
                        "image_url": "/api/graspscope/scene?f=" + sid + ".jpg",
                    }
                )
    except (OSError, ValueError, TypeError, KeyError):
        previews = []
        cases = []

    payload = {
        "metric": metric,
        "alphas": alphas,
        "frontier": frontier.to_dict(),
        "failure_decomposition": decomp,
        "adjacent_pairs": adj_pairs,
        "gate": gate,
        "real_anchor": real_anchor,
        "scenes": previews,
        "cases": cases,
        "params": {
            "exec_success": args.exec_success,
            "phantom_rate": args.phantom_rate,
            "max_fail_rate": args.max_fail_rate,
            "n_scenes": args.n_scenes,
            "seed": args.seed,
        },
    }
    payload["exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "frontier.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # summary
    lines = [
        "# GraspScope closed-loop summary",
        "",
        f"- metric: {metric}",
        f"- exec_success anchor: {args.exec_success}",
        f"- phantom_rate: {args.phantom_rate}",
        "",
        "| alpha | failure_rate | success_rate | CI |",
        "|---|---|---|---|",
    ]
    for p in frontier.sorted_points():
        r = p.failure_rate
        lo, hi = wilson_ci(p.n_scenarios, round(r * p.n_scenarios))
        lines.append(
            f"| {p.tier} ({p.coverage}) | {r:.3f} | {1-r:.3f} | "
            f"({lo:.3f}, {hi:.3f}) |"
        )
    lines += [
        "",
        f"cliff_tier: {frontier.cliff_tier}  coverage={frontier.cliff_coverage}  "
        f"separation={frontier.cliff_separation}",
        f"gate (fail<={args.max_fail_rate}): {gate}",
        f"real anchor: {real_anchor}",
        "",
        "adjacent-tier significance:",
    ]
    for pair in adj_pairs:
        lines.append(
            f"- {pair['from']} -> {pair['to']}: fail {pair['fail_rate_from']} -> "
            f"{pair['fail_rate_to']}, p_fisher={pair['p_fisher']}, q_fdr={pair['q_fdr']}"
        )
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[closedloop] artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
