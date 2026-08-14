#!/usr/bin/env python3
"""Render figures for the OpenVocab-GraspGate technical report.

Figures (written under docs/paper/graspgate/figures/):

- fig1_pipeline.png         method pipeline diagram
- fig2_alpha_cliff.png      main coverage arm: failure rate + Wilson CIs + gate/cliff
- fig3_failure_decomp.png   stacked failure composition per alpha
- fig4_cross_domain.png     driving vs grasping frontiers, side by side
- fig5_real_anchor.png      synthetic alpha=1.0 vs real anchor, with CIs
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opengate.closedloop.frontier import build_frontier, load_scenarios

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "graspgate" / "figures"

C_FAIL = "#d1495b"
C_OK = "#2a9d8f"
C_OOV = "#30638e"
C_CLIFF = "#b45309"
C_EMPTY = "#e9c46a"
C_WRONG = "#d1495b"
C_DROP = "#30638e"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
    }
)


def load_grasp() -> dict:
    return json.loads((ROOT / "data" / "grasp_closedloop" / "frontier.json").read_text(encoding="utf-8"))


def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.axis("off")
    boxes = [
        ("COCO crops", 0.02, "real product crops"),
        ("scene composer\n(α by construction)", 0.22, "750 synth + 149 real"),
        ("YOLO-World\nopen-vocab detector", 0.42, "profiles: recall / loc / confusion"),
        ("closed-loop\nGraspEnv", 0.62, "perceive→plan→execute"),
        ("frontier + gate", 0.82, "cliff + coverage_min"),
    ]
    for label, x, sub in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.14, 0.32, fc="#f2f2f2", ec="#444", lw=1))
        ax.text(x + 0.07, 0.55, label, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + 0.07, 0.43, sub, ha="center", va="center", fontsize=6.5, color="#555")
        if x < 0.82:
            ax.annotate("", xy=(x + 0.155, 0.51), xytext=(x + 0.147, 0.51),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#444"))
    ax.text(0.5, 0.06, "one audit pipeline, two domains (driving nuPlan · grasping GraspEnv)",
            ha="center", fontsize=8, color="#666")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def fig2_alpha_cliff() -> None:
    d = load_grasp()
    curve = sorted(d["frontier"]["curve"], key=lambda p: p["coverage"])
    covs = np.array([p["coverage"] for p in curve])
    fails = np.array([p["failure_rate"] for p in curve])
    lo = np.array([p["failure_rate_ci_lo"] for p in curve])
    hi = np.array([p["failure_rate_ci_hi"] for p in curve])
    gate = d.get("gate", {})
    cliff_cov = d["frontier"]["cliff_coverage"]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.fill_between(covs, lo, hi, color=C_OOV, alpha=0.15, label="95% Wilson CI")
    ax.plot(covs, fails, "-o", color=C_FAIL, lw=2, ms=5, label="grasp failure rate")
    if gate.get("coverage_min") is not None:
        ax.axvline(gate["coverage_min"], color="#1b7a5a", ls="--", lw=1.4)
        ax.text(gate["coverage_min"] + 0.012, 0.62, f"gate α≥{gate['coverage_min']:.2f}",
                color="#1b7a5a", fontsize=8, rotation=90)
    ax.axvline(cliff_cov, color=C_CLIFF, ls=":", lw=1.4)
    ax.text(cliff_cov + 0.012, 0.04, f"cliff α={cliff_cov:g}", color=C_CLIFF, fontsize=8, rotation=90)
    ax.set_xlabel("vocabulary coverage α")
    ax.set_ylabel("grasp failure rate")
    ax.set_ylim(0, 0.65)
    ax.set_xticks(covs)
    ax.set_xticklabels([f"{c:g}" for c in covs])
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_alpha_cliff.png", bbox_inches="tight")
    plt.close(fig)


def fig3_failure_decomp() -> None:
    d = load_grasp()
    dec = d["failure_decomposition"]
    tiers = [f"alpha_{a}" for a in (0.2, 0.4, 0.6, 0.8, 1.0)]
    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    empty = [dec[t]["empty_grasp"] for t in tiers]
    wrong = [dec[t]["wrong_object"] for t in tiers]
    drop = [dec[t]["drop"] for t in tiers]

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x = np.arange(len(tiers))
    ax.bar(x, empty, color=C_EMPTY, label="empty grasp")
    ax.bar(x, wrong, bottom=empty, color=C_WRONG, label="wrong object")
    ax.bar(x, drop, bottom=np.array(empty) + np.array(wrong), color=C_DROP, label="drop")
    for i, a in enumerate(alphas):
        total = empty[i] + wrong[i] + drop[i]
        ax.text(i, total + 0.015, f"{100*total:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a:g}" for a in alphas])
    ax.set_ylabel("share of attempts")
    ax.set_ylim(0, 0.68)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_failure_decomp.png", bbox_inches="tight")
    plt.close(fig)


def fig4_cross_domain() -> None:
    # driving
    drv = ROOT / "data" / "closedloop" / "metrics.json"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    if drv.is_file():
        scenarios = load_scenarios(drv)
        fr = build_frontier(scenarios, metric="collision_rate")
        pts = sorted(fr.sorted_points(), key=lambda p: p.coverage)
        ax = axes[0]
        ax.plot([p.coverage for p in pts], [p.collision_rate for p in pts], "-o",
                color=C_OOV, lw=2, ms=5)
        ax.set_title("driving (nuPlan)")
        ax.set_xlabel("coverage α")
        ax.set_ylabel("collision rate")
        ax.axvline(fr.cliff_coverage, color=C_CLIFF, ls=":", lw=1.3)
        ax.text(fr.cliff_coverage + 0.01, 0.03, f"cliff α={fr.cliff_coverage:g}",
                color=C_CLIFF, fontsize=8, rotation=90)
        ax.set_ylim(0, 0.8)
    # grasping
    d = load_grasp()
    curve = sorted(d["frontier"]["curve"], key=lambda p: p["coverage"])
    ax = axes[1]
    ax.plot([p["coverage"] for p in curve], [p["failure_rate"] for p in curve], "-o",
            color=C_FAIL, lw=2, ms=5)
    ax.set_title("grasping (GraspEnv)")
    ax.set_xlabel("coverage α")
    ax.set_ylabel("failure rate")
    ax.axvline(d["frontier"]["cliff_coverage"], color=C_CLIFF, ls=":", lw=1.3)
    ax.text(d["frontier"]["cliff_coverage"] + 0.01, 0.03,
            f"cliff α={d['frontier']['cliff_coverage']:g}", color=C_CLIFF, fontsize=8, rotation=90)
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_cross_domain.png", bbox_inches="tight")
    plt.close(fig)


def fig5_real_anchor() -> None:
    d = load_grasp()
    synth = next(p for p in d["frontier"]["curve"] if p["coverage"] == 1.0)
    real = d.get("real_anchor", {})
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    labels = ["synthetic α=1.0\n(n=150)", f"real COCO\n(n={real.get('n', '?')})"]
    su = [1 - synth["failure_rate"], real.get("success_rate", 0)]
    lo = [1 - synth["failure_rate_ci_hi"], real.get("success_ci", [0, 0])[0]]
    hi = [1 - synth["failure_rate_ci_lo"], real.get("success_ci", [0, 0])[1]]
    err_lo = np.array(su) - np.array(lo)
    err_hi = np.array(hi) - np.array(su)
    ax.bar(labels, su, yerr=[err_lo, err_hi], capsize=5, color=[C_OK, C_DROP], alpha=0.85)
    ax.set_ylabel("grasp success rate")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(su):
        ax.text(i, v + 0.04, f"{100*v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.5, 0.5, "coverage gap\ndoes not explain\nthe whole gap",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_real_anchor.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_pipeline()
    fig2_alpha_cliff()
    fig3_failure_decomp()
    fig4_cross_domain()
    fig5_real_anchor()
    print(f"[paper-figures] written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
