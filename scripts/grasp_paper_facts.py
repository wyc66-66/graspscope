#!/usr/bin/env python3
"""Collect every number that the GraspScope paper cites.

Reads only real artifacts:
- data/grasp_closedloop/frontier.json  grasp-domain closed-loop results
- data/grasp_gate/profiles.json         grasp perception audit profiles
- data/grasp_synth/synthetic_corpus.json  corpus stats
- data/grasp_real/coco/annotations.json real pack stats

Prints a stable facts table used by the paper and docs. Never fabricates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = Path(_WORKSPACE)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def main() -> int:
    facts: list[tuple[str, str]] = []

    # ---- grasp closed-loop frontier ----
    gr = ROOT / "data" / "grasp_closedloop" / "frontier.json"
    if gr.is_file():
        d = json.loads(gr.read_text(encoding="utf-8"))
        facts.append(("grasp|cliff_tier", d["frontier"]["cliff_tier"]))
        facts.append(("grasp|cliff_coverage", f"{d['frontier']['cliff_coverage']:g}"))
        facts.append(("grasp|cliff_separation", f"{d['frontier']['cliff_separation']:.1f}x"))
        for p in sorted(d["frontier"]["curve"], key=lambda x: x["coverage"]):
            facts.append((f"grasp|failure_rate@{p['coverage']:g}", pct(p["failure_rate"], 1)))
            facts.append(
                (f"grasp|failure_ci@{p['coverage']:g}",
                 f"[{pct(p['failure_rate_ci_lo'], 1)}, {pct(p['failure_rate_ci_hi'], 1)}]")
            )
        g = d.get("gate") or {}
        facts.append(("grasp|gate_max_fail", pct(g.get("max_fail_rate", 0), 0)))
        facts.append(("grasp|gate_coverage_min", f"{g.get('coverage_min', 'NA'):g}"))
        ra = d.get("real_anchor") or {}
        facts.append(("grasp|real_success", pct(ra.get("success_rate", 0), 0)))
        facts.append(("grasp|real_n", str(ra.get("n", "?"))))
        dec = d.get("failure_decomposition") or {}
        for tier, comp in sorted(dec.items()):
            for k in ("empty_grasp", "wrong_object", "drop"):
                facts.append((f"grasp|{tier}|{k}", pct(comp.get(k, 0), 1)))
    else:
        facts.append(("grasp", "MISSING frontier.json"))

    # ---- corpus stats ----
    corpus = ROOT / "data" / "grasp_synth" / "synthetic_corpus.json"
    if corpus.is_file():
        c = json.loads(corpus.read_text(encoding="utf-8"))
        n = sum(len(v) for v in c.get("cells", {}).values())
        facts.append(("corpus|synth_scenes", str(n)))
        facts.append(("corpus|alphas", ",".join(str(a) for a in c.get("alpha_grid", []))))
    real_ann = ROOT / "data" / "grasp_real" / "coco" / "annotations.json"
    if real_ann.is_file():
        r = json.loads(real_ann.read_text(encoding="utf-8"))
        facts.append(("corpus|real_images", str(len(r.get("images", [])))))
        facts.append(("corpus|real_annotations", str(len(r.get("annotations", [])))))

    # ---- perception audit ----
    profs = ROOT / "data" / "grasp_gate" / "profiles.json"
    if profs.is_file():
        p = json.loads(profs.read_text(encoding="utf-8"))
        for tier, prof in p.items():
            if isinstance(prof, dict) and "classes" in prof:
                cls = prof["classes"]
                recs = [c.get("recall", 0) for c in cls.values() if isinstance(c, dict)]
                locs = [c.get("loc_recall") for c in cls.values() if isinstance(c, dict)]
                if recs:
                    facts.append((f"percept|{tier}|strict_recall_mean",
                                  pct(sum(recs) / len(recs), 1)))
                if locs and any(x is not None for x in locs):
                    locs2 = [x for x in locs if x is not None]
                    facts.append((f"percept|{tier}|loc_recall_mean",
                                  pct(sum(locs2) / len(locs2), 1)))

    # ---- sensitivity sweeps ----
    det = ROOT / "data" / "grasp_detector_scale" / "frontier.json"
    if det.is_file():
        d = json.loads(det.read_text(encoding="utf-8"))
        facts.append(("sens|detector_l|cliff_coverage",
                      f"{d['frontier']['cliff_coverage']:g}"))
        facts.append(("sens|detector_l|gate_coverage_min",
                      f"{d['gate']['coverage_min']:g}"))
        facts.append(("sens|detector_l|failure_rate@1.0",
                      pct(next(p["failure_rate"] for p in d["frontier"]["curve"]
                               if p["coverage"] == 1.0), 1)))
        facts.append(("sens|detector_l|failure_rate@0.2",
                      pct(next(p["failure_rate"] for p in d["frontier"]["curve"]
                               if p["coverage"] == 0.2), 1)))
    v8 = ROOT / "data" / "grasp_vocab_size" / "frontier.json"
    if v8.is_file():
        d = json.loads(v8.read_text(encoding="utf-8"))
        facts.append(("sens|vocab8|cliff_coverage",
                      f"{d['frontier']['cliff_coverage']:g}"))
        facts.append(("sens|vocab8|gate_coverage_min",
                      f"{d['gate']['coverage_min']:g}"))
        facts.append(("sens|vocab8|failure_rate@0.2",
                      pct(next(p["failure_rate"] for p in d["frontier"]["curve"]
                               if p["coverage"] == 0.2), 1)))

    print("# GraspScope paper facts (auto-collected)\n")
    for k, v in facts:
        print(f"{k:<45} {v}")
    print("\n# end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
