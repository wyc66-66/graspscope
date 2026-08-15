# GraspScope

Closed-loop deployability audit for **open-vocabulary robotic grasping**:
turn the vocabulary decision — *which object names do I ship to the robot?* —
from intuition into a number with a confidence interval.

Same methodology as the grasp-deployability audit: profile the
perception stack → inject measured failures into a closed-loop simulator → sweep
vocabulary coverage α → find the safety cliff → derive a deployment gate.

## What was measured

**Corpus.** 750 synthetic shelf scenes composited from real COCO product crops
(960×540, 5 coverage tiers × 150, α ∈ {0.2, 0.4, 0.6, 0.8, 1.0}) plus 149 real
COCO scenes. Deployment vocabulary V: 12 classes
(bottle / cup / bowl / book / banana / apple / orange / carrot / cake / donut /
sports ball / vase). Perception: YOLO-World `yolov8s`, imgsz 640, conf 0.15.

**Closed-loop grasp sweep** (exec success anchored at 0.95, phantom 0.35,
5-seed aggregation):

| α | grasp failure | 95% Wilson CI |
|---|---|---|
| 1.0 | 9.9% | [7.9, 12.2] |
| 0.8 | 12.5% | [10.4, 15.1] |
| 0.6 | 29.5% | [26.3, 32.8] |
| 0.4 | 30.4% | [27.2, 33.8] |
| 0.2 | **57.2%** | [53.6, 60.7] |

Three findings:

1. **Safety cliff at α = 0.4** (separation 5.5×, Fisher p < 10⁻⁵). Perception
   recall itself is flat across α — the cliff is a *system* effect of
   vocabulary coverage on grasp reliability, not the detector changing.
2. **Deployment gate: α ≥ 0.653** for a 25% failure bound (linear
   interpolation of the frontier). Below that, ship a bigger vocabulary.
3. **Failure composition shifts.** Low α: *empty grasp* dominates (target never
   seen). High α: *wrong-object grasp* dominates (detector confuses classes).
   Two different mitigations.
4. **Real anchor: 72% success on real COCO scenes**, well below synthetic
   α=1.0 (90%). Vocabulary coverage is necessary but not sufficient — detector
   quality matters on real data.
5. **Sensitivity.** The law's form survives both sweeps:
   - *Detector scale* (s → l): gate relaxes **0.653 → 0.484** (better
     perception needs less vocabulary coverage; cliff stays at α=0.4).
   - *Vocabulary size* (12 → 8): gate **0.653 → 0.545** (dropping the four
     confusable classes reduces label confusion). Vocabulary *composition*
     matters as much as vocabulary count.

## Reproduce

```bash
pip install -e ".[ui,perception,paper]"   # + torch/ultralytics for the audit step

# 1. corpus (requires COCO val2017 annotations under data/coco/)
python scripts/grasp_build_corpus.py

# 2. perception audit (GPU) -> data/grasp_gate/
python scripts/grasp_audit_perception.py

# 3. closed-loop sweep -> data/grasp_closedloop/frontier.json
python scripts/grasp_run_closedloop.py

# 4. sensitivity sweeps -> data/grasp_detector_scale/, data/grasp_vocab_size/
python scripts/grasp_sensitivity_sweep.py --gpu-model-dir <dir-with-weights>

# 5. dashboard
python -m graspscope ui   # http://127.0.0.1:8787/graspscope

# 6. paper facts + figures + PDF (docs/paper/graspscope/)
python scripts/grasp_paper_facts.py
python scripts/render_graspscope_paper_figures.py
python scripts/render_graspscope_paper.py
```

The committed `data/` already contains the generated corpus, perception
profiles, closed-loop results, and both sensitivity sweeps, so steps 1–4 are
only needed to regenerate.

## Paper

[docs/paper/graspscope/graspscope_paper.pdf](docs/paper/graspscope/graspscope_paper.pdf)
— full English technical report (arXiv-style) with all figures.

## Why a deployability audit

Deploying an open-vocabulary grasp stack is a *vocabulary* decision: which
object names to ship. The identical pipeline — `PerceptionErrorProfile` →
closed-loop simulator → coverage sweep → cliff + gate — turns that decision
into a number with a confidence interval. The statistical machinery (Wilson CI,
Fisher-exact, BH-FDR) is deliberately small and transparent, so the audit is a
procedure any deployer can rerun on their own scenes and vocabulary.
---

## Live report

The technical report, figures and every number are served at **[https://wyc66-66.github.io/graspscope/](https://wyc66-66.github.io/graspscope/)**.
