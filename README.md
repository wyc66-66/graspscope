# OpenVocab-GraspGate

Closed-loop deployability audit for **open-vocabulary robotic grasping**:
turn the vocabulary decision — *which object names do I ship to the robot?* —
from intuition into a number with a confidence interval.

Same methodology as the driving-domain gate (nuScenes + nuPlan): profile the
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

## Reproduce

```bash
pip install -e ".[ui,perception,paper]"   # + torch/ultralytics for the audit step

# 1. corpus (requires COCO val2017 annotations under data/coco/)
python scripts/grasp_build_corpus.py

# 2. perception audit (GPU) -> data/grasp_gate/
python scripts/grasp_audit_perception.py

# 3. closed-loop sweep -> data/grasp_closedloop/frontier.json
python scripts/grasp_run_closedloop.py

# 4. dashboard
python -m opengate ui   # http://127.0.0.1:8787/graspgate

# 5. paper facts + figures + PDF (docs/paper/graspgate/)
python scripts/grasp_paper_facts.py
python scripts/render_graspgate_paper_figures.py
python scripts/render_graspgate_paper.py
```

The committed `data/` already contains the generated corpus, perception
profiles, and closed-loop results, so steps 1–2 are only needed to regenerate.

## Paper

[docs/paper/graspgate/graspgate_paper.pdf](docs/paper/graspgate/graspgate_paper.pdf)
— full English technical report (arXiv-style) with all figures.

## Why a cross-domain gate

The identical pipeline — `PerceptionErrorProfile` → closed-loop sim → coverage
sweep → cliff + gate — runs on autonomous driving (nuPlan, cliff separation
5.6×) and on robotic grasping (GraspEnv, 5.5×). Same qualitative law, same
statistical machinery (Wilson CI, Fisher-exact, BH-FDR). That is the point:
deployability auditing generalizes across embodied domains.
