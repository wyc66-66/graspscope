"""Closed-loop reliability study: perception error models, profiles, and frontier analysis.

This subpackage is the GraspScope research core. It answers:

    H1: closed-loop reliability (grasp failure / empty grasp / wrong-object rate)
        degrades monotonically with vocabulary coverage and has a sharp
        threshold ("safety cliff").
    H2: per-class localization recall and label-confusion measured on the target
        scenes can predict the cliff location, i.e. they are effective
        pre-deployment gates for grasp reliability.

Every tier rate carries a Wilson 95% CI and the cliff carries a bootstrap 95% CI
so the frontier supports statistically honest comparison across coverage tiers.

The modules here are intentionally pure Python + numpy (no GPU dependency) so CI
can unit-test them on synthetic fixtures; the perception audit that produces the
profiles lives in ``scripts/grasp_audit_perception.py``.
"""
