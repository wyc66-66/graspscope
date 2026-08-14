"""Closed-loop safety study: perception error models, profiles, and frontier analysis.

This subpackage is the OpenGate 0.9.1 research core. It answers:

    H1: closed-loop safety (collision/intervention/failure rate) degrades
        monotonically with perception vocabulary coverage and has a sharp
        threshold ("safety cliff").
    H2: OpenGate's EpisodicAP / OOV-FP measured on real driving frames can
        predict the cliff location, i.e. it is an effective pre-deployment
        safety gate for planners.

Every tier rate carries a Wilson 95% CI and the cliff carries a bootstrap 95% CI
so the frontier supports statistically honest comparison across coverage tiers.

The modules here are intentionally pure Python + numpy (no nuPlan / GPU
dependency) so CI can unit-test them on synthetic fixtures; the nuPlan
closed-loop engine under ``wsl2/nuplan_closedloop/`` only adapts
:mod:`perception_model` into the nuPlan ``BoxPerception`` interface.
"""
