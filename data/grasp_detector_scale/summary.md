# GraspScope closed-loop summary

- metric: failure_rate
- exec_success anchor: 0.92
- phantom_rate: 0.35

| alpha | failure_rate | success_rate | CI |
|---|---|---|---|
| alpha_1.0 (1.0) | 0.079 | 0.921 | (0.061, 0.100) |
| alpha_0.8 (0.8) | 0.081 | 0.919 | (0.064, 0.103) |
| alpha_0.6 (0.6) | 0.211 | 0.789 | (0.183, 0.241) |
| alpha_0.4 (0.4) | 0.279 | 0.721 | (0.248, 0.312) |
| alpha_0.2 (0.2) | 0.505 | 0.495 | (0.470, 0.541) |

cliff_tier: alpha_0.4  coverage=0.4  separation=5.655
gate (fail<=0.25): {'coverage_min': 0.484, 'found': True, 'interp_from': 0.279, 'interp_to': 0.211, 'tier_below': 'alpha_0.4', 'tier_above': 'alpha_0.6', 'max_fail_rate': 0.25, 'cliff_tier': 'alpha_0.4'}
real anchor: {'n': 50, 'success_rate': 0.66, 'success_ci': [0.5215, 0.7756], 'failure_rate': 0.34}

adjacent-tier significance:
- alpha_0.2 -> alpha_0.4: fail 0.5053 -> 0.2787, p_fisher=0.0, q_fdr=0.0
- alpha_0.4 -> alpha_0.6: fail 0.2787 -> 0.2107, p_fisher=0.00264, q_fdr=0.00353
- alpha_0.6 -> alpha_0.8: fail 0.2107 -> 0.0813, p_fisher=0.0, q_fdr=0.0
- alpha_0.8 -> alpha_1.0: fail 0.0813 -> 0.0787, p_fisher=0.92422, q_fdr=0.92422