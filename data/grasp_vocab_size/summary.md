# GraspScope closed-loop summary

- metric: failure_rate
- exec_success anchor: 0.92
- phantom_rate: 0.35

| alpha | failure_rate | success_rate | CI |
|---|---|---|---|
| alpha_1.0 (1.0) | 0.071 | 0.929 | (0.054, 0.091) |
| alpha_0.8 (0.8) | 0.123 | 0.877 | (0.101, 0.148) |
| alpha_0.6 (0.6) | 0.223 | 0.777 | (0.194, 0.254) |
| alpha_0.4 (0.4) | 0.321 | 0.679 | (0.289, 0.356) |
| alpha_0.2 (0.2) | 0.545 | 0.455 | (0.510, 0.581) |

cliff_tier: alpha_0.4  coverage=0.4  separation=5.311
gate (fail<=0.25): {'coverage_min': 0.545, 'found': True, 'interp_from': 0.321, 'interp_to': 0.223, 'tier_below': 'alpha_0.4', 'tier_above': 'alpha_0.6', 'max_fail_rate': 0.25, 'cliff_tier': 'alpha_0.4'}
real anchor: {'n': 50, 'success_rate': 0.56, 'success_ci': [0.4231, 0.6884], 'failure_rate': 0.44}

adjacent-tier significance:
- alpha_0.2 -> alpha_0.4: fail 0.5453 -> 0.3213, p_fisher=0.0, q_fdr=0.0
- alpha_0.4 -> alpha_0.6: fail 0.3213 -> 0.2227, p_fisher=2e-05, q_fdr=3e-05
- alpha_0.6 -> alpha_0.8: fail 0.2227 -> 0.1227, p_fisher=0.0, q_fdr=0.0
- alpha_0.8 -> alpha_1.0: fail 0.1227 -> 0.0707, p_fisher=0.00085, q_fdr=0.00085