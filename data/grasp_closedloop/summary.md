# OpenVocab-GraspGate closed-loop summary

- metric: failure_rate
- exec_success anchor: 0.95
- phantom_rate: 0.35

| alpha | failure_rate | success_rate | CI |
|---|---|---|---|
| alpha_1.0 (1.0) | 0.099 | 0.901 | (0.079, 0.122) |
| alpha_0.8 (0.8) | 0.125 | 0.875 | (0.104, 0.151) |
| alpha_0.6 (0.6) | 0.295 | 0.705 | (0.263, 0.328) |
| alpha_0.4 (0.4) | 0.304 | 0.696 | (0.272, 0.338) |
| alpha_0.2 (0.2) | 0.572 | 0.428 | (0.536, 0.607) |

cliff_tier: alpha_0.4  coverage=0.4  separation=5.5
gate (fail<=0.25): {'coverage_min': 0.653, 'found': True, 'interp_from': 0.295, 'interp_to': 0.125, 'tier_below': 'alpha_0.6', 'tier_above': 'alpha_0.8', 'max_fail_rate': 0.25, 'cliff_tier': 'alpha_0.4'}
real anchor: {'n': 50, 'success_rate': 0.72, 'success_ci': [0.5833, 0.8253], 'failure_rate': 0.28}

adjacent-tier significance:
- alpha_0.2 -> alpha_0.4: fail 0.572 -> 0.304, p_fisher=0.0, q_fdr=0.0
- alpha_0.4 -> alpha_0.6: fail 0.304 -> 0.2947, p_fisher=0.73518, q_fdr=0.73518
- alpha_0.6 -> alpha_0.8: fail 0.2947 -> 0.1253, p_fisher=0.0, q_fdr=0.0
- alpha_0.8 -> alpha_1.0: fail 0.1253 -> 0.0987, p_fisher=0.1196, q_fdr=0.15947