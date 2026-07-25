# Reproducibility — arXiv:1803.03635

**Overall verdict: NOT_REPRODUCED**

This is *reproduction* (re-running the authors' own artifacts), not
replication. Tolerances were registered before the run.

## Claim / result matrix

| Claim | Metric | Claimed | Observed | Tolerance | Verdict |
|---|---|---|---|---|---|
| Winning tickets learn faster: at Pm=21.1 | early_stop_reduction_pct_pm21_1 | 38.0 | 46.67 | 15.0 (abs) | REPRODUCED |
| Test accuracy at early stop improves by  | test_acc_delta_pp_pm13_5 | 0.3 | -0.05 | 0.3 (abs) | NOT_REPRODUCED |
| By Pm=3.6% the accuracy gain is gone: te | test_acc_delta_pp_pm3_6 | 0.0 | -0.07 | 0.3 (abs) | REPRODUCED |
| The original initialisation matters: at  | speedup_ticket_vs_random_pm21_1 | 2.51 | 1.675 | 0.9 (abs) | REPRODUCED |
| At Pm=21% the winning ticket is about ha | acc_gap_ticket_minus_random_pp_pm21_1 | 0.5 | 0.36 | 0.4 (abs) | REPRODUCED |
| Winning tickets generalise better, not j | max_test_acc_delta_pp_final_iter | 0.35 | 0.23 | 0.3 (abs) | REPRODUCED |
| Structural check of the pruning schedule | pm_percent_pm21_1 | 21.1 | 21.072 | 0.05 (abs) | REPRODUCED |

## Environment & command

- Image: `localhost/repro-base:py312`
- Command: `python /inputs/imp.py --data /inputs/mnist --out /output/metrics.json --seed 0 --iters 50000 --max-round 20`
- Artifact manifest hash: `999c9ca4eb33af00d73c2cdb71a3987b843d200931ce8dacb805393cb7f95f8c`
- Sandbox policy hash: `c32ebb9dcd7dcc21a399cbc6cfe45458e1a14c065b6cc0262701e9e679c922df`
- Wall seconds: 13388.74
- Outcome: COMPLETED

## Containment & limitations

- Runs are **containerized, not sandboxed**: no network, all Linux
  capabilities dropped, non-root user, read-only root filesystem, and
  CPU/memory/pid caps. On a host whose user is in the `docker` group the
  orchestrator is root-equivalent; a kernel escape reaches the host.
- Provenance is SLSA build level L1 (self-attested, unhosted).
- A `FAILED_SAFELY` outcome means a resource/time cap was hit, not that
  the result is wrong.
