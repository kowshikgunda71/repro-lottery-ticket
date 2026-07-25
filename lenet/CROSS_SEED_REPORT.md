# Lottery Ticket reproduction — 3 seeds

seed 0: 21 levels, 3.72 h
seed 1: 21 levels, 3.69 h
seed 2: 21 levels, 3.72 h

## Pre-registered claims

| claim | metric | paper | tol | seed 0 | seed 1 | seed 2 | mean | verdict |
|---|---|---|---|---|---|---|---|---|
| C1 | early_stop_reduction_pct_pm21_1 | 38.0 | ±15.0 | 46.7 | 0 | -2.56 | 14.7 | NOT_REPRODUCED (mean); 1/3 seeds pass |
| C2 | test_acc_delta_pp_pm13_5 | 0.3 | ±0.3 | -0.05 | 0.31 | 0.39 | 0.217 | REPRODUCED (mean); 2/3 seeds pass |
| C3 | test_acc_delta_pp_pm3_6 | 0.0 | ±0.3 | -0.07 | 0.06 | 0.31 | 0.1 | REPRODUCED (mean); 2/3 seeds pass |
| C4 | speedup_ticket_vs_random_pm21_1 | 2.51 | ±0.9 | 1.68 | 1.54 | 1.75 | 1.65 | REPRODUCED (mean); 2/3 seeds pass |
| C5 | acc_gap_ticket_minus_random_pp_pm21_1 | 0.5 | ±0.4 | 0.36 | 0.38 | 0.32 | 0.353 | REPRODUCED (mean); 3/3 seeds pass |
| C6 | max_test_acc_delta_pp_final_iter | 0.35 | ±0.3 | 0.23 | 0.42 | 0.13 | 0.26 | REPRODUCED (mean); 3/3 seeds pass |
| C7 | pm_percent_pm21_1 | 21.1 | ±0.05 | 21.1 | 21.1 | 21.1 | 21.1 | REPRODUCED (mean); 3/3 seeds pass |

**Overall (mean across seeds): 6/7 claims within the pre-registered tolerance.**

## Sparsity ladder (test accuracy at early stop, %)

| round | Pm % | seed 0 | seed 1 | seed 2 | mean | Δ vs unpruned |
|---|---|---|---|---|---|---|
| 0 | 100.00 | 98.24 | 98.01 | 97.86 | 98.04 | +0.00 |
| 1 | 80.04 | 98.14 | 97.93 | 98.07 | 98.05 | +0.01 |
| 2 | 64.06 | 98.21 | 98.16 | 97.92 | 98.10 | +0.06 |
| 3 | 51.28 | 98.01 | 97.91 | 98.03 | 97.98 | -0.05 |
| 4 | 41.05 | 98.18 | 98.01 | 98.20 | 98.13 | +0.09 |
| 5 | 32.87 | 98.17 | 98.28 | 98.20 | 98.22 | +0.18 |
| 6 | 26.32 | 98.14 | 98.25 | 98.23 | 98.21 | +0.17 |
| 7 | 21.07 | 98.29 | 98.36 | 98.38 | 98.34 | +0.31 |
| 8 | 16.88 | 98.35 | 98.14 | 98.28 | 98.26 | +0.22 |
| 9 | 13.52 | 98.19 | 98.32 | 98.25 | 98.25 | +0.22 |
| 10 | 10.83 | 98.32 | 98.30 | 98.25 | 98.29 | +0.25 |
| 11 | 8.68 | 98.22 | 98.17 | 98.15 | 98.18 | +0.14 |
| 12 | 6.95 | 98.32 | 98.06 | 98.30 | 98.23 | +0.19 |
| 13 | 5.57 | 98.17 | 98.00 | 98.22 | 98.13 | +0.09 |
| 14 | 4.47 | 98.24 | 98.01 | 98.19 | 98.15 | +0.11 |
| 15 | 3.58 | 98.17 | 98.07 | 98.17 | 98.14 | +0.10 |
| 16 | 2.87 | 97.98 | 97.76 | 97.88 | 97.87 | -0.16 |
| 17 | 2.31 | 97.90 | 97.72 | 97.84 | 97.82 | -0.22 |
| 18 | 1.85 | 97.56 | 97.44 | 97.47 | 97.49 | -0.55 |
| 19 | 1.49 | 97.37 | 97.31 | 97.30 | 97.33 | -0.71 |
| 20 | 1.19 | 96.97 | 97.03 | 96.81 | 96.94 | -1.10 |

## Random-reinitialisation control (the paper's central test)

- seed 0, Pm=21.07%: ticket early-stop @4000 acc 98.29%  vs  reinit @6700 acc 97.93%  → 1.68x faster, +0.36 pp
- seed 0, Pm=3.58%: ticket early-stop @4000 acc 98.17%  vs  reinit @15600 acc 97.30%  → 3.90x faster, +0.87 pp
- seed 1, Pm=21.07%: ticket early-stop @4100 acc 98.36%  vs  reinit @6300 acc 97.98%  → 1.54x faster, +0.38 pp
- seed 1, Pm=3.58%: ticket early-stop @4600 acc 98.07%  vs  reinit @17800 acc 97.02%  → 3.87x faster, +1.05 pp
- seed 2, Pm=21.07%: ticket early-stop @4000 acc 98.38%  vs  reinit @7000 acc 98.06%  → 1.75x faster, +0.32 pp
- seed 2, Pm=3.58%: ticket early-stop @5800 acc 98.17%  vs  reinit @13900 acc 97.17%  → 2.40x faster, +1.00 pp
