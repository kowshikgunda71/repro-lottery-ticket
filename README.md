# Reproduction: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Verdict: NOT_REPRODUCED**  (6/7 claims reproduced within their pre-registered tolerance)

An independent *replication* (ACM "Results Replicated") — the experiment was re-implemented from the paper's text by a different team, without using the authors' code, and the reported numbers were checked against tolerances registered **before** the run. See [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) for the 3-seed results, the one claim that did not replicate, and the scope limits.

## Paper reproduced

> Frankle, Jonathan; Carbin, Michael (2019) *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*. ICLR 2019. https://doi.org/10.48550/arXiv.1803.03635

Original work by the authors above; all credit for the research is theirs. This repository is an independent reproduction, not the original work, and does not redistribute the paper's code, data, or models — see [ACQUISITION.md](ACQUISITION.md). See [CITATION.cff](CITATION.cff) to cite both this reproduction and the original paper.

Produced with [paper-repro-gym](https://github.com/kowshikgunda71/paper-repro-gym), a gated, containerized reproduction workbench.

## Results (reported honestly)

Every registered claim is shown with its verdict — reproduced, **not reproduced**, partial, or inconclusive alike. A failure to reproduce is a real, reportable result and is never hidden.

| Claim | Metric | Claimed | Observed | Tolerance | Verdict |
|---|---|---|---|---|---|
| Winning tickets learn faster: at Pm=21.1 | early_stop_reduction_pct_pm21_1 | 38.0 | 46.67 | 15.0 (abs) | REPRODUCED |
| Test accuracy at early stop improves by  | test_acc_delta_pp_pm13_5 | 0.3 | -0.05 | 0.3 (abs) | NOT_REPRODUCED |
| By Pm=3.6% the accuracy gain is gone: te | test_acc_delta_pp_pm3_6 | 0.0 | -0.07 | 0.3 (abs) | REPRODUCED |
| The original initialisation matters: at  | speedup_ticket_vs_random_pm21_1 | 2.51 | 1.675 | 0.9 (abs) | REPRODUCED |
| At Pm=21% the winning ticket is about ha | acc_gap_ticket_minus_random_pp_pm21_1 | 0.5 | 0.36 | 0.4 (abs) | REPRODUCED |
| Winning tickets generalise better, not j | max_test_acc_delta_pp_final_iter | 0.35 | 0.23 | 0.3 (abs) | REPRODUCED |
| Structural check of the pruning schedule | pm_percent_pm21_1 | 21.1 | 21.072 | 0.05 (abs) | REPRODUCED |

## How it was reproduced

- `experiment.json` — the exact image and command that was run.
- `claims.json` — the claims and tolerances, registered before the run.
- `code/` — the reproduction harness (the scripts that were run). This is the reproducer's own code; the paper's artifacts are **not** redistributed (see [ACQUISITION.md](ACQUISITION.md)).

## Evidence in this repo

- `claim_result_matrix.json` — claimed vs observed vs pre-registered tolerance
- `experiment_manifest.json` — image (by digest), command, hashes, resource use, boundary
- `provenance.json` — SLSA-subset build provenance
- `REPRODUCIBILITY.md` — how to reproduce this reproduction
- `logs/` — raw run record, stdout, stderr

## Reproduce it yourself

Clone [paper-repro-gym](https://github.com/kowshikgunda71/paper-repro-gym), acquire the artifacts per ACQUISITION.md,
and run the command in `experiment_manifest.json` on a hardened boundary.

A failure to reproduce is a real, reportable result — this record states the
verdict honestly, whatever it was.
