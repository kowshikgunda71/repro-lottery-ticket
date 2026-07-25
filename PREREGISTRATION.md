# Pre-registration — Lottery Ticket Hypothesis, CIFAR-10 (Conv-2/4/6)

**This document was written and published BEFORE any training run happened.**
It states what will be run, what is predicted, and how each prediction will be
scored. Nothing in `claims.json` may be edited after results exist; the results
will be published whatever they show.

Replication target:

> Jonathan Frankle and Michael Carbin. **The Lottery Ticket Hypothesis: Finding
> Sparse, Trainable Neural Networks.** ICLR 2019. arXiv:1803.03635. Section 3
> (convolutional networks on CIFAR-10).

This is the companion to the already-published LeNet/MNIST half:
[repro-lottery-ticket-lenet](https://github.com/kowshikgunda71/repro-lottery-ticket-lenet)
(6/7 claims replicated across 3 seeds).

This is a **replication**, not a reproduction: the harness (`imp_conv.py`) was
written from the paper's text. No code, weights, or data of the authors' is used
or redistributed.

## What will run

| | Conv-2 | Conv-4 | Conv-6 |
|---|---|---|---|
| modules (2 conv layers + maxpool each) | 64,64 | +128,128 | +256,256 |
| fully-connected | 256, 256, 10 | 256, 256, 10 | 256, 256, 10 |
| weights (computed) | 4,300,992 | 2,425,024 | 2,261,184 |
| iterations / batch | 20,000 / 60 | 25,000 / 60 | 30,000 / 60 |
| optimizer | Adam 2e-4 | Adam 3e-4 | Adam 3e-4 |
| per-round pruning | conv 10%, fc 20%, out 10% | conv 10%, fc 20%, out 10% | conv 15%, fc 20%, out 10% |
| ladder depth | rounds 0–18 | 0–22 | 0–22 |

3 seeds per architecture, 1 random-reinitialisation control per claim rung.
Iterative magnitude pruning, layer-wise; survivors reset to θ₀. Early stop =
iteration of minimum validation loss, applied retroactively, never influencing
training. 45,000 train / 5,000 validation / 10,000 test. Evaluation every 100
iterations.

Run on Kaggle T4 (the local gym sandbox grants 4 CPUs and no GPU; these sweeps
are 110–578 CPU-hours each, so they cannot run there). Results return via
`gym import-run`, which records the boundary honestly as `external:Kaggle`,
never claiming containment it did not provide.

## Predictions

16 claims are registered in [`claims.json`](claims.json), each with the paper's
verbatim quote, a tolerance, and a written rationale for that tolerance. Nine
are empirical (the paper's headline numbers: 3.5×/3.4pp for Conv-2, 3.5×/3.5pp
for Conv-4, 2.5×/3.3pp for Conv-6, plus the Pm>2% crossover, the 100%
training-accuracy condition, and the random-reinit control). Seven are
structural checks of the architecture and pruning ladder.

**Two claims are registered as predicted FAILURES, before running anything:**

- **S4 — Conv-6 total weights = 1.7M.** Figure 2 states this. The architecture
  Appendix H.1 describes computes to 2,261,184. It is registered under the same
  ±50,000 tolerance as the Conv-2 and Conv-4 counts, which pass exactly, and it
  will fail by 561,184.
- **S7 — Conv-6 Pm at round 10 = 15.1%.** Registered under the same ±0.1pp
  tolerance as the Conv-2 (8.8%) and Conv-4 (9.2%) rungs, which pass. Ours gives
  15.29% and will fail by 0.19pp.

### Why we predict the paper is wrong here

Three independent lines of evidence, all from the paper's own numbers:

1. **Parameter arithmetic.** The 'same'-padding, pool-after-every-module reading
   reproduces Figure 2 exactly for LeNet (266,200 = "266K"), Conv-2 (4,300,992 =
   "4.3M") and Conv-4 (2,425,024 = "2.4M"). Only Conv-6 misses. An 8-way sweep
   over padding × pool window × trailing-pool found *no* configuration yielding
   1.7M, and every configuration that lowers Conv-6 also breaks Conv-2 and
   Conv-4. Reaching 1.7M would require an fc1 input of 1,903.9 units — not an
   integer, and no spatial map over 256 channels.
2. **The paper's own sparsity labels.** Conv-6's printed rungs (Pm = 26.4% and
   15.1%) land at 26.61% / 15.29% under our 2.26M reading (0.2pp off) but at
   28.32% / 13.99% under a 1.7M reading (1.9–2.0pp off). The labels corroborate
   ~2.3M and refute 1.7M.
3. **Appendix H.6's FC shares.** It states fully-connected layers are "99%, 89%,
   and 35%" of Conv-2/4/6. Ours give 99.10% and 89.29% — exact — and 49.4% for
   Conv-6.

So the defect is localised to Conv-6's fully-connected/total accounting and
propagates into Section 3's "nearly two thirds" and Appendix H.6's "35%". The
convolutional count (1.1M) is correct, which is why S3 is registered as an
expected *pass* — its passing is what makes S4's failure diagnostic rather than
a sign of a broken implementation.

We will build the 2,261,184-weight network, because that is what the paper
*describes*, and report the mismatch.

## Stated assumptions (things the paper does not specify)

- **Input preprocessing is NOT STATED anywhere in the paper** for the CIFAR-10
  experiments — zero occurrences of preprocess / whiten / ZCA / GCN / mean
  subtraction across the full text and all appendices. We use plain [0,1]
  scaling of raw pixels, with no normalisation and no data augmentation. This is
  an assumption, not a reading, and it is the single largest threat to
  comparability of absolute accuracies.
- ReLU activations (not stated; VGG convention).
- 'same' padding (not stated; forced by the parameter arithmetic above).
- No batch normalisation, no weight decay (not mentioned).
- The 5,000-example validation split is a fixed random draw (seed 12345) shared
  across seeds, so seeds differ in initialisation and batch order, not in split.

## Declared deviations from the paper's protocol

- **3 seeds, not 5.** Matches the published LeNet half so the two are
  comparable, and fits the free Kaggle quota with slack for a failed push.
- **1 random-reinitialisation draw per rung, not 3** (the paper averages
  5 trials × 3 reinits = 15 runs per reinit point).
- Ladder depth stops at the Pm>2% boundary each claim needs, not below it.

## How results will be scored

`aggregate.py` combines the per-architecture, per-seed metrics into the 16
registered metrics (means across seeds, as the paper averages trials), and
`gym import-run` scores them against `claims.json` with the tolerances fixed
here. A metric a run fails to produce is scored INCONCLUSIVE, never a pass.

Every verdict will be published — including, and especially, the ones that fail.
