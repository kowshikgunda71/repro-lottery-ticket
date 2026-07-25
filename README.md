# Replicating *The Lottery Ticket Hypothesis*

An independent, **pre-registered** replication of both halves of:

> Jonathan Frankle and Michael Carbin. **The Lottery Ticket Hypothesis: Finding
> Sparse, Trainable Neural Networks.** ICLR 2019.
> [arXiv:1803.03635](https://arxiv.org/abs/1803.03635) ·
> [doi:10.48550/arXiv.1803.03635](https://doi.org/10.48550/arXiv.1803.03635)

All credit for the research is the original authors'. This repository is a
replication — it contains the reproducer's own harness and evidence, and
redistributes no code, data, or weights of the authors'.

This is a **replication**, not a reproduction, in the ACM sense ("Results
Replicated"): the experiments were re-implemented from the paper's text by a
different team on different artifacts. The authors' code was not used.

| | [`lenet/`](lenet/) — LeNet-300-100 / MNIST | [`cifar/`](cifar/) — Conv-2/4/6 / CIFAR-10 |
|---|---|---|
| status | **complete, 5 seeds** | pre-registered, runs in flight |
| claims | 7 | 16 |
| result | **6/7 replicated** on the 5-seed mean | pending |
| boundary | rootless podman, contained | Kaggle T4, recorded as `external:` |

## What is actually being claimed here

Every number below was registered — with its verbatim quote from the paper, a
tolerance, and a written rationale for that tolerance — **before the run that
tests it existed**. Nothing was retuned after seeing a result.

**LeNet/MNIST (done).** Six of seven claims replicate on the mean of five seeds.
The one that does not is the paper's "38% earlier early stopping" magnitude: it
came out at 14.7% over three seeds and 21.4% over five, against a pre-registered
tolerance of 38±15. We report it as not replicated **at this sample size**, and
specifically do *not* claim the paper is wrong — the quantity is a ratio of two
individually noisy single runs. Details, including a correction to our own n=3
diagnosis, are in [`lenet/REPRODUCTION_NOTES.md`](lenet/REPRODUCTION_NOTES.md).

**CIFAR-10 (running).** Sixteen claims, of which **two are registered as
predicted failures** before any run: the paper's stated Conv-6 parameter count
(1.7M) contradicts the architecture its own appendix describes (2,261,184), and
that propagates into one of its printed sparsity labels. Three independent lines
of evidence are laid out in [`cifar/PREREGISTRATION.md`](cifar/PREREGISTRATION.md).
Predicting a failure in public, before running, is the point — it is what makes
the eventual result informative rather than post-hoc.

## Why both halves live in one repository

They are one paper. Splitting a replication across repositories by experiment
makes the claim ledger harder to read than the paper it is checking.

## Beyond the paper's own question

[`USE_CASES.md`](USE_CASES.md) lists five things this work is useful for that the
original paper was not written to serve — a measured compression schedule with a
located accuracy cliff, a CI canary for silent training-pipeline regressions, a
way to size the seed count a convergence-speed claim needs, a transferable
template for auditing anyone's numeric claims, and a pre-submission
parameter-count check that catches a class of error peer review reliably misses.
Each points at evidence in this repo, and the caveats are stated with them.

## Known defects in this replication

Errata are published, not patched away. See
[`lenet/ERRATUM.md`](lenet/ERRATUM.md): two of the seven LeNet claims are
one-sided in the paper's wording ("more than 0.3", "up to 0.35") and were
registered two-sided. Rescored correctly the verdicts are unchanged and the 6/7
headline stands — but the registration was right by luck, not construction, and
under a different draw it would have produced a false failure. The scorer now
supports one-sided and interval claims; the original registration is left
unedited, because a pre-registration rewritten after results exist is worthless.

## Honest results

Every registered claim is published with its verdict — replicated, **not
replicated**, or inconclusive. A failure is a real, reportable result and is
never dropped. Where our own earlier analysis turned out to be wrong (the n=3
variance diagnosis), the correction is published beside it rather than replacing
it.

## Reproducing this replication

Each half documents its own environment and commands. Both were produced with
[paper-repro-gym](https://github.com/kowshikgunda71/paper-repro-gym), which fixes
claims and tolerances before a run, scores against them afterwards, and records
the containment boundary honestly — including when a run happened on external
compute it did not control.

See [`CITATION.cff`](lenet/CITATION.cff) to cite both this replication and the
original paper.
