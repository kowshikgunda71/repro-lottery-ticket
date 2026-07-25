# Five uses for this work that the original paper wasn't written for

Frankle & Carbin wrote *The Lottery Ticket Hypothesis* to argue a scientific
point: that a randomly-initialised dense network contains a sparse subnetwork
which, trained from **its original initialisation**, matches the full network.
That is a claim about *why* neural networks are trainable.

This repository is a replication of that claim — but the artifacts it produced
(a from-scratch IMP harness, a 21-rung sparsity ladder measured across 5 seeds,
a random-reinitialisation control, and the pre-registration machinery around
them) are useful for several things the paper was not aimed at. Each item below
points at evidence actually in this repo.

---

### 1. A measured compression schedule with a known accuracy cliff

The paper argues tickets *exist*. What a practitioner shipping to a phone or a
microcontroller needs is *where the cliff is*, in numbers, on their own hardware.

[`lenet/CROSS_SEED_REPORT_N5.md`](lenet/CROSS_SEED_REPORT_N5.md) is exactly that
table: accuracy at every rung from 100% down to 1.19% density, averaged over 5
seeds. It shows accuracy **above** baseline from ~80% down to ~3.6% density
(peaking around +0.29pp at 21%), then falling off — −0.45pp at 1.85%, −1.06pp at
1.19%. So on LeNet-300-100/MNIST you can delete ~79% of the weights and do
slightly *better*, and the honest budget before real degradation is roughly 3%.

Useful because it is a measured curve with seed-level variance, not a rule of
thumb. `code/imp.py` re-runs it on a different architecture.

### 2. A sensitive canary for silent training-pipeline regressions

The paper's central control — train the same mask from the original init vs. from
a fresh random init — turns out to be a very sensitive integrity check, and
nobody needs to care about the hypothesis to use it.

Across every seed and both sparsity levels tested, the ticket beat its
randomly-reinitialised twin **10 times out of 10**: +0.31 to +1.28pp accuracy and
1.54× to 3.90× faster to early stop. That gap is large, consistent, and it
depends on the initialisation being preserved end-to-end.

Which makes it a CI assertion. If someone changes your seeding, your init scheme,
your checkpoint restore, or your data order in a way that silently decouples
weights from their initialisation, this gap collapses toward zero long before
your top-line accuracy moves enough to notice.

### 3. Sizing how many seeds your own benchmark actually needs

The most useful number here is one that *failed*.

Claim C1 — "38% earlier early stopping" — did not replicate at n=3 or n=5, and
[`lenet/REPRODUCTION_NOTES.md`](lenet/REPRODUCTION_NOTES.md) shows why: the
unpruned network's early-stopping iteration came out 7500/4100/3900/4700/4400
across seeds — a **30% coefficient of variation** on the denominator alone.

If you are about to report "our method converges X% faster" from a single run,
that is the number to look at first. Iteration-of-minimum-validation-loss is a
noisy argmin, and a ratio of two of them is noisier still. The measured
dispersion here lets you compute the sample size a convergence-speed claim needs
*before* you run the experiment.

### 4. A worked template for evaluating someone else's numeric claims

Strip out the lottery tickets and what remains is a procedure for holding a
claimed number to account:

1. quote the claim verbatim, with its source
2. fix the tolerance and its justification **before** you measure
3. publish the registration with a timestamp
4. score mechanically and publish the result whichever way it goes

That transfers directly to evaluating a vendor's accuracy/latency claims before a
pilot, validating a model card, or checking an internal team's benchmark. The
tolerances are the hard part, and
[`paper-repro-gym`](https://github.com/kowshikgunda71/paper-repro-gym) now derives
them by rule from the claim's own reported precision rather than by judgement —
which is what stops "we set the bar where we knew we'd clear it".

### 5. A pre-submission consistency check for your own papers

The most transferable finding here needed no GPU at all.

[`cifar/PREREGISTRATION.md`](cifar/PREREGISTRATION.md) documents that the paper's
stated Conv-6 parameter count (1.7M) contradicts the architecture its own
appendix describes (2,261,184) — established three independent ways, including
against the paper's own printed sparsity labels. It was found by arithmetic,
before any run.

Any paper stating an architecture *and* a parameter count is checkable this way
in seconds, and the check catches a class of error that peer review reliably
misses because reviewers rarely multiply the layer shapes. Run it on your own
draft before submitting. `code/imp_conv.py --selfcheck` shows the pattern:
compute the count from the described architecture and assert it against the one
you plan to print.

---

### A caveat that applies to all five

These are uses of *this replication's* artifacts and measurements, on
LeNet-300-100/MNIST and (in progress) CIFAR-10 Conv-2/4/6, at the specific
hyperparameters the paper describes. Nothing here establishes that the numbers
transfer to your architecture, your data, or your optimiser — items 1 and 3 in
particular are measurements, not laws. Re-run `code/imp.py` on your own setup
rather than importing the constants.

Item 2 and item 5 are the two that generalise cleanly, because they are
*procedures*, not values.
