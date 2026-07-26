# CIFAR-10 figures — PARTIAL, runs still in flight

**These are interim figures from an incomplete sweep. No claim in
[`../claims.json`](../claims.json) has been scored yet.**

Present here: `conv2/` (3 of 3 seeds, complete) and `conv4/` (1 of 3 seeds).
`conv6/` has not run. Scoring happens only when a full architecture's seeds are
in, because the registered claims are defined on the cross-seed mean.

Rendered on CPU from `metrics.json` by `gym figures` — measured values only, no
smoothing, no interpolation between rungs, no trend fitting. Each architecture is
plotted separately; runs are never pooled across architectures.

One observation already visible, and reported here rather than saved for a
convenient moment: our Conv-2 baseline sits near **69.8%**, where the paper's
Figure 2 is closer to 77%. That gap is consistent with the preprocessing
assumption declared in [`../PREREGISTRATION.md`](../PREREGISTRATION.md) — the
paper states no CIFAR-10 preprocessing anywhere, and we registered plain [0,1]
pixel scaling with no normalisation and no augmentation. It is a gap in our
assumption before it is a gap in the paper, and the registered claims are all
*relative* quantities precisely because absolute accuracies were never going to
be comparable.
