# Where this work is useful — and where the evidence runs out

This file has two halves and they are not equally reliable.

**Part A — Evidenced.** Where pruning and the Lottery Ticket line of work
actually stand in 2026, each claim with a source that was opened and checked.
Some of it argues *against* using this technique.

**Part B — Unvalidated suggestions.** Extrapolations from this replication's own
artifacts. This replication tested the paper's claims; it did **not** test any of
these. They are hypotheses, not results, and the original authors neither
proposed nor endorsed them.

---

# Part A — What is actually true about pruning in 2026 (evidenced)

## A1. The flagship commercial sparsity stack is dead

Neural Magic — the company built on running pruned/sparse models fast on
commodity CPUs — was acquired by Red Hat, and its entire sparsity product line
(DeepSparse, SparseML, SparseZoo, Sparsify) was **end-of-lifed on 2025-06-02**,
with the repository archived the following day.
Source: <https://github.com/neuralmagic/deepsparse>

The successor library from the same team (Red Hat AI + the vLLM Project),
[`llm-compressor`](https://github.com/vllm-project/llm-compressor), is
**quantization-first**: it ships no general pruning method in its headline
algorithms.

**What this means for you:** if you came here looking for a production path to
sparse inference, the commercial vehicle for it has shut down. Quantization is
where that engineering effort went.

## A2. The scale limitation is real, but it was fixed — do not repeat the popular overstatement

The widely-repeated claim is "LTH doesn't scale." The accurate version:

- Gale, Elsen & Hooker, *The State of Sparsity in Deep Neural Networks*
  ([arXiv:1902.09574](https://arxiv.org/abs/1902.09574), Feb 2019) could not
  reproduce the winning-ticket phenomenon on ResNet-50/ImageNet or
  Transformer/WMT14. That is a **null result against rewinding to
  initialization**, not a refutation of the hypothesis.
- One month later, Frankle et al. showed that rewinding to an *early training
  iteration* rather than to initialization **does** find winning tickets at
  ResNet-50/ImageNet scale
  ([arXiv:1912.05671](https://arxiv.org/abs/1912.05671)).

So the hypothesis survives with a procedural amendment. Anyone telling you it
was debunked is quoting the 2019 null result and skipping the 2019 fix.

## A3. What actually ships for compressing large models

Not iterative magnitude pruning from initialization. The methods in production
use are **one-shot post-training** pruning (SparseGPT
[arXiv:2301.00774](https://arxiv.org/abs/2301.00774), Wanda) and **structured
pruning plus knowledge distillation** — see NVIDIA's Llama-3.1-Minitron work.
None of them rewind to an initialization.

**Honest consequence for this repository:** the LTH replication has no
downstream production consumer. Its value here is pedagogical and as a test
fixture for the pre-registration tooling. That is stated plainly rather than
dressed up.

---

# Part B — Unvalidated suggestions

> ### ⚠️ None of the following has been tested.
> These are extrapolations from artifacts this replication happened to produce.
> Treat each as a hypothesis. Items marked **procedure** are more likely to
> transfer than items marked **measurement**, because a measurement is a number
> from one architecture, one dataset and one set of hyperparameters.

### B1. A CI canary for silent training-pipeline regressions — *procedure*

The paper's central control (same mask, original init vs. a fresh one) is a
sensitive integrity check, and you do not have to care about the hypothesis to
use it. Across every seed and both sparsity levels tested, the ticket beat its
randomly-reinitialised twin **10 times out of 10**: +0.31 to +1.28pp accuracy,
1.54×–3.90× faster to early stop.

That gap depends on the initialisation surviving end-to-end. If a change to
seeding, init, checkpoint restore, or data order silently decouples weights from
their initialisation, this collapses toward zero long before top-line accuracy
moves enough to notice.

### B2. Sizing the seed count your own benchmark needs — *measurement*

The most useful number here is one that failed. The unpruned network's
early-stopping iteration came out **7500 / 4100 / 3900 / 4700 / 4400** across
seeds — a ~30% coefficient of variation on the denominator alone.

If you are about to report "our method converges X% faster" from a single run,
look at that first. Iteration-of-minimum-validation-loss is a noisy argmin, and
a ratio of two of them is noisier still. See
[`lenet/REPRODUCTION_NOTES.md`](lenet/REPRODUCTION_NOTES.md).

### B3. A pre-submission parameter-count check — *procedure, and the strongest item here*

Needed no GPU. This paper's printed Conv-6 parameter count (1.7M) contradicts the
architecture its own appendix describes (2,261,184) — established three
independent ways. `code/imp_conv.py --selfcheck` shows the pattern: compute the
count from the architecture you describe and assert it against the one you print.

Peer review reliably misses this class of error because reviewers rarely multiply
layer shapes. The generalized version is `gym audit` in
[paper-repro-gym](https://github.com/kowshikgunda71/paper-repro-gym).

### B4. A compression schedule with a located cliff — *measurement, and see A1*

[`lenet/CROSS_SEED_REPORT_N5.md`](lenet/CROSS_SEED_REPORT_N5.md) gives accuracy
at every rung from 100% down to 1.19% density across 5 seeds: above baseline from
~80% down to ~3.6% density, then −0.45pp at 1.85% and −1.06pp at 1.19%.

Read this alongside **A1** and **A3**: it is a measured curve on
LeNet-300-100/MNIST, and the production tooling for exploiting sparsity has moved
elsewhere. Useful as a teaching curve; not a deployment recipe.

### B5. A template for auditing anyone's numeric claim — *procedure*

Strip out the lottery tickets and a general procedure remains: quote the claim
verbatim, fix the tolerance *and its justification* before measuring, publish the
registration with a timestamp, score mechanically, publish either way. That
transfers to vendor accuracy claims, model cards, and internal benchmarks.
