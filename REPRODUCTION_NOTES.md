# Reproduction notes — The Lottery Ticket Hypothesis (LeNet-300-100 / MNIST)

Independent reproduction of the Section 2 results of:

> Jonathan Frankle and Michael Carbin. **The Lottery Ticket Hypothesis: Finding
> Sparse, Trainable Neural Networks.** ICLR 2019. arXiv:1803.03635.

Original authors are credited in `CITATION.cff`. This repository contains only
the reproducer's harness and evidence — no code or artifact of the authors is
redistributed.

## Headline

**6 of 7 pre-registered claims reproduce on the 3-seed mean. One does not:
the specific "38% earlier early stopping" magnitude (C1).**

Each individual seed's bundle scores `NOT_REPRODUCED`, because the gym's overall
verdict is the worst claim in that run and every seed misses at least one of the
seven. That is the scorer being strict, not a collapse of the result: the
paper's central hypothesis — that the *original initialisation* of a pruned
subnetwork is what makes it trainable — reproduced on every seed, at every
sparsity we tested it.

## What the tolerances were, and when they were fixed

All seven claims, their verbatim quotes from the paper, their tolerances, and a
written rationale for each tolerance were committed to `claims.json` **before any
training run started**. Nothing was retuned after seeing a result. The paper
prints no absolute LeNet accuracy anywhere — only relative deltas and ratios —
so every claim is registered as a relative quantity. Registering an absolute
accuracy target would have meant inventing a number the paper never states.

## Results

| claim | what the paper says | paper | tol | seed 0 | seed 1 | seed 2 | mean | verdict |
|---|---|---|---|---|---|---|---|---|
| C1 | early stop 38% earlier at Pm=21.1% | 38 | ±15 | 46.7 | 0.0 | −2.6 | **14.7** | **NOT_REPRODUCED** |
| C2 | accuracy +>0.3pp at Pm=13.5% | 0.3 | ±0.3 | −0.05 | 0.31 | 0.39 | 0.22 | REPRODUCED |
| C3 | accuracy back to baseline at Pm=3.6% | 0.0 | ±0.3 | −0.07 | 0.06 | 0.31 | 0.10 | REPRODUCED |
| C4 | ticket 2.51× faster than random reinit | 2.51 | ±0.9 | 1.68 | 1.54 | 1.75 | 1.65 | REPRODUCED |
| C5 | ticket +0.5pp over random reinit | 0.5 | ±0.4 | 0.36 | 0.38 | 0.32 | 0.35 | REPRODUCED |
| C6 | +0.35pp still present at iteration 50k | 0.35 | ±0.3 | 0.23 | 0.42 | 0.13 | 0.26 | REPRODUCED |
| C7 | pruning ladder reaches Pm=21.1% | 21.1 | ±0.05 | 21.072 | 21.072 | 21.072 | 21.072 | REPRODUCED |

### The one that failed, and why

C1 is a ratio whose denominator is the *unpruned* network's early-stopping
iteration. Across three seeds that denominator came out **7500, 4100, 3900** — a
1.9× spread from nothing but the random seed. Seed 0's late baseline manufactures
a 47% "speedup"; seeds 1 and 2, whose baselines happened to stop early, show
none. The numerator is stable (early stop at Pm=21.1% was 4000, 4100, 4000 —
a 2.5% spread across seeds); essentially all of C1's variance comes from the
single baseline run it is divided by.

This is a measurement-noise result, not a contradiction of the paper. The paper
reports this number as an average of **five** trials and plots it with min/max
error bars; we ran one trial per seed. We therefore report C1 as not reproduced
**at this sample size**, and specifically do *not* claim the paper is wrong.

A caveat we cannot resolve: our absolute early-stopping iterations (3.9k–7.5k)
may be earlier than the authors', but since the paper prints no absolute
iteration counts, there is no number to check against. If their baseline stopped
substantially later, the 38% reduction would be measured on a different scale
than ours.

### The claim that matters most reproduced cleanly

The paper's central control — same pruning mask, weights resampled instead of
reset to θ₀ — held on every seed and got *stronger* with sparsity:

| | Pm=21.07% | Pm=3.58% |
|---|---|---|
| seed 0 | 1.68× faster, +0.36pp | 3.90× faster, +0.87pp |
| seed 1 | 1.54× faster, +0.38pp | 3.87× faster, +1.05pp |
| seed 2 | 1.75× faster, +0.32pp | 2.40× faster, +1.00pp |

At Pm=3.58% the winning ticket reaches minimum validation loss ~2.4–3.9× faster
and lands ~1 percentage point higher than the identical architecture with fresh
weights. 6/6 of these comparisons favour the winning ticket.

### The full sparsity curve reproduces qualitatively

Mean test accuracy at early stop across 3 seeds, relative to the unpruned net:
rises to a peak of **+0.31pp at Pm=21.1%**, stays positive down to Pm≈3.6%, then
falls off a cliff — −0.55pp at 1.9% and −1.10pp at 1.19%, where early stopping
also climbs back to ~10k iterations (learning slows). That is the shape the paper
describes. Our accuracy peak sits at Pm≈21% (seeds 1, 2) or 16.9% (seed 0) rather
than the paper's 13.5%, a difference well inside the seed-to-seed spread.

A useful methodological observation: the between-seed spread of the *unpruned
baseline* accuracy (97.86–98.24%, 0.38pp) is as large as the effects being
measured (0.3–0.5pp). Every delta claim inherits that noise, and the sign of a
seed's deltas tracks its baseline draw — seed 2 drew the lowest baseline and
shows the most positive deltas; seed 0 drew the highest and shows negative ones.
The paper's five-trial averaging is load-bearing, not ceremonial.

## Method

Harness written from the paper's text (`code/imp.py`); no author code was used.
Every protocol element was taken from the paper and is quoted in the harness
docstring:

- 784-300-100-10 ReLU MLP; 266,200 weights, biases never pruned (matches the
  paper's "266K").
- Gaussian Glorot initialisation (**normal**, not the more common uniform).
- Adam, lr 1.2e-3, batch 60, 50,000 iterations per training run, run to the end
  regardless of when early stopping occurs.
- 5,000 randomly-sampled validation examples, 55,000 train, 10,000 test.
- Validation and test evaluated every 100 iterations; early stop = iteration of
  minimum validation loss, applied *retroactively* (it never influences training).
- Iterative magnitude pruning, layer-wise and independent: 20% of surviving
  weights per round in the hidden layers, 10% in the output layer; survivors
  reset to θ₀.
- Control arm: identical mask, freshly resampled initialisation.

**Structural pre-check.** That pruning schedule is arithmetic, so it can be
verified before spending any compute: it must land on the sparsity labels the
paper prints. It does — 51.28/21.07/13.52/3.58/1.85/1.19% against the paper's
51.3/21.1/13.5/3.6/1.9/1.2%. `python imp.py --selfcheck` asserts this, plus that
magnitude pruning removes exactly the smallest surviving weights. A failure here
would mean the other six claims were measuring the wrong thing.

## Scope and limitations

- **3 seeds, not 5.** The paper averages 5 trials. This is the main reason C1 is
  reported as not reproduced at this sample size.
- **LeNet-300-100 / MNIST only.** The paper's CIFAR-10 convolutional results
  (Conv-2/4/6, Resnet-18, VGG-19) were not attempted.
- **Iterative pruning only**; the one-shot comparison was not run.
- **One random-reinitialisation draw per level**, against the paper's three.
- **Absolute accuracies are not comparable** to the paper because it prints none.
- Validation split is a fixed random 5,000 (seed 12345) shared by all three runs,
  so seeds differ in initialisation and batch order but not in the split.
- The container image is a locally-built base (`digest_pinned: false`); the
  `Containerfile` is included in `code/` so the environment can be rebuilt.

## Reproducing this reproduction

```bash
podman build -t repro-base:py312 -f code/Containerfile .
python code/imp.py --selfcheck                      # structural check, no compute
python code/imp.py --data <mnist-dir> --out metrics.json --seed 0
```

MNIST was acquired from the CVDF mirror
(`storage.googleapis.com/cvdf-datasets/mnist/`) with SHA-256 verification; the
checksums are in `experiment.json`. Each run took ~3.7 h on 4 CPU cores of an
NVIDIA GB10 (aarch64), CPU-only, no network inside the container.
