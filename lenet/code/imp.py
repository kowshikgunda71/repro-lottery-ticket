"""Iterative magnitude pruning on LeNet-300-100 / MNIST.

Independent reproduction harness for Frankle & Carbin, "The Lottery Ticket
Hypothesis: Finding Sparse, Trainable Neural Networks" (ICLR 2019,
arXiv:1803.03635). Written from the paper's text; no code from the authors is
used, and no artifact of theirs is redistributed.

Protocol, each element quoted from the paper (Section 2, Figure 2, Appendix G):
  - 784-300-100-10 fully-connected ReLU net; "All/Conv Weights 266K" -> biases
    are NOT pruned (784*300 + 300*100 + 100*10 = 266,200).
  - "Initializations are Gaussian Glorot" -> normal, std sqrt(2/(fan_in+fan_out)).
  - "Iterations/Batch 50K / 60 ... Optimizer Adam 1.2e-3".
  - "We randomly sampled a 5,000-example validation set from the training set and
    used the remaining 55,000 training examples as our training set".
  - "a pruning rate of 20% per iteration (10% for the output layer) ... Each
    layer of the network is pruned independently."
  - "the network is trained for 50,000 training iterations regardless of when
    early-stopping occurs ... early-stopping times are determined retroactively".
  - "The particular early-stopping criterion we employ throughout this paper is
    the iteration of minimum validation loss during training."
  - "We evaluate validation and test performance every 100 iterations."
  - Winning ticket: surviving weights are RESET to their values in theta_0.
    Control: same mask, freshly resampled initialisation.

Round k therefore leaves Pm = (265200*0.8^k + 1000*0.9^k)/266200 of the weights:
k=3 -> 51.3%, 7 -> 21.1%, 9 -> 13.5%, 15 -> 3.6%, 18 -> 1.9%, 20 -> 1.2% --
exactly the sparsity labels the paper prints. `--selfcheck` asserts this.

CPU-only, stdlib + numpy, no network. metrics.json is rewritten after every
level, so a run cut short still yields scoreable evidence.
"""

from __future__ import annotations

import os

# Must precede `import numpy`: OpenBLAS sizes its thread pool at import time from
# the HOST core count, which inside a 4-CPU cgroup means ~20 threads fighting for
# 4 cores -- measured 8.2x slower on the evaluation GEMMs. The gym builds the
# container argv itself and passes no -e flags, so the cap belongs here.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse                                                    # noqa: E402
import gzip                                                        # noqa: E402
import json                                                        # noqa: E402
import struct                                                      # noqa: E402
import time                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

SHAPES = [(784, 300), (300, 100), (100, 10)]
PRUNE_RATE = [0.20, 0.20, 0.10]           # per round, per layer, of the survivors
TOTAL_W = sum(a * b for a, b in SHAPES)   # 266_200
VAL_SPLIT_SEED = 12345                    # fixed across seeds: one held-out 5k set

# Pm label -> pruning round. The labels are the paper's; k is arithmetic.
LEVELS = {0: "100_0", 3: "51_3", 7: "21_1", 9: "13_5", 15: "3_6", 18: "1_9", 20: "1_2"}
REINIT_AT = (7, 15)                       # paper's random-reinit control points


def load_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        buf = fh.read()
    magic, = struct.unpack(">I", buf[:4])
    if magic == 2051:                                    # images
        n, r, c = struct.unpack(">III", buf[4:16])
        return np.frombuffer(buf, np.uint8, offset=16).reshape(n, r * c)
    if magic == 2049:                                    # labels
        return np.frombuffer(buf, np.uint8, offset=8)
    raise ValueError(f"{path}: bad IDX magic {magic}")


def mnist(root: Path):
    x = load_idx(root / "train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y = load_idx(root / "train-labels-idx1-ubyte.gz").astype(np.int64)
    xt = load_idx(root / "t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    yt = load_idx(root / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    if len(y) != 60000 or len(yt) != 10000:
        raise ValueError(f"unexpected MNIST sizes: {len(y)}/{len(yt)}")
    perm = np.random.default_rng(VAL_SPLIT_SEED).permutation(60000)
    va, tr = perm[:5000], perm[5000:]
    return (np.ascontiguousarray(x[tr]), y[tr],
            np.ascontiguousarray(x[va]), y[va], xt, yt)


def init_weights(rng):
    """Gaussian Glorot, as stated in Figure 2's caption."""
    W = [rng.normal(0.0, np.sqrt(2.0 / (a + b)), (a, b)).astype(np.float32) for a, b in SHAPES]
    return W, [np.zeros(b, np.float32) for _, b in SHAPES]


def forward(W, b, x):
    h1 = np.maximum(x @ W[0] + b[0], 0.0)
    h2 = np.maximum(h1 @ W[1] + b[1], 0.0)
    return h1, h2, h2 @ W[2] + b[2]


def softmax_xent(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    loss = float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())
    return p, loss


def evaluate(W, b, x, y, chunk=5000):
    loss, correct = 0.0, 0
    for i in range(0, len(y), chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk]
        _, _, logits = forward(W, b, xb)
        _, l = softmax_xent(logits, yb)
        loss += l * len(yb)
        correct += int((logits.argmax(1) == yb).sum())
    return loss / len(y), correct / len(y)


def train(W_init, b_init, masks, data, *, seed, iters, batch=60, lr=1.2e-3, eval_every=100):
    """One training run from a fixed init under a fixed mask.

    Returns (early_stop_record, final_record). Validation/test are measured every
    `eval_every` iterations and never influence training -- early stopping is
    applied retroactively, as the paper specifies.
    """
    xtr, ytr, xva, yva, xte, yte = data
    rng = np.random.default_rng(seed)
    W = [w.copy() * m for w, m in zip(W_init, masks)]
    b = [v.copy() for v in b_init]
    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(v) for v in b]
    vb = [np.zeros_like(v) for v in b]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best = {"val_loss": float("inf"), "iter": 0, "test_acc": 0.0}
    n = len(ytr)
    perm, pos = rng.permutation(n), 0
    for t in range(1, iters + 1):
        if pos + batch > n:
            perm, pos = rng.permutation(n), 0
        idx = perm[pos:pos + batch]
        pos += batch
        xb, yb = xtr[idx], ytr[idx]

        h1, h2, logits = forward(W, b, xb)
        d, _ = softmax_xent(logits, yb)
        d[np.arange(batch), yb] -= 1.0
        d /= batch
        gW2, gb2 = h2.T @ d, d.sum(0)
        d2 = (d @ W[2].T) * (h2 > 0)
        gW1, gb1 = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ W[1].T) * (h1 > 0)
        gW0, gb0 = xb.T @ d1, d1.sum(0)

        c1, c2 = 1 - beta1 ** t, 1 - beta2 ** t
        for i, (gw, gb) in enumerate(((gW0, gb0), (gW1, gb1), (gW2, gb2))):
            gw = gw * masks[i]                       # pruned weights get no gradient
            mW[i] = beta1 * mW[i] + (1 - beta1) * gw
            vW[i] = beta2 * vW[i] + (1 - beta2) * gw * gw
            W[i] -= lr * (mW[i] / c1) / (np.sqrt(vW[i] / c2) + eps)
            W[i] *= masks[i]                         # ... and stay at zero
            mb[i] = beta1 * mb[i] + (1 - beta1) * gb
            vb[i] = beta2 * vb[i] + (1 - beta2) * gb * gb
            b[i] -= lr * (mb[i] / c1) / (np.sqrt(vb[i] / c2) + eps)

        if t % eval_every == 0 or t == iters:
            vl, _ = evaluate(W, b, xva, yva)
            if vl < best["val_loss"]:
                _, ta = evaluate(W, b, xte, yte)
                best = {"val_loss": vl, "iter": t, "test_acc": ta}

    _, final_test = evaluate(W, b, xte, yte)
    _, final_train = evaluate(W, b, xtr[:10000], ytr[:10000])
    return best, {"test_acc": final_test, "train_acc": final_train}, W


def prune(W, masks):
    """Magnitude-prune the surviving weights of each layer, independently."""
    out = []
    for w, m, rate in zip(W, masks, PRUNE_RATE):
        alive = m.astype(bool)
        k = int(round(rate * int(alive.sum())))
        if k <= 0:
            out.append(m.copy())
            continue
        cut = np.partition(np.abs(w[alive]), k - 1)[k - 1]
        new = m.copy()
        new[alive & (np.abs(w) <= cut)] = 0.0
        out.append(new)
    return out


def pm_percent(masks):
    return 100.0 * sum(int(m.sum()) for m in masks) / TOTAL_W


def derive(lv: dict) -> dict:
    """Turn the per-level records into the metrics the paper's claims are about.

    The paper prints RELATIVE quantities for LeNet (deltas, ratios, percentages)
    and never an absolute accuracy, so these are what a claim can be scored on.
    """
    m: dict = {}
    if 0 not in lv:
        return m
    base = lv[0]["ticket"]
    for k, tag in LEVELS.items():
        if k in lv:
            r = lv[k]["ticket"]
            m[f"pm_percent_pm{tag}"] = round(lv[k]["pm"], 3)
            m[f"test_acc_pm{tag}"] = round(100 * r["test_acc"], 3)
            m[f"early_stop_iter_pm{tag}"] = r["early_stop_iter"]
            m[f"test_acc_delta_pp_pm{tag}"] = round(100 * (r["test_acc"] - base["test_acc"]), 3)
            if base["early_stop_iter"]:
                m[f"early_stop_reduction_pct_pm{tag}"] = round(
                    100 * (1 - r["early_stop_iter"] / base["early_stop_iter"]), 2)
    for k in REINIT_AT:
        if k in lv and "random" in lv[k]:
            tag, t_, r_ = LEVELS[k], lv[k]["ticket"], lv[k]["random"]
            m[f"test_acc_random_reinit_pm{tag}"] = round(100 * r_["test_acc"], 3)
            m[f"acc_gap_ticket_minus_random_pp_pm{tag}"] = round(
                100 * (t_["test_acc"] - r_["test_acc"]), 3)
            if t_["early_stop_iter"]:
                m[f"speedup_ticket_vs_random_pm{tag}"] = round(
                    r_["early_stop_iter"] / t_["early_stop_iter"], 3)
    deltas = [100 * (v["ticket"]["final_test_acc"] - base["final_test_acc"])
              for k, v in lv.items() if k > 0]
    if deltas:
        m["max_test_acc_delta_pp_final_iter"] = round(max(deltas), 3)
    m["min_train_acc_final_all_levels"] = round(
        100 * min(v["ticket"]["final_train_acc"] for v in lv.values()), 3)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/inputs/mnist")
    ap.add_argument("--out", default="/output/metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--max-round", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    data = mnist(Path(args.data))
    W0, b0 = init_weights(np.random.default_rng(args.seed))
    masks = [np.ones(s, np.float32) for s in SHAPES]
    lv: dict = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        out.write_text(json.dumps({
            **derive(lv),
            "_levels": [lv[k] for k in sorted(lv)],
            "_config": {"seed": args.seed, "iters": args.iters, "max_round": args.max_round,
                        "batch": 60, "lr": 1.2e-3, "optimizer": "adam", "init": "gaussian_glorot",
                        "val_split_seed": VAL_SPLIT_SEED,
                        "wall_seconds": round(time.time() - t0, 1)},
        }, indent=2), encoding="utf-8")

    W_end = None
    for k in range(args.max_round + 1):
        if k > 0:
            masks = prune(W_end, masks)
        pm = pm_percent(masks)
        best, final, W_end = train(W0, b0, masks, data, seed=args.seed * 1000 + k, iters=args.iters)
        lv[k] = {"round": k, "pm": pm, "ticket": {
            "early_stop_iter": best["iter"], "test_acc": best["test_acc"],
            "val_loss": best["val_loss"], "final_test_acc": final["test_acc"],
            "final_train_acc": final["train_acc"]}}
        print(f"[{time.time()-t0:7.0f}s] round {k:2d} Pm={pm:6.2f}%  early_stop@{best['iter']:6d}"
              f"  test={100*best['test_acc']:.2f}%  final={100*final['test_acc']:.2f}%", flush=True)
        flush()

        if k in REINIT_AT:
            Wr, br = init_weights(np.random.default_rng(args.seed + 90000 + k))
            rb, rf, _ = train(Wr, br, masks, data, seed=args.seed * 1000 + k, iters=args.iters)
            lv[k]["random"] = {"early_stop_iter": rb["iter"], "test_acc": rb["test_acc"],
                               "val_loss": rb["val_loss"], "final_test_acc": rf["test_acc"],
                               "final_train_acc": rf["train_acc"]}
            print(f"[{time.time()-t0:7.0f}s] round {k:2d} Pm={pm:6.2f}%  RANDOM-REINIT "
                  f"early_stop@{rb['iter']:6d}  test={100*rb['test_acc']:.2f}%", flush=True)
            flush()

    flush()
    print(json.dumps(derive(lv), indent=2))
    return 0


def _selfcheck() -> int:
    """One runnable check: magnitude pruning must drop exactly the smallest
    survivors at the stated per-layer rate, and the resulting sparsity ladder
    must land on the paper's own printed Pm labels."""
    rng = np.random.default_rng(0)
    W, _ = init_weights(rng)
    m = [np.ones(s, np.float32) for s in SHAPES]
    assert TOTAL_W == 266200, TOTAL_W
    for k in range(1, 21):
        m = prune(W, m)
        for w, mm in zip(W, m):
            assert set(np.unique(mm)) <= {0.0, 1.0}
            alive, dead = np.abs(w[mm.astype(bool)]), np.abs(w[~mm.astype(bool)])
            assert alive.min() >= dead.max() - 1e-9, "magnitude pruning kept the wrong weights"
        pm = pm_percent(m)
        expected = 100.0 * (265200 * 0.8 ** k + 1000 * 0.9 ** k) / TOTAL_W
        assert abs(pm - expected) < 0.05, f"round {k}: Pm {pm:.3f} != {expected:.3f}"
        if k in LEVELS:
            print(f"round {k:2d}: Pm={pm:.2f}%  (paper prints {LEVELS[k].replace('_', '.')}%)")
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_selfcheck() if "--selfcheck" in sys.argv else main())
