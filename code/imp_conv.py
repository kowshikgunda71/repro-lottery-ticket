"""Iterative magnitude pruning on Conv-2/4/6 / CIFAR-10.

Independent replication harness for Section 3 of Frankle & Carbin, "The Lottery
Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (ICLR 2019,
arXiv:1803.03635). Written from the paper's text; no author code is used.

Protocol, quoted from the paper:
  - Appendix H.1: "Each module has two layers of 3x3 convolutional filters
    followed by a maxpool layer with stride 2. After all of the modules are two
    fully-connected layers of size 256 followed by an output layer of size 10."
    Conv-2/4/6 = 1/2/3 modules, channels 64,64 | 128,128 | 256,256.
  - Figure 2: Iterations/Batch 20K/25K/30K at batch 60; Adam 2e-4 (Conv-2),
    3e-4 (Conv-4, Conv-6); Gaussian Glorot init.
  - Appendix H.4: "we select an iterative convolutional pruning rate of 10% for
    Conv-2, 10% for Conv-4, and 15% for Conv-6"; fully-connected 20%.
  - Appendix H.1: "The output layer is pruned at half of the rate of the
    fully-connected layers" -> 10% per round.
  - Pruning is layer-wise; survivors reset to theta_0. Control = same mask,
    resampled init.
  - Early stop = iteration of minimum validation loss, applied retroactively;
    validation/test evaluated every 100 iterations; 45k train / 5k val / 10k test.

ARCHITECTURE NOTE (a discrepancy in the paper, resolved by its own numbers):
Figure 2 states Conv-6 has "1.7M / 1.1M" weights. The conv count matches exactly
(1,144,512), but the stated total does not: the architecture Appendix H.1
describes gives 2,261,184. The reading used here is the one that reproduces
Figure 2 for LeNet (266,200), Conv-2 (4,300,992) and Conv-4 (2,425,024) exactly,
AND reproduces the Pm values the paper prints for Conv-6 (round 10 -> 15.29% vs
printed 15.1%; round 7 -> 26.61% vs printed 26.4%) far better than any reading
that yields 1.7M (13.99% / 28.32%). See REPRODUCTION_NOTES.md.

Runs on GPU (Kaggle T4) or CPU. Writes metrics.json after every level.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = {
    # name: (conv modules, iterations, adam lr, conv prune rate/round)
    "conv2": ([[64, 64]], 20000, 2e-4, 0.10),
    "conv4": ([[64, 64], [128, 128]], 25000, 3e-4, 0.10),
    "conv6": ([[64, 64], [128, 128], [256, 256]], 30000, 3e-4, 0.15),
}
FC_RATE, OUT_RATE = 0.20, 0.10
VAL_SPLIT_SEED = 12345

# Pm values the paper prints, and the round that produces each (arithmetic,
# verified against the paper's own labels -- see the module docstring).
CLAIM_ROUNDS = {
    "conv2": {11: "8_8", 14: "4_6"},
    "conv4": {12: "9_2", 11: "11_1"},
    "conv6": {10: "15_1", 7: "26_4"},
}


class ConvNet(nn.Module):
    def __init__(self, modules_spec):
        super().__init__()
        layers, cin, spatial = [], 3, 32
        for mod in modules_spec:
            for cout in mod:
                layers.append(nn.Conv2d(cin, cout, 3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                cin = cout
            layers.append(nn.MaxPool2d(2, 2))
            spatial //= 2
        self.features = nn.Sequential(*layers)
        self.flat = cin * spatial * spatial
        self.fc1 = nn.Linear(self.flat, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def glorot_normal_(model):
    """'Initializations are Gaussian Glorot' (Figure 2) -- normal, not uniform."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            nn.init.normal_(m.weight, 0.0, (2.0 / (fan_in + fan_out)) ** 0.5)
            nn.init.zeros_(m.bias)


def prunable(model):
    """(name, module, per-round rate) for every weight tensor that is pruned.
    Biases are never pruned -- Figure 2's counts are weights only."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            out.append((name, m, "conv"))
        elif isinstance(m, nn.Linear):
            out.append((name, m, "out" if name == "out" else "fc"))
    return out


def load_cifar(root: Path):
    """CIFAR-10 python batches -> float32 [0,1], NCHW. 45k train / 5k val / 10k test."""
    root = Path(root)
    tar = root / "cifar-10-python.tar.gz"
    if tar.exists():
        with tarfile.open(tar) as tf:
            def rd(n):
                f = tf.extractfile(f"cifar-10-batches-py/{n}")
                return pickle.load(f, encoding="bytes")
            batches = [rd(f"data_batch_{i}") for i in range(1, 6)]
            test = rd("test_batch")
    else:
        d = root / "cifar-10-batches-py"
        def rd2(n):
            with open(d / n, "rb") as f:
                return pickle.load(f, encoding="bytes")
        batches = [rd2(f"data_batch_{i}") for i in range(1, 6)]
        test = rd2("test_batch")

    x = np.concatenate([b[b"data"] for b in batches]).reshape(-1, 3, 32, 32)
    y = np.concatenate([np.array(b[b"labels"]) for b in batches])
    xt = test[b"data"].reshape(-1, 3, 32, 32)
    yt = np.array(test[b"labels"])
    assert x.shape == (50000, 3, 32, 32) and xt.shape == (10000, 3, 32, 32), (x.shape, xt.shape)

    perm = np.random.default_rng(VAL_SPLIT_SEED).permutation(50000)
    va, tr = perm[:5000], perm[5000:]
    f = lambda a: torch.tensor(a.astype(np.float32) / 255.0)
    g = lambda a: torch.tensor(a.astype(np.int64))
    return (f(x[tr]), g(y[tr]), f(x[va]), g(y[va]), f(xt), g(yt))


@torch.no_grad()
def evaluate(model, x, y, dev, bs=1000):
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(y), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        logits = model(xb)
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    model.train()
    return loss / len(y), correct / len(y)


def train(model, init_state, masks, data, *, lr, iters, dev, seed, batch=60, eval_every=100):
    xtr, ytr, xva, yva, xte, yte = data
    model.load_state_dict(init_state)
    with torch.no_grad():
        for (name, m, _), mask in zip(prunable(model), masks):
            m.weight.mul_(mask)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(ytr)
    best = {"val_loss": float("inf"), "iter": 0, "test_acc": 0.0}
    # ponytail: whole dataset lives on the GPU (~740MB of 16GB), so batch
    # assembly is a device-side gather. Host-side .to() per batch made the
    # sweep input-bound on Kaggle's T4 -- the one condition that decides
    # whether this fits the weekly quota.
    perm, pos = torch.randperm(n, generator=g).to(xtr.device), 0

    for t in range(1, iters + 1):
        if pos + batch > n:
            perm, pos = torch.randperm(n, generator=g).to(xtr.device), 0
        idx = perm[pos:pos + batch]; pos += batch
        xb, yb = xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        with torch.no_grad():                       # pruned weights get no gradient
            for (_, m, _), mask in zip(prunable(model), masks):
                m.weight.grad.mul_(mask)
        opt.step()
        with torch.no_grad():                       # ... and stay at zero
            for (_, m, _), mask in zip(prunable(model), masks):
                m.weight.mul_(mask)

        if t % eval_every == 0 or t == iters:
            vl, _ = evaluate(model, xva, yva, dev)
            if vl < best["val_loss"]:
                _, ta = evaluate(model, xte, yte, dev)
                best = {"val_loss": vl, "iter": t, "test_acc": ta}

    _, final_test = evaluate(model, xte, yte, dev)
    _, final_train = evaluate(model, xtr[:10000], ytr[:10000], dev)
    return best, {"test_acc": final_test, "train_acc": final_train}


def prune_masks(model, masks, conv_rate):
    """Layer-wise magnitude pruning of the surviving weights."""
    rates = {"conv": conv_rate, "fc": FC_RATE, "out": OUT_RATE}
    new = []
    for (name, m, kind), mask in zip(prunable(model), masks):
        w = m.weight.detach().abs()
        alive = mask.bool()
        k = int(round(rates[kind] * int(alive.sum())))
        if k <= 0:
            new.append(mask.clone()); continue
        cut = torch.kthvalue(w[alive].flatten(), k).values
        nm = mask.clone()
        nm[alive & (w <= cut)] = 0.0
        new.append(nm)
    return new


def pm_percent(model, masks):
    alive = sum(int(m.sum()) for m in masks)
    total = sum(mm.weight.numel() for _, mm, _ in prunable(model))
    return 100.0 * alive / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--data", default="/kaggle/input/cifar10-python")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-round", type=int, default=20)
    ap.add_argument("--iters", type=int, default=0, help="override (smoke tests only)")
    ap.add_argument("--reinit-at", type=int, nargs="*", default=None)
    args = ap.parse_args()

    spec, iters, lr, conv_rate = ARCHS[args.arch]
    if args.iters:
        iters = args.iters
    reinit_at = args.reinit_at if args.reinit_at is not None else sorted(CLAIM_ROUNDS[args.arch])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    t0 = time.time()
    data = tuple(v.to(dev) for v in load_cifar(args.data))
    model = ConvNet(spec).to(dev)
    glorot_normal_(model)
    n_params = sum(m.weight.numel() for _, m, _ in prunable(model))
    n_conv = sum(m.weight.numel() for _, m, kind in prunable(model) if kind == "conv")
    print(f"{args.arch}: {n_params:,} prunable weights, flat={model.flat}, dev={dev}", flush=True)

    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    masks = [torch.ones_like(m.weight) for _, m, _ in prunable(model)]
    lv: dict = {}
    out = Path(args.out)

    def flush():
        m: dict = {"_arch": args.arch, "_params": n_params, "_conv_params": n_conv,
                   "_config": {"seed": args.seed, "iters": iters, "lr": lr,
                               "conv_rate": conv_rate, "fc_rate": FC_RATE, "out_rate": OUT_RATE,
                               "device": str(dev), "wall_seconds": round(time.time() - t0, 1)},
                   "_levels": [lv[k] for k in sorted(lv)]}
        if 0 in lv:
            b = lv[0]["ticket"]
            for k, v in lv.items():
                tag = CLAIM_ROUNDS[args.arch].get(k)
                if not tag:
                    continue
                m[f"pm_percent_pm{tag}"] = round(v["pm"], 3)
                m[f"test_acc_delta_pp_pm{tag}"] = round(100 * (v["ticket"]["test_acc"] - b["test_acc"]), 3)
                if v["ticket"]["early_stop_iter"]:
                    m[f"speedup_vs_unpruned_pm{tag}"] = round(
                        b["early_stop_iter"] / v["ticket"]["early_stop_iter"], 3)
            m["max_test_acc_delta_pp"] = round(
                max(100 * (v["ticket"]["test_acc"] - b["test_acc"]) for v in lv.values()), 3)
            m["max_speedup_vs_unpruned"] = round(
                max(b["early_stop_iter"] / v["ticket"]["early_stop_iter"]
                    for v in lv.values() if v["ticket"]["early_stop_iter"]), 3)
            above = [v["pm"] for v in lv.values()
                     if v["ticket"]["test_acc"] >= b["test_acc"]]
            m["min_pm_still_above_baseline"] = round(min(above), 3) if above else None
            m["min_train_acc_final_pm_ge_2"] = round(
                100 * min(v["ticket"]["final_train_acc"] for v in lv.values() if v["pm"] >= 2.0), 3)
        out.write_text(json.dumps(m, indent=2))

    for k in range(args.max_round + 1):
        if k > 0:
            masks = prune_masks(model, masks, conv_rate)
        pm = pm_percent(model, masks)
        best, final = train(model, init_state, masks, data, lr=lr, iters=iters, dev=dev,
                            seed=args.seed * 1000 + k)
        lv[k] = {"round": k, "pm": pm, "ticket": {
            "early_stop_iter": best["iter"], "test_acc": best["test_acc"],
            "val_loss": best["val_loss"], "final_test_acc": final["test_acc"],
            "final_train_acc": final["train_acc"]}}
        print(f"[{time.time()-t0:7.0f}s] r{k:2d} Pm={pm:6.2f}% es@{best['iter']:6d} "
              f"test={100*best['test_acc']:.2f}% final={100*final['test_acc']:.2f}%", flush=True)
        flush()

        if k in reinit_at:
            rmodel = ConvNet(spec).to(dev)
            torch.manual_seed(args.seed + 90000 + k)
            glorot_normal_(rmodel)
            rstate = {kk: v.clone() for kk, v in rmodel.state_dict().items()}
            rb, rf = train(rmodel, rstate, masks, data, lr=lr, iters=iters, dev=dev,
                           seed=args.seed * 1000 + k)
            lv[k]["random"] = {"early_stop_iter": rb["iter"], "test_acc": rb["test_acc"],
                               "final_test_acc": rf["test_acc"], "final_train_acc": rf["train_acc"]}
            print(f"[{time.time()-t0:7.0f}s] r{k:2d} REINIT es@{rb['iter']:6d} "
                  f"test={100*rb['test_acc']:.2f}%", flush=True)
            flush()

    flush()
    print("done:", out)
    return 0


def _selfcheck() -> int:
    """One runnable check: the architectures must reproduce Figure 2's weight
    counts, and the pruning ladder must land on the Pm values the paper prints."""
    expect = {"conv2": 4300992, "conv4": 2425024, "conv6": 2261184}
    printed = {"conv2": {11: 8.8, 14: 4.6}, "conv4": {12: 9.2, 11: 11.1},
               "conv6": {10: 15.1, 7: 26.4}}
    for name, (spec, _, _, conv_rate) in ARCHS.items():
        model = ConvNet(spec)
        glorot_normal_(model)
        n = sum(m.weight.numel() for _, m, _ in prunable(model))
        assert n == expect[name], f"{name}: {n:,} != {expect[name]:,}"
        masks = [torch.ones_like(m.weight) for _, m, _ in prunable(model)]
        for k in range(1, 21):
            masks = prune_masks(model, masks, conv_rate)
            for (_, m, _), mask in zip(prunable(model), masks):
                assert set(mask.unique().tolist()) <= {0.0, 1.0}
            if k in printed[name]:
                pm = pm_percent(model, masks)
                assert abs(pm - printed[name][k]) < 0.35, f"{name} r{k}: {pm:.2f} vs {printed[name][k]}"
                print(f"{name} round {k:2d}: Pm={pm:6.2f}%  (paper prints {printed[name][k]}%)")
        if name == "conv6":
            print(f"{name}: {n:,} weights — Figure 2 states 1.7M, which this architecture "
                  f"does NOT produce. See the module docstring: this reading is the one that "
                  f"reproduces Figure 2 for Lenet/Conv-2/Conv-4 exactly and the printed Conv-6 "
                  f"Pm labels far better than any 1.7M reading.")
        else:
            print(f"{name}: {n:,} weights matches Figure 2")
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_selfcheck() if "--selfcheck" in sys.argv else main())
