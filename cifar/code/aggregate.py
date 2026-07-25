"""Combine per-architecture, per-seed CIFAR runs into the single metrics.json
that `gym import-run` scores against the pre-registered claims.

Usage: python aggregate.py out.json metrics-conv2-seed0.json [more...]

Cross-seed statistics are means, matching how the paper reports (it averages
five trials). A claim metric whose inputs are missing is simply absent from the
output, so the gym scores it INCONCLUSIVE -- never a silent pass or fail.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

runs = defaultdict(list)                      # arch -> [metrics dict per seed]
for p in sys.argv[2:]:
    d = json.loads(Path(p).read_text())
    runs[d["_arch"]].append(d)

out, prov = {}, {}


def mean_over_seeds(arch, fn, label):
    vals = [v for v in (fn(d) for d in runs.get(arch, [])) if v is not None]
    if not vals:
        return None
    prov[label] = {"arch": arch, "seeds": len(vals), "values": [round(v, 4) for v in vals]}
    return st.mean(vals)


# --- structural claims: deterministic, identical across seeds -----------------
for arch, key in (("conv2", "conv2_total_weights"), ("conv4", "conv4_total_weights"),
                  ("conv6", "conv6_total_weights")):
    v = mean_over_seeds(arch, lambda d: d.get("_params"), key)
    if v is not None:
        out[key] = v
v = mean_over_seeds("conv6", lambda d: d.get("_conv_params"), "conv6_conv_weights")
if v is not None:
    out["conv6_conv_weights"] = v

for arch, tag, key in (("conv2", "8_8", "conv2_pm_percent_round11"),
                       ("conv4", "9_2", "conv4_pm_percent_round12"),
                       ("conv6", "15_1", "conv6_pm_percent_round10")):
    v = mean_over_seeds(arch, lambda d, t=tag: d.get(f"pm_percent_pm{t}"), key)
    if v is not None:
        out[key] = v

# --- empirical claims ---------------------------------------------------------
for arch in ("conv2", "conv4", "conv6"):
    v = mean_over_seeds(arch, lambda d: d.get("max_speedup_vs_unpruned"), f"{arch}_max_early_stop_speedup")
    if v is not None:
        out[f"{arch}_max_early_stop_speedup"] = v
    v = mean_over_seeds(arch, lambda d: d.get("max_test_acc_delta_pp"), f"{arch}_max_test_acc_delta_pp")
    if v is not None:
        out[f"{arch}_max_test_acc_delta_pp"] = v

# "All three networks remain above their original average test accuracy when
# Pm > 2%" -- the binding case is the architecture that crosses back earliest.
crossovers = [mean_over_seeds(a, lambda d: d.get("min_pm_still_above_baseline"), f"_cross_{a}")
              for a in ("conv2", "conv4", "conv6")]
crossovers = [c for c in crossovers if c is not None]
if len(crossovers) == 3:
    out["worst_arch_accuracy_crossover_pm_percent"] = max(crossovers)

# "training accuracy reaches 100% for all networks when Pm >= 2%"
trains = [mean_over_seeds(a, lambda d: d.get("min_train_acc_final_pm_ge_2"), f"_train_{a}")
          for a in ("conv2", "conv4", "conv6")]
trains = [t for t in trains if t is not None]
if len(trains) == 3:
    out["min_final_train_acc_pct_above_2pct"] = min(trains)

# "the winning ticket beats the same mask randomly reinitialised" at the
# sparsest rung tested per architecture -- counted across architectures.
def beats_reinit(d):
    lv = [l for l in d["_levels"] if "random" in l]
    if not lv:
        return None
    sparsest = min(lv, key=lambda l: l["pm"])
    return 1.0 if sparsest["ticket"]["test_acc"] > sparsest["random"]["test_acc"] else 0.0

wins = [mean_over_seeds(a, beats_reinit, f"_beats_{a}") for a in ("conv2", "conv4", "conv6")]
wins = [w for w in wins if w is not None]
if len(wins) == 3:
    # an architecture counts as a win when it holds on a majority of its seeds
    out["num_archs_ticket_beats_reinit_at_sparsest_rung"] = float(sum(w > 0.5 for w in wins))

out["_provenance"] = prov
out["_runs"] = {a: len(v) for a, v in runs.items()}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if not k.startswith("_")}, indent=2))
print(f"\n-> {sys.argv[1]}  ({sum(len(v) for v in runs.values())} runs across {len(runs)} architectures)")
