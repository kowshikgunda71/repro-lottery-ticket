"""Generate the Kaggle batch notebooks that run the CIFAR sweeps.

Usage: python make_kernels.py <kaggle-username> [outdir]

One kernel per architecture (each gets its own 12 h session, and the sweep is
inherently sequential so there is nothing to gain from packing them together).
The harness is embedded in the notebook because Kaggle notebooks have NO
internet by default -- it is gated behind phone verification -- so nothing may
be cloned or downloaded at run time. CIFAR-10 arrives as an attached dataset
for the same reason.

Push with:  kaggle kernels push -p <outdir>/<arch>
Poll with:  kaggle kernels status <user>/<slug>
Fetch with: kaggle kernels output <user>/<slug> -p <dest>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = (HERE / "imp_conv.py").read_text()
SEEDS = [0, 1, 2]
# Round at which each architecture's ladder passes Pm = 2%, the depth claim C7
# needs. Arithmetic, computed from the per-layer rates -- see imp_conv.py.
MAX_ROUND = {"conv2": 18, "conv4": 22, "conv6": 22}


def notebook(arch: str) -> dict:
    runs = "\n".join(
        f'!python imp_conv.py --arch {arch} --seed {s} --max-round {MAX_ROUND[arch]} \\\n'
        f'    --data /kaggle/input/cifar10-python --out /kaggle/working/metrics-{arch}-seed{s}.json'
        for s in SEEDS)
    cells = [
        ("markdown", f"""# Lottery Ticket Hypothesis — {arch} replication (CIFAR-10)

Independent replication of Section 3 of Frankle & Carbin, *The Lottery Ticket
Hypothesis* (ICLR 2019, [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)).

Claims and tolerances were registered **before** this ran. Harness written from
the paper's text; no author code is used. Outputs `metrics-{arch}-seed*.json`,
scored offline against the pre-registered claims.

CIFAR-10 is the attached `pankrzysiu/cifar10-python` dataset (Krizhevsky 2009);
it is read here and never redistributed."""),
        ("code", f"import torch, os\n"
                 f"print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),\n"
                 f"      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\n"
                 f"print(os.listdir('/kaggle/input'))"),
        ("code", "%%writefile imp_conv.py\n" + HARNESS),
        ("code", "!python imp_conv.py --selfcheck"),
        ("code", runs),
        ("code", "import json, glob\n"
                 "for f in sorted(glob.glob('/kaggle/working/metrics-*.json')):\n"
                 "    d = json.load(open(f))\n"
                 "    print(f, d['_config']['wall_seconds']/3600, 'h,', len(d['_levels']), 'levels')"),
    ]
    return {
        "cells": [{"cell_type": t, "metadata": {},
                   "source": s.splitlines(keepends=True),
                   **({"outputs": [], "execution_count": None} if t == "code" else {})}
                  for t, s in cells],
        "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    user = sys.argv[1]
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else HERE / "kernels")
    for arch in ("conv2", "conv4", "conv6"):
        d = outdir / arch
        d.mkdir(parents=True, exist_ok=True)
        (d / "notebook.ipynb").write_text(json.dumps(notebook(arch), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/lth-{arch}-cifar-replication",
            "title": f"LTH {arch} CIFAR replication",
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "false",
            "enable_gpu": "true",
            "enable_internet": "false",          # no internet: data is attached
            "dataset_sources": ["pankrzysiu/cifar10-python"],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=2))
        print(f"{d}  ->  kaggle kernels push -p {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
