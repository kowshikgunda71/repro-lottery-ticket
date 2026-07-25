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

# A kernel can request `enable_gpu` and still be scheduled on the CPU-only image
# (whose torch has no CUDA) when the account may not attach accelerators. Fail on
# line one with the actual reason rather than 12 h later, or -- worse -- silently
# producing CPU timings that would mis-size the whole sweep.
GPU_ASSERT = """import torch
print('torch', torch.__version__)
assert torch.cuda.is_available(), (
    'NO ACCELERATOR ATTACHED: this kernel requested a GPU but was scheduled on the '
    'CPU image. Kaggle gates GPU/TPU behind phone verification -- kaggle.com/settings.')
print(torch.cuda.get_device_name(0))"""


def notebook(arch: str) -> dict:
    if arch == "calib":
        return calib_notebook()
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
        ("code", GPU_ASSERT + "\nimport os; print(os.listdir('/kaggle/input'))"),
        ("code", "%%writefile imp_conv.py\n" + HARNESS),
        ("code", "!python imp_conv.py --selfcheck"),
        ("code", runs),
        ("code", "import json, glob\n"
                 "for f in sorted(glob.glob('/kaggle/working/metrics-*.json')):\n"
                 "    d = json.load(open(f))\n"
                 "    print(f, d['_config']['wall_seconds']/3600, 'h,', len(d['_levels']), 'levels')"),
    ]
    return _nb(cells)


def calib_notebook() -> dict:
    """Cheap timing kernel. The weekly GPU quota is 30 h and a session is capped
    at 12 h, so the sweep must be sized before any of it is committed -- this
    measures ms/iteration on the actual T4 and projects each architecture."""
    bench = "\n".join(
        f'!python imp_conv.py --arch {a} --benchmark 2000 --max-round {MAX_ROUND[a]} '
        f'--data /kaggle/input/cifar10-python'
        for a in ("conv2", "conv4", "conv6"))
    cells = [
        ("markdown", """# LTH CIFAR replication — GPU calibration

Times 2,000 training iterations of Conv-2/4/6 on this session's GPU and projects
the full iterative-magnitude-pruning sweep. No claims are scored here; this
exists so the real runs can be sized against the weekly quota. Minutes, not
hours."""),
        ("code", GPU_ASSERT),
        ("code", "%%writefile imp_conv.py\n" + HARNESS),
        ("code", "!python imp_conv.py --selfcheck"),
        ("code", bench),
    ]
    return _nb(cells)


def _nb(cells) -> dict:
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
    for arch in ("calib", "conv2", "conv4", "conv6"):
        d = outdir / arch
        d.mkdir(parents=True, exist_ok=True)
        (d / "notebook.ipynb").write_text(json.dumps(notebook(arch), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/lth-{arch}-cifar-replication",
            "title": f"LTH {arch} CIFAR replication",
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            # Private until the operator approves publishing. Running compute and
            # publishing the harness are separate decisions; flip this only with the
            # pre-registration push.
            "is_private": "true",
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
