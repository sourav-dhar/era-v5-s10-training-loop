"""Execute training_loop.ipynb headlessly.

    python tools/run_notebook.py --profile quick                # CPU smoke test of every cell
    python tools/run_notebook.py --profile full                 # the reported GPU run
    python tools/run_notebook.py --profile mfu --tags core,mfu  # only the cells tagged core/mfu

Markdown cells are kept only when a full run is requested; with --tags, only matching code cells run.
"""
import argparse
import os
import time
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def run(nb_in: Path, nb_out: Path, profile: str, tags=None, timeout=3600, cwd: Path = None):
    nb = nbf.read(nb_in, as_version=4)
    if tags:
        keep = set(tags)
        # a cell runs only if every one of its tags was requested ([core,grad] is skipped by --tags core,mfu)
        nb.cells = [c for c in nb.cells if c.cell_type == "code" and c.metadata.get("tags") and set(c.metadata["tags"]) <= keep]
    os.environ["S10_PROFILE"] = profile
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    t0 = time.time()
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(cwd or nb_in.parent)}})
    finally:
        nbf.write(nb, nb_out)
        print(f"wrote {nb_out} ({time.time() - t0:.0f}s)")
    return nb_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--tags", default="")
    ap.add_argument("--inp", default="training_loop.ipynb")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    inp = root / a.inp
    out = Path(a.out) if a.out else root / f"training_loop.{a.profile}.ipynb"
    run(inp, out, a.profile, [t for t in a.tags.split(",") if t], a.timeout, root)
