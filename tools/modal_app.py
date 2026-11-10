"""Run the notebook on Modal GPUs and pull the executed notebook, figures and results back into the repo.

    modal run tools/modal_app.py                 # A100 MFU pass, then the full T4 run
    modal run tools/modal_app.py --which t4      # only the full T4 run
    modal run tools/modal_app.py --which a100    # only the A100 MFU pass
"""
import io
import sys
import tarfile
import traceback
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("era-v5-s10-training-loop")
vol = modal.Volume.from_name("era-v5-s10-runs", create_if_missing=True)   # results survive a dropped local client
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")                       # torch.compile / Triton needs a C compiler
    .pip_install("torch==2.5.1", "numpy<2.3", "pandas", "matplotlib", "jinja2",
                 "nbformat", "nbconvert", "ipykernel", "matplotlib-inline")
    .add_local_dir(ROOT, "/repo", ignore=[".git", "data", "figures", "results", "docs", "**/__pycache__",
                                          "training_loop.*.ipynb", "*.png"])
)


def _execute(profile, tags, extra_files, save_as=None):
    import os
    os.chdir("/repo")
    for d in ("figures", "results"):
        Path(d).mkdir(exist_ok=True)
    for rel, blob in (extra_files or {}).items():
        Path(rel).write_bytes(blob)
    sys.path.insert(0, "/repo/tools")
    from run_notebook import run
    out = Path(f"/repo/training_loop.{profile}.ipynb")
    err = None
    try:
        run(Path("/repo/training_loop.ipynb"), out, profile, tags, timeout=7200, cwd=Path("/repo"))
    except Exception:
        err = traceback.format_exc()[-4000:]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in ["figures", "results", out.name]:
            if Path(p).exists():
                tar.add(p)
    if save_as:   # fetch later with: modal volume get era-v5-s10-runs <save_as>
        Path(f"/out/{save_as}").write_bytes(buf.getvalue())
        if err:
            Path(f"/out/{save_as}.err.txt").write_text(err)
        vol.commit()
    return buf.getvalue(), err


@app.function(image=image, gpu="T4", cpu=8.0, memory=16384, timeout=3 * 3600, volumes={"/out": vol})
def run_t4(extra_files: dict = None):
    return _execute("full", None, extra_files, save_as="t4.tar.gz")


@app.function(image=image, gpu="A100", cpu=8.0, memory=16384, timeout=3600, volumes={"/out": vol})
def run_a100_mfu():
    return _execute("mfu", ["core", "mfu"], None, save_as="a100.tar.gz")


def _unpack(blob, dest: Path):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


@app.local_entrypoint()
def main(which: str = "both"):
    staging = ROOT / "runs"
    extra = {}
    if which in ("both", "a100"):
        blob, err = run_a100_mfu.remote()
        _unpack(blob, staging / "a100")
        print("A100 MFU pass:", "FAILED\n" + err if err else "ok")
        mfu = staging / "a100" / "results" / "mfu_A100.json"
        if mfu.exists():
            extra["results/mfu_A100.json"] = mfu.read_bytes()
    if which in ("both", "t4"):
        prev = ROOT / "results" / "mfu_A100.json"
        if "results/mfu_A100.json" not in extra and prev.exists():
            extra["results/mfu_A100.json"] = prev.read_bytes()
        blob, err = run_t4.remote(extra)
        _unpack(blob, staging / "t4")
        print("T4 full run:", "FAILED\n" + err if err else "ok")
