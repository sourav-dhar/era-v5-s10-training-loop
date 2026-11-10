"""Build training_loop.ipynb from tools/notebook_cells.md.

Cell markers (each on its own line):
    %%md            -> markdown cell
    %%code [tags]   -> code cell, tags are comma separated (e.g. core, shapes, mfu)

Tags let the Modal runner execute only some sections (e.g. the MFU-only A100 run).
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "notebook_cells.md"
OUT = ROOT / "training_loop.ipynb"
MARK = re.compile(r"^%%(md|code)(?:\s+\[([^\]]*)\])?\s*$")


def build(src: Path = SRC, out: Path = OUT) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    kind, tags, buf = None, [], []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body:
            return
        cell = nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body)
        if tags:
            cell.metadata["tags"] = tags
        nb.cells.append(cell)

    for line in src.read_text(encoding="utf-8").splitlines():
        m = MARK.match(line)
        if m:
            flush()
            kind = m.group(1)
            tags = [t.strip() for t in (m.group(2) or "").split(",") if t.strip()]
            buf = []
        else:
            buf.append(line)
    flush()
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    print(build(*(Path(a) for a in sys.argv[1:])))
