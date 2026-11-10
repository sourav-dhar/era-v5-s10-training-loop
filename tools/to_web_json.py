"""Copy results/*.json to docs/data/ as strict JSON: NaN/Infinity (valid for Python, invalid for browsers) become null."""
import json, math, sys
from pathlib import Path


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob("*.json")):
    (dst / p.name).write_text(json.dumps(clean(json.loads(p.read_text(encoding="utf-8"))), allow_nan=False, ensure_ascii=False), encoding="utf-8")
    print("web json:", p.name)
