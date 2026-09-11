# src/aggregate.py
# 读 results/*.json，输出 Markdown 结果表
import json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from src.data import ROOT

rows = []
for f in sorted(glob.glob(str(ROOT / "results/*.json"))):
    r = json.load(open(f))
    if "test_acc" not in r:
        continue
    rows.append(
        {
            "run": r.get("run") or Path(f).stem,
            "val": r.get("best_val_acc", r.get("val_acc")),
            "test": r["test_acc"],
            "model": r.get("args", {}).get("model", "tfidf"),
            "max_len": r.get("args", {}).get("max_len", "-"),
            "seed": r.get("args", {}).get("seed", "-"),
        }
    )
df = pd.DataFrame(rows)
print(df.to_markdown(index=False, floatfmt=".4f"))

b = df[df.run.str.startswith("bert_base_s")]
if len(b) >= 2:
    print(
        f"\nbert-base 3 seeds: test {b.test.mean():.4f} ± {b.test.std():.4f}  (n={len(b)})"
    )
df.to_csv(ROOT / "results/summary.csv", index=False)
