# src/stats.py
import pandas as pd

TEXT = "sentence"  # ← 改
LABEL = "label"  # ← 改

tr = pd.read_parquet("data/train.parquet")
te = pd.read_parquet("data/dev.parquet")
print("train:", tr.shape, " dev:", te.shape)

print("\n=== 1. 标签分布（比例）===")
print(tr[LABEL].value_counts(normalize=True).round(4))

print("\n=== 2. 文本长度（字符数）分位数 ===")
lens = tr[TEXT].astype(str).str.len()
print(lens.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).round(1))

print("\n=== 3. 脏数据 ===")
print("重复文本:", tr.duplicated(subset=[TEXT]).sum())
print("空文本  :", (lens == 0).sum())
print("train/dev 重叠:", len(set(tr[TEXT]) & set(te[TEXT])))

print("\n=== 4. 肉眼看样本 ===")
for lab in sorted(tr[LABEL].unique()):
    print(f"\n--- label = {lab} ---")
    for t in tr[tr[LABEL] == lab][TEXT].sample(5, random_state=0):
        print("-", str(t)[:150].replace("\n", " "))
