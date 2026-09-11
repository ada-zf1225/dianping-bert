# src/confidence.py
# 模型置信度和准确率的关系：p_pos 越接近 0.5 的样本，是不是越判不准
import pandas as pd

from src.data import ROOT

df = pd.read_csv(ROOT / "results/bert_base_s42_val_pred.csv")
df["conf"] = (df.p_pos - 0.5).abs() * 2  # 0 = 完全不确定，1 = 完全确定
df["correct"] = (df.pred == df.label).astype(int)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
df["bin"] = pd.cut(df.conf, bins, include_lowest=True)
tab = df.groupby("bin", observed=True).agg(
    n=("correct", "size"), acc=("correct", "mean")
)
tab["share"] = tab.n / tab.n.sum()
print(tab.round(3))
low = df[df.conf < 0.4]
print(
    f"\n低置信（conf<0.4）样本占 {len(low) / len(df):.1%}，其中准确率 {low.correct.mean():.1%}"
)
print(
    f"高置信（conf>=0.8）样本占 {(df.conf >= 0.8).mean():.1%}，其中准确率 {df[df.conf >= 0.8].correct.mean():.1%}"
)
tab.to_csv(ROOT / "results/confidence_bins.csv")
