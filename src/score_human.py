# src/score_human.py
# 三个一致率 + 人和标签不一致的样本
import pandas as pd
from src.data import ROOT

human = pd.read_csv(ROOT / "results/human_blind_done.csv")
ans = pd.read_csv(ROOT / "results/human_answer.csv")
m = ans.merge(human[["id", "human"]], on="id")

print(f"人   vs 标签  一致率: {(m.human == m.label).mean():.1%}")
print(f"BERT vs 标签  一致率: {(m.pred == m.label).mean():.1%}")
print(f"人   vs BERT  一致率: {(m.human == m.pred).mean():.1%}")
print(f"\n人和标签不一致的样本（{(m.human != m.label).sum()} 条）：")
for _, r in m[m.human != m.label].iterrows():
    print(
        f"[label={r.label} human={r.human} bert={r.pred} p_pos={r.p_pos:.2f}] {r.sentence[:120]}"
    )
