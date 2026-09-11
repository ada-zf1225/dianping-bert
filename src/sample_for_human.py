# src/sample_for_human.py
# 从验证集预测里随机抽 50 条，隐藏标签导出给人盲标，答案另存
import pandas as pd
from src.data import ROOT

df = pd.read_csv(ROOT / "results/bert_base_s42_val_pred.csv")
sample = df.sample(50, random_state=7)[
    ["sentence", "label", "pred", "p_pos"]
].reset_index(drop=True)
sample[["sentence"]].to_csv(ROOT / "results/human_blind.csv", index_label="id")
sample.to_csv(ROOT / "results/human_answer.csv", index_label="id")
print(
    "已导出 results/human_blind.csv（50 条，无标签）和 results/human_answer.csv（答案，先别看）"
)
