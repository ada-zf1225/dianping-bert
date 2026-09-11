# src/plots.py
# 从 results/ 生成 README 里的全部图片，输出到 docs/figs/
import glob
import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import ROOT

OUT = ROOT / "docs/figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Noto Sans CJK JP",
                               "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 150
C_BASE, C_BERT, C_REF, C_GREY = "#8c8c8c", "#2f6db5", "#d1495b", "#cccccc"


def load_runs():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/*.json"))):
        r = json.load(open(f))
        if "test_acc" not in r or "history" not in r:
            continue
        a = r["args"]
        rows.append({"run": r["run"], "model": a["model"], "max_len": a["max_len"], "seed": a["seed"],
                     "val": r["best_val_acc"], "test": r["test_acc"],
                     "hist": [h["val_acc"] for h in r["history"]]})
    return pd.DataFrame(rows)


# ---------- 图 1：主结果 ----------
def fig_main(runs):
    grid = json.load(open(ROOT / "results/baseline_tfidf_grid.json"))
    best = max(grid, key=lambda r: r["val_acc"])
    base = best["test_acc"]
    b = runs[runs.run.str.startswith("bert_base_s")]
    rob = runs[runs.run.str.startswith("roberta")].test.iloc[0]
    l512 = runs[runs.run == "bert_base_len512"].test.iloc[0]
    ref = 0.7869

    names = ["TF-IDF + LR\n(基线)", "BERT-base\n(3 seed 均值)", "RoBERTa-wwm", "BERT-base\nmax_len 512"]
    vals = [base, b.test.mean(), rob, l512]
    errs = [0, b.test.std(), 0, 0]
    colors = [C_BASE, C_BERT, C_BERT, C_BERT]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(names, [v * 100 for v in vals], yerr=[e * 100 for e in errs], capsize=6,
                  color=colors, width=0.6, edgecolor="white")
    ax.axhline(ref * 100, color=C_REF, ls="--", lw=1.5)
    ax.text(3.35, ref * 100 + 0.15, f"达摩院 StructBERT 参照 {ref*100:.1f}%", color=C_REF, ha="right", fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 0.25, f"{v*100:.1f}", ha="center", fontsize=10)
    ax.set_ylim(70, 81)
    ax.set_ylabel("测试集准确率 (%)")
    ax.set_title("所有模型都停在 77–78%", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "main_results.png")
    plt.close(fig)


# ---------- 图 2：基线网格热力图 ----------
def fig_grid():
    grid = json.load(open(ROOT / "results/baseline_tfidf_grid.json"))
    df = pd.DataFrame(grid)
    df["ngram"] = df.ngram.apply(lambda x: f"(1,{x[1]})")
    piv = df.pivot(index="ngram", columns="C", values="val_acc") * 100
    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(piv.values, cmap="Blues", vmin=71, vmax=75)
    ax.set_xticks(range(len(piv.columns)), [f"C={c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i}-gram" for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 73.8 else "black", fontsize=10)
    ax.set_title("TF-IDF + LR 网格搜索（验证集准确率 %）", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_grid.png")
    plt.close(fig)


# ---------- 图 3：训练曲线（各 run 的验证集准确率随 epoch） ----------
def fig_curves(runs):
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, r in runs.iterrows():
        ep = range(1, len(r["hist"]) + 1)
        style = "-" if r.model.startswith("bert") and r.max_len == 256 else "--"
        ax.plot(ep, [h * 100 for h in r["hist"]], style, marker="o", lw=1.6, label=r.run)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("epoch")
    ax.set_ylabel("验证集准确率 (%)")
    ax.set_ylim(75, 79)
    ax.set_title("第 1 个 epoch 之后验证集几乎不再变化", fontsize=12)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "val_curves.png")
    plt.close(fig)


# ---------- 图 4：置信度分档 ----------
def fig_confidence():
    df = pd.read_csv(ROOT / "results/confidence_bins.csv")
    labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.bar(labels, df.share * 100, color=C_GREY, width=0.6, label="样本占比")
    ax1.set_ylabel("样本占比 (%)", color="#666")
    ax1.set_xlabel("模型置信度  |p − 0.5| × 2")
    ax2 = ax1.twinx()
    ax2.plot(labels, df.acc * 100, "o-", color=C_BERT, lw=2, ms=7, label="该区间准确率")
    ax2.axhline(50, color=C_REF, ls=":", lw=1.2)
    ax2.text(4.4, 51, "随机水平 50%", color=C_REF, ha="right", fontsize=9)
    for x, a in zip(labels, df.acc):
        ax2.text(x, a * 100 + 1.5, f"{a*100:.0f}%", ha="center", color=C_BERT, fontsize=10)
    ax2.set_ylim(40, 100)
    ax2.set_ylabel("准确率 (%)", color=C_BERT)
    ax1.set_title("模型越不确定的样本，判得越接近抛硬币", fontsize=12, pad=10)
    for a in (ax1, ax2):
        a.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "confidence_bins.png")
    plt.close(fig)


# ---------- 图 5：三方一致率 ----------
def fig_agreement():
    ans = pd.read_csv(ROOT / "results/human_answer.csv")
    hum = pd.read_csv(ROOT / "results/human_blind_done.csv")
    m = ans.merge(hum[["id", "human"] + (["unsure"] if "unsure" in hum.columns else [])], on="id")
    a_hl = (m.human == m.label).mean()
    a_bl = (m.pred == m.label).mean()
    a_hb = (m.human == m.pred).mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = ["标注者 vs 标签", "BERT vs 标签", "标注者 vs BERT"]
    vals = [a_hl, a_bl, a_hb]
    bars = ax.barh(names, [v * 100 for v in vals], color=[C_GREY, C_GREY, C_BERT], height=0.55)
    for bar, v in zip(bars, vals):
        ax.text(v * 100 + 1, bar.get_y() + bar.get_height() / 2, f"{v*100:.0f}%", va="center", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_xlabel("一致率 (%)，50 条盲标样本")
    ax.invert_yaxis()
    ax.set_title("标注者和模型彼此一致，却都和标签不一致", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "agreement.png")
    plt.close(fig)


if __name__ == "__main__":
    runs = load_runs()
    fig_main(runs)
    fig_grid()
    fig_curves(runs)
    fig_confidence()
    fig_agreement()
    print("saved to", OUT, sorted(p.name for p in OUT.iterdir()))
