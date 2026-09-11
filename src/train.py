# src/train.py
# BERT 微调：预训练主干 + 分类头，按验证集选最优 epoch，最后测一次测试集，导出预测
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data import ROOT, load_splits, make_loader


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    """跑一遍 loader，返回 (准确率, 预测列表, 好评概率列表)"""
    model.eval()
    preds, probs, labels = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits  # (B, 2)
        p = torch.softmax(logits.float(), dim=-1)[:, 1]  # 判为"好评"的概率
        preds += logits.argmax(-1).tolist()
        probs += p.tolist()
        labels += batch["labels"].tolist()
    model.train()
    acc = sum(int(a == b) for a, b in zip(preds, labels)) / len(labels)
    return acc, preds, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-chinese")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n_train", type=int, default=0, help="只用前 n 条训练，0 = 全量（调试用）"
    )
    p.add_argument("--run", default="bert_base")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    use_bf16 = device.type == "cuda"
    print("device:", device, " bf16:", use_bf16)

    # 数据
    tok = AutoTokenizer.from_pretrained(args.model)
    train, val, test = load_splits(args.seed)
    if args.n_train:
        train = train.iloc[: args.n_train]
    tr_loader = make_loader(train, tok, args.max_len, args.batch, shuffle=True)
    va_loader = make_loader(val, tok, args.max_len, 64, shuffle=False)
    te_loader = make_loader(test, tok, args.max_len, 64, shuffle=False)

    # 模型：预训练主干 + 随机初始化的 768→2 分类头
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    ).to(device)

    # 优化器：AdamW，LayerNorm 和 bias 不做 weight decay
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {
            "params": [
                q
                for n, q in model.named_parameters()
                if not any(k in n for k in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                q for n, q in model.named_parameters() if any(k in n for k in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    total_steps = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    # 训练
    best_val, best_state, history = 0.0, None, []
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0, running = time.time(), 0.0
        for step, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                out = model(**batch)  # 前向：有 labels 就自动算交叉熵
            out.loss.backward()  # 反向：算所有参数的梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += out.loss.item()
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(tr_loader)} loss {running / 50:.4f}"
                )
                running = 0.0
        val_acc, _, _ = evaluate(model, va_loader, device)
        history.append(
            {"epoch": epoch, "val_acc": val_acc, "sec": round(time.time() - t0)}
        )
        print(f"== epoch {epoch} val_acc {val_acc:.4f} ({history[-1]['sec']}s)")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # 用验证集最优的参数，测一次验证集和测试集，并导出预测
    model.load_state_dict(best_state)
    val_acc, va_pred, va_prob = evaluate(model, va_loader, device)
    test_acc, te_pred, te_prob = evaluate(model, te_loader, device)
    print(f"best val {val_acc:.4f}  test {test_acc:.4f}")

    Path(ROOT / "results").mkdir(exist_ok=True)
    val.assign(pred=va_pred, p_pos=va_prob).to_csv(
        ROOT / f"results/{args.run}_val_pred.csv", index=False
    )
    test.assign(pred=te_pred, p_pos=te_prob).to_csv(
        ROOT / f"results/{args.run}_test_pred.csv", index=False
    )
    json.dump(
        {
            "run": args.run,
            "args": vars(args),
            "history": history,
            "best_val_acc": val_acc,
            "test_acc": test_acc,
        },
        open(ROOT / f"results/{args.run}.json", "w"),
        ensure_ascii=False,
        indent=2,
    )
    print(
        "saved:",
        f"results/{args.run}.json",
        f"results/{args.run}_val_pred.csv",
        f"results/{args.run}_test_pred.csv",
    )


if __name__ == "__main__":
    main()
