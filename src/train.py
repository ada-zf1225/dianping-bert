# src/train.py
# BERT 微调：加载预训练主干 + 分类头，训练若干 epoch，按验证集选最优，最后测一次测试集
import argparse, json, time, random
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from src.data import load_splits, make_loader, ROOT


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
    """跑一遍 loader，返回准确率。不算梯度。"""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits  # (B, 2)
        pred = logits.argmax(dim=-1)  # 取概率大的那一类
        correct += (pred == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    model.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-chinese")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n_train", type=int, default=0, help="只用前 n 条训练，0 = 全量（调试用）"
    )
    p.add_argument("--run", default="bert_base")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print("device:", device)

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
                p
                for n, p in model.named_parameters()
                if not any(k in n for k in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(k in n for k in no_decay)
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
        val_acc = evaluate(model, va_loader, device)
        history.append(
            {"epoch": epoch, "val_acc": val_acc, "sec": round(time.time() - t0)}
        )
        print(f"== epoch {epoch} val_acc {val_acc:.4f} ({history[-1]['sec']}s)")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # 用验证集最优的那个 epoch 的参数测一次测试集
    model.load_state_dict(best_state)
    test_acc = evaluate(model, te_loader, device)
    print(f"best val {best_val:.4f}  test {test_acc:.4f}")

    Path(ROOT / "results").mkdir(exist_ok=True)
    json.dump(
        {
            "run": args.run,
            "args": vars(args),
            "history": history,
            "best_val_acc": best_val,
            "test_acc": test_acc,
        },
        open(ROOT / f"results/{args.run}.json", "w"),
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    main()
