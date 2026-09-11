# src/check_loader.py
# 验证 data.py：取一个 batch 看形状和内容
from transformers import AutoTokenizer

from src.data import load_splits, make_loader

tok = AutoTokenizer.from_pretrained("bert-base-chinese")
train, val, test = load_splits()
print("train/val/test:", len(train), len(val), len(test))
print("val 标签比例:", val["label"].mean().round(3))

loader = make_loader(train, tok, max_len=256, batch_size=4, shuffle=True)
print("batch 数:", len(loader))

batch = next(iter(loader))
for k, v in batch.items():
    print(f"{k:16s} {tuple(v.shape)}")
print("第 0 条 input_ids 前 12 个:", batch["input_ids"][0, :12].tolist())
print("第 0 条 mask 之和（真实 token 数）:", batch["attention_mask"][0].sum().item())
print("labels:", batch["labels"].tolist())
