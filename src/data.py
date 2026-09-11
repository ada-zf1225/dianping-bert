# src/data.py
# 数据层：读数据、切验证集、Dataset、动态 padding 的 DataLoader
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
TEXT, LABEL = "sentence", "label"


def load_splits(seed: int = 42):
    """返回 train / val / test 三个 DataFrame。切分规则和基线完全一致。"""
    df = pd.read_parquet(ROOT / "data/train.parquet")
    test = pd.read_parquet(ROOT / "data/dev.parquet")
    train, val = train_test_split(
        df, test_size=0.1, stratify=df[LABEL], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test


class ReviewDataset(Dataset):
    """只存原始文本和标签，编码放到 collate 里做（一批一批编，才能动态 padding）"""

    def __init__(self, df: pd.DataFrame):
        self.texts = df[TEXT].astype(str).tolist()
        self.labels = df[LABEL].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tok, max_len: int):
    """返回一个函数：把 [(文本, 标签), ...] 变成一个 batch 的张量字典"""

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels)
        return enc

    return collate


def make_loader(df, tok, max_len: int, batch_size: int, shuffle: bool):
    return DataLoader(
        ReviewDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(tok, max_len),
    )
