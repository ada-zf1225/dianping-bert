import pandas as pd
import pytest
import torch
from transformers import BertTokenizerFast

from src.data import LABEL, TEXT, Collate, ReviewDataset, make_loader


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "菜", "好", "吃", "难"]
    p = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    p.write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(p))


@pytest.fixture
def df():
    return pd.DataFrame({TEXT: ["菜好吃", "菜难吃", "好吃好吃好吃", "难吃"], LABEL: [1, 0, 1, 0]})


def test_dataset_len_and_getitem(df):
    ds = ReviewDataset(df)
    assert len(ds) == 4
    text, label = ds[1]
    assert text == "菜难吃" and label == 0


def test_collate_shapes_and_padding(tok, df):
    batch = Collate(tok, max_len=16)(list(ReviewDataset(df)))
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    assert ids.shape == mask.shape
    assert ids.shape[0] == 4 and labels.shape == (4,)
    assert ids.shape[1] == 8          # 动态 padding：最长 6 字 + CLS + SEP
    assert (ids[:, 0] == 2).all()     # [CLS] = 2
    assert mask[3].sum().item() == 4  # "难吃" 2 字 + 2 特殊 token
    assert (ids[3, 4:] == 0).all()    # PAD = 0


def test_truncation(tok, df):
    batch = Collate(tok, max_len=4)(list(ReviewDataset(df)))
    assert batch["input_ids"].shape[1] == 4
    assert (batch["input_ids"][:, -1] == 3).all()  # 截断后仍以 [SEP]=3 结尾


def test_loader_iterates(tok, df):
    loader = make_loader(df, tok, max_len=16, batch_size=2, shuffle=False, num_workers=0)
    batches = list(loader)
    assert len(batches) == 2
    assert torch.equal(batches[0]["labels"], torch.tensor([1, 0]))
