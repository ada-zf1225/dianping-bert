# src/explore_bert.py
# 看 tokenizer 把一句话变成什么，看 BERT 输出什么形状
import torch
from transformers import AutoTokenizer, AutoModel

name = "bert-base-chinese"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

s = "菜不好吃，就是上菜的速度超慢"

# 1. 分词：句子 → token 列表
tokens = tok.tokenize(s)
print("tokens:", tokens)
print("token 数:", len(tokens))

# 2. 编码：token → 编号，并加上 [CLS] 和 [SEP]
enc = tok(s, return_tensors="pt")
print("input_ids:", enc["input_ids"])
print("还原:", tok.convert_ids_to_tokens(enc["input_ids"][0]))
print("attention_mask:", enc["attention_mask"])

# 3. 词表大小 / 特殊 token 的编号
print("词表大小:", tok.vocab_size)
print(
    "[CLS]=", tok.cls_token_id, " [SEP]=", tok.sep_token_id, " [PAD]=", tok.pad_token_id
)

# 4. 前向：看输出形状
with torch.no_grad():
    out = model(**enc)
print("last_hidden_state:", out.last_hidden_state.shape)  # (1, 序列长, 768)
print("[CLS] 向量前 5 个数:", out.last_hidden_state[0, 0, :5])

# 5. 参数量
print("参数量:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# 6. 一个 batch 里两句长度不同怎么办：padding
batch = tok(
    ["菜好吃", "菜不好吃，就是上菜的速度超慢"], padding=True, return_tensors="pt"
)
print("batch input_ids:\n", batch["input_ids"])
print("batch attention_mask:\n", batch["attention_mask"])
