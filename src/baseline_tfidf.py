# src/baseline_tfidf.py
# 传统流水线：字 n-gram TF-IDF 向量化 → 逻辑回归
import json  # noqa: I001
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TEXT, LABEL = "sentence", "label"
df = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/dev.parquet")

# 从 train 切 10% 做内部验证；stratify 保证正负比例一致；seed 固定保证可复现
train, val = train_test_split(df, test_size=0.1, stratify=df[LABEL], random_state=42)

# 向量化：按字切，取 1 到 2 个字的组合作为特征
vec = TfidfVectorizer(
    analyzer="char", ngram_range=(1, 2), min_df=2, max_features=200_000
)
X_tr = vec.fit_transform(train[TEXT])  # 只在训练集上 fit，统计词表和 IDF
X_val = vec.transform(val[TEXT])
X_te = vec.transform(test[TEXT])
print("特征维度:", X_tr.shape[1])

# 训练模型：逻辑回归就是 y = sigmoid(Wx + b)
clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_tr, train[LABEL])

acc_val = clf.score(X_val, val[LABEL])
acc_te = clf.score(X_te, test[LABEL])
print(f"val acc = {acc_val:.4f}   test(dev.csv) acc = {acc_te:.4f}")

json.dump(
    {"model": "tfidf_char12_lr", "val_acc": acc_val, "test_acc": acc_te},
    open("results/baseline_tfidf.json", "w"),
    ensure_ascii=False,
    indent=2,
)
