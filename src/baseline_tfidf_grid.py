# src/baseline_tfidf_grid.py
# 扫 C 和 n-gram 范围，看基线的上限在哪
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TEXT, LABEL = "sentence", "label"
df = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/dev.parquet")
train, val = train_test_split(df, test_size=0.1, stratify=df[LABEL], random_state=42)

results = []
for ngram in [(1, 1), (1, 2), (1, 3)]:
    vec = TfidfVectorizer(
        analyzer="char", ngram_range=ngram, min_df=2, max_features=500_000
    )
    X_tr, X_val, X_te = (
        vec.fit_transform(train[TEXT]),
        vec.transform(val[TEXT]),
        vec.transform(test[TEXT]),
    )
    for C in [0.3, 1, 3, 10]:
        clf = LogisticRegression(C=C, max_iter=2000).fit(X_tr, train[LABEL])
        r = {
            "ngram": ngram,
            "C": C,
            "n_feat": X_tr.shape[1],
            "val_acc": round(clf.score(X_val, val[LABEL]), 4),
            "test_acc": round(clf.score(X_te, test[LABEL]), 4),
        }
        print(r)
        results.append(r)

best = max(results, key=lambda r: r["val_acc"])
print("best by val:", best)
json.dump(
    results, open("results/baseline_tfidf_grid.json", "w"), ensure_ascii=False, indent=2
)
