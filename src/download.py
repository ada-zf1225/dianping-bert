# src/download.py
# 作用：读 data/raw 下的 csv，看清结构，存成 parquet
import pandas as pd

for split in ["train", "dev"]:
    df = pd.read_csv(f"data/raw/{split}.csv")
    print(f"[{split}] shape={df.shape}")
    print(f"[{split}] columns={df.columns.tolist()}")
    print(df.head(3))
    print()
    df.to_parquet(f"data/{split}.parquet", index=False)

print("done")
