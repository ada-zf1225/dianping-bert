from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["菜好吃", "菜不好吃", "菜好贵"]
vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
