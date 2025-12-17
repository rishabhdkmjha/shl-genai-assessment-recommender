import pandas as pd
from backend.rag import recommend_urls

def recall_at_k(predicted, actual, k=10):
    predicted = predicted[:k]
    actual = set(actual)
    if not actual:
        return 0.0
    return len(set(predicted) & actual) / len(actual)

def mean_recall_at_10(train_path="data/train.csv"):
    df = pd.read_csv(train_path)
    recalls = []

    for query, group in df.groupby("query"):
        actual_urls = group["assessment_url"].tolist()
        predicted_urls = recommend_urls(query, k=10)
        recalls.append(recall_at_k(predicted_urls, actual_urls, k=10))

    return sum(recalls) / len(recalls)

if __name__ == "__main__":
    print("Mean Recall@10:", mean_recall_at_10())
