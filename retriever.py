from backend.embeddings import build_embeddings

# Load everything once
model, index, metadata = build_embeddings()

def retrieve(query, k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results
