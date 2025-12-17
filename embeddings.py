import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CATALOG_PATH = "data/catalog.json"

def build_embeddings():
    # Load catalog
    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)

    documents = []
    metadata = []

    for a in catalog:
        text = f"""
        Assessment Name: {a['name']}
        Description: {a['description']}
        Test Type: {a['test_type']}
        """
        documents.append(text.strip())
        metadata.append(a)

    # Build embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, index, metadata
