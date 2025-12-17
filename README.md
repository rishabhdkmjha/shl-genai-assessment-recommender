# SHL GenAI Assessment Recommender

This project implements a retrieval-based GenAI system to recommend SHL assessments from natural language job descriptions.

## Approach
- Semantic embeddings using SentenceTransformers
- FAISS vector search for retrieval
- RAG-style post-processing (deduplication + balance between Knowledge & Skills and Personality & Behavior)
- Evaluation using Mean Recall@10

## Structure
- `backend/` – core retrieval and recommendation logic
- `data/` – catalog, train/test data, submission output
- `eval/` – evaluation scripts

## Output
Final recommendations are provided in `data/submission.csv` following the required format.
