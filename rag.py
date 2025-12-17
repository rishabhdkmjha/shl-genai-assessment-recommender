from backend.retriever import retrieve

def rag_recommend_no_llm(query, k=20):
    retrieved = retrieve(query, k)

    seen = set()
    results = []
    has_k = False
    has_p = False

    for r in retrieved:
        if r["name"] in seen:
            continue
        seen.add(r["name"])

        if r["test_type"] == "Knowledge & Skills":
            has_k = True
        if r["test_type"] == "Personality & Behavior":
            has_p = True

        results.append({
            "name": r["name"],
            "url": r["url"],
            "test_type": r["test_type"],
            "justification": "Relevant based on semantic similarity to job requirements"
        })

        if len(results) >= 5 and has_k and has_p:
            break

    return results


def recommend_urls(query, k=10):
    results = rag_recommend_no_llm(query, k=20)
    return [r["url"] for r in results][:k]
