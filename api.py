from fastapi import FastAPI
from backend.rag import recommend_urls

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(query: str):
    return recommend_urls(query)
