from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vector_store import initialize_database, search_documents

app = FastAPI(
    title="Embeddings + Vector Database API",
    description="Task 4 - Semantic similarity search using embeddings and ChromaDB",
    version="1.0.0"
)

# Initialize the vector database when the application starts
collection = initialize_database()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=8)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Embeddings and vector database API is running"
    }


@app.get("/documents/count")
def document_count():
    return {
        "document_count": collection.count()
    }


@app.post("/search")
def search(request: SearchRequest):
    try:
        results = search_documents(
            collection,
            request.query,
            request.top_k
        )

        return {
            "query": request.query,
            "results": results
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}"
        )