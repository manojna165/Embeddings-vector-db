import json
import chromadb
from embeddings import create_embeddings, create_query_embedding

DB_PATH = "./chroma_db"
COLLECTION_NAME = "sample_documents"

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=DB_PATH)


def initialize_database():
    """Load documents, create embeddings, and store them in ChromaDB."""

    # Delete old collection so every run starts clean
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}}
    )

    # Load sample documents
    with open("documents.json", "r", encoding="utf-8") as file:
        documents = json.load(file)

    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    # Generate embeddings
    embeddings = create_embeddings(texts)

    # Store documents + embeddings in ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings
    )

    return collection


def search_documents(collection, query, top_k=3):
    """Find the most similar documents for a query."""

    query_embedding = create_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    matches = []

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]

        matches.append({
            "rank": i + 1,
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": round(distance, 4),
            "similarity": round(1 - distance, 4)
        })

    return matches