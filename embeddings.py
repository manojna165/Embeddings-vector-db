from sentence_transformers import SentenceTransformer

# Load the local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):
    """Convert text into numerical vector embeddings."""
    return model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()


def create_query_embedding(query):
    """Create an embedding for a search query."""
    return model.encode(
        query,
        normalize_embeddings=True
    ).tolist()