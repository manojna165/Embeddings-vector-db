# Task 4 - Embeddings + Vector Database

## Objective

This project demonstrates how to create text embeddings, store them in a vector database, and perform similarity-based retrieval.

## Technologies Used

- Python
- Sentence Transformers
- ChromaDB
- FastAPI
- Pydantic

## Architecture

Documents
    ↓
Embedding Model
    ↓
Vector Embeddings
    ↓
ChromaDB
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Relevant Documents

## Embedding Model

The project uses:

`all-MiniLM-L6-v2`

from Sentence Transformers.

The model converts text into numerical vector representations.

## Vector Database

ChromaDB is used to store:

- Document IDs
- Document text
- Embeddings

Cosine distance is used for similarity search.

A lower distance means the documents are more similar to the query.

## API Endpoints

### GET /

Checks whether the API is running.

### GET /documents/count

Returns the number of documents stored in the vector database.

### POST /search

Searches for documents that are semantically similar to the given query.

Example request:

```json
{
  "query": "What is FastAPI?",
  "top_k": 3
}