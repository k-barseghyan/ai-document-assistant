# AI Document Assistant — RAG Demo Application

A compact, local-first RAG demo application for document ingestion, vector retrieval, and source-grounded question answering using FastAPI, Ollama, Qdrant, and Docker.

The project demonstrates the complete document RAG flow:

- upload documents
- extract and chunk text
- generate embeddings
- store chunks in a vector database
- retrieve relevant chunks
- answer questions with source references
- expose the AI logic as an HTTP API

## Current Features

- FastAPI application structure
- Pydantic request/response DTOs
- request validation
- explicit service layer
- custom exception handling
- router-based endpoint organization
- Swagger/OpenAPI documentation

## Tech Stack

- Python
- FastAPI
- Pydantic
- Uvicorn
- Ollama
- Qdrant
- Docker and Docker Compose

## Question Answering

`POST /questions/ask`

Request example:

```json
{
  "question": "What does the document say about RAG?",
  "answer_mode": "rag_only"
}
```

`answer_mode` values:

`rag_only`:

- default
- answers only from uploaded document context
- returns "I do not know based on the uploaded documents." when there is no relevant document context

`hybrid`:

- uses uploaded document context first
- may use general model knowledge when document context is missing or incomplete
- sources still contain only uploaded-document chunks

## Dev RAG Evaluation

`POST /dev/rag/evaluate`

Use this dev-only endpoint to inspect retrieval quality separately from answer-generation quality before changing the RAG stack.

```bash
curl -X POST http://localhost:8000/dev/rag/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the document say about refunds?",
    "answer_mode": "rag_only",
    "expected_keywords": ["refund", "receipt"]
  }'
```

Response shape:

```json
{
  "question": "What does the document say about refunds?",
  "answer_mode": "rag_only",
  "retrieval_limit": 5,
  "context_limit": 3,
  "min_relevance_score": 0.55,
  "retrieved_count": 2,
  "used_context_count": 1,
  "matches": [
    {
      "score": 0.82,
      "filename": "policy.txt",
      "document_id": "doc-123",
      "chunk_index": 0,
      "char_count": 1180,
      "used_in_context": true,
      "preview": "Refunds are available within 30 days with the original receipt..."
    }
  ],
  "answer": "Refunds are available within 30 days with the original receipt.",
  "missing_expected_keywords_in_context": [],
  "missing_expected_keywords_in_answer": []
}
```

`missing_expected_keywords_in_context` checks only the retrieved chunks that were actually used in the prompt. `missing_expected_keywords_in_answer` checks the generated answer text. If a keyword is missing from context, retrieval or filtering likely needs attention; if it is present in context but missing from the answer, answer generation likely needs attention.

## Run Locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Optional RAG retrieval settings:

```bash
RAG_RETRIEVAL_LIMIT=5
RAG_CONTEXT_LIMIT=3
RAG_MIN_RELEVANCE_SCORE=0.55
```
