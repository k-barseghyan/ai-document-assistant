# AI Document Assistant

A production-style AI backend service built with FastAPI.

The goal of this project is to build an AI document assistant step by step:

- upload documents
- extract text
- chunk documents
- generate embeddings
- store chunks in a vector database
- retrieve relevant chunks
- answer questions with source references
- expose the AI logic as an HTTP API
- later integrate it with a Spring Boot backend

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
