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

## Run Locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate