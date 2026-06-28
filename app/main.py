from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.exceptions import EmptyAnswerError, LLMClientError
from app.routers.questions import router as questions_router

app = FastAPI(title="AI Document Assistant")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.exception_handler(EmptyAnswerError)
def empty_answer_error_handler(request: Request, exc: EmptyAnswerError):
    return JSONResponse(
        status_code=500,
        content={
            "error": "ANSWER_GENERATION_FAILED",
            "message": str(exc),
        },
    )


@app.exception_handler(LLMClientError)
def llm_client_error_handler(request: Request, exc: LLMClientError):
    return JSONResponse(
        status_code=502,
        content={
            "error": "LLM_REQUEST_FAILED",
            "message": str(exc),
        },
    )


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(questions_router)
