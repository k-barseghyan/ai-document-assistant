from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import EmptyAnswerError
from app.routers.questions import router as questions_router

app = FastAPI(title="AI Document Assistant")


@app.exception_handler(EmptyAnswerError)
def empty_answer_error_handler(request: Request, exc: EmptyAnswerError):
    return JSONResponse(
        status_code=500,
        content={
            "error": "ANSWER_GENERATION_FAILED",
            "message": str(exc),
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(questions_router)