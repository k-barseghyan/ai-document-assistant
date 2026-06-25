from fastapi import APIRouter, Depends

from app.schemas import AnswerResponse, QuestionRequest
from app.services import QuestionService, get_question_service

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post("/ask", response_model=AnswerResponse)
def ask(
    request: QuestionRequest,
    question_service: QuestionService = Depends(get_question_service),
):
    return question_service.answer_question(request)    