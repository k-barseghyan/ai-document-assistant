from app.exceptions import EmptyAnswerError
from app.schemas import AnswerResponse, QuestionRequest


class QuestionService:
    def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        if request.question.lower() == "simulate failure":
            raise EmptyAnswerError("Could not generate an answer")

        return AnswerResponse(answer=f"You asked: {request.question}")


def get_question_service() -> QuestionService:
    return QuestionService()