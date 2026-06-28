from app.llm.ollama_client import OllamaClient
from app.schemas import AnswerResponse, QuestionRequest


class QuestionService:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        answer = self.llm_client.generate(request.question)
        return AnswerResponse(answer=answer)


def get_question_service() -> QuestionService:
    return QuestionService(llm_client=OllamaClient())
