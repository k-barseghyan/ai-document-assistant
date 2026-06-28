from app.llm.ollama_client import OllamaClient
from app.schemas import AnswerResponse, ChatRequest, ChatResponse, QuestionRequest


DEFAULT_CHAT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant. Answer clearly and naturally."
)


class QuestionService:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        answer = self.llm_client.generate(request.question)
        return AnswerResponse(answer=answer)


class ChatService:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def answer_message(self, request: ChatRequest) -> ChatResponse:
        messages = [message.model_dump() for message in request.messages]

        if not any(message["role"] == "system" for message in messages):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": DEFAULT_CHAT_SYSTEM_MESSAGE,
                },
            )

        answer = self.llm_client.chat(messages)
        return ChatResponse(answer=answer)


def get_question_service() -> QuestionService:
    return QuestionService(llm_client=OllamaClient())


def get_chat_service() -> ChatService:
    return ChatService(llm_client=OllamaClient())
