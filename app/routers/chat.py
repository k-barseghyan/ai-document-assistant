from fastapi import APIRouter, Depends

from app.schemas import ChatRequest, ChatResponse
from app.services import ChatService, get_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/message", response_model=ChatResponse)
def chat_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    return chat_service.answer_message(request)
