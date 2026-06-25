from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="User question about uploaded documents"
    )


class AnswerResponse(BaseModel):
    answer: str