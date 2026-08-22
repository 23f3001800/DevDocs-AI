from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    # Structured output so Pydantic validates types at parse time (e.g. a
    # non-numeric confidence is caught immediately).
    answer: str = Field(description="Grounded answer from context")
    sources: list[str] = Field(description="List of file paths that support the answer")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0=not in docs, 1.0=perfectly grounded")
    has_answer: bool = Field(description="False if docs don't contain relevant info")
