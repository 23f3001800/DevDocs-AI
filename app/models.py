
from pydantic import BaseModel, Field

class RAGResponse(BaseModel):
    # WHY structured output instead of raw string?
    # Pydantic validates types at parse time — if Claude returns
    # confidence="high" instead of 0.85, we catch it immediately.
    answer: str = Field(description="Grounded answer from context")
    sources: list[str] = Field(
        description="List of file paths that support the answer"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.0=not in docs, 1.0=perfectly grounded"
    )
    has_answer: bool = Field(
        description="False if docs don't contain relevant info"
    )