from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Literal

class IngestUrlRequest(BaseModel):
    url: HttpUrl
    source_title: Optional[str] = None

class SiftRequest(BaseModel):
    query: str = Field(..., min_length=3)
    mode: Literal["key_points", "study_guide", "quiz"] = "study_guide"
    k: int = Field(8, ge=1, le=25)

class SiftResponse(BaseModel):
    answer: str
    sources: List[dict]