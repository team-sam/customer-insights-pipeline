from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class FeedbackSource(str, Enum):
    REVIEW = "review"
    RETURN = "return"
    CHAT = "chat"


class FeedbackRecord(BaseModel):
    """Raw customer feedback record."""
    feedback_id: str
    text: str
    source: FeedbackSource
    created_at: datetime
    product_id: Optional[str] = None
    category: Optional[str] = None
    rating: Optional[int] = None
    
    class Config:
        use_enum_values = True


class EmbeddingRecord(BaseModel):
    """Embedding vector for feedback."""
    feedback_id: str
    vector: List[float] = Field(..., min_length=1536, max_length=1536)
    model: str
    created_at: datetime


class ClusterRecord(BaseModel):
    """Discovered theme/cluster."""
    cluster_id: int
    label: str
    description: str
    sample_feedback_ids: List[str]
    record_count: int
    period_start: datetime
    period_end: datetime
    created_at: datetime


class TagRecord(BaseModel):
    """Assigned tag from taxonomy."""
    feedback_id: str
    tag_name: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime