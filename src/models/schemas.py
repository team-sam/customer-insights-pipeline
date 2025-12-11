from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, List
from enum import Enum



class FeedbackRecord(BaseModel):
    """Raw customer feedback record."""
    model_config = ConfigDict(use_enum_values=True)
    
    feedback_id: str
    text: str
    source: str
    created_at: datetime
    product_id: Optional[str] = None
    style: Optional[str] = None
    category: Optional[str] = None
    rating: Optional[int] = None
    cluster_id: Optional[int] = None


class EmbeddingRecord(BaseModel):
    """Embedding vector for feedback."""
    feedback_id: str
    vector: List[float] = Field(..., min_length=1536, max_length=1536)
    source: str
    model: str
    created_at: datetime
    feedback_text: str
    style: Optional[str] = None


class ClusterRecord(BaseModel):
    """Discovered theme/cluster."""
    cluster_id: int
    label: str
    description: str
    style: Optional[str] = None
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