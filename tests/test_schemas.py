"""Unit tests for data schemas."""
import pytest
from datetime import datetime
from src.models.schemas import FeedbackRecord, EmbeddingRecord, ClusterRecord, TagRecord


class TestFeedbackRecord:
    """Test FeedbackRecord schema."""
    
    def test_feedback_record_creation(self):
        """Test creating a valid FeedbackRecord."""
        record = FeedbackRecord(
            feedback_id="test-123",
            text="Great product!",
            source="review",
            created_at=datetime.now()
        )
        assert record.feedback_id == "test-123"
        assert record.text == "Great product!"
        assert record.source == "review"
        assert record.product_id is None
        assert record.category is None
        assert record.rating is None
        assert record.cluster_id is None
    
    def test_feedback_record_with_optional_fields(self):
        """Test FeedbackRecord with all optional fields."""
        now = datetime.now()
        record = FeedbackRecord(
            feedback_id="test-456",
            text="Good shoes",
            source="chat",
            created_at=now,
            product_id="SKU-789",
            category="Footwear",
            rating=5,
            cluster_id="source_review.style_chelsea.0.1"
        )
        assert record.product_id == "SKU-789"
        assert record.category == "Footwear"
        assert record.rating == 5
        assert record.cluster_id == "source_review.style_chelsea.0.1"


class TestEmbeddingRecord:
    """Test EmbeddingRecord schema."""
    
    def test_embedding_record_creation(self):
        """Test creating a valid EmbeddingRecord."""
        vector = [0.1] * 1536
        record = EmbeddingRecord(
            feedback_id="test-123",
            vector=vector,
            source="review",
            model="text-embedding-3-small",
            created_at=datetime.now(),
            feedback_text="Great product!"
        )
        assert record.feedback_id == "test-123"
        assert len(record.vector) == 1536
        assert record.source == "review"
        assert record.model == "text-embedding-3-small"
        assert record.feedback_text == "Great product!"
    
    def test_embedding_record_invalid_vector_length(self):
        """Test that EmbeddingRecord rejects invalid vector lengths."""
        with pytest.raises(ValueError):
            EmbeddingRecord(
                feedback_id="test-123",
                vector=[0.1] * 100,  # Wrong length
                source="review",
                model="text-embedding-3-small",
                created_at=datetime.now()
            )


class TestClusterRecord:
    """Test ClusterRecord schema."""
    
    def test_cluster_record_creation(self):
        """Test creating a valid ClusterRecord."""
        now = datetime.now()
        record = ClusterRecord(
            cluster_id="source_review.style_chelsea.0",
            label="Waterproof Issues",
            description="Customers reporting leaking problems",
            source="review",
            style="chelsea",
            cluster_depth=2,
            sample_feedback_ids=["fb-1", "fb-2", "fb-3"],
            record_count=25,
            period_start=now,
            period_end=now,
            created_at=now
        )
        assert record.cluster_id == "source_review.style_chelsea.0"
        assert record.label == "Waterproof Issues"
        assert record.source == "review"
        assert record.style == "chelsea"
        assert record.cluster_depth == 2
        assert len(record.sample_feedback_ids) == 3
        assert record.record_count == 25


class TestTagRecord:
    """Test TagRecord schema."""
    
    def test_tag_record_creation(self):
        """Test creating a valid TagRecord."""
        record = TagRecord(
            feedback_id="test-123",
            tag_name="Waterproof Leak",
            confidence_score=0.95,
            created_at=datetime.now()
        )
        assert record.feedback_id == "test-123"
        assert record.tag_name == "Waterproof Leak"
        assert record.confidence_score == 0.95
    
    def test_tag_record_confidence_validation(self):
        """Test that confidence score is validated to be between 0 and 1."""
        with pytest.raises(ValueError):
            TagRecord(
                feedback_id="test-123",
                tag_name="Test",
                confidence_score=1.5,  # Invalid: > 1.0
                created_at=datetime.now()
            )
        
        with pytest.raises(ValueError):
            TagRecord(
                feedback_id="test-123",
                tag_name="Test",
                confidence_score=-0.1,  # Invalid: < 0.0
                created_at=datetime.now()
            )
