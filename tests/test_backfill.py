"""Unit tests for the BackfillPipeline class."""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from src.pipelines.backfill import BackfillPipeline
from src.config.settings import Settings
from src.models.schemas import FeedbackRecord, EmbeddingRecord, TagRecord


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Settings)
    config.openai_api_key = "test-api-key"
    config.openai_embedding_model = "text-embedding-3-small"
    config.openai_llm_model = "gpt-5-nano"
    config.batch_size = 100
    return config


@pytest.fixture
def sample_feedback_records():
    """Create sample feedback records for testing."""
    return [
        FeedbackRecord(
            feedback_id="fb001",
            text="Shoes are leaking after 2 weeks",
            source="review",
            created_at=datetime(2024, 1, 1)
        ),
        FeedbackRecord(
            feedback_id="fb002",
            text="Great shoes, very comfortable!",
            source="review",
            created_at=datetime(2024, 1, 2)
        ),
        FeedbackRecord(
            feedback_id="fb003",
            text="Size runs small, had to return",
            source="return",
            created_at=datetime(2024, 1, 3)
        )
    ]


@pytest.fixture
def sample_vectors():
    """Create sample embedding vectors."""
    return [
        [0.1] * 1536,
        [0.2] * 1536,
        [0.3] * 1536
    ]


class TestBackfillPipeline:
    """Test BackfillPipeline class."""
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_pipeline_initialization(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent, mock_config
    ):
        """Test BackfillPipeline initializes correctly."""
        pipeline = BackfillPipeline(mock_config)
        
        assert pipeline.config == mock_config
        mock_sql_client.assert_called_once_with(mock_config)
        mock_cosmos_client.assert_called_once_with(mock_config)
        mock_embedder.assert_called_once_with(mock_config)
        mock_chat_agent.assert_called_once_with(mock_config)
        assert len(pipeline.categories) == 10
        assert "Waterproof Leak" in pipeline.categories
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_pipeline_initialization_with_custom_categories(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent, mock_config
    ):
        """Test BackfillPipeline initializes with custom categories."""
        custom_categories = ["Category1", "Category2"]
        pipeline = BackfillPipeline(mock_config, categories=custom_categories)
        
        assert pipeline.categories == custom_categories
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_run_with_no_records(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent, 
        mock_config
    ):
        """Test pipeline run with no records to process."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = []
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline
        pipeline = BackfillPipeline(mock_config)
        stats = pipeline.run()
        
        # Verify
        assert stats["total_records"] == 0
        assert stats["embeddings_created"] == 0
        assert stats["tags_created"] == 0
        assert stats["errors"] == 0
        
        mock_sql_instance.connect.assert_called_once()
        mock_cosmos_instance.connect.assert_called_once()
        mock_sql_instance.close.assert_called_once()
        mock_cosmos_instance.close.assert_called_once()
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_run_with_single_batch(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test pipeline run with single batch of records."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = sample_feedback_records
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_chat_instance = mock_chat_agent.return_value
        mock_chat_instance.tag_feedback_batch.return_value = [
            ["Waterproof Leak"],
            ["Positive Feedback"],
            ["Sizes not standard"]
        ]
        
        # Run pipeline
        pipeline = BackfillPipeline(mock_config)
        stats = pipeline.run()
        
        # Verify
        assert stats["total_records"] == 3
        assert stats["embeddings_created"] == 3
        assert stats["tags_created"] == 3
        assert stats["errors"] == 0
        
        # Verify embedding generation
        mock_embedder_instance.embed_texts.assert_called_once()
        call_args = mock_embedder_instance.embed_texts.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0] == "Shoes are leaking after 2 weeks"
        
        # Verify embeddings saved
        mock_cosmos_instance.insert_embeddings.assert_called_once()
        embedding_records = mock_cosmos_instance.insert_embeddings.call_args[0][0]
        assert len(embedding_records) == 3
        assert embedding_records[0].feedback_id == "fb001"
        assert embedding_records[0].vector == sample_vectors[0]
        
        # Verify tagging
        mock_chat_instance.tag_feedback_batch.assert_called_once()
        
        # Verify tags saved
        mock_sql_instance.insert_tags.assert_called_once()
        tag_records = mock_sql_instance.insert_tags.call_args[0][0]
        assert len(tag_records) == 3
        assert tag_records[0].feedback_id == "fb001"
        assert tag_records[0].tag_name == "Waterproof Leak"
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_run_with_multiple_batches(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent,
        mock_config
    ):
        """Test pipeline run with multiple batches."""
        # Create 5 feedback records (will be split into 2 batches with batch_size=3)
        feedback_records = [
            FeedbackRecord(
                feedback_id=f"fb{i:03d}",
                text=f"Feedback text {i}",
                source="review",
                created_at=datetime(2024, 1, i+1)
            )
            for i in range(5)
        ]
        
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = feedback_records
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        # Return different vectors for each batch
        mock_embedder_instance.embed_texts.side_effect = [
            [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],  # Batch 1
            [[0.4] * 1536, [0.5] * 1536]  # Batch 2
        ]
        
        mock_chat_instance = mock_chat_agent.return_value
        mock_chat_instance.tag_feedback_batch.side_effect = [
            [["Waterproof Leak"], ["Quality Issues"], ["Positive Feedback"]],  # Batch 1
            [["Sizes not standard"], ["Too Heavy"]]  # Batch 2
        ]
        
        # Run pipeline with batch_size=3
        pipeline = BackfillPipeline(mock_config)
        stats = pipeline.run(batch_size=3)
        
        # Verify
        assert stats["total_records"] == 5
        assert stats["embeddings_created"] == 5
        assert stats["tags_created"] == 5
        assert stats["errors"] == 0
        
        # Verify 2 batches were processed
        assert mock_embedder_instance.embed_texts.call_count == 2
        assert mock_cosmos_instance.insert_embeddings.call_count == 2
        assert mock_chat_instance.tag_feedback_batch.call_count == 2
        assert mock_sql_instance.insert_tags.call_count == 2
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_run_with_limit(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent,
        mock_config
    ):
        """Test pipeline run with limit parameter."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = []
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline with limit
        pipeline = BackfillPipeline(mock_config)
        pipeline.run(limit=50)
        
        # Verify limit was passed to get_new_feedback
        mock_sql_instance.get_new_feedback.assert_called_once_with(limit=50)
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_run_with_error_handling(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent,
        mock_config, sample_feedback_records
    ):
        """Test pipeline handles errors gracefully."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = sample_feedback_records
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        # Make embedder raise an exception
        mock_embedder_instance.embed_texts.side_effect = Exception("API Error")
        
        # Run pipeline
        pipeline = BackfillPipeline(mock_config)
        stats = pipeline.run()
        
        # Verify error was tracked
        assert stats["total_records"] == 3
        assert stats["embeddings_created"] == 0
        assert stats["tags_created"] == 0
        assert stats["errors"] == 3
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_process_embeddings_batch(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test _process_embeddings_batch method."""
        # Setup mocks
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Create pipeline and call method
        pipeline = BackfillPipeline(mock_config)
        result = pipeline._process_embeddings_batch(sample_feedback_records)
        
        # Verify
        assert len(result) == 3
        assert all(isinstance(r, EmbeddingRecord) for r in result)
        assert result[0].feedback_id == "fb001"
        assert result[0].vector == sample_vectors[0]
        assert result[0].source == "review"
        assert result[0].model == "text-embedding-3-small"
        
        mock_embedder_instance.embed_texts.assert_called_once()
        mock_cosmos_instance.insert_embeddings.assert_called_once_with(result)
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_process_tags_batch(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent,
        mock_config, sample_feedback_records
    ):
        """Test _process_tags_batch method."""
        # Setup mocks
        mock_chat_instance = mock_chat_agent.return_value
        mock_chat_instance.tag_feedback_batch.return_value = [
            ["Waterproof Leak"],
            ["Positive Feedback", "Comfortable"],
            ["Sizes not standard"]
        ]
        
        mock_sql_instance = mock_sql_client.return_value
        
        # Create pipeline and call method
        pipeline = BackfillPipeline(mock_config)
        result = pipeline._process_tags_batch(sample_feedback_records)
        
        # Verify
        assert len(result) == 4  # First has 1 tag, second has 2 tags, third has 1 tag
        assert all(isinstance(r, TagRecord) for r in result)
        assert result[0].feedback_id == "fb001"
        assert result[0].tag_name == "Waterproof Leak"
        assert result[0].confidence_score == 1.0
        assert result[1].feedback_id == "fb002"
        assert result[1].tag_name == "Positive Feedback"
        assert result[2].feedback_id == "fb002"
        assert result[2].tag_name == "Comfortable"
        
        mock_chat_instance.tag_feedback_batch.assert_called_once()
        mock_sql_instance.insert_tags.assert_called_once_with(result)
    
    @patch('src.pipelines.backfill.ChatAgent')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_connections_closed_after_run(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_chat_agent,
        mock_config
    ):
        """Test that database connections are closed even on error."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.side_effect = Exception("Database error")
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline (should raise exception but still close connections)
        pipeline = BackfillPipeline(mock_config)
        with pytest.raises(Exception):
            pipeline.run()
        
        # Verify connections were closed
        mock_sql_instance.close.assert_called_once()
        mock_cosmos_instance.close.assert_called_once()
