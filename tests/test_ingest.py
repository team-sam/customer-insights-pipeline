"""Unit tests for the IngestionPipeline class."""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timedelta, timezone
from src.pipelines.ingest import IngestionPipeline
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
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        FeedbackRecord(
            feedback_id="fb002",
            text="Great shoes, very comfortable!",
            source="review",
            created_at=datetime(2024, 1, 2, tzinfo=timezone.utc)
        ),
        FeedbackRecord(
            feedback_id="fb003",
            text="Size runs small, had to return",
            source="return",
            created_at=datetime(2024, 1, 3, tzinfo=timezone.utc)
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


class TestIngestionPipeline:
    """Test IngestionPipeline class."""
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_pipeline_initialization(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger, mock_config
    ):
        """Test IngestionPipeline initializes correctly."""
        pipeline = IngestionPipeline(mock_config)
        
        assert pipeline.config == mock_config
        mock_sql_client.assert_called_once_with(mock_config)
        mock_cosmos_client.assert_called_once_with(mock_config)
        mock_embedder.assert_called_once_with(mock_config)
        mock_tagger.assert_called_once_with(mock_config, custom_categories=None)
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_pipeline_initialization_with_custom_categories(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger, mock_config
    ):
        """Test IngestionPipeline initializes with custom categories."""
        custom_categories = ["Category1", "Category2"]
        pipeline = IngestionPipeline(mock_config, categories=custom_categories)
        
        mock_tagger.assert_called_once_with(mock_config, custom_categories=custom_categories)
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_run_with_no_records(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger, 
        mock_config
    ):
        """Test pipeline run with no records to process."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = []
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline
        pipeline = IngestionPipeline(mock_config)
        stats = pipeline.run()
        
        # Verify
        assert stats["total_records"] == 0
        assert stats["embeddings_created"] == 0
        assert stats["tags_created"] == 0
        assert stats["errors"] == 0
        assert stats["start_date"] is None
        assert stats["end_date"] is None
        
        mock_sql_instance.connect.assert_called_once()
        mock_cosmos_instance.connect.assert_called_once()
        mock_sql_instance.close.assert_called_once()
        mock_cosmos_instance.close.assert_called_once()
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_run_with_single_batch(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test pipeline run with single batch of records."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = sample_feedback_records
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.return_value = [
            ["Waterproof Leak"],
            ["Positive Feedback"],
            ["Sizes not standard"]
        ]
        
        # Run pipeline
        pipeline = IngestionPipeline(mock_config)
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
        mock_tagger_instance.tag_batch.assert_called_once()
        
        # Verify tags saved
        mock_sql_instance.insert_tags.assert_called_once()
        tag_records = mock_sql_instance.insert_tags.call_args[0][0]
        assert len(tag_records) == 3
        assert tag_records[0].feedback_id == "fb001"
        assert tag_records[0].tag_name == "Waterproof Leak"
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_run_with_date_range(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test pipeline run with specific date range."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = sample_feedback_records
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.return_value = [
            ["Tag1"], ["Tag2"], ["Tag3"]
        ]
        
        # Define date range
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        
        # Run pipeline with date range
        pipeline = IngestionPipeline(mock_config)
        stats = pipeline.run(start_date=start_date, end_date=end_date)
        
        # Verify date range was passed to get_new_feedback
        mock_sql_instance.get_new_feedback.assert_called_once_with(
            start_date=start_date,
            end_date=end_date,
            limit=None
        )
        
        # Verify stats include date range
        assert stats["start_date"] == start_date.isoformat()
        assert stats["end_date"] == end_date.isoformat()
        assert stats["total_records"] == 3
    
    @patch('src.pipelines.ingest.datetime')
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_run_with_days_back(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_datetime, mock_config, sample_feedback_records, sample_vectors
    ):
        """Test pipeline run with days_back parameter."""
        # Mock datetime.now to return a fixed time
        fixed_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_now
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = sample_feedback_records
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.return_value = [
            ["Tag1"], ["Tag2"], ["Tag3"]
        ]
        
        # Run pipeline with days_back
        pipeline = IngestionPipeline(mock_config)
        stats = pipeline.run(days_back=7)
        
        # Verify get_new_feedback was called with calculated date range
        call_kwargs = mock_sql_instance.get_new_feedback.call_args[1]
        assert 'start_date' in call_kwargs
        assert 'end_date' in call_kwargs
        
        # Start date should be 7 days before end date
        start_date = call_kwargs['start_date']
        end_date = call_kwargs['end_date']
        assert end_date == fixed_now
        assert start_date == fixed_now - timedelta(days=7)
        
        # Verify stats
        assert stats["total_records"] == 3
        assert stats["start_date"] is not None
        assert stats["end_date"] is not None
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_run_with_limit(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config
    ):
        """Test pipeline run with limit parameter."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = []
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline with limit
        pipeline = IngestionPipeline(mock_config)
        pipeline.run(limit=50)
        
        # Verify limit was passed to get_new_feedback
        mock_sql_instance.get_new_feedback.assert_called_once()
        call_kwargs = mock_sql_instance.get_new_feedback.call_args[1]
        assert call_kwargs['limit'] == 50
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_run_with_multiple_batches(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config
    ):
        """Test pipeline run with multiple batches."""
        # Create 5 feedback records (will be split into 2 batches with batch_size=3)
        feedback_records = [
            FeedbackRecord(
                feedback_id=f"fb{i:03d}",
                text=f"Feedback text {i}",
                source="review",
                created_at=datetime(2024, 1, i+1, tzinfo=timezone.utc)
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
        
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.side_effect = [
            [["Waterproof Leak"], ["Quality Issues"], ["Positive Feedback"]],  # Batch 1
            [["Sizes not standard"], ["Too Heavy"]]  # Batch 2
        ]
        
        # Run pipeline with batch_size=3
        pipeline = IngestionPipeline(mock_config)
        stats = pipeline.run(batch_size=3)
        
        # Verify
        assert stats["total_records"] == 5
        assert stats["embeddings_created"] == 5
        assert stats["tags_created"] == 5
        assert stats["errors"] == 0
        
        # Verify 2 batches were processed
        assert mock_embedder_instance.embed_texts.call_count == 2
        assert mock_cosmos_instance.insert_embeddings.call_count == 2
        assert mock_tagger_instance.tag_batch.call_count == 2
        assert mock_sql_instance.insert_tags.call_count == 2
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_run_with_error_handling(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
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
        pipeline = IngestionPipeline(mock_config)
        stats = pipeline.run()
        
        # Verify error was tracked
        assert stats["total_records"] == 3
        assert stats["embeddings_created"] == 0
        assert stats["tags_created"] == 0
        assert stats["errors"] == 3
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_connections_closed_after_run(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config
    ):
        """Test that database connections are closed even on error."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.side_effect = Exception("Database error")
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline (should raise exception but still close connections)
        pipeline = IngestionPipeline(mock_config)
        with pytest.raises(Exception):
            pipeline.run()
        
        # Verify connections were closed
        mock_sql_instance.close.assert_called_once()
        mock_cosmos_instance.close.assert_called_once()
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_process_embeddings_batch(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test _process_embeddings_batch method."""
        # Setup mocks
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Create pipeline and call method
        pipeline = IngestionPipeline(mock_config)
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
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_process_tags_batch(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records
    ):
        """Test _process_tags_batch method."""
        # Setup mocks
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.return_value = [
            ["Waterproof Leak"],
            ["Positive Feedback", "Comfortable"],
            ["Sizes not standard"]
        ]
        
        mock_sql_instance = mock_sql_client.return_value
        
        # Create pipeline and call method
        pipeline = IngestionPipeline(mock_config)
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
        
        mock_tagger_instance.tag_batch.assert_called_once()
        mock_sql_instance.insert_tags.assert_called_once_with(result)
