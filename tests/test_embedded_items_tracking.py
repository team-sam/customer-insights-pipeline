"""Unit tests for embedded items tracking functionality."""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timezone
from src.pipelines.ingest import IngestionPipeline
from src.config.settings import Settings
from src.models.schemas import FeedbackRecord, EmbeddingRecord


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


class TestEmbeddedItemsTracking:
    """Test embedded items tracking functionality."""
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_initialize_embedded_items_table(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config
    ):
        """Test that embedded items table is initialized during run."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = []
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline
        pipeline = IngestionPipeline(mock_config)
        pipeline.run()
        
        # Verify table initialization was called
        mock_sql_instance.initialize_embedded_items_table.assert_called_once()
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_sync_embeddings_to_sql_server(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test that embedding metadata is synced to SQL Server after Cosmos insertion."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = sample_feedback_records
        mock_sql_instance.get_embedded_feedback_ids.return_value = []
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.return_value = [["Tag1"], ["Tag2"], ["Tag3"]]
        
        # Run pipeline
        pipeline = IngestionPipeline(mock_config)
        stats = pipeline.run()
        
        # Verify embeddings were synced to SQL Server
        mock_sql_instance.insert_embedded_items.assert_called_once()
        
        # Verify the correct data was passed
        call_args = mock_sql_instance.insert_embedded_items.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0]['feedback_id'] == "fb001"
        assert call_args[1]['feedback_id'] == "fb002"
        assert call_args[2]['feedback_id'] == "fb003"
        assert all('embedded_at' in item for item in call_args)
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_skip_embedded_items_disabled(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test that all items are processed when skip_embedded is False."""
        # Setup mocks
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = sample_feedback_records
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = sample_vectors
        
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.return_value = [["Tag1"], ["Tag2"], ["Tag3"]]
        
        # Run pipeline with skip_embedded=False (default)
        pipeline = IngestionPipeline(mock_config, skip_embedded=False)
        stats = pipeline.run()
        
        # Verify get_all_embedded_ids was NOT called on CosmosClient
        mock_cosmos_instance.get_all_embedded_ids.assert_not_called()
        
        # Verify sync_embeddings_from_cosmos was NOT called
        mock_sql_instance.sync_embeddings_from_cosmos.assert_not_called()
        
        # Verify get_new_feedback was called with skip_embedded=False
        mock_sql_instance.get_new_feedback.assert_called_once()
        call_kwargs = mock_sql_instance.get_new_feedback.call_args[1]
        assert call_kwargs['skip_embedded'] == False
        
        # Verify all 3 records were processed
        assert stats["total_records"] == 3
        assert stats["embeddings_created"] == 3
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_skip_embedded_items_enabled(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records, sample_vectors
    ):
        """Test that already-embedded items are skipped when skip_embedded is True."""
        # Setup mocks - simulating that SQL query returns only 1 record after filtering
        one_record = [sample_feedback_records[2]]  # Only fb003
        
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = one_record
        mock_sql_instance.sync_embeddings_from_cosmos.return_value = 2  # 2 items synced
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        mock_cosmos_instance.get_all_embedded_ids.return_value = [
            ("fb001", datetime(2024, 1, 1, tzinfo=timezone.utc)),
            ("fb002", datetime(2024, 1, 2, tzinfo=timezone.utc))
        ]
        
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed_texts.return_value = [[0.3] * 1536]  # Only 1 vector needed
        
        mock_tagger_instance = mock_tagger.return_value
        mock_tagger_instance.tag_batch.return_value = [["Tag3"]]
        
        # Run pipeline with skip_embedded=True
        pipeline = IngestionPipeline(mock_config, skip_embedded=True)
        stats = pipeline.run()
        
        # Verify get_all_embedded_ids was called on CosmosClient
        mock_cosmos_instance.get_all_embedded_ids.assert_called_once()
        
        # Verify sync_embeddings_from_cosmos was called with the embedded items
        mock_sql_instance.sync_embeddings_from_cosmos.assert_called_once()
        
        # Verify get_new_feedback was called with skip_embedded=True
        mock_sql_instance.get_new_feedback.assert_called_once()
        call_kwargs = mock_sql_instance.get_new_feedback.call_args[1]
        assert call_kwargs['skip_embedded'] == True
        
        # Verify only 1 record was processed (fb003)
        assert stats["total_records"] == 1
        assert stats["embeddings_created"] == 1
        
        # Verify embed_texts was called with only fb003's text
        call_args = mock_embedder_instance.embed_texts.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] == "Size runs small, had to return"
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_skip_embedded_all_items_already_embedded(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config, sample_feedback_records
    ):
        """Test behavior when all items have already been embedded."""
        # Setup mocks - SQL query returns empty list after filtering
        mock_sql_instance = mock_sql_client.return_value
        mock_sql_instance.get_new_feedback.return_value = []
        mock_sql_instance.sync_embeddings_from_cosmos.return_value = 3  # All 3 items synced
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        mock_cosmos_instance.get_all_embedded_ids.return_value = [
            ("fb001", datetime(2024, 1, 1, tzinfo=timezone.utc)),
            ("fb002", datetime(2024, 1, 2, tzinfo=timezone.utc)),
            ("fb003", datetime(2024, 1, 3, tzinfo=timezone.utc))
        ]
        
        mock_cosmos_instance = mock_cosmos_client.return_value
        
        # Run pipeline with skip_embedded=True
        pipeline = IngestionPipeline(mock_config, skip_embedded=True)
        stats = pipeline.run()
        
        # Verify no records were processed
        assert stats["total_records"] == 0
        assert stats["embeddings_created"] == 0
        assert stats["tags_created"] == 0
    
    @patch('src.pipelines.ingest.FeedbackTagger')
    @patch('src.pipelines.ingest.Embedder')
    @patch('src.pipelines.ingest.CosmosClient')
    @patch('src.pipelines.ingest.SQLClient')
    def test_sync_embeddings_with_multiple_batches(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger,
        mock_config
    ):
        """Test that embedding sync works correctly with multiple batches."""
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
            [["Tag1"], ["Tag2"], ["Tag3"]],  # Batch 1
            [["Tag4"], ["Tag5"]]  # Batch 2
        ]
        
        # Run pipeline with batch_size=3
        pipeline = IngestionPipeline(mock_config)
        stats = pipeline.run(batch_size=3)
        
        # Verify insert_embedded_items was called twice (once per batch)
        assert mock_sql_instance.insert_embedded_items.call_count == 2
        
        # Verify correct items in each batch
        first_batch = mock_sql_instance.insert_embedded_items.call_args_list[0][0][0]
        assert len(first_batch) == 3
        assert first_batch[0]['feedback_id'] == "fb000"
        assert first_batch[1]['feedback_id'] == "fb001"
        assert first_batch[2]['feedback_id'] == "fb002"
        
        second_batch = mock_sql_instance.insert_embedded_items.call_args_list[1][0][0]
        assert len(second_batch) == 2
        assert second_batch[0]['feedback_id'] == "fb003"
        assert second_batch[1]['feedback_id'] == "fb004"
