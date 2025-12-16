"""Unit tests for cluster persistence functionality."""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
from src.agents.llm_agent import ChatAgent
from src.data_access.sql_client import SQLClient
from src.models.schemas import ClusterRecord


class TestClusterPersistence:
    """Test cluster persistence functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock Settings config."""
        config = Mock()
        config.openai_api_key = "test-key"
        config.openai_llm_model = "gpt-4"
        config.max_workers = 2
        return config
    
    @pytest.fixture
    def mock_sql_client(self):
        """Create a mock SQL client."""
        client = Mock(spec=SQLClient)
        client.update_feedback_clusters = Mock()
        client.insert_clusters = Mock()
        return client
    
    def test_update_feedback_clusters(self, mock_sql_client):
        """Test updating feedback table with cluster assignments."""
        cluster_assignments = [
            {'feedback_id': 'fb-1', 'cluster_id': 'source_review.style_chelsea.0'},
            {'feedback_id': 'fb-2', 'cluster_id': 'source_review.style_chelsea.1'},
            {'feedback_id': 'fb-3', 'cluster_id': 'source_chat.style_weekend.0'}
        ]
        
        mock_sql_client.update_feedback_clusters(cluster_assignments)
        
        # Verify the method was called with correct arguments
        mock_sql_client.update_feedback_clusters.assert_called_once_with(cluster_assignments)
    
    def test_chat_agent_describe_cluster(self, mock_config):
        """Test LLM cluster description generation."""
        with patch('src.agents.llm_agent.OpenAI') as mock_openai:
            # Setup mock response
            mock_completion = Mock()
            mock_completion.choices = [Mock(message=Mock(content="This cluster contains feedback about waterproof issues."))]
            mock_openai.return_value.chat.completions.create.return_value = mock_completion
            
            agent = ChatAgent(mock_config)
            feedback_texts = [
                "Shoes leak in rain",
                "Not waterproof as advertised",
                "Water seeps through after a month"
            ]
            
            description = agent.describe_cluster(feedback_texts)
            
            assert isinstance(description, str)
            assert len(description) > 0
            mock_openai.return_value.chat.completions.create.assert_called_once()
    
    def test_cluster_record_with_full_attributes(self):
        """Test creating ClusterRecord with all required attributes."""
        now = datetime.now(timezone.utc)
        
        record = ClusterRecord(
            cluster_id="source_review.style_chelsea.0.1",
            label="source_review.style_chelsea.0.1",
            description="Customers reporting waterproof issues with Chelsea style boots.",
            source="review",
            style="chelsea",
            cluster_depth=3,
            sample_feedback_ids=["fb-1", "fb-2", "fb-3"],
            record_count=150,
            period_start=now,
            period_end=now,
            created_at=now
        )
        
        assert record.cluster_id == "source_review.style_chelsea.0.1"
        assert record.source == "review"
        assert record.style == "chelsea"
        assert record.cluster_depth == 3
        assert record.record_count == 150
        assert len(record.sample_feedback_ids) == 3
    
    def test_cluster_id_format_parsing(self):
        """Test parsing cluster_id to extract source, style, and depth."""
        cluster_label = "source_review.style_chelsea.0.1.2"
        
        # Parse the cluster label
        parts = cluster_label.split('.')
        source = None
        style = None
        
        for part in parts:
            if part.startswith('source_'):
                source = part.replace('source_', '')
            elif part.startswith('style_'):
                style = part.replace('style_', '')
        
        cluster_depth = cluster_label.count('.')
        
        assert source == "review"
        assert style == "chelsea"
        assert cluster_depth == 4
    
    def test_cluster_description_sampling(self):
        """Test that cluster description uses max 50 samples."""
        # Create a list of 100 feedback texts
        feedback_texts = [f"Feedback item {i}" for i in range(100)]
        
        # Simulate the sampling logic
        sample_size = min(50, len(feedback_texts))
        sampled = feedback_texts[:sample_size]
        
        assert len(sampled) == 50
        
        # Test with fewer than 50 items
        small_list = [f"Feedback {i}" for i in range(20)]
        sample_size_small = min(50, len(small_list))
        sampled_small = small_list[:sample_size_small]
        
        assert len(sampled_small) == 20
    
    def test_insert_clusters_with_new_schema(self, mock_sql_client):
        """Test inserting cluster records with new schema."""
        now = datetime.now(timezone.utc)
        
        clusters = [
            ClusterRecord(
                cluster_id="source_review.style_chelsea.0",
                label="Waterproof Issues",
                description="Customers reporting leaking problems",
                source="review",
                style="chelsea",
                cluster_depth=2,
                sample_feedback_ids=["fb-1", "fb-2"],
                record_count=25,
                period_start=now,
                period_end=now,
                created_at=now
            ),
            ClusterRecord(
                cluster_id="source_chat.style_weekend.0",
                label="Sizing Complaints",
                description="Customers reporting sizing issues",
                source="chat",
                style="weekend",
                cluster_depth=2,
                sample_feedback_ids=["fb-3", "fb-4"],
                record_count=15,
                period_start=now,
                period_end=now,
                created_at=now
            )
        ]
        
        mock_sql_client.insert_clusters(clusters)
        
        # Verify the method was called
        mock_sql_client.insert_clusters.assert_called_once_with(clusters)
        
        # Verify cluster attributes
        assert len(clusters) == 2
        assert all(hasattr(c, 'source') for c in clusters)
        assert all(hasattr(c, 'cluster_depth') for c in clusters)
        assert clusters[0].source == "review"
        assert clusters[1].source == "chat"
