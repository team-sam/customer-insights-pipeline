"""Integration tests to verify BackfillPipeline uses FeedbackTagger categories."""
import pytest
from unittest.mock import Mock, patch
from src.pipelines.backfill import BackfillPipeline
from src.config.settings import Settings
from src.tagging.tagger import FeedbackTagger


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Settings)
    config.openai_api_key = "test-api-key"
    config.openai_embedding_model = "text-embedding-3-small"
    config.openai_llm_model = "gpt-5-nano"
    config.batch_size = 100
    return config


class TestBackfillTaggerIntegration:
    """Test BackfillPipeline integration with FeedbackTagger."""
    
    @patch('src.pipelines.backfill.FeedbackTagger')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_backfill_uses_tagger_default_categories(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger, mock_config
    ):
        """Test that BackfillPipeline uses FeedbackTagger's default categories when none provided."""
        # Initialize pipeline without custom categories
        pipeline = BackfillPipeline(mock_config)
        
        # Verify FeedbackTagger was instantiated with custom_categories=None
        mock_tagger.assert_called_once_with(mock_config, custom_categories=None)
        
    @patch('src.pipelines.backfill.FeedbackTagger')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_backfill_passes_custom_categories_to_tagger(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger, mock_config
    ):
        """Test that BackfillPipeline passes custom categories to FeedbackTagger."""
        custom_categories = ["Category1", "Category2", "Category3"]
        
        # Initialize pipeline with custom categories
        pipeline = BackfillPipeline(mock_config, categories=custom_categories)
        
        # Verify FeedbackTagger was instantiated with the custom categories
        mock_tagger.assert_called_once_with(mock_config, custom_categories=custom_categories)
    
    @patch('src.pipelines.backfill.FeedbackTagger')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_backfill_uses_tagger_not_chatagent(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger, mock_config
    ):
        """Test that BackfillPipeline uses FeedbackTagger instead of ChatAgent."""
        # Initialize pipeline
        pipeline = BackfillPipeline(mock_config)
        
        # Verify pipeline has tagger attribute (not llm_agent)
        assert hasattr(pipeline, 'tagger')
        assert not hasattr(pipeline, 'llm_agent')
        
    @patch('src.pipelines.backfill.FeedbackTagger')
    @patch('src.pipelines.backfill.Embedder')
    @patch('src.pipelines.backfill.CosmosClient')
    @patch('src.pipelines.backfill.SQLClient')
    def test_backfill_no_hardcoded_categories(
        self, mock_sql_client, mock_cosmos_client, mock_embedder, mock_tagger, mock_config
    ):
        """Test that BackfillPipeline does not have hardcoded categories."""
        # Initialize pipeline
        pipeline = BackfillPipeline(mock_config)
        
        # Verify pipeline does not have categories attribute
        # Categories should now be managed by the FeedbackTagger
        assert not hasattr(pipeline, 'categories')
    
    def test_tagger_has_comprehensive_categories(self):
        """Test that FeedbackTagger has comprehensive default categories."""
        # Verify FeedbackTagger has more comprehensive categories than the old backfill had
        tagger_categories = FeedbackTagger.DEFAULT_CATEGORIES
        
        # The tagger should have 30 categories (much more than the 10 that backfill had)
        assert len(tagger_categories) == 30
        
        # Verify some key categories are present
        expected_categories = [
            "Waterproof Leak",
            "Upper Knit Separation",
            "Insole Issue",
            "Inner Lining Rip",
            "Glue Gap",
            "Discolouration",
            "Sizes not standard",
            "Toe Area too narrow",
            "Too Heavy"
        ]
        
        for category in expected_categories:
            assert category in tagger_categories, f"Expected category '{category}' not found in tagger categories"
