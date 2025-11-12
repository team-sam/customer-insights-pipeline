"""Unit tests for the ChatAgent class."""
import pytest
import os
from src.agents.llm_agent import ChatAgent
from src.config.settings import Settings


@pytest.fixture
def real_config(pytestconfig):
    """Create a real configuration with OpenAI API key from command line or environment."""
    # Try to get API key from command line option first, then environment
    api_key = getattr(pytestconfig.option, "openai_api_key", None) or Settings().openai_api_key
    if not api_key:
        pytest.skip("OpenAI API key not provided via --openai-api-key or environment variable")
    
    # Create a minimal Settings object with required fields
    return type('Settings', (), {
        'openai_api_key': api_key,
        'openai_llm_model': 'gpt-3.5-turbo',
        'openai_embedding_model': 'text-embedding-3-small',
        'sql_server_host': 'dummy',
        'sql_server_database': 'dummy',
        'sql_server_username': 'dummy',
        'sql_server_password': 'dummy',
        'postgres_host': 'dummy',
        'postgres_database': 'dummy',
        'postgres_username': 'dummy',
        'postgres_password': 'dummy',
    })()


class TestChatAgent:
    """Test ChatAgent class."""
    
    def test_agent_initialization(self, real_config):
        """Test ChatAgent initializes correctly."""
        agent = ChatAgent(real_config)
        assert agent.config == real_config
        assert agent.model == "gpt-3.5-turbo"
        assert agent.client is not None
    
    def test_chat_single(self, real_config):
        """Test single prompt chat with real API."""
        agent = ChatAgent(real_config)
        result = agent.chat_single("Say 'Hello' and nothing else.")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_chat_with_messages(self, real_config):
        """Test chat with message list with real API."""
        agent = ChatAgent(real_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Reply with just 'Good' and nothing else."}
        ]
        result = agent.chat(messages)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_label_cluster(self, real_config):
        """Test cluster labeling with real API."""
        agent = ChatAgent(real_config)
        feedback_texts = [
            "Shoes leaked after one day",
            "Water got through the material",
            "Not waterproof at all"
        ]
        label = agent.label_cluster(feedback_texts)
        
        # Check that we get a reasonable label back
        assert isinstance(label, str)
        assert len(label) > 0
        # The label should be relatively short (2-5 words as per prompt)
        assert len(label.split()) <= 10  # Allow some flexibility
    
    def test_tag_feedback_single_tag(self, real_config):
        """Test tagging feedback with single tag using real API."""
        agent = ChatAgent(real_config)
        categories = ["Waterproof Leak", "Sizes not standard", "Too Heavy"]
        result = agent.tag_feedback("Water got in", categories, allow_multiple=False)
        
        assert isinstance(result, list)
        assert len(result) == 1
        # Result should be one of the categories or "Uncategorized"
        assert result[0] in categories or result[0] == "Uncategorized"
    
    def test_tag_feedback_multiple_tags(self, real_config):
        """Test tagging feedback with multiple tags using real API."""
        agent = ChatAgent(real_config)
        categories = ["Waterproof Leak", "Sizes not standard", "Too Heavy"]
        result = agent.tag_feedback("Leaking and too small", categories, allow_multiple=True)
        
        assert isinstance(result, list)
        assert len(result) >= 1  # Should have at least one tag
        # All results should be valid categories or "Uncategorized"
        for tag in result:
            assert tag in categories or tag == "Uncategorized"
    
    def test_parse_json_array_valid_json(self, real_config):
        """Test parsing valid JSON array."""
        agent = ChatAgent(real_config)
        categories = ["Cat1", "Cat2", "Cat3"]
        result = agent._parse_json_array('["Cat1", "Cat2"]', categories)
        
        assert result == ["Cat1", "Cat2"]
    
    def test_parse_json_array_with_markdown(self, real_config):
        """Test parsing JSON array with markdown code blocks."""
        agent = ChatAgent(real_config)
        categories = ["Cat1", "Cat2", "Cat3"]
        result = agent._parse_json_array('```json\n["Cat1", "Cat3"]\n```', categories)
        
        assert result == ["Cat1", "Cat3"]
    
    def test_parse_json_array_empty(self, real_config):
        """Test parsing empty JSON array."""
        agent = ChatAgent(real_config)
        categories = ["Cat1", "Cat2"]
        result = agent._parse_json_array('[]', categories)
        
        assert result == []
    
    def test_parse_single_tag_exact_match(self, real_config):
        """Test parsing single tag with exact match."""
        agent = ChatAgent(real_config)
        categories = ["Waterproof Leak", "Too Heavy"]
        result = agent._parse_single_tag("Waterproof Leak", categories)
        
        assert result == "Waterproof Leak"
    
    def test_parse_single_tag_case_insensitive(self, real_config):
        """Test parsing single tag with case-insensitive match."""
        agent = ChatAgent(real_config)
        categories = ["Waterproof Leak", "Too Heavy"]
        result = agent._parse_single_tag("waterproof leak", categories)
        
        assert result == "Waterproof Leak"
    
    def test_parse_single_tag_no_match(self, real_config):
        """Test parsing single tag with no match returns Uncategorized."""
        agent = ChatAgent(real_config)
        categories = ["Waterproof Leak", "Too Heavy"]
        result = agent._parse_single_tag("Something Random", categories)
        
        assert result == "Uncategorized"
    
    def test_tag_feedback_batch(self, real_config):
        """Test batch tagging of feedback with real API."""
        agent = ChatAgent(real_config)
        feedback_texts = ["Leaking shoes", "Very heavy"]
        categories = ["Waterproof Leak", "Too Heavy", "Sizes not standard"]
        result = agent.tag_feedback_batch(feedback_texts, categories, allow_multiple=True)
        
        assert len(result) == 2
        # Each result should be a list of tags
        for tags in result:
            assert isinstance(tags, list)
            assert len(tags) >= 1
            # All tags should be valid categories or "Uncategorized"
            for tag in tags:
                assert tag in categories or tag == "Uncategorized"
