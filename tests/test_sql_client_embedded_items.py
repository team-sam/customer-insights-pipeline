"""Unit tests for SQL client embedded items methods."""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timezone
from src.data_access.sql_client import SQLClient
from src.config.settings import Settings


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Settings)
    config.sql_server_host = "test-server"
    config.sql_server_username = "test-user"
    config.sql_server_password = "test-pass"
    config.sql_server_database = "test-db"
    return config


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)
    return conn, cursor


def _create_mock_connection():
    """Helper to create mock connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)
    return conn, cursor


class TestSQLClientEmbeddedItems:
    """Test SQL client methods for embedded items tracking."""
    
    @patch('src.data_access.sql_client.pymssql.connect')
    def test_initialize_embedded_items_table(self, mock_connect, mock_config):
        """Test creating the embedded_items table."""
        conn, cursor = _create_mock_connection()
        mock_connect.return_value = conn
        
        client = SQLClient(mock_config)
        client.initialize_embedded_items_table()
        
        # Verify connection was established
        mock_connect.assert_called_once()
        
        # Verify SQL was executed
        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args[0][0]
        assert "CREATE TABLE customer_insights.embedded_items" in call_args
        assert "feedback_id VARCHAR(255) PRIMARY KEY" in call_args
        assert "embedded_at DATETIME NOT NULL" in call_args
        
        # Verify commit was called
        conn.commit.assert_called_once()
    
    @patch('src.data_access.sql_client.pymssql.connect')
    def test_insert_embedded_items_single(self, mock_connect, mock_config):
        """Test inserting a single embedded item."""
        conn, cursor = _create_mock_connection()
        mock_connect.return_value = conn
        
        client = SQLClient(mock_config)
        client.connect()
        
        items = [
            {
                'feedback_id': 'fb001',
                'embedded_at': datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
            }
        ]
        
        client.insert_embedded_items(items)
        
        # Verify cursor.execute was called once
        cursor.execute.assert_called_once()
        
        # Verify the SQL statement uses MERGE
        call_args = cursor.execute.call_args[0][0]
        assert "MERGE INTO customer_insights.embedded_items" in call_args
        
        # Verify the values passed
        call_values = cursor.execute.call_args[0][1]
        assert call_values == ('fb001', datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc))
        
        # Verify commit was called
        conn.commit.assert_called_once()
    
    @patch('src.data_access.sql_client.pymssql.connect')
    def test_insert_embedded_items_multiple(self, mock_connect, mock_config):
        """Test inserting multiple embedded items."""
        conn, cursor = _create_mock_connection()
        mock_connect.return_value = conn
        
        client = SQLClient(mock_config)
        client.connect()
        
        items = [
            {'feedback_id': 'fb001', 'embedded_at': datetime(2024, 1, 15, tzinfo=timezone.utc)},
            {'feedback_id': 'fb002', 'embedded_at': datetime(2024, 1, 16, tzinfo=timezone.utc)},
            {'feedback_id': 'fb003', 'embedded_at': datetime(2024, 1, 17, tzinfo=timezone.utc)},
        ]
        
        client.insert_embedded_items(items)
        
        # Verify cursor.execute was called 3 times (once per item)
        assert cursor.execute.call_count == 3
        
        # Verify commit was called once at the end
        conn.commit.assert_called_once()
    
    @patch('src.data_access.sql_client.pymssql.connect')
    def test_get_embedded_feedback_ids_empty(self, mock_connect, mock_config):
        """Test getting embedded feedback IDs when table is empty."""
        conn, cursor = _create_mock_connection()
        mock_connect.return_value = conn
        cursor.fetchall.return_value = []
        
        client = SQLClient(mock_config)
        client.connect()
        
        result = client.get_embedded_feedback_ids()
        
        # Verify query was executed
        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args[0][0]
        assert "SELECT feedback_id" in call_args
        assert "FROM customer_insights.embedded_items" in call_args
        
        # Verify empty list is returned
        assert result == []
    
    @patch('src.data_access.sql_client.pymssql.connect')
    def test_get_embedded_feedback_ids_with_data(self, mock_connect, mock_config):
        """Test getting embedded feedback IDs when table has data."""
        conn, cursor = _create_mock_connection()
        mock_connect.return_value = conn
        
        # Mock fetchall to return rows
        cursor.fetchall.return_value = [
            ('fb001',),
            ('fb002',),
            ('fb003',),
        ]
        
        client = SQLClient(mock_config)
        client.connect()
        
        result = client.get_embedded_feedback_ids()
        
        # Verify correct IDs are returned
        assert result == ['fb001', 'fb002', 'fb003']
    
    @patch('src.data_access.sql_client.pymssql.connect')
    def test_insert_embedded_items_with_existing_connection(self, mock_connect, mock_config):
        """Test that insert_embedded_items uses existing connection if available."""
        conn, cursor = _create_mock_connection()
        mock_connect.return_value = conn
        
        client = SQLClient(mock_config)
        # Pre-establish connection
        client.connect()
        mock_connect.reset_mock()
        
        items = [{'feedback_id': 'fb001', 'embedded_at': datetime(2024, 1, 15, tzinfo=timezone.utc)}]
        client.insert_embedded_items(items)
        
        # Verify connect was NOT called again
        mock_connect.assert_not_called()
        
        # Verify execute was still called
        cursor.execute.assert_called_once()
