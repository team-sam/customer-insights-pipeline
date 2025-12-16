"""Unit tests for the RecursiveClusteringPipeline class."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.pipelines.umap_clustering import RecursiveClusteringPipeline


class TestRecursiveClusteringPipeline:
    """Test RecursiveClusteringPipeline class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock Settings config."""
        config = Mock()
        config.sql_connection_string = "mock_connection"
        config.cosmos_endpoint = "mock_endpoint"
        config.cosmos_key = "mock_key"
        config.cosmos_database = "mock_db"
        config.cosmos_container = "mock_container"
        return config
    
    @pytest.fixture
    def pipeline(self, mock_config):
        """Create a RecursiveClusteringPipeline instance for testing."""
        # Mock the SQL and Cosmos clients
        with patch('src.pipelines.umap_clustering.SQLClient'):
            with patch('src.pipelines.umap_clustering.CosmosClient'):
                pipeline = RecursiveClusteringPipeline(
                    config=mock_config,
                    recursive_depth=3,
                    min_cluster_size_pct=0.02,
                    min_sample_pct=0.003,
                    hdbscan_metric="euclidean",
                    local_mode=False
                )
                return pipeline
    
    def test_constructor_stores_base_parameters(self, pipeline):
        """Test that constructor stores base parameters correctly."""
        assert hasattr(pipeline, 'base_umap_params')
        assert hasattr(pipeline, 'base_min_cluster_pct')
        assert hasattr(pipeline, 'base_min_sample_pct')
        assert hasattr(pipeline, 'hdbscan_metric')
        
        assert pipeline.base_min_cluster_pct == 0.02
        assert pipeline.base_min_sample_pct == 0.003
        assert pipeline.hdbscan_metric == "euclidean"
    
    def test_get_adaptive_umap_params_basic(self, pipeline):
        """Test basic functionality of _get_adaptive_umap_params."""
        params = pipeline._get_adaptive_umap_params(depth=0, n_points=1000)
        
        assert 'n_neighbors' in params
        assert 'min_dist' in params
        assert isinstance(params['n_neighbors'], int)
        assert params['n_neighbors'] >= 5
        assert params['n_neighbors'] <= 30
    
    def test_get_adaptive_umap_params_square_root_scaling(self, pipeline):
        """Test that n_neighbors scales with square root of points."""
        # For 10000 points: sqrt(10000) / 2 = 50, capped at 30
        params_large = pipeline._get_adaptive_umap_params(depth=0, n_points=10000)
        assert params_large['n_neighbors'] == 30  # Should be capped
        
        # For 100 points: sqrt(100) / 2 = 5
        params_small = pipeline._get_adaptive_umap_params(depth=0, n_points=100)
        assert params_small['n_neighbors'] >= 5
        
        # For 400 points: sqrt(400) / 2 = 10
        params_medium = pipeline._get_adaptive_umap_params(depth=0, n_points=400)
        assert params_medium['n_neighbors'] == 10
    
    def test_get_adaptive_umap_params_depth_adjustment(self, pipeline):
        """Test that depth reduces n_neighbors."""
        params_depth0 = pipeline._get_adaptive_umap_params(depth=0, n_points=1000)
        params_depth1 = pipeline._get_adaptive_umap_params(depth=1, n_points=1000)
        params_depth2 = pipeline._get_adaptive_umap_params(depth=2, n_points=1000)
        
        # Deeper levels should have smaller n_neighbors
        assert params_depth1['n_neighbors'] < params_depth0['n_neighbors']
        assert params_depth2['n_neighbors'] < params_depth1['n_neighbors']
    
    def test_get_adaptive_umap_params_min_dist(self, pipeline):
        """Test that min_dist adjusts based on cluster size."""
        params_tiny = pipeline._get_adaptive_umap_params(depth=0, n_points=50)
        params_small = pipeline._get_adaptive_umap_params(depth=0, n_points=200)
        params_large = pipeline._get_adaptive_umap_params(depth=0, n_points=1000)
        
        assert params_tiny['min_dist'] == 0.0  # < 100
        assert params_small['min_dist'] == 0.05  # 100-500
        assert params_large['min_dist'] == 0.1  # > 500
    
    def test_get_adaptive_umap_params_percentage_constraint(self, pipeline):
        """Test that n_neighbors never exceeds 5% of dataset."""
        params = pipeline._get_adaptive_umap_params(depth=0, n_points=100)
        # 5% of 100 = 5
        assert params['n_neighbors'] <= 5
    
    def test_get_adaptive_hdbscan_params_basic(self, pipeline):
        """Test basic functionality of _get_adaptive_hdbscan_params."""
        params = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=1000)
        
        assert 'min_cluster_size' in params
        assert 'min_samples' in params
        assert 'metric' in params
        assert params['metric'] == "euclidean"
        assert isinstance(params['min_cluster_size'], int)
        assert isinstance(params['min_samples'], int)
    
    def test_get_adaptive_hdbscan_params_depth_decay(self, pipeline):
        """Test that min_cluster_size decreases with depth."""
        params_depth0 = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=1000)
        params_depth1 = pipeline._get_adaptive_hdbscan_params(depth=1, n_points=1000)
        params_depth2 = pipeline._get_adaptive_hdbscan_params(depth=2, n_points=1000)
        
        # Deeper levels should have smaller min_cluster_size
        assert params_depth1['min_cluster_size'] < params_depth0['min_cluster_size']
        assert params_depth2['min_cluster_size'] < params_depth1['min_cluster_size']
    
    def test_get_adaptive_hdbscan_params_size_bounds(self, pipeline):
        """Test that min_cluster_size respects size-based bounds."""
        # Very small clusters (< 100)
        params_tiny = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=50)
        assert params_tiny['min_cluster_size'] >= 5
        assert params_tiny['min_cluster_size'] <= 50 // 5
        
        # Small clusters (100-500)
        params_small = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=200)
        assert params_small['min_cluster_size'] >= 8
        assert params_small['min_cluster_size'] <= 200 // 8
        
        # Medium clusters (500-2000)
        params_medium = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=1000)
        assert params_medium['min_cluster_size'] >= 10
        assert params_medium['min_cluster_size'] <= 1000 // 10
        
        # Large clusters (> 2000)
        params_large = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=5000)
        assert params_large['min_cluster_size'] >= 15
        assert params_large['min_cluster_size'] <= 5000 // 10
    
    def test_get_adaptive_hdbscan_params_min_samples(self, pipeline):
        """Test that min_samples is properly calculated."""
        params = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=1000)
        
        # min_samples should be at least 3 and at most min_cluster_size
        assert params['min_samples'] >= 3
        assert params['min_samples'] <= params['min_cluster_size']
        # Should be approximately half of min_cluster_size
        assert params['min_samples'] <= params['min_cluster_size'] // 2 + 1
    
    def test_get_adaptive_hdbscan_params_epsilon_addition(self, pipeline):
        """Test that cluster_selection_epsilon is added for deep small clusters."""
        # Depth 2, small cluster (< 500)
        params_deep_small = pipeline._get_adaptive_hdbscan_params(depth=2, n_points=200)
        assert 'cluster_selection_epsilon' in params_deep_small
        assert params_deep_small['cluster_selection_epsilon'] == 0.1
        
        # Depth 1, should not have epsilon
        params_shallow = pipeline._get_adaptive_hdbscan_params(depth=1, n_points=200)
        assert 'cluster_selection_epsilon' not in params_shallow
        
        # Depth 2, large cluster (>= 500)
        params_deep_large = pipeline._get_adaptive_hdbscan_params(depth=2, n_points=1000)
        assert 'cluster_selection_epsilon' not in params_deep_large
    
    def test_should_recurse_max_depth(self, pipeline):
        """Test that _should_recurse respects max depth."""
        # At max depth, should not recurse
        assert pipeline._should_recurse(depth=3, n_points=1000) == False
        
        # Below max depth, should recurse (assuming other conditions pass)
        assert pipeline._should_recurse(depth=2, n_points=1000) == True
    
    def test_should_recurse_min_points_threshold(self, pipeline):
        """Test that _should_recurse respects adaptive minimum points threshold."""
        # Depth 0: threshold = 50 * (1.3^0) = 50
        assert pipeline._should_recurse(depth=0, n_points=100) == True
        assert pipeline._should_recurse(depth=0, n_points=40) == False
        
        # Depth 1: threshold = 50 * (1.3^1) = 65
        assert pipeline._should_recurse(depth=1, n_points=100) == True
        assert pipeline._should_recurse(depth=1, n_points=60) == False
        
        # Depth 2: threshold = 50 * (1.3^2) = 84.5
        assert pipeline._should_recurse(depth=2, n_points=100) == True
        assert pipeline._should_recurse(depth=2, n_points=80) == False
    
    def test_should_recurse_quality_checks(self, pipeline):
        """Test that _should_recurse checks cluster quality metrics."""
        # Test over-fragmentation: n_clusters > n_points / 10
        quality_fragmented = {'noise_ratio': 0.1, 'mean_persistence': 0.5}
        assert pipeline._should_recurse(
            depth=1, n_points=100, n_clusters=15, cluster_quality=quality_fragmented
        ) == False
        
        # Test excessive noise: noise_ratio > 0.5
        quality_noisy = {'noise_ratio': 0.6, 'mean_persistence': 0.5}
        assert pipeline._should_recurse(
            depth=1, n_points=1000, n_clusters=5, cluster_quality=quality_noisy
        ) == False
        
        # Test instability: mean_persistence < 0.05
        quality_unstable = {'noise_ratio': 0.1, 'mean_persistence': 0.04}
        assert pipeline._should_recurse(
            depth=1, n_points=1000, n_clusters=5, cluster_quality=quality_unstable
        ) == False
        
        # Test good quality: should recurse
        quality_good = {'noise_ratio': 0.2, 'mean_persistence': 0.5}
        assert pipeline._should_recurse(
            depth=1, n_points=1000, n_clusters=5, cluster_quality=quality_good
        ) == True
    
    def test_should_recurse_without_quality_metrics(self, pipeline):
        """Test that _should_recurse works without quality metrics."""
        # Should only check depth and points
        assert pipeline._should_recurse(depth=1, n_points=1000) == True
        assert pipeline._should_recurse(depth=1, n_points=40) == False
        assert pipeline._should_recurse(depth=3, n_points=1000) == False
    
    def test_custom_hdbscan_metric(self, mock_config):
        """Test that custom hdbscan_metric is properly stored."""
        with patch('src.pipelines.umap_clustering.SQLClient'):
            with patch('src.pipelines.umap_clustering.CosmosClient'):
                pipeline = RecursiveClusteringPipeline(
                    config=mock_config,
                    hdbscan_metric="cosine"
                )
                
                params = pipeline._get_adaptive_hdbscan_params(depth=0, n_points=1000)
                assert params['metric'] == "cosine"
    
    def test_style_based_clustering_partitioning(self, mock_config):
        """Test that clustering partitions data by both source and style fields."""
        with patch('src.pipelines.umap_clustering.SQLClient') as mock_sql:
            with patch('src.pipelines.umap_clustering.CosmosClient') as mock_cosmos:
                pipeline = RecursiveClusteringPipeline(
                    config=mock_config,
                    recursive_depth=1,
                    local_mode=False
                )
                
                # Mock data with different sources and styles
                mock_embeddings = [
                    ('id1', [0.1] * 1536, 'review', 'text1', 'style_a'),
                    ('id2', [0.2] * 1536, 'review', 'text2', 'style_a'),
                    ('id3', [0.3] * 1536, 'chat', 'text3', 'style_a'),
                    ('id4', [0.4] * 1536, 'chat', 'text4', 'style_b'),
                    ('id5', [0.5] * 1536, 'return', 'text5', 'style_b'),
                    ('id6', [0.6] * 1536, 'review', 'text6', None),
                ]
                
                mock_cosmos_instance = mock_cosmos.return_value
                mock_cosmos_instance.get_all_embeddings.return_value = mock_embeddings
                
                # Run the pipeline
                result = pipeline.run()
                
                # Verify clustering was called
                assert result['total_records'] == 6
                
                # Check that cluster labels contain both source and style prefixes
                cluster_labels = list(result['clusters'].keys())
                
                # Verify we have source_ and style_ in labels
                has_source_prefix = any('source_' in label for label in cluster_labels)
                has_style_prefix = any('style_' in label for label in cluster_labels)
                
                assert has_source_prefix, f"Expected source_ prefix in labels, got: {cluster_labels}"
                assert has_style_prefix, f"Expected style_ prefix in labels, got: {cluster_labels}"
                
                # Verify hierarchical structure: source_{source}.style_{style}
                for label in cluster_labels:
                    if not label.endswith('.unclustered'):
                        # Should start with source_
                        assert label.startswith('source_'), f"Expected label to start with 'source_', got: {label}"
                        # Should contain .style_ 
                        assert '.style_' in label, f"Expected label to contain '.style_', got: {label}"
