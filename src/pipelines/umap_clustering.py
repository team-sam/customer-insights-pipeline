"""
Customer feedback clustering pipeline using UMAP + HDBSCAN with recursive clustering.
Based on the iterative dimensionality reduction and density-based clustering approach.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
import logging
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast

import pandas as pd
import numpy as np
import umap
import hdbscan

from src.config.settings import Settings
from src.data_access.sql_client import SQLClient
from src.data_access.cosmos_client import CosmosClient
from src.models.schemas import FeedbackRecord, ClusterRecord
from src.agents.llm_agent import ChatAgent

logger = logging.getLogger(__name__)


class RecursiveClusteringPipeline:
    """
    Pipeline for clustering feedback using UMAP + HDBSCAN with recursive zooming.
    Implements the approach from the article for customer segmentation.
    """

    def __init__(
        self,
        config: Settings,
        umap_params: Optional[Dict[str, Any]] = None,
        recursive_depth: int = 1,
        min_cluster_size_pct: float = 0.02,
        min_sample_pct: float = 0.003,
        hdbscan_metric: str = "euclidean",
        local_mode: bool = False,
        output_dir: str = "./cluster_output"
    ):
        self.config = config
        self.sql_client = SQLClient(config)
        self.cosmos_client = CosmosClient(config)
        
        # Default UMAP parameters (base values for adaptation)
        self.base_umap_params = umap_params or {
            'n_neighbors': 15,
            'n_components': 8,
            'metric': 'cosine'
        }
        
        
        self.recursive_depth = recursive_depth
        self.base_min_cluster_pct = min_cluster_size_pct
        self.base_min_sample_pct = min_sample_pct 
        self.hdbscan_metric = hdbscan_metric
        self.cluster_hierarchy = {}
        self.local_mode = local_mode
        self.output_dir = output_dir
        
        
        if self.local_mode:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Local mode enabled. Outputs will be saved to: {self.output_dir}")
            
            # Import visualization libraries only in local mode
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                self.plt = plt
                self.sns = sns
                self.sns.set_style("whitegrid")
            except ImportError:
                logger.warning("matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
                self.local_mode = False


    def _get_adaptive_umap_params(self, depth: int, n_points: int) -> Dict[str, Any]:
        """
        Calculate UMAP parameters dynamically based on cluster size and depth.
        
        Args:
            depth: Current recursion depth
            n_points: Number of points in the current cluster
            
        Returns:
            Complete dictionary of UMAP parameters ready to pass to UMAP constructor
        """
        # Start with a copy of base parameters
        params = self.base_umap_params.copy()
        
        # Calculate size-based n_neighbors using square root rule
        n_neighbors = np.sqrt(n_points) / 2
        
        # Apply depth adjustment multiplier (15% reduction per level)
        n_neighbors = n_neighbors * (0.85 ** depth)
        
        # Enforce practical bounds
        n_neighbors = np.clip(n_neighbors, 5, 30)
        
        # Percentage-based constraint: never exceed 5% of dataset size
        max_neighbors = int(n_points * 0.05)
        n_neighbors = min(n_neighbors, max_neighbors)

        min_neighbors_required = 2 * params['n_components']
        if n_neighbors < min_neighbors_required:
            n_neighbors = min_neighbors_required 
        
        # Convert to integer
        params['n_neighbors'] = int(n_neighbors)
        
        # Adjust min_dist based on cluster size
        if n_points < 100:
            params['min_dist'] = 0.0
            params['spread'] = 1.0
        elif n_points < 500:
            params['min_dist'] = 0.05
            params['spread'] = 1.5
        else:
            params['min_dist'] = 0.1
            params['spread'] = 2.0

        return params


    def _get_adaptive_hdbscan_params(self, depth: int, n_points: int) -> Dict[str, Any]:
        """
        Calculate HDBSCAN parameters dynamically based on cluster size and depth.
        
        Args:
            depth: Current recursion depth
            n_points: Number of points in the current cluster
            
        Returns:
            Complete dictionary of HDBSCAN parameters ready to pass to HDBSCAN constructor
        """
        # Calculate depth-adjusted percentage (exponential decay)
        adjusted_pct = self.base_min_cluster_pct * (0.6 ** depth)
        
        # Convert to absolute number
        min_cluster_size = int(adjusted_pct * n_points)
        
        # Define size-based bounds
        if n_points < 100:
            min_bound, max_bound = 5, n_points // 5
        elif n_points < 500:
            min_bound, max_bound = 8, n_points // 8
        elif n_points < 2000:
            min_bound, max_bound = 10, n_points // 10
        else:
            min_bound, max_bound = 15, n_points // 10
        
        # Clip to bounds
        min_cluster_size = np.clip(min_cluster_size, min_bound, max_bound)
        
        # Calculate min_samples (half of min_cluster_size, but at least 3)
        min_samples = max(3, min(min_cluster_size // 2, min_cluster_size))
        
        # Build parameter dictionary
        params = {
            'min_cluster_size': int(min_cluster_size),
            'min_samples': int(min_samples),
            'metric': self.hdbscan_metric
        }
        
        # Add cluster_selection_epsilon for deeper levels with small clusters
        if depth >= 2 and n_points < 500:
            params['cluster_selection_epsilon'] = 0.1
        
        return params

    def _should_recurse(self, depth: int, n_points: int, n_clusters: int = 0, 
                    cluster_quality: Optional[Dict[str, float]] = None) -> bool:
        """Make smart decisions about when to stop recursing."""
        
        # Check max depth
        if depth >= self.recursive_depth:
            return False
        
        # Calculate adaptive minimum points threshold
        min_points_threshold = 50 * (1.3 ** depth)
        if n_points < min_points_threshold:
            return False
        
        # Check cluster quality if provided
        if cluster_quality is not None:
            # NEW: More aggressive persistence check
            mean_persistence = cluster_quality.get('mean_persistence', 1.0)
            
            # At depth 0-1: Be lenient (threshold = 0.05)
            # At depth 2-3: Be stricter (threshold = 0.10)
            # At depth 4+: Be very strict (threshold = 0.15)
            persistence_threshold = 0.05 + (0.025 * min(depth, 4))
            
            if mean_persistence < persistence_threshold:
                logger.info(f"Stopping recursion at depth {depth}: mean_persistence={mean_persistence:.3f} < threshold={persistence_threshold:.3f}")
                return False
            
            # Check for over-fragmentation
            if n_clusters > n_points / 10:
                return False
            
            # Check for excessive noise
            if cluster_quality.get('noise_ratio', 0) > 0.5:
                return False
        
        return True

    def _evaluate_umap_quality(self, original_embeddings: np.ndarray, reduced_embeddings: np.ndarray, n_neighbors: int = 15) -> Dict[str, float]:

        from sklearn.manifold import trustworthiness

        n_points = original_embeddings.shape[0]
        if n_points > 5000:
            sample_size = 5000
            idx = np.random.RandomState(42).choice(n_points, sample_size, replace=False)
            trust = trustworthiness(original_embeddings[idx], reduced_embeddings[idx], n_neighbors=n_neighbors)
        else:
            trust = trustworthiness(original_embeddings, reduced_embeddings, n_neighbors=n_neighbors)

        return {
            'trustworthiness': float(trust)
        }

    def _evaluate_clustering_quality(self, labels: np.ndarray, clusterer: hdbscan.HDBSCAN) -> Dict[str, float]:
        """
        Evaluate clustering quality with metrics appropriate for UMAP + HDBSCAN.
        """
        metrics = {}
        
        # HDBSCAN Native Metrics (best for density-based clustering)
        metrics['relative_validity'] = float(clusterer.relative_validity_) if hasattr(clusterer, 'relative_validity_') else None
        
        # Cluster persistence (how stable/strong are the clusters?)
        if hasattr(clusterer, 'cluster_persistence_'):
            metrics['min_persistence'] = float(clusterer.cluster_persistence_.min())
            metrics['mean_persistence'] = float(clusterer.cluster_persistence_.mean())
        
        # Membership probabilities (how confident is HDBSCAN about assignments?)
        if hasattr(clusterer, 'probabilities_'):
            metrics['mean_probability'] = float(clusterer.probabilities_.mean())
            metrics['low_confidence_ratio'] = float((clusterer.probabilities_ < 0.5).sum() / len(clusterer.probabilities_))
        
        # Basic cluster statistics
        mask = labels != -1
        metrics['n_clusters'] = len(np.unique(labels[mask])) if mask.sum() > 0 else 0
        metrics['noise_ratio'] = float((labels == -1).sum() / len(labels))
        
        # Cluster size distribution
        if metrics['n_clusters'] > 0:
            cluster_sizes = pd.Series(labels[mask]).value_counts()
            metrics['largest_cluster_ratio'] = float(cluster_sizes.max() / mask.sum())
            metrics['smallest_cluster_size'] = int(cluster_sizes.min())
            metrics['mean_cluster_size'] = float(cluster_sizes.mean())
            metrics['cluster_size_std'] = float(cluster_sizes.std())
        
        return metrics

    def _visualize_clusters(self, reduced_data: np.ndarray, labels: np.ndarray, parent_label: str, depth: int, df_subset: pd.DataFrame = None):
        """Create visualizations for clusters (local mode only)."""
        if not self.local_mode:
            return
            
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        fig, axes = self.plt.subplots(1, 2, figsize=(16, 6))

        scatter = axes[0].scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=20
        )

        axes[0].set_title(f'Clusters at {parent_label} (Depth {depth})\n{n_clusters} clusters found')
        axes[0].set_xlabel('UMAP Component 1')
        axes[0].set_ylabel('UMAP Component 2')
        self.plt.colorbar(scatter, ax=axes[0], label='Cluster ID')
        
        # Plot 2: Cluster size distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_counts = cluster_counts[cluster_counts.index != -1]  # Exclude noise
        axes[1].bar(range(len(cluster_counts)), cluster_counts.values)
        axes[1].set_title(f'Cluster Size Distribution (Depth {depth})')
        axes[1].set_xlabel('Cluster ID')
        axes[1].set_ylabel('Number of Reviews')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add noise count if exists
        noise_count = (labels == -1).sum()
        if noise_count > 0:
            axes[1].text(0.02, 0.98, f'Noise points: {noise_count}', 
                        transform=axes[1].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.plt.tight_layout()
        
        # Save figure
        safe_label = parent_label.replace('.', '_')
        filename = f"{self.output_dir}/clusters_{safe_label}_depth{depth}.png"
        self.plt.savefig(filename, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        logger.info(f"Saved visualization: {filename}")

    def _analyze_cluster_content(self, df_subset: pd.DataFrame, parent_label: str, depth: int):
        """Analyze and export cluster content (local mode only)."""
        if not self.local_mode or df_subset is None:
            return
        
        # Check if we have feedback_text column (we won't in new implementation)
        has_text = 'feedback_text' in df_subset.columns
        
        # Group by cluster and get sample reviews
        analysis = []
        for cluster_id in df_subset['cluster_label'].unique():
            cluster_df = df_subset[df_subset['cluster_label'] == cluster_id]
            
            # Get sample feedback text if available
            sample_texts = []
            if has_text:
                sample_texts = cluster_df['feedback_text'].head(10).tolist()
            
            analysis.append({
                'cluster_label': cluster_id,
                'size': len(cluster_df),
                'percentage': f"{len(cluster_df) / len(df_subset) * 100:.1f}%",
                'sample_count': len(sample_texts),
                'samples': sample_texts if has_text else []  # Empty list if no text
            })
            # Save to CSV
        analysis_df = pd.DataFrame(analysis)
        safe_label = parent_label.replace('.', '_')
        filename = f"{self.output_dir}/cluster_analysis_{safe_label}_depth{depth}.csv"
        analysis_df.to_csv(filename, index=False)
        
        # Also save detailed cluster assignments
        if 'feedback_id' in df_subset.columns:
            detail_filename = f"{self.output_dir}/cluster_assignments_{safe_label}_depth{depth}.csv"
            export_cols = ['feedback_id', 'cluster_label']
            if 'feedback_text' in df_subset.columns:
                export_cols.append('feedback_text')
            
            df_subset[export_cols].to_csv(detail_filename, index=False)
            logger.info(f"Saved cluster assignments: {detail_filename}")
        
        logger.info(f"Saved cluster analysis: {filename}")



    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate overall summary report (local mode only)."""
        if not self.local_mode:
            return
        
        report = []
        report.append("=" * 80)
        report.append("CLUSTERING SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal Reviews Analyzed: {len(df)}")
        report.append(f"Total Unique Clusters: {df['cluster_label'].nunique()}")
        report.append(f"Recursive Depth: {self.recursive_depth}")
        
        # Source breakdown
        if 'source' in df.columns:
            report.append(f"\n" + "-" * 80)
            report.append("BREAKDOWN BY SOURCE")
            report.append("-" * 80)
            source_stats = df.groupby('source').agg({
                'cluster_label': 'nunique',
                'feedback_id': 'count'
            }).rename(columns={'cluster_label': 'n_clusters', 'feedback_id': 'n_reviews'})
            
            for source, row in source_stats.iterrows():
                source_label = str(source) if source is not None else "none"
                report.append(f"\nSource: {source_label}")
                report.append(f"  Reviews: {row['n_reviews']}")
                report.append(f"  Clusters: {row['n_clusters']}")
        
        # Style breakdown
        if 'style' in df.columns:
            report.append(f"\n" + "-" * 80)
            report.append("BREAKDOWN BY STYLE")
            report.append("-" * 80)
            style_stats = df.groupby('style').agg({
                'cluster_label': 'nunique',
                'feedback_id': 'count'
            }).rename(columns={'cluster_label': 'n_clusters', 'feedback_id': 'n_reviews'})
            
            for style, row in style_stats.iterrows():
                style_label = str(style) if style is not None else "none"
                report.append(f"\nStyle: {style_label}")
                report.append(f"  Reviews: {row['n_reviews']}")
                report.append(f"  Clusters: {row['n_clusters']}")
        
        report.append(f"\n" + "-" * 80)
        report.append("QUALITY METRICS BY DEPTH")
        report.append("-" * 80)
    
        for label, metrics in self.cluster_hierarchy.items():
            if 'cluster_quality' in metrics:
                cq = metrics['cluster_quality']
                um = metrics.get('umap_metrics', {})
                report.append(f"\n{label} (Depth {metrics['depth']}):")
                report.append(f"  Points: {metrics['n_points']}")
                report.append(f"  Clusters: {metrics['n_clusters']}")
                if um.get('trustworthiness') is not None:
                    report.append(f"  UMAP Trustworthiness: {um['trustworthiness']:.3f}")
                if cq.get('relative_validity') is not None:
                    report.append(f"  Relative Validity: {cq['relative_validity']:.3f}")
                if cq.get('mean_persistence') is not None:
                    report.append(f"  Mean Persistence: {cq['mean_persistence']:.3f}")
                if cq.get('mean_probability') is not None:
                    report.append(f"  Mean Probability: {cq['mean_probability']:.3f}")
                report.append(f"  Noise Ratio: {cq['noise_ratio']:.1%}")
        
        # Breakdown by depth
        report.append(f"\n" + "-" * 80)
        report.append("CLUSTER BREAKDOWN BY DEPTH")
        report.append("-" * 80)
        
        depth_stats = df.groupby('cluster_depth').agg({
            'cluster_label': 'nunique',
            'feedback_id': 'count'
        }).rename(columns={'cluster_label': 'n_clusters', 'feedback_id': 'n_reviews'})
        
        for depth, row in depth_stats.iterrows():
            report.append(f"\nDepth {depth}:")
            report.append(f"  Clusters: {row['n_clusters']}")
            report.append(f"  Reviews: {row['n_reviews']}")
        
        report.append(f"\n" + "-" * 80)
        report.append("TOP 10 LARGEST CLUSTERS")
        report.append("-" * 80)
        
        # Top clusters
        top_clusters = df['cluster_label'].value_counts().head(10)
        for i, (label, count) in enumerate(top_clusters.items(), 1):
            pct = count / len(df) * 100
            report.append(f"{i}. {label}: {count} reviews ({pct:.1f}%)")
        
        # Save report
        report_text = "\n".join(report)
        filename = f"{self.output_dir}/summary_report.txt"
        with open(filename, 'w') as f:
            f.write(report_text)
        
        # Also print to console
        print("\n" + report_text)
        logger.info(f"Saved summary report: {filename}")



    def _recursive_cluster(self, embeddings: np.ndarray, indices: np.ndarray, parent_label: str = "root", current_depth: int = 0, feedback_ids: Optional[np.ndarray] = None, feedback_texts: Optional[np.ndarray] = None ) -> Dict[int, str]:

        # Check if we should recurse (initial check with n_clusters=0)
        if not self._should_recurse(current_depth, len(indices), n_clusters=0):
            return {idx: parent_label for idx in indices}
        
        logger.info(f"Clustering {len(indices)} points at depth {current_depth} (parent: {parent_label})")
        
        subset_embeddings = embeddings[indices]
        
        # Get adaptive UMAP parameters and apply UMAP
        umap_params = self._get_adaptive_umap_params(current_depth, len(indices))
        logger.info(f"Applying UMAP with adaptive params: {umap_params}")
        reducer = umap.UMAP(**umap_params)
        reduced_data = reducer.fit_transform(subset_embeddings)
     
        if self.local_mode:
            umap_metrics = self._evaluate_umap_quality(subset_embeddings, reduced_data, n_neighbors=umap_params['n_neighbors'])
            logger.info(f"UMAP trustworthiness at depth {current_depth}: {umap_metrics['trustworthiness']:.3f}")

        # Get adaptive HDBSCAN parameters and apply HDBSCAN
        hdbscan_params = self._get_adaptive_hdbscan_params(current_depth, len(indices))
        logger.info(f"Applying HDBSCAN with adaptive params: {hdbscan_params}")
        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        labels = clusterer.fit_predict(reduced_data)
        
        if not hasattr(clusterer, 'labels_') or len(labels) == 0:
            logger.warning(f"HDBSCAN failed at {parent_label} (depth {current_depth})")
            return {idx: f"{parent_label}.failed" for idx in indices}

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Early exit for degenerate cases
        if n_clusters == 0:
            logger.warning(f"HDBSCAN found only noise at {parent_label}")
            return {idx: f"{parent_label}.noise" for idx in indices}

        if n_clusters == 1:
            logger.info(f"Only 1 cluster at {parent_label}, no subdivision possible")
            return {idx: f"{parent_label}.0" for idx in indices}

        logger.info(f"Found {n_clusters} clusters at depth {current_depth}")

        
        # Evaluate clustering quality and store metrics
        cluster_metrics = self._evaluate_clustering_quality(labels, clusterer)
        logger.info(f"Clustering metrics at depth {current_depth}: n_clusters={cluster_metrics['n_clusters']}, noise_ratio={cluster_metrics['noise_ratio']:.2%}")
        
        # Store parameters and metrics in hierarchy
        hierarchy_entry = {
            'n_points': len(indices),
            'n_clusters': cluster_metrics['n_clusters'],
            'depth': current_depth,
            'umap_params': umap_params,
            'hdbscan_params': hdbscan_params,
            'cluster_quality': cluster_metrics
        }
        
        if self.local_mode:
            hierarchy_entry['umap_metrics'] = umap_metrics
            
        self.cluster_hierarchy[parent_label] = hierarchy_entry

        # Visualize in local mode
        if self.local_mode and feedback_ids is not None:

            viz_data = {
                'feedback_id': feedback_ids[indices],  
                'temp_cluster': labels                  
            }
            df_viz = pd.DataFrame(viz_data)
            
            self._visualize_clusters(reduced_data, labels, parent_label, current_depth, df_viz)
        
        # Build hierarchical labels
        cluster_mapping = {}
        unique_labels = set(labels)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_indices = indices[cluster_mask]
            
            if cluster_id == -1:
                # Noise points
                hierarchical_label = f"{parent_label}.noise"
                cluster_mapping.update({idx: hierarchical_label for idx in cluster_indices})
            else:
                hierarchical_label = f"{parent_label}.{cluster_id}"
                
                # Use intelligent recursion decision
                should_recurse = self._should_recurse(
                    current_depth + 1,
                    len(cluster_indices),
                    n_clusters=n_clusters,
                    cluster_quality=cluster_metrics

                )
                
                if should_recurse:
                    sub_clusters = self._recursive_cluster(
                        embeddings,
                        cluster_indices,
                        hierarchical_label,
                        current_depth + 1,
                        feedback_ids,
                        feedback_texts
                    )
                    cluster_mapping.update(sub_clusters)
                else:
                    cluster_mapping.update({idx: hierarchical_label for idx in cluster_indices})
        
             
        if self.local_mode and feedback_ids is not None:
            analysis_data = {
                'feedback_id': feedback_ids[indices],
                'cluster_label': [cluster_mapping[idx] for idx in indices]
            }
            if feedback_texts is not None:
                analysis_data['feedback_text'] = feedback_texts[indices]
            
            df_analysis = pd.DataFrame(analysis_data)
            self._analyze_cluster_content(df_analysis, parent_label, current_depth)
        
        return cluster_mapping



    def run(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Execute the UMAP + HDBSCAN clustering pipeline with recursive zooming.
        """

        logger.info(f"Clustering with UMAP + HDBSCAN (recursive depth: {self.recursive_depth})")

        # Fetch feedback data
        self.sql_client.connect()
        self.cosmos_client.connect()

        logger.info(f"Start date: {start_date}, End date: {end_date}")
        feedback_records = self.cosmos_client.get_all_embeddings(
            start_date=start_date,
            end_date=end_date,
            print_query=self.local_mode
            
        )

        logger.info(f"Fetched {len(feedback_records)} feedback records for clustering")

        if not feedback_records:
            logger.info("No records to cluster.")
            return {
                "total_records": 0,
                "clusters": {},
                "hierarchy": {},
                "start_date": start_date,
                "end_date": end_date
            }

        # Convert to DataFrame

        df = pd.DataFrame(feedback_records)
        
        df = df.rename(columns={0: 'feedback_id', 1: 'vector', 2: 'source', 3: 'feedback_text', 4: 'style'})
        
        if self.local_mode:
            print(df.head())
            print(f"\nStyle distribution:\n{df['style'].value_counts()}")
    

        def parse_vector(vec_str):
            """Parse string vector to numpy array."""
            if isinstance(vec_str, str):
                return np.array(ast.literal_eval(vec_str), dtype=np.float32)
            return np.array(vec_str, dtype=np.float32)

        embeddings_list = [parse_vector(v) for v in df['vector']]
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")


        # Partition by source first, then by style within each source
        sources = df['source'].unique()
        logger.info(f"Found {len(sources)} unique sources: {sources}")
        
        all_cluster_mappings = {}
        
        for source in sources:
            source_label = str(source) if source is not None else "none"
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing source: {source_label}")
            logger.info(f"{'='*80}")
            
            # Get data for this source
            source_mask = df['source'] == source
            source_df = df[source_mask]
            
            # Now partition by style within this source
            styles = source_df['style'].unique()
            logger.info(f"Found {len(styles)} unique styles within source {source_label}: {styles}")
            
            for style in styles:
                style_label = str(style) if style is not None else "none"
                logger.info(f"\n{'-'*80}")
                logger.info(f"Clustering source: {source_label}, style: {style_label}")
                logger.info(f"{'-'*80}")
                
                # Get indices for this source+style combination
                style_mask = (df['source'] == source) & (df['style'] == style)
                combo_indices = df[style_mask].index.to_numpy()
                
                logger.info(f"Processing {len(combo_indices)} records for source: {source_label}, style: {style_label}")
                
                # Run recursive clustering for this source+style partition
                combo_embeddings = embeddings[combo_indices]
                local_indices = np.arange(len(combo_indices))
                
                # Create source+style-specific data for visualizations
                combo_feedback_ids = df.loc[style_mask, 'feedback_id'].to_numpy()
                combo_feedback_texts = df.loc[style_mask, 'feedback_text'].to_numpy()

                cluster_mapping = self._recursive_cluster(
                    combo_embeddings, 
                    local_indices, 
                    parent_label=f"source_{source_label}.style_{style_label}",
                    feedback_ids=combo_feedback_ids,
                    feedback_texts=combo_feedback_texts
                )
                
                # Map local indices back to global indices
                for local_idx, cluster_label in cluster_mapping.items():
                    global_idx = combo_indices[local_idx]
                    all_cluster_mappings[global_idx] = cluster_label
        
        # Add cluster labels to dataframe
        df['cluster_label'] = df.index.map(all_cluster_mappings)
        
        # Handle any records that didn't get assigned (too few points to cluster)
        unassigned_mask = df['cluster_label'].isna()
        if unassigned_mask.any():
            logger.warning(f"Found {unassigned_mask.sum()} records that could not be clustered (too few points)")
            # Assign them to a special "unclustered" label based on their source and style
            for idx in df[unassigned_mask].index:
                source = df.loc[idx, 'source']
                style = df.loc[idx, 'style']
                source_label = str(source) if source is not None else "none"
                style_label = str(style) if style is not None else "none"
                df.loc[idx, 'cluster_label'] = f"source_{source_label}.style_{style_label}.unclustered"
        
        df['cluster_depth'] = df['cluster_label'].apply(lambda x: x.count('.'))

        logger.info(f"\nClustering complete: {df['cluster_label'].nunique()} unique clusters found across {len(sources)} sources and multiple styles")

        # Generate summary report if in local mode
        if self.local_mode:
            self._generate_summary_report(df)

        # Prepare result summary
        cluster_stats = df.groupby('cluster_label').agg({
            'feedback_id': ['count', list],
            'cluster_depth': 'first'
        }).to_dict('index')

        result = {
            "total_records": len(df),
            "n_clusters": df['cluster_label'].nunique(),
            "clusters": {
                label: {
                    'count': stats[('feedback_id', 'count')],
                    'feedback_ids': stats[('feedback_id', 'list')],
                    'depth': stats[('cluster_depth', 'first')]
                }
                for label, stats in cluster_stats.items()
            },
            "hierarchy": self.cluster_hierarchy,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "algorithm": "UMAP + HDBSCAN (recursive)",
            "recursive_depth": self.recursive_depth,
            "local_mode": self.local_mode,
            "output_dir": self.output_dir if self.local_mode else None
        }

        # Optionally persist results
        if not self.local_mode:
            self._save_results(df, start_date, end_date)

        return result

    def _save_results(self, df: pd.DataFrame, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """Save clustering results back to database."""
        
        # Step 1: Update feedback table with cluster_id assignments
        logger.info("Updating feedback table with cluster assignments...")
        cluster_assignments = [
            {'feedback_id': row['feedback_id'], 'cluster_id': row['cluster_label']}
            for _, row in df.iterrows()
        ]
        self.sql_client.update_feedback_clusters(cluster_assignments)
        logger.info(f"Updated {len(cluster_assignments)} feedback records with cluster assignments")
        
        # Step 2: Generate cluster descriptions using LLM
        logger.info("Generating cluster descriptions using LLM...")
        chat_agent = ChatAgent(self.config)
        
        # Group by cluster_label and prepare cluster records
        cluster_groups = df.groupby('cluster_label')
        cluster_records = []
        
        def generate_description_for_cluster(cluster_label, cluster_df):
            """Helper function to generate description for a single cluster."""
            try:
                # Sample up to 50 reviews for description
                sample_size = min(50, len(cluster_df))
                # Use head() instead of sample() to avoid ValueError with small DataFrames
                sample_reviews = cluster_df['feedback_text'].head(sample_size).tolist()
                
                # Generate description using LLM
                description = chat_agent.describe_cluster(sample_reviews)
                
                return cluster_label, description
            except Exception as e:
                logger.error(f"Error generating description for cluster {cluster_label}: {e}")
                return cluster_label, f"Cluster containing {len(cluster_df)} feedback items"
        
        # Use ThreadPoolExecutor to parallelize LLM calls
        cluster_descriptions = {}
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(generate_description_for_cluster, label, group): label
                for label, group in cluster_groups
            }
            
            for future in as_completed(futures):
                try:
                    label, description = future.result()
                    cluster_descriptions[label] = description
                    logger.info(f"Generated description for cluster: {label}")
                except Exception as e:
                    label = futures[future]
                    logger.error(f"Failed to generate description for cluster {label}: {e}")
                    cluster_descriptions[label] = f"Cluster {label}"
        
        # Step 3: Create ClusterRecord objects and insert into database
        logger.info("Inserting cluster metadata into database...")
        
        # Use provided date range or default to current time
        default_start = start_date if start_date else datetime.now(timezone.utc)
        default_end = end_date if end_date else datetime.now(timezone.utc)
        
        for cluster_label, cluster_df in cluster_groups:
            # Parse cluster information
            cluster_depth = cluster_label.count('.')
            
            # Extract source and style from cluster_label
            # Format: source_{source}.style_{style}.{hierarchy}
            parts = cluster_label.split('.')
            source = 'unknown'  # Default value
            style = None  # Can be None
            
            for part in parts:
                if part.startswith('source_'):
                    source = part.replace('source_', '')
                elif part.startswith('style_'):
                    style = part.replace('style_', '')
            
            # Get sample feedback IDs (up to 50)
            sample_feedback_ids = cluster_df['feedback_id'].head(50).tolist()
            
            cluster_record = ClusterRecord(
                cluster_id=cluster_label,
                label=cluster_label,
                description=cluster_descriptions.get(cluster_label, f"Cluster {cluster_label}"),
                source=source,
                style=style,
                cluster_depth=cluster_depth,
                sample_feedback_ids=sample_feedback_ids,
                record_count=len(cluster_df),
                period_start=default_start,
                period_end=default_end,
                created_at=datetime.now(timezone.utc)
            )
            cluster_records.append(cluster_record)
        
        # Insert all cluster records
        self.sql_client.insert_clusters(cluster_records)
        logger.info(f"Inserted {len(cluster_records)} cluster records into database")


def main():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Print to console
        ]
    )
    parser = argparse.ArgumentParser(description="Run UMAP + HDBSCAN recursive clustering pipeline.")

    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, default=None, help="Max records to cluster.")
    parser.add_argument("--recursive-depth", type=int, default=5, help="How many levels to recurse (1 = no recursion).")
    #parser.add_argument("--min-cluster-size", type=int, default=20, help="Minimum cluster size for HDBSCAN.")
    parser.add_argument("--min-cluster-pct", type=float, default=0.02, help="Min cluster size as percentage of data.")
    parser.add_argument("--min-sample-pct", type=float, default=0.003, help="Min samples as percentage of data for HDBSCAN.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--n-components", type=int, default=3, help="UMAP n_components (dimensions).")
    parser.add_argument("--local", action='store_true', help="Enable local mode with visualizations and analysis.")
    parser.add_argument("--output-dir", type=str, default="./cluster_output", help="Output directory for local mode files.")

    args = parser.parse_args()

    config = Settings()

    # Parse dates
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None

    # Configure parameters
    umap_params = {
        'n_neighbors': args.n_neighbors,
        'n_components': args.n_components,
        'metric': 'cosine',
        'random_state': 42
    }

    

    pipeline = RecursiveClusteringPipeline(
        config,
        umap_params=umap_params,
        recursive_depth=args.recursive_depth,
        min_cluster_size_pct=args.min_cluster_pct,
        min_sample_pct=args.min_sample_pct,
        hdbscan_metric='euclidean',
        local_mode=args.local,
        output_dir=args.output_dir
    )

    result = pipeline.run(
        start_date=start_date,
        end_date=end_date,
        
    )
    
    if args.local:
        print(f"\nClustering Results:")
        print(f"Total records: {result['total_records']}")
        print(f"Clusters found: {result['n_clusters']}")
        print(f"Recursive depth: {result['recursive_depth']}")
        print(f"\nTop 10 clusters:")
        sorted_clusters = sorted(result['clusters'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        for label, info in sorted_clusters:
            print(f"  {label}: {info['count']} reviews (depth {info['depth']})")


if __name__ == "__main__":
    main()