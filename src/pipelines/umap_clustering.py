"""
Customer feedback clustering pipeline using UMAP + HDBSCAN with recursive clustering.
Based on the iterative dimensionality reduction and density-based clustering approach.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
import logging
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import ast

import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler


from src.config.settings import Settings
from src.data_access.sql_client import SQLClient
from src.data_access.cosmos_client import CosmosClient
from src.models.schemas import FeedbackRecord

logger = logging.getLogger(__name__)


class RecursiveClusteringPipeline:
    """
    Pipeline for clustering feedback using UMAP + HDBSCAN with recursive zooming.
    Implements the approach from the article for customer segmentation.
    """

    def __init__(self, config: Settings, umap_params: Optional[Dict[str, Any]] = None,  recursive_depth: int = 1, min_cluster_size_pct: float = 0.02, min_sample_pct: float = 0.003, local_mode: bool = False, output_dir: str = "./cluster_output"):
        self.config = config
        self.sql_client = SQLClient(config)
        self.cosmos_client = CosmosClient(config)
        
        # Default UMAP parameters
        self.umap_params = umap_params or {
            'n_neighbors': 15,
            'n_components': 2,
            'metric': 'cosine'
        }
        
        
        self.recursive_depth = recursive_depth
        self.min_cluster_size_pct = min_cluster_size_pct
        self.min_sample_pct = min_sample_pct # Minimum samples as percentage of data for HDBSCAN
        self.cluster_hierarchy = {}
        self.local_mode = local_mode
        self.output_dir = output_dir
        
        # Storage for visualization data
        self.viz_data = []
        
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


    def _evaluate_umap_quality(self, original_embeddings: np.ndarray, reduced_embeddings: np.ndarray, n_neighbors: int = 15) -> Dict[str, float]:

        from sklearn.manifold import trustworthiness
        
        trust = trustworthiness(original_embeddings, reduced_embeddings, n_neighbors=n_neighbors)
        
        return {
            'trustworthiness': float(trust)
        }

    def _evaluate_clustering_quality( self, original_embeddings: np.ndarray, reduced_data: np.ndarray,  labels: np.ndarray, clusterer: hdbscan.HDBSCAN) -> Dict[str, float]:
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

    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        logger.info(f"Applying UMAP with params: {self.umap_params}")
        reducer = umap.UMAP(**self.umap_params)
        return reducer.fit_transform(embeddings)

    def _apply_hdbscan(self, reduced_data: np.ndarray) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """Apply HDBSCAN clustering and return labels + clusterer object."""
        params = {}
        params['min_samples'] = int(len(reduced_data) * self.min_sample_pct)
        params['min_cluster_size'] = int(len(reduced_data) * self.min_cluster_size_pct)
        params["metric"] = "euclidean"
            
        logger.info(f"Applying HDBSCAN with params: {params}")
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(reduced_data)
        
        return labels, clusterer

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
        
        # Group by cluster and get sample reviews
        analysis = []
        for cluster_id in df_subset['cluster_label'].unique():
            cluster_df = df_subset[df_subset['cluster_label'] == cluster_id]
            
            # Get sample feedback text if available
            sample_texts = []
            if 'feedback_text' in cluster_df.columns:
                sample_texts = cluster_df['feedback_text'].head(10).tolist()
            
            analysis.append({
                'cluster_label': cluster_id,
                'size': len(cluster_df),
                'percentage': f"{len(cluster_df) / len(df_subset) * 100:.1f}%",
                'sample_count': len(sample_texts),
                'samples': sample_texts
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



    def _recursive_cluster(self, embeddings: np.ndarray, indices: np.ndarray, parent_label: str = "root", current_depth: int = 0, df: pd.DataFrame = None) -> Dict[int, str]:

        if current_depth >= self.recursive_depth or len(indices) <int(len(indices) * self.min_cluster_size_pct):
            return {idx: parent_label for idx in indices}
        
        logger.info(f"Clustering {len(indices)} points at depth {current_depth} (parent: {parent_label})")
        
        subset_embeddings = embeddings[indices]
        
        reduced_data = self._apply_umap(subset_embeddings)
     
        if self.local_mode:
            umap_metrics = self._evaluate_umap_quality(subset_embeddings, reduced_data)
            logger.info(f"UMAP trustworthiness at depth {current_depth}: {umap_metrics['trustworthiness']:.3f}")

        # Apply HDBSCAN with dynamic min_cluster_size based on subset size
        min_size =  int(len(indices) * self.min_cluster_size_pct)
        
        labels, clusterer = self._apply_hdbscan(reduced_data, min_cluster_size=min_size)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Found {n_clusters} clusters at depth {current_depth}")
        
        # Evaluate clustering quality and store metrics
        if self.local_mode:
            cluster_metrics = self._evaluate_clustering_quality(subset_embeddings, reduced_data, labels, clusterer)
            logger.info(f"Clustering metrics at depth {current_depth}:")
            
            self.cluster_hierarchy[parent_label] = {
                'n_points': len(indices),
                'n_clusters': cluster_metrics['n_clusters'],
                'depth': current_depth,
                'umap_metrics': umap_metrics,
                'cluster_quality': cluster_metrics
            }

            # Visualize
            df_subset = df.iloc[indices].copy()
            temp_labels = labels.copy()
            df_subset['temp_cluster'] = temp_labels
            
            self._visualize_clusters(reduced_data, labels, parent_label, current_depth, df_subset)
        
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
                
                # Recurse if we haven't reached max depth and cluster is large enough
                if current_depth < self.recursive_depth - 1 and len(cluster_indices) >= min_size * 2:
                    sub_clusters = self._recursive_cluster(
                        embeddings,
                        cluster_indices,
                        hierarchical_label,
                        current_depth + 1,
                        df
                    )
                    cluster_mapping.update(sub_clusters)
                else:
                    cluster_mapping.update({idx: hierarchical_label for idx in cluster_indices})
        
        # Analyze cluster content if in local mode
        if self.local_mode and df is not None:
            df_subset = df.iloc[indices].copy()
            df_subset['cluster_label'] = df_subset.index.map(cluster_mapping)
            self._analyze_cluster_content(df_subset, parent_label, current_depth)
        
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
        df = df.rename(columns={0: 'feedback_id', 1: 'vector', 2: 'source'})
        
        if self.local_mode:
            print(df.head())
    

        def parse_vector(vec_str):
            """Parse string vector to numpy array."""
            if isinstance(vec_str, str):
                return np.array(ast.literal_eval(vec_str), dtype=np.float32)
            return np.array(vec_str, dtype=np.float32)

        # Parallel processing with optional progress bar
        with ThreadPoolExecutor(max_workers=8) as executor:
            if self.local_mode:
                from tqdm import tqdm
                embeddings_list = list(
                    tqdm(
                        executor.map(parse_vector, df['vector']),
                        total=len(df),
                        desc="Parsing embeddings"
                    )
                )
            else:
                embeddings_list = list(executor.map(parse_vector, df['vector']))

        embeddings = np.array(embeddings_list)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        # Run recursive clustering
        indices = np.arange(len(embeddings))
        cluster_mapping = self._recursive_cluster(embeddings, indices, df=df)

        # Add cluster labels to dataframe
        df['cluster_label'] = df.index.map(cluster_mapping)
        df['cluster_depth'] = df['cluster_label'].apply(lambda x: x.count('.'))

        logger.info(f"Clustering complete: {df['cluster_label'].nunique()} unique clusters found")

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
            self._save_results(df)

        return result

    def _save_results(self, df: pd.DataFrame):
        """Save clustering results back to database."""
        # Example: update SQL or Cosmos with cluster labels
        cluster_data = df[['feedback_id', 'cluster_label', 'cluster_depth']].to_dict('records')
        # self.sql_client.update_feedback_clusters(cluster_data)
        logger.info(f"Saved clustering results for {len(cluster_data)} records")


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
    parser.add_argument("--n-components", type=int, default=2, help="UMAP n_components (dimensions).")
    parser.add_argument("--local", action='store_true', help="Enable local mode with visualizations and analysis.")
    parser.add_argument("--output-dir", type=str, default="./cluster_output", help="Output directory for local mode files.")
    parser.add_argument("--hdbscan-metric", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Distance metric for HDBSCAN.")

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