"""
Flexible clustering pipeline for feedback records over any date range.
Supports dynamic selection of clustering algorithm and flexible look-back configuration.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import logging
import argparse

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from src.config.settings import Settings
from src.data_access.sql_client import SQLClient
from src.data_access.cosmos_client import CosmosClient
from src.models.schemas import FeedbackRecord

logger = logging.getLogger(__name__)


CLUSTERING_ALGOS = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "agglomerative": AgglomerativeClustering,
}


class ClusteringPipeline:
    """
    Pipeline for clustering feedback records over a user-specified date range
    with a selectable clustering algorithm.
    """

    def __init__(self, config: Settings, algorithm: str = "kmeans", algo_params: Optional[Dict[str, Any]] = None):
        self.config = config
        self.sql_client = SQLClient(config)
        self.cosmos_client = CosmosClient(config)
        self.algorithm = algorithm.lower()
        if self.algorithm not in CLUSTERING_ALGOS:
            raise ValueError(f"Unsupported clustering algorithm '{self.algorithm}'. Supported: {list(CLUSTERING_ALGOS.keys())}")
        self.algo_params = algo_params or {}

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days_back: Optional[int] = None,
        features: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """
        Execute the clustering pipeline.

        Args:
            start_date: Filter feedback on or after this date (inclusive).
            end_date: Filter feedback on or before this date (inclusive).
            days_back: Look-back window in days from now (alternative to start_date/end_date).
            features: List of features (col names) to use for clustering.
            limit: Max records to process.

        Returns:
            Dictionary of clustering stats/results.
        """
        if days_back is not None:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)
            logger.info(f"Using look-back window: last {days_back} days")

        logger.info(f"Clustering with algorithm: {self.algorithm} (params: {self.algo_params})")

        # Fetch the feedback data
        self.sql_client.connect()
        self.cosmos_client.connect()
        feedback_records = self.cosmos_client.get_embeddings_by_date_range(
            start_date=start_date,
            end_date=end_date,
            source_filter=None,
        )

        logger.info(f"Fetched {len(feedback_records)} feedback records for clustering")

        if not feedback_records:
            logger.info("No records to cluster.")
            return {"total_records": 0, "clusters": {}, "start_date": start_date, "end_date": end_date}

        # Convert to DataFrame
        df = pd.DataFrame([f.__dict__ for f in feedback_records])

        # Auto-detect features if not provided
        if not features:
            if {'embedding', 'vector'}.intersection(df.columns):
                features = list({'embedding', 'vector'}.intersection(df.columns))
            else:
                raise ValueError("No valid feature columns ('embedding', 'vector') found for clustering.")

        feature_matrix = df[features].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x, axis=1).tolist()

        if len(feature_matrix) == 0:
            raise ValueError("No valid feature data for clustering.")

        # Create and fit clusterer
        clusterer_cls = CLUSTERING_ALGOS[self.algorithm]
        clusterer = clusterer_cls(**self.algo_params)
        labels = clusterer.fit_predict(feature_matrix)
        df["cluster_id"] = labels
        
        logger.info(f"Clustering complete ({df['cluster_id'].nunique()} clusters found)")

        # Persist results if needed (not implemented here, could write to Cosmos or SQL)
        # Example: self.sql_client.update_feedback_clusters(df[['feedback_id', 'cluster_id']])

        # Prepare result summary
        result = {
            "total_records": len(df),
            "n_clusters": df['cluster_id'].nunique(),
            "clusters": df.groupby("cluster_id")["feedback_id"].apply(list).to_dict(),
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "algorithm": self.algorithm,
        }
        return result


def main():
    parser = argparse.ArgumentParser(description="Run feedback clustering pipeline.")
    parser.add_argument("--algorithm", type=str, default="kmeans", choices=CLUSTERING_ALGOS.keys(), help="Clustering algorithm to use.")
    parser.add_argument("--lookback", type=int, help="Number of days to look back.")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD).")
    parser.add_argument("--features", type=str, nargs="+", help="Feature columns for clustering.")
    parser.add_argument("--limit", type=int, default=None, help="Max records to cluster.")
    parser.add_argument("--algo-param", action='append', type=str, help="Algorithm parameter as key=value.")

    args = parser.parse_args()

    # Parse settings, e.g. from environment or config file
    config = Settings()

    # Parse date args
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None

    # Convert algo param strings into dict
    algo_params = {}
    if args.algo_param:
        for param_str in args.algo_param:
            k, v = param_str.split('=', 1)
            try:
                # Try to convert number values
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            algo_params[k] = v

    pipeline = ClusteringPipeline(
        config,
        algorithm=args.algorithm,
        algo_params=algo_params
    )

    result = pipeline.run(
        start_date=start_date,
        end_date=end_date,
        days_back=args.lookback,
        features=args.features,
        limit=args.limit,
    )
    print(result)


if __name__ == "__main__":
    main()