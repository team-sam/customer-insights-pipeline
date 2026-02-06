"""
Re-run clustering on a subset of feedback data.

This script performs local clustering only - results are NOT saved back to the database.
Use this for exploratory analysis with different parameters.

Usage:
    python recluster.py --style Weekend --source review --output ./my_clusters/
    python recluster.py --start-date 2024-01-01 --end-date 2024-03-31 --output ./q1_clusters/
    python recluster.py --limit 5000 --min-cluster-size 50 --recursive-depth 2
"""

import os
import argparse
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Optional imports for clustering
try:
    import umap
    import hdbscan
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("Warning: umap-learn and/or hdbscan not installed.")
    print("Install with: pip install umap-learn hdbscan")

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def get_cosmos_connection():
    """Create Cosmos DB (PostgreSQL) connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        database=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USERNAME"),
        password=os.getenv("POSTGRES_PASSWORD"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
    )


def get_embeddings(conn, source=None, style=None, start_date=None, end_date=None, limit=None):
    """Fetch embeddings from Cosmos DB."""
    query = "SELECT feedback_id, vector, source, feedback_text, style FROM embeddings WHERE 1=1"
    params = []

    if source:
        query += " AND source = %s"
        params.append(source)

    if style:
        query += " AND style = %s"
        params.append(style)

    if start_date:
        query += " AND created_at >= %s"
        params.append(start_date)

    if end_date:
        query += " AND created_at <= %s"
        params.append(end_date)

    if limit:
        query += f" LIMIT {limit}"

    with conn.cursor() as cursor:
        cursor.execute(query, tuple(params) if params else None)
        rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=["feedback_id", "vector", "source", "feedback_text", "style"])


def run_umap(vectors, n_neighbors=15, n_components=2, min_dist=0.1):
    """Reduce dimensionality with UMAP."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(vectors)


def run_hdbscan(embeddings, min_cluster_size=100, min_samples=10):
    """Cluster with HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def recursive_cluster(df, vectors_2d, prefix="root", depth=0, max_depth=2, min_cluster_size=100):
    """Recursively cluster data."""
    if depth >= max_depth or len(df) < min_cluster_size * 2:
        df["cluster_label"] = prefix
        return df

    # Run HDBSCAN
    labels = run_hdbscan(vectors_2d, min_cluster_size=min_cluster_size)
    df["cluster_label"] = [f"{prefix}.{l}" if l >= 0 else f"{prefix}.noise" for l in labels]

    # Recursively cluster each non-noise cluster
    result_dfs = []
    for label in df["cluster_label"].unique():
        subset = df[df["cluster_label"] == label].copy()
        subset_vectors = vectors_2d[df["cluster_label"] == label]

        if "noise" not in label and len(subset) >= min_cluster_size * 2:
            # Recurse
            subset = recursive_cluster(
                subset, subset_vectors, prefix=label, depth=depth + 1,
                max_depth=max_depth, min_cluster_size=min_cluster_size
            )
        result_dfs.append(subset)

    return pd.concat(result_dfs, ignore_index=True)


def plot_clusters(df, vectors_2d, output_dir):
    """Generate cluster visualization."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available, skipping plots")
        return

    plt.figure(figsize=(12, 8))

    # Get unique clusters and assign colors
    unique_labels = df["cluster_label"].unique()
    colors = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, colors))

    for label in unique_labels:
        mask = df["cluster_label"] == label
        color = color_map[label]
        alpha = 0.3 if "noise" in label else 0.6
        plt.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=[color],
            label=f"{label} ({mask.sum()})",
            alpha=alpha,
            s=10,
        )

    plt.title("UMAP + HDBSCAN Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "cluster_visualization.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    if not CLUSTERING_AVAILABLE:
        print("Error: Clustering libraries not available.")
        print("Install with: pip install umap-learn hdbscan")
        return

    parser = argparse.ArgumentParser(description="Re-cluster feedback data")
    parser.add_argument("--source", help="Filter by source (review/return/chat)")
    parser.add_argument("--style", help="Filter by product style")
    parser.add_argument("--start-date", help="Filter by start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Filter by end date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Max records to cluster")
    parser.add_argument("--min-cluster-size", type=int, default=100, help="Min HDBSCAN cluster size")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--recursive-depth", type=int, default=1, help="Recursive clustering depth")
    parser.add_argument("--output", default="./cluster_output", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Fetch data
    print("Connecting to Cosmos DB...")
    conn = get_cosmos_connection()

    print("Fetching embeddings...")
    df = get_embeddings(
        conn,
        source=args.source,
        style=args.style,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )
    conn.close()

    if len(df) == 0:
        print("No data found with the given filters.")
        return

    print(f"Loaded {len(df)} embeddings")

    # Convert vectors to numpy array
    vectors = np.array(df["vector"].tolist())
    print(f"Vector shape: {vectors.shape}")

    # Run UMAP
    print(f"Running UMAP (n_neighbors={args.n_neighbors})...")
    vectors_2d = run_umap(vectors, n_neighbors=args.n_neighbors)

    # Run clustering
    print(f"Running HDBSCAN (min_cluster_size={args.min_cluster_size}, depth={args.recursive_depth})...")
    df = recursive_cluster(
        df, vectors_2d,
        max_depth=args.recursive_depth,
        min_cluster_size=args.min_cluster_size,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Clustering Results:")
    print("=" * 60)
    cluster_counts = df["cluster_label"].value_counts()
    print(f"\nFound {len(cluster_counts)} clusters:\n")
    print(cluster_counts.to_string())

    # Save results
    assignments_path = os.path.join(args.output, "cluster_assignments.csv")
    df[["feedback_id", "cluster_label", "source", "style", "feedback_text"]].to_csv(
        assignments_path, index=False
    )
    print(f"\nSaved assignments to {assignments_path}")

    # Generate cluster analysis
    analysis_data = []
    for label in cluster_counts.index:
        subset = df[df["cluster_label"] == label]
        samples = subset["feedback_text"].head(5).tolist()
        analysis_data.append({
            "cluster_label": label,
            "count": len(subset),
            "sample_1": samples[0] if len(samples) > 0 else "",
            "sample_2": samples[1] if len(samples) > 1 else "",
            "sample_3": samples[2] if len(samples) > 2 else "",
        })

    analysis_df = pd.DataFrame(analysis_data)
    analysis_path = os.path.join(args.output, "cluster_analysis.csv")
    analysis_df.to_csv(analysis_path, index=False)
    print(f"Saved analysis to {analysis_path}")

    # Generate visualization
    plot_clusters(df, vectors_2d, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
