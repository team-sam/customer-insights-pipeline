"""
Read and analyze cluster results from SQL Server.

Usage:
    python read_clusters.py
    python read_clusters.py --start-date 2024-01-01 --end-date 2024-03-31
    python read_clusters.py --style Weekend --source review
    python read_clusters.py --export clusters_report.csv
"""

import os
import argparse
import pymssql
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Create SQL Server connection."""
    return pymssql.connect(
        server=os.getenv("SQL_SERVER_HOST"),
        port=int(os.getenv("SQL_SERVER_PORT", 1433)),
        database=os.getenv("SQL_SERVER_DATABASE"),
        user=os.getenv("SQL_SERVER_USERNAME"),
        password=os.getenv("SQL_SERVER_PASSWORD"),
    )


def get_clusters(
    conn,
    start_date=None,
    end_date=None,
    style=None,
    source=None,
    depth=None,
):
    """Query clusters with optional filters."""
    query = """
        SELECT
            cluster_id,
            cluster_label,
            cluster_description,
            record_count,
            style,
            source,
            cluster_depth,
            period_start,
            period_end,
            created_at
        FROM customer_insights.clusters
        WHERE 1=1
    """
    params = []

    if start_date:
        query += " AND period_start >= %s"
        params.append(start_date)

    if end_date:
        query += " AND period_end <= %s"
        params.append(end_date)

    if style:
        query += " AND style = %s"
        params.append(style)

    if source:
        query += " AND source = %s"
        params.append(source)

    if depth is not None:
        query += " AND cluster_depth = %s"
        params.append(depth)

    query += " ORDER BY record_count DESC"

    return pd.read_sql(query, conn, params=params if params else None)


def get_cluster_feedback_samples(conn, cluster_id, limit=5):
    """Get sample feedback for a cluster."""
    query = """
        SELECT TOP %s
            feedback_id,
            feedback_text,
            feedback_source,
            rating,
            created_at
        FROM customer_insights.feedback
        WHERE cluster_id = %s
        ORDER BY created_at DESC
    """
    return pd.read_sql(query, conn, params=(limit, cluster_id))


def main():
    parser = argparse.ArgumentParser(description="Read cluster results")
    parser.add_argument("--start-date", help="Filter by period start (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Filter by period end (YYYY-MM-DD)")
    parser.add_argument("--style", help="Filter by product style")
    parser.add_argument("--source", help="Filter by feedback source (review/return/chat)")
    parser.add_argument("--depth", type=int, help="Filter by cluster depth (0=top-level)")
    parser.add_argument("--export", help="Export to CSV file")
    parser.add_argument("--samples", action="store_true", help="Show sample feedback for each cluster")
    args = parser.parse_args()

    conn = get_connection()

    print("Fetching clusters...")
    df = get_clusters(
        conn,
        start_date=args.start_date,
        end_date=args.end_date,
        style=args.style,
        source=args.source,
        depth=args.depth,
    )

    print(f"\nFound {len(df)} clusters\n")
    print("=" * 80)

    # Summary statistics
    print(f"Total feedback items clustered: {df['record_count'].sum():,}")
    print(f"Unique styles: {df['style'].nunique()}")
    print(f"Unique sources: {df['source'].nunique()}")
    print(f"Max depth: {df['cluster_depth'].max()}")
    print("=" * 80)

    # Top clusters
    print("\nTop 10 Clusters by Size:\n")
    top_clusters = df.nlargest(10, "record_count")[
        ["cluster_id", "cluster_description", "record_count", "style", "source"]
    ]
    print(top_clusters.to_string(index=False))

    # Show samples if requested
    if args.samples:
        print("\n" + "=" * 80)
        print("Sample Feedback for Top Clusters:\n")
        for _, row in top_clusters.head(5).iterrows():
            print(f"\n--- {row['cluster_id']} ({row['record_count']} items) ---")
            print(f"Description: {row['cluster_description'][:200]}...")
            samples = get_cluster_feedback_samples(conn, row["cluster_id"])
            for _, sample in samples.iterrows():
                print(f"  - {sample['feedback_text'][:100]}...")

    # Export if requested
    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\nExported to {args.export}")

    conn.close()


if __name__ == "__main__":
    main()
