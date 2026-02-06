"""
Query feedback with various filters.

Usage:
    python read_feedback.py --cluster-id "source_review.style_Weekend.0.1"
    python read_feedback.py --source review --limit 100
    python read_feedback.py --tag "Waterproof Leak" --min-confidence 0.8
    python read_feedback.py --style Weekend --export feedback.csv
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


def get_feedback(
    conn,
    cluster_id=None,
    source=None,
    style=None,
    tag=None,
    min_confidence=0.7,
    start_date=None,
    end_date=None,
    limit=1000,
):
    """Query feedback with filters."""
    query = """
        SELECT DISTINCT
            f.feedback_id,
            f.feedback_text,
            f.feedback_source,
            f.created_at,
            f.sku,
            f.category,
            f.rating,
            f.cluster_id,
            i.Style as style
        FROM customer_insights.feedback f
        LEFT JOIN dbo.inventory_info i ON f.sku = i.SKU
    """

    if tag:
        query += """
        JOIN customer_insights.tags t ON f.feedback_id = t.feedback_id
        """

    query += " WHERE 1=1"
    params = []

    if cluster_id:
        query += " AND f.cluster_id = %s"
        params.append(cluster_id)

    if source:
        query += " AND f.feedback_source = %s"
        params.append(source)

    if style:
        query += " AND i.Style = %s"
        params.append(style)

    if tag:
        query += " AND t.tag_name = %s AND t.confidence_score >= %s"
        params.extend([tag, min_confidence])

    if start_date:
        query += " AND f.created_at >= %s"
        params.append(start_date)

    if end_date:
        query += " AND f.created_at <= %s"
        params.append(end_date)

    query += f" ORDER BY f.created_at DESC"

    if limit:
        # Use subquery for TOP with other clauses
        query = f"SELECT TOP {limit} * FROM ({query}) AS subquery"

    return pd.read_sql(query, conn, params=params if params else None)


def get_feedback_tags(conn, feedback_ids):
    """Get tags for a list of feedback IDs."""
    if not feedback_ids:
        return pd.DataFrame()

    placeholders = ",".join(["%s"] * len(feedback_ids))
    query = f"""
        SELECT feedback_id, tag_name, confidence_score
        FROM customer_insights.tags
        WHERE feedback_id IN ({placeholders})
        ORDER BY feedback_id, confidence_score DESC
    """
    return pd.read_sql(query, conn, params=feedback_ids)


def main():
    parser = argparse.ArgumentParser(description="Query feedback")
    parser.add_argument("--cluster-id", help="Filter by cluster ID")
    parser.add_argument("--source", help="Filter by source (review/return/chat)")
    parser.add_argument("--style", help="Filter by product style")
    parser.add_argument("--tag", help="Filter by tag name")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Min tag confidence")
    parser.add_argument("--start-date", help="Filter by start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Filter by end date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=100, help="Max records to return")
    parser.add_argument("--export", help="Export to CSV file")
    parser.add_argument("--include-tags", action="store_true", help="Include tags in output")
    args = parser.parse_args()

    conn = get_connection()

    print("Fetching feedback...")
    df = get_feedback(
        conn,
        cluster_id=args.cluster_id,
        source=args.source,
        style=args.style,
        tag=args.tag,
        min_confidence=args.min_confidence,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )

    print(f"\nFound {len(df)} feedback records\n")
    print("=" * 80)

    # Summary
    if len(df) > 0:
        print(f"Sources: {df['feedback_source'].value_counts().to_dict()}")
        print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        if df["rating"].notna().any():
            print(f"Avg rating: {df['rating'].mean():.2f}")
        print("=" * 80)

        # Show sample records
        print("\nSample Feedback:\n")
        for _, row in df.head(10).iterrows():
            print(f"[{row['feedback_source']}] {row['feedback_text'][:100]}...")
            if row["cluster_id"]:
                print(f"  Cluster: {row['cluster_id']}")
            print()

    # Include tags if requested
    if args.include_tags and len(df) > 0:
        print("\nFetching tags...")
        tags_df = get_feedback_tags(conn, df["feedback_id"].tolist())
        if len(tags_df) > 0:
            # Pivot tags to columns
            tags_pivot = tags_df.pivot_table(
                index="feedback_id",
                columns="tag_name",
                values="confidence_score",
                aggfunc="first",
            )
            df = df.merge(tags_pivot, left_on="feedback_id", right_index=True, how="left")

    # Export if requested
    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\nExported to {args.export}")

    conn.close()


if __name__ == "__main__":
    main()
