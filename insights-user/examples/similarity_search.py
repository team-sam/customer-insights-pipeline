"""
Search for similar feedback using vector embeddings.

Usage:
    python similarity_search.py --query "shoes are too narrow"
    python similarity_search.py --query "water leaking" --source review --limit 20
    python similarity_search.py --feedback-id "abc123" --limit 10
"""

import os
import argparse
import pymssql
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_sql_connection():
    """Create SQL Server connection."""
    return pymssql.connect(
        server=os.getenv("SQL_SERVER_HOST"),
        port=int(os.getenv("SQL_SERVER_PORT", 1433)),
        database=os.getenv("SQL_SERVER_DATABASE"),
        user=os.getenv("SQL_SERVER_USERNAME"),
        password=os.getenv("SQL_SERVER_PASSWORD"),
    )


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


def get_embedding_by_id(cosmos_conn, feedback_id):
    """Get embedding vector for a feedback ID."""
    query = "SELECT vector FROM embeddings WHERE feedback_id = %s"
    with cosmos_conn.cursor() as cursor:
        cursor.execute(query, (feedback_id,))
        result = cursor.fetchone()
        return result[0] if result else None


def search_similar(cosmos_conn, query_vector, source=None, limit=10):
    """Find similar feedback using cosine distance."""
    query = """
        SELECT
            feedback_id,
            feedback_text,
            source,
            style,
            vector <=> %s::vector AS distance
        FROM embeddings
        WHERE 1=1
    """
    params = [query_vector]

    if source:
        query += " AND source = %s"
        params.append(source)

    query += " ORDER BY distance LIMIT %s"
    params.append(limit)

    with cosmos_conn.cursor() as cursor:
        cursor.execute(query, tuple(params))
        columns = ["feedback_id", "feedback_text", "source", "style", "distance"]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def get_feedback_details(sql_conn, feedback_ids):
    """Get additional feedback details from SQL Server."""
    if not feedback_ids:
        return pd.DataFrame()

    placeholders = ",".join(["%s"] * len(feedback_ids))
    query = f"""
        SELECT
            f.feedback_id,
            f.rating,
            f.cluster_id,
            f.created_at,
            c.cluster_description
        FROM customer_insights.feedback f
        LEFT JOIN customer_insights.clusters c ON f.cluster_id = c.cluster_id
        WHERE f.feedback_id IN ({placeholders})
    """
    return pd.read_sql(query, sql_conn, params=feedback_ids)


def main():
    parser = argparse.ArgumentParser(description="Search for similar feedback")
    parser.add_argument("--query", help="Text query to search for")
    parser.add_argument("--feedback-id", help="Find items similar to this feedback ID")
    parser.add_argument("--source", help="Filter by source (review/return/chat)")
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    parser.add_argument("--export", help="Export to CSV")
    args = parser.parse_args()

    if not args.query and not args.feedback_id:
        print("Error: Must provide either --query or --feedback-id")
        return

    cosmos_conn = get_cosmos_connection()
    sql_conn = get_sql_connection()

    # Get query vector
    if args.feedback_id:
        print(f"Finding items similar to feedback: {args.feedback_id}")
        query_vector = get_embedding_by_id(cosmos_conn, args.feedback_id)
        if query_vector is None:
            print(f"Error: Feedback ID '{args.feedback_id}' not found in embeddings")
            return
    else:
        # For text queries, we need OpenAI API to generate embedding
        # This requires OPENAI_API_KEY in .env
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                input=args.query,
            )
            query_vector = response.data[0].embedding
            print(f"Searching for: '{args.query}'")
        except Exception as e:
            print(f"Error: Could not generate embedding for query. {e}")
            print("Note: Text search requires OPENAI_API_KEY in .env")
            print("Alternatively, use --feedback-id to find similar items to existing feedback.")
            return

    # Search
    print(f"\nSearching (source={args.source or 'all'}, limit={args.limit})...\n")
    results = search_similar(cosmos_conn, query_vector, source=args.source, limit=args.limit)

    if len(results) == 0:
        print("No results found.")
        return

    # Get additional details from SQL Server
    details = get_feedback_details(sql_conn, results["feedback_id"].tolist())
    if len(details) > 0:
        results = results.merge(details, on="feedback_id", how="left")

    # Display results
    print("=" * 80)
    print(f"Found {len(results)} similar items:\n")

    for i, row in results.iterrows():
        similarity = 1 - row["distance"]  # Convert distance to similarity
        print(f"{i+1}. [{row['source']}] (similarity: {similarity:.3f})")
        print(f"   {row['feedback_text'][:150]}...")
        if row.get("cluster_id"):
            print(f"   Cluster: {row['cluster_id']}")
        print()

    # Export if requested
    if args.export:
        results.to_csv(args.export, index=False)
        print(f"Exported to {args.export}")

    cosmos_conn.close()
    sql_conn.close()


if __name__ == "__main__":
    main()
