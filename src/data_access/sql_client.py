import pymssql
from typing import List, Optional
from datetime import datetime
from src.config.settings import Settings
from src.models.schemas import FeedbackRecord, TagRecord, ClusterRecord

class SQLClient:
    """SQL Server client for feedback data and results."""

    def __init__(self, config: Settings):
        self.config = config
        self.conn = None

    def connect(self) -> None:
        """Establish database connection."""
        self.conn = pymssql.connect(
            server=self.config.sql_server_host,
            user=self.config.sql_server_username,
            password=self.config.sql_server_password,
            database=self.config.sql_server_database
        )

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def get_new_feedback(self, last_processed_date: Optional[datetime] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, limit: Optional[int] = None) -> List[FeedbackRecord]:

        if not self.conn:
            self.connect()

        query = """
			SELECT feedback_id, feedback_text, feedback_source, created_at, customer_insights.feedback.sku, customer_insights.feedback.category, rating, cluster_id,[inventory_info].Style as style
            FROM customer_insights.feedback
			LEFT JOIN [dbo].[inventory_info] AS [inventory_info] ON TRIM(customer_insights.feedback.[sku]) = TRIM([inventory_info].[SKU])
            WHERE created_at IS NOT NULL
				AND [inventory_info].Style is not null
              AND feedback_text IS NOT NULL
              AND LTRIM(RTRIM(feedback_text)) <> ''
              AND feedback_text LIKE '%[A-Za-z0-9]%'
            AND UPPER(LTRIM(RTRIM(feedback_text))) NOT IN ('NA', 'N/A', 'N.A.', 'N.A', 'NOT APPLICABLE')
              AND feedback_source IS NOT NULL
        """

        params = []
        if last_processed_date:
            query += " AND created_at > %s"
            params.append(last_processed_date)
        
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)

        if limit:
            query = f"SELECT TOP {limit} * FROM ({query}) AS subquery"

        with self.conn.cursor(as_dict=True) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                FeedbackRecord(
                    feedback_id=row['feedback_id'],
                    text=row['feedback_text'],
                    source=row['feedback_source'],
                    created_at=row['created_at'],
                    product_id=row.get('sku'),
                    style=row.get('style'),
                    category=row.get('category'),
                    rating=row.get('rating'),
                    cluster_id=row.get('cluster_id')
                )
                for row in rows
            ]

    def get_feedback_by_ids(self, feedback_ids: List[str]) -> List[FeedbackRecord]:
        """
        Retrieve specific feedback records by IDs.
        """
        if not self.conn:
            self.connect()

        placeholders = ','.join(['%s'] * len(feedback_ids))
        query = f"""
			SELECT feedback_id, feedback_text, feedback_source, created_at, customer_insights.feedback.sku, customer_insights.feedback.category, rating, cluster_id,[inventory_info].Style as style
            FROM customer_insights.feedback
			LEFT JOIN [dbo].[inventory_info] AS [inventory_info] ON TRIM(customer_insights.feedback.[sku]) = TRIM([inventory_info].[SKU])
            WHERE feedback_id IN ({placeholders}) AND [inventory_info].Style is not null
        """

        with self.conn.cursor(as_dict=True) as cursor:
            cursor.execute(query, feedback_ids)
            rows = cursor.fetchall()

            return [
                FeedbackRecord(
                    feedback_id=row['feedback_id'],
                    text=row['feedback_text'],
                    source=row['feedback_source'],
                    created_at=row['created_at'],
                    product_id=row.get('sku'),
                    style=row.get('style'),
                    category=row.get('category'),
                    rating=row.get('rating'),
                    cluster_id=row.get('cluster_id')
                )
                for row in rows
            ]


    def insert_tags(self, tags: List[TagRecord]) -> None:
        """
        Insert tag assignments.
        """
        if not self.conn:
            self.connect()

        query = """
            INSERT INTO customer_insights.tags (feedback_id, tag_name, confidence_score, created_at)
            VALUES (%s, %s, %s, %s)
        """

        with self.conn.cursor() as cursor:
            for tag in tags:
                cursor.execute(
                    query,
                    (tag.feedback_id, tag.tag_name, tag.confidence_score, tag.created_at)
                )
            self.conn.commit()



    def insert_clusters(self, clusters: List[ClusterRecord]) -> None:
        """
        Insert cluster metadata.
        """
        if not self.conn:
            self.connect()

        query = """
            MERGE INTO customer_insights.clusters AS target
            USING (VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)) AS source 
                (cluster_id, cluster_label, cluster_description, sample_feedback_ids, record_count, 
                 period_start, period_end, created_at, style)
            ON target.cluster_id = source.cluster_id
            WHEN MATCHED THEN
                UPDATE SET 
                    cluster_label = source.cluster_label,
                    cluster_description = source.cluster_description,
                    sample_feedback_ids = source.sample_feedback_ids,
                    record_count = source.record_count,
                    period_start = source.period_start,
                    period_end = source.period_end,
                    created_at = source.created_at,
                    style = source.style
            WHEN NOT MATCHED THEN
                INSERT (cluster_id, cluster_label, cluster_description, sample_feedback_ids, record_count,
                        period_start, period_end, created_at, style)
                VALUES (source.cluster_id, source.cluster_label, source.cluster_description, 
                        source.sample_feedback_ids, source.record_count,
                        source.period_start, source.period_end, source.created_at, source.style);
        """

        with self.conn.cursor() as cursor:
            for cluster in clusters:
                sample_ids = ','.join(cluster.sample_feedback_ids)
                cursor.execute(
                    query,
                    (cluster.cluster_id, cluster.label, cluster.description, sample_ids,
                     cluster.record_count, cluster.period_start, cluster.period_end, 
                     cluster.created_at, cluster.style)
                )
            self.conn.commit()


