import pymssql
from typing import List, Optional, Tuple
from datetime import datetime
from src.config.settings import Settings
from src.models.schemas import FeedbackRecord, TagRecord, ClusterRecord
from tqdm import tqdm

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
            self.conn = None

    def reconnect(self) -> None:
        """Close existing connection and establish a new one."""
        self.close()
        self.connect()

    def initialize_embedded_items_table(self) -> None:
        """
        Create the embedded_items tracking table if it doesn't exist.
        This table tracks which feedback items have been embedded in Cosmos DB.
        """
        if not self.conn:
            self.connect()

        query = """
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'embedded_items' AND schema_id = SCHEMA_ID('customer_insights'))
            BEGIN
                CREATE TABLE customer_insights.embedded_items (
                    feedback_id VARCHAR(255) PRIMARY KEY,
                    embedded_at DATETIME NOT NULL
                )
            END
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.conn.commit()

    def insert_embedded_items(self, items: List[dict]) -> None:
        """
        Insert or update embedded item records to track what has been embedded.

        Args:
            items: List of dicts with feedback_id and embedded_at datetime
        """
        if not items:
            return

        if not self.conn:
            self.connect()

        query = """
            MERGE INTO customer_insights.embedded_items AS target
            USING (VALUES (%s, %s)) AS source (feedback_id, embedded_at)
            ON target.feedback_id = source.feedback_id
            WHEN MATCHED THEN
                UPDATE SET embedded_at = source.embedded_at
            WHEN NOT MATCHED THEN
                INSERT (feedback_id, embedded_at)
                VALUES (source.feedback_id, source.embedded_at);
        """

        params = [(item['feedback_id'], item['embedded_at']) for item in items]

        with self.conn.cursor() as cursor:
            cursor.executemany(query, params)
            self.conn.commit()

    def get_embedded_feedback_ids(self) -> List[str]:
        """
        Get list of feedback IDs that have already been embedded.
        
        Returns:
            List of feedback_id strings
        """
        if not self.conn:
            self.connect()

        query = """
            SELECT feedback_id
            FROM customer_insights.embedded_items
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    
    def sync_embeddings_from_cosmos(self, embedded_items: List[Tuple[str, str]]) -> int:
        """
        Sync embedded items from Cosmos DB to SQL Server tracking table.
        This is used at the start of the ingestion process to populate the tracking table.

        Args:
            embedded_items: List of tuples (feedback_id, created_at/embedded_at)

        Returns:
            Number of items synced
        """
        if not embedded_items:
            return 0

        if not self.conn:
            self.connect()

        query = """
            MERGE INTO customer_insights.embedded_items AS target
            USING (VALUES (%s, %s)) AS source (feedback_id, embedded_at)
            ON target.feedback_id = source.feedback_id
            WHEN MATCHED THEN
                UPDATE SET embedded_at = source.embedded_at
            WHEN NOT MATCHED THEN
                INSERT (feedback_id, embedded_at)
                VALUES (source.feedback_id, source.embedded_at);
        """

        with self.conn.cursor() as cursor:
            cursor.executemany(query, embedded_items)
            self.conn.commit()

        return len(embedded_items)

    def get_new_feedback(self, last_processed_date: Optional[datetime] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, limit: Optional[int] = None, skip_embedded: bool = False) -> List[FeedbackRecord]:
        """
        Retrieve feedback records based on date range and optional filters.
        
        Args:
            last_processed_date: Only get records created after this date
            start_date: Start date for filtering feedback (inclusive)
            end_date: End date for filtering feedback (inclusive)
            limit: Maximum number of records to return
            skip_embedded: If True, exclude items that have already been embedded (uses LEFT JOIN with embedded_items table)
        
        Returns:
            List of FeedbackRecord objects
        """
        if not self.conn:
            self.connect()

        query = """
			SELECT feedback_id, feedback_text, feedback_source, created_at, customer_insights.feedback.sku, customer_insights.feedback.category, rating, cluster_id,[inventory_info].Style as style
            FROM customer_insights.feedback
			LEFT JOIN [dbo].[inventory_info] AS [inventory_info] ON customer_insights.feedback.[sku] = [inventory_info].[SKU]
        """
        
        # Add LEFT JOIN to exclude embedded items if skip_embedded is True
        if skip_embedded:
            query += """
			LEFT JOIN customer_insights.embedded_items AS [embedded_items] ON customer_insights.feedback.feedback_id = [embedded_items].feedback_id
            """
        
        query += """
            WHERE created_at IS NOT NULL
				AND [inventory_info].Style is not null
              AND feedback_text IS NOT NULL
              AND LTRIM(RTRIM(feedback_text)) <> ''
              AND feedback_text LIKE '%[A-Za-z0-9]%'
            AND UPPER(LTRIM(RTRIM(feedback_text))) NOT IN ('NA', 'N/A', 'N.A.', 'N.A', 'NOT APPLICABLE')
              AND feedback_source IS NOT NULL
        """
        
        # Add filter to exclude embedded items
        if skip_embedded:
            query += " AND [embedded_items].feedback_id IS NULL"

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
			LEFT JOIN [dbo].[inventory_info] AS [inventory_info] ON customer_insights.feedback.[sku] = [inventory_info].[SKU]
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
        Insert or update tag assignments.
        """
        if not tags:
            return

        if not self.conn:
            self.connect()

        query = """
            MERGE INTO customer_insights.tags AS target
            USING (VALUES (%s, %s, %s, %s)) AS source
                (feedback_id, tag_name, confidence_score, created_at)
            ON target.feedback_id = source.feedback_id AND target.tag_name = source.tag_name
            WHEN MATCHED THEN
                UPDATE SET
                    confidence_score = source.confidence_score,
                    created_at = source.created_at
            WHEN NOT MATCHED THEN
                INSERT (feedback_id, tag_name, confidence_score, created_at)
                VALUES (source.feedback_id, source.tag_name, source.confidence_score, source.created_at);
        """

        params = [(tag.feedback_id, tag.tag_name, tag.confidence_score, tag.created_at) for tag in tags]

        with self.conn.cursor() as cursor:
            cursor.executemany(query, params)
            self.conn.commit()




    def update_feedback_clusters(self, cluster_assignments: List[dict]) -> None:
        """
        Update feedback records with cluster assignments.

        Args:
            cluster_assignments: List of dicts with feedback_id and cluster_id
        """
        if not cluster_assignments:
            return

        query = """
            UPDATE customer_insights.feedback
            SET cluster_id = %s
            WHERE feedback_id = %s
        """

        params = [(a['cluster_id'], a['feedback_id']) for a in cluster_assignments]

        batch_size = 100
        total_batches = (len(params) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(params), batch_size), 
                    desc="Updating feedback clusters", 
                    total=total_batches):
            self.reconnect()
            
            batch = params[i:i + batch_size]
            
            try:
                with self.conn.cursor() as cursor:
                    cursor.executemany(query, batch)
                    self.conn.commit()
            except Exception as e:
                print(f"\nError on batch {i//batch_size + 1}: {e}")
                self.reconnect()
                with self.conn.cursor() as cursor:
                    cursor.executemany(query, batch)
                    self.conn.commit()


    def insert_clusters(self, clusters: List[ClusterRecord]) -> None:
        """Insert cluster metadata."""
        if not clusters:
            return

        query = """
            MERGE INTO customer_insights.clusters AS target
            USING (VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)) AS source
                (cluster_id, cluster_label, cluster_description, sample_feedback_ids, record_count,
                period_start, period_end, created_at, style, source, cluster_depth)
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
                    style = source.style,
                    source = source.source,
                    cluster_depth = source.cluster_depth
            WHEN NOT MATCHED THEN
                INSERT (cluster_id, cluster_label, cluster_description, sample_feedback_ids, record_count,
                        period_start, period_end, created_at, style, source, cluster_depth)
                VALUES (source.cluster_id, source.cluster_label, source.cluster_description,
                        source.sample_feedback_ids, source.record_count,
                        source.period_start, source.period_end, source.created_at, source.style,
                        source.source, source.cluster_depth);
        """

        params = [
            (cluster.cluster_id, cluster.label, cluster.description,
            ','.join(cluster.sample_feedback_ids), cluster.record_count,
            cluster.period_start, cluster.period_end, cluster.created_at,
            cluster.style, cluster.source, cluster.cluster_depth)
            for cluster in clusters
        ]

        batch_size = 100
        total_batches = (len(params) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(params), batch_size), 
                    desc="Inserting clusters", 
                    total=total_batches):
            self.reconnect()
            
            batch = params[i:i + batch_size]
            
            try:
                with self.conn.cursor() as cursor:
                    cursor.executemany(query, batch)
                    self.conn.commit()
            except Exception as e:
                print(f"\nError on batch {i//batch_size + 1}: {e}")
                self.reconnect()
                with self.conn.cursor() as cursor:
                    cursor.executemany(query, batch)
                    self.conn.commit()