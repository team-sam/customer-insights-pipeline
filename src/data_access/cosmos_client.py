# src/data_access/cosmos_client.py
"""
PostgreSQL vector database client for embeddings.
"""

import psycopg2
from psycopg2.extras import execute_values
from typing import List, Tuple, Optional
from src.config.settings import Settings
from src.models.schemas import EmbeddingRecord


class CosmosClient:
    """PostgreSQL vector database client for embeddings."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.conn = None
    
    def connect(self) -> None:
        """Establish database connection."""
        self.conn = psycopg2.connect(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            database=self.config.postgres_database,
            user=self.config.postgres_username,
            password=self.config.postgres_password,
            sslmode=self.config.postgres_sslmode
        )
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def initialize_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        if not self.conn:
            self.connect()
        
        schema_sql = """
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS embeddings (
            feedback_id VARCHAR(255) PRIMARY KEY,
            vector vector(1536),
            model VARCHAR(100),
            source VARCHAR(50),
            created_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
        ON embeddings USING ivfflat (vector vector_cosine_ops)
        WITH (lists = 1000);

        CREATE INDEX IF NOT EXISTS embeddings_source_idx ON embeddings(source);
        """
        
        with self.conn.cursor() as cursor:
            cursor.execute(schema_sql)
            self.conn.commit()
    
    def insert_embeddings(self, records: List[EmbeddingRecord]) -> None:
        """
        Insert embedding records into vector database.
        
        Args:
            records: List of EmbeddingRecord objects
        """
        if not self.conn:
            self.connect()
        
        values = [
            (r.feedback_id, r.vector, r.model, r.source, r.created_at)
            for r in records
        ]
        
        query = """
            INSERT INTO embeddings (feedback_id, vector, model, source, created_at)
            VALUES %s
            ON CONFLICT (feedback_id) DO UPDATE 
            SET vector = EXCLUDED.vector, 
                model = EXCLUDED.model,
                source = EXCLUDED.source,
                created_at = EXCLUDED.created_at
        """
        
        with self.conn.cursor() as cursor:
            execute_values(cursor, query, values)
            self.conn.commit()
    
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        distance_threshold: Optional[float] = None,
        source_filter: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar vectors using cosine distance.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            distance_threshold: Optional maximum distance filter
            source_filter: Optional filter by source (review, return, chat)
        
        Returns:
            List of (feedback_id, distance) tuples
        """
        if not self.conn:
            self.connect()
        
        query = """
            SELECT feedback_id, vector <=> %s::vector AS distance
            FROM embeddings
            WHERE 1=1
        """
        params = [query_vector]
        
        if source_filter:
            query += " AND source = %s"
            params.append(source_filter)
        
        if distance_threshold:
            query += " AND vector <=> %s::vector < %s"
            params.extend([query_vector, distance_threshold])
        
        query += " ORDER BY distance LIMIT %s"
        params.append(limit)
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            return cursor.fetchall()
    
    def get_embeddings_by_ids(self, feedback_ids: List[str]) -> List[Tuple[str, List[float], str]]:
        """
        Retrieve embeddings for specific feedback IDs.
        
        Args:
            feedback_ids: List of feedback IDs
        
        Returns:
            List of (feedback_id, vector, source) tuples
        """
        if not self.conn:
            self.connect()
        
        query = """
            SELECT feedback_id, vector, source
            FROM embeddings
            WHERE feedback_id = ANY(%s)
        """
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, (feedback_ids,))
            return cursor.fetchall()
    
    def get_all_embeddings(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_filter: Optional[str] = None
    ) -> List[Tuple[str, List[float], str]]:
        """
        Retrieve all embeddings, optionally filtered by date range and source.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            source_filter: Optional filter by source (review, return, chat)
        
        Returns:
            List of (feedback_id, vector, source) tuples
        """
        if not self.conn:
            self.connect()
        
        query = "SELECT feedback_id, vector, source FROM embeddings WHERE 1=1"
        params = []
        
        if source_filter:
            query += " AND source = %s"
            params.append(source_filter)
        
        if start_date and end_date:
            query += " AND created_at BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        elif start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        elif end_date:
            query += " AND created_at <= %s"
            params.append(end_date)
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()