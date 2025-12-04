"""
Flexible ingestion pipeline to process feedback records over any date range.
Supports backfill (all records), daily ingestion (recent records), or custom date ranges.
"""

from typing import List, Optional
from datetime import datetime, timedelta, timezone
import logging
import argparse

from src.config.settings import Settings
from src.data_access.sql_client import SQLClient
from src.data_access.cosmos_client import CosmosClient
from src.embedding.embedder import Embedder
from src.agents.llm_agent import FeedbackTagger
from src.models.schemas import FeedbackRecord, EmbeddingRecord, TagRecord


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting and processing feedback records over a flexible date range."""
    
    def __init__(self, config: Settings, categories: Optional[List[str]] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Application settings
            categories: List of tag categories for LLM tagging. If None, uses default categories from FeedbackTagger.
        """
        self.config = config
        self.sql_client = SQLClient(config)
        self.cosmos_client = CosmosClient(config)
        self.embedder = Embedder(config)
        self.tagger = FeedbackTagger(config, custom_categories=categories)
    
    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days_back: Optional[int] = None,
        batch_size: Optional[int] = None,
        limit: Optional[int] = None
    ) -> dict:
        """
        Execute the ingestion pipeline.
        
        Args:
            start_date: Start date for filtering feedback (inclusive)
            end_date: End date for filtering feedback (inclusive)
            days_back: Number of days back from now to process (alternative to start_date/end_date)
            batch_size: Number of records to process in each batch (default from config)
            limit: Maximum number of records to process (None = all matching records)
        
        Returns:
            Dictionary with processing statistics including date range
        """
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Calculate date range if days_back is provided
        if days_back is not None:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)
            logger.info(f"Processing feedback from last {days_back} days")
        
        # Log the date range being processed
        if start_date and end_date:
            logger.info(f"Starting ingestion pipeline for date range: {start_date.date()} to {end_date.date()}")
        elif start_date:
            logger.info(f"Starting ingestion pipeline from: {start_date.date()}")
        elif end_date:
            logger.info(f"Starting ingestion pipeline until: {end_date.date()}")
        else:
            logger.info("Starting ingestion pipeline (full backfill - no date filters)")
        
        try:
            # Connect to databases
            self.sql_client.connect()
            self.cosmos_client.connect()
            
            # Fetch feedback records based on date range
            logger.info("Fetching feedback records from SQL database")
            feedback_records = self.sql_client.get_new_feedback(
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            total_records = len(feedback_records)
            logger.info(f"Found {total_records} feedback records to process")
            
            if total_records == 0:
                logger.info("No records to process")
                return {
                    "total_records": 0,
                    "embeddings_created": 0,
                    "tags_created": 0,
                    "errors": 0,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                }
            
            # Process in batches
            embeddings_created = 0
            tags_created = 0
            errors = 0
            
            for i in range(0, total_records, batch_size):
                batch = feedback_records[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_records + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)")

                embeddings_batch = self._process_embeddings_batch(batch)
                embeddings_created += len(embeddings_batch)

                tags_batch = self._process_tags_batch(batch)
                tags_created += len(tags_batch)
                    

            logger.info(
                f"Ingestion complete: {embeddings_created} embeddings, "
                f"{tags_created} tags, {errors} errors"
            )
            
            return {
                "total_records": total_records,
                "embeddings_created": embeddings_created,
                "tags_created": tags_created,
                "errors": errors,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        
        finally:
            # Close connections
            self.sql_client.close()
            self.cosmos_client.close()
    
    def _process_embeddings_batch(self, feedback_batch: List[FeedbackRecord]) -> List[EmbeddingRecord]:
        """
        Generate and store embeddings for a batch of feedback records.
        
        Args:
            feedback_batch: List of feedback records
        
        Returns:
            List of created embedding records
        """
        # Extract texts for embedding
        texts = [record.text for record in feedback_batch]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} records")
        vectors = self.embedder.embed_texts(texts)
        
        # Create embedding records
        embedding_records = []

        for feedback, vector in zip(feedback_batch, vectors):
            embedding_record = EmbeddingRecord(
                feedback_id=feedback.feedback_id,
                vector=vector,
                source=feedback.source,
                model=self.config.openai_embedding_model,
                created_at=datetime.now(timezone.utc)
            )
            embedding_records.append(embedding_record)
        
        # Save to Cosmos DB
        logger.info(f"Saving {len(embedding_records)} embeddings to Cosmos DB")
        self.cosmos_client.insert_embeddings(embedding_records)
        
        return embedding_records
    
    def _process_tags_batch(self, feedback_batch: List[FeedbackRecord]) -> List[TagRecord]:
        """
        Generate and store tags for a batch of feedback records.
        
        Args:
            feedback_batch: List of feedback records
        
        Returns:
            List of created tag records
        """
        # Extract texts for tagging
        texts = [record.text for record in feedback_batch]
        
        # Generate tags using FeedbackTagger
        logger.info(f"Generating tags for {len(texts)} records")
        tag_lists = self.tagger.tag_batch(texts, allow_multiple=True)
        
        # Create tag records
        tag_records = []
        for feedback, tags in zip(feedback_batch, tag_lists):
            for tag_name in tags:
                tag_record = TagRecord(
                    feedback_id=feedback.feedback_id,
                    tag_name=tag_name,
                    confidence_score=1.0,  # LLM doesn't provide confidence scores
                    created_at=datetime.now(timezone.utc)
                )
                tag_records.append(tag_record)
        
        # Save to SQL database
        logger.info(f"Saving {len(tag_records)} tags to SQL database")
        self.sql_client.insert_tags(tag_records)
        
        return tag_records


def main():
    """Main entry point for running the ingestion pipeline with CLI arguments."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run the customer feedback ingestion pipeline with flexible date range options.'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        help='Number of days back to process feedback (e.g., 1 for daily ingestion)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for processing feedback in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for processing feedback in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of records to process'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Number of records to process in each batch'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.days_back and (args.start_date or args.end_date):
        parser.error("Cannot specify both --days-back and --start-date/--end-date")
    
    # Parse dates if provided
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            parser.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD")
    
    if args.end_date:
        try:
            # Set to end of day
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d').replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
        except ValueError:
            parser.error(f"Invalid end date format: {args.end_date}. Use YYYY-MM-DD")
    
    # Load configuration
    config = Settings()
    
    # Run ingestion pipeline
    pipeline = IngestionPipeline(config)
    stats = pipeline.run(
        start_date=start_date,
        end_date=end_date,
        days_back=args.days_back,
        batch_size=args.batch_size,
        limit=args.limit
    )
    
    # Print results
    print("\n" + "="*60)
    print("INGESTION PIPELINE RESULTS")
    print("="*60)
    if stats['start_date'] or stats['end_date']:
        if stats['start_date'] and stats['end_date']:
            print(f"Date range: {stats['start_date'][:10]} to {stats['end_date'][:10]}")
        elif stats['start_date']:
            print(f"Start date: {stats['start_date'][:10]}")
        elif stats['end_date']:
            print(f"End date: {stats['end_date'][:10]}")
    else:
        print("Date range: Full backfill (no date filter)")
    print(f"Total records processed: {stats['total_records']}")
    print(f"Embeddings created: {stats['embeddings_created']}")
    print(f"Tags created: {stats['tags_created']}")
    print(f"Errors: {stats['errors']}")
    print("="*60)


if __name__ == "__main__":
    main()
