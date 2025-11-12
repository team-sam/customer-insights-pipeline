# src/pipelines/backfill.py
"""
Backfill pipeline to process all historical feedback records.
Generates embeddings and tags for existing feedback in the database.
"""

from typing import List, Optional
from datetime import datetime, timezone
import logging

from src.config.settings import Settings
from src.data_access.sql_client import SQLClient
from src.data_access.cosmos_client import CosmosClient
from src.embedding.embedder import Embedder
from src.agents.llm_agent import ChatAgent
from src.models.schemas import FeedbackRecord, EmbeddingRecord, TagRecord


logger = logging.getLogger(__name__)


class BackfillPipeline:
    """Pipeline for backfilling embeddings and tags for historical feedback."""
    
    def __init__(self, config: Settings, categories: Optional[List[str]] = None):
        """
        Initialize the backfill pipeline.
        
        Args:
            config: Application settings
            categories: List of tag categories for LLM tagging. If None, uses default categories.
        """
        self.config = config
        self.sql_client = SQLClient(config)
        self.cosmos_client = CosmosClient(config)
        self.embedder = Embedder(config)
        self.llm_agent = ChatAgent(config)
        
        # Default categories for Vessi feedback tagging
        self.categories = categories or [
            "Waterproof Leak",
            "Sizes not standard",
            "Too Heavy",
            "Toe Area too narrow",
            "Uncomfortable",
            "Quality Issues",
            "Positive Feedback",
            "Delivery Issues",
            "Customer Service",
            "Uncategorized"
        ]
    
    def run(self, batch_size: Optional[int] = None, limit: Optional[int] = None) -> dict:
        """
        Execute the backfill pipeline.
        
        Args:
            batch_size: Number of records to process in each batch (default from config)
            limit: Maximum number of records to process (None = all records)
        
        Returns:
            Dictionary with processing statistics
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        logger.info("Starting backfill pipeline")
        
        try:
            # Connect to databases
            self.sql_client.connect()
            self.cosmos_client.connect()
            
            # Fetch all feedback records
            logger.info("Fetching feedback records from SQL database")
            feedback_records = self.sql_client.get_new_feedback(limit=limit)
            total_records = len(feedback_records)
            logger.info(f"Found {total_records} feedback records to process")
            
            if total_records == 0:
                logger.info("No records to process")
                return {
                    "total_records": 0,
                    "embeddings_created": 0,
                    "tags_created": 0,
                    "errors": 0
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
                
                try:
                    # Generate embeddings
                    embeddings_batch = self._process_embeddings_batch(batch)
                    embeddings_created += len(embeddings_batch)
                    
                    # Generate tags
                    tags_batch = self._process_tags_batch(batch)
                    tags_created += len(tags_batch)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {str(e)}")
                    errors += len(batch)
            
            logger.info(
                f"Backfill complete: {embeddings_created} embeddings, "
                f"{tags_created} tags, {errors} errors"
            )
            
            return {
                "total_records": total_records,
                "embeddings_created": embeddings_created,
                "tags_created": tags_created,
                "errors": errors
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
        
        # Generate tags using LLM batch processing
        logger.info(f"Generating tags for {len(texts)} records")
        tag_lists = self.llm_agent.tag_feedback_batch(
            texts, 
            self.categories, 
            allow_multiple=True
        )
        
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
    """Main entry point for running the backfill pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = Settings()
    
    # Run backfill pipeline
    pipeline = BackfillPipeline(config)
    stats = pipeline.run()
    
    # Print results
    print("\n" + "="*50)
    print("BACKFILL PIPELINE RESULTS")
    print("="*50)
    print(f"Total records processed: {stats['total_records']}")
    print(f"Embeddings created: {stats['embeddings_created']}")
    print(f"Tags created: {stats['tags_created']}")
    print(f"Errors: {stats['errors']}")
    print("="*50)


if __name__ == "__main__":
    main()
