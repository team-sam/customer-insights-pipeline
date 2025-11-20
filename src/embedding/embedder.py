# src/embedding/embedder.py
from openai import OpenAI, RateLimitError
from typing import List
from src.config.settings import Settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """OpenAI embedding client."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_embedding_model
        self.max_workers = config.max_workers
    
    def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using multithreading.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per API call (default: 100)
        
        Returns:
            List of embedding vectors in the same order as input texts
        """
        if not texts:
            return []
        
        # If texts fit in a single batch, process directly (no threading overhead)
        if len(texts) <= batch_size:
            return self._embed_batch(texts)
        
        # Split texts into batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append((i, batch))
        
        # Process batches in parallel using ThreadPoolExecutor
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self._embed_batch, batch): (batch_idx, batch)
                for batch_idx, batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    results_dict[batch_idx] = batch_embeddings
                except Exception as e:
                    # Handle errors gracefully - raise with context
                    raise RuntimeError(f"Error processing embedding batch starting at index {batch_idx}: {e}") from e
        
        # Reconstruct results in original order
        all_embeddings = []
        for batch_idx in sorted(results_dict.keys()):
            all_embeddings.extend(results_dict[batch_idx])
        
        return all_embeddings
    
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (internal method for threading).
        Uses exponential backoff retry logic for rate limit errors.
        
        Args:
            batch: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        max_retries = 5
        base_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                return [item.embedding for item in response.data]
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    # Last attempt, raise the error
                    raise
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit on embedding batch. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                # For other errors, raise immediately
                raise
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        Uses exponential backoff retry logic for rate limit errors.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector
        """
        embeddings = self._embed_batch([text])
        return embeddings[0]