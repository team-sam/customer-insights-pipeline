# src/embedding/embedder.py
from openai import OpenAI
from typing import List
from src.config.settings import Settings


class Embedder:
    """OpenAI embedding client."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_embedding_model
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        # Extract vectors in original order
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        
        return response.data[0].embedding