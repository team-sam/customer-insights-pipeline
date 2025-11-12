# scripts/test_pipeline_integration.py
"""
End-to-end integration test:
1. Pull feedback from SQL Server
2. Generate embeddings
3. Store in Cosmos DB
4. Search with new text
"""

from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import Settings
from src.data_access.sql_client import SQLClient
from src.data_access.cosmos_client import CosmosClient
from src.embedding.embedder import Embedder
from src.models.schemas import EmbeddingRecord

# tests/integration/test_embedder.py

from src.embedding.embedder import Embedder
from src.config.settings import Settings


def config():
    """Load actual configuration from .env."""
    return Settings()


def embedder(config):
    """Create embedder instance."""
    return Embedder(config)


def test_embed_single_real(embedder):
    """Test single text embedding with real API."""
    text = "This is a test sentence."
    result = embedder.embed_single(text)
    
    assert len(result) == 1536
    assert all(isinstance(x, float) for x in result)
    assert any(x != 0 for x in result)  # Verify not all zeros


def test_embed_texts_real(embedder):
    """Test batch text embedding with real API."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    result = embedder.embed_texts(texts)
    
    assert len(result) == 3
    assert all(len(vec) == 1536 for vec in result)
    assert all(isinstance(x, float) for vec in result for x in vec)
    
    # Verify vectors are different
    assert result[0] != result[1]
    assert result[1] != result[2]


def test_embed_similar_texts_have_similar_vectors(embedder):
    """Test that similar texts produce similar embeddings."""
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on a mat."
    text3 = "The stock market crashed today."
    
    vectors = embedder.embed_texts([text1, text2, text3])
    
    # Cosine similarity function
    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        return dot / (mag1 * mag2)
    
    sim_cat_sentences = cosine_similarity(vectors[0], vectors[1])
    sim_cat_market = cosine_similarity(vectors[0], vectors[2])
    
    # Similar sentences should be more similar than unrelated ones
    assert sim_cat_sentences > sim_cat_market
    assert sim_cat_sentences > 0.8  # High similarity threshold


def test_embed_empty_string(embedder):
    """Test embedding an empty string."""
    result = embedder.embed_single("")
    
    assert len(result) == 1536
    assert all(isinstance(x, float) for x in result)


    
def main():

    config = Settings()
    sql_client = SQLClient(config)
    cosmos_client = CosmosClient(config)
    embedder = Embedder(config)
    
    try:
        # Step 1: Pull feedback from SQL Server
        print("\n1. Fetching feedback from SQL Server...")
        sql_client.connect()
        feedback_records = sql_client.get_new_feedback(limit=1)
        print(f"   Retrieved {len(feedback_records)} records")
        
        if not feedback_records:
            print("   No feedback found in database. Exiting.")
            return
        
        # Display sample
        print(f"\n   Sample record:")
        print(f"   ID: {feedback_records[0].feedback_id}")
        print(f"   Text: {feedback_records[0].text[:100]}...")
        print(f"   Source: {feedback_records[0].source}")
        
        # Step 2: Generate embeddings
        print("\n2. Generating embeddings...")
        texts = [record.text for record in feedback_records]
        vectors = embedder.embed_texts(texts)
        print(f"   Generated {len(vectors)} embeddings")
        print(f"   Vector dimension: {len(vectors[0])}")
        
        # Create embedding records
        embedding_records = []
        for record, vector in zip(feedback_records, vectors):
            # Print raw values before creating EmbeddingRecord
            print("Raw values before EmbeddingRecord:")
            print(f"  feedback_id: {record.feedback_id}")
            print(f"  vector (first 5): {vector[:5]}")
            print(f"  model: {config.openai_embedding_model}")
            print(f"  source: {str(record.source) if record.source is not None else 'unknown'}")
            print(f"  created_at: {record.created_at}")

            embedding_record = EmbeddingRecord(
            feedback_id=record.feedback_id,
            vector=vector,
            model=config.openai_embedding_model,
            source=str(record.source) if record.source is not None else "unknown",
            created_at=record.created_at
            )
            print(f"EmbeddingRecord:\n"
              f"  feedback_id: {embedding_record.feedback_id}\n"
              f"  model: {embedding_record.model}\n"
              f"  source: {embedding_record.source}\n"
              f"  created_at: {embedding_record.created_at}\n"
              f"  vector (first 5): {embedding_record.vector[:5]}")
            
            
            embedding_records.append(embedding_record)
        
    
        # Step 3: Store in Cosmos DB
        print("\n3. Storing embeddings in Cosmos DB...")
        cosmos_client.connect()
        cosmos_client.initialize_schema()
       #print(embedding_records)
        for record in embedding_records:
            print(f"EmbeddingRecord source: {record.source}")
        print(type(embedding_records))
   

        cosmos_client.insert_embeddings(embedding_records)
        print(f"   Inserted {len(embedding_records)} embeddings")
        
        # Step 4: Search with new text
        print("\n4. Testing vector search...")
        query_text = "product quality issue"
        print(f"   Query: '{query_text}'")
        
        query_vector = embedder.embed_single(query_text)
        results = cosmos_client.search_similar(query_vector, limit=5)
        
        print(f"\n   Found {len(results)} similar records:")
        for i, (feedback_id, distance) in enumerate(results, 1):
            # Get original text
            original = next(
                (r for r in feedback_records if r.feedback_id == feedback_id),
                None
            )
            print(f"\n   {i}. ID: {feedback_id}")
            print(f"      Distance: {distance:.4f}")
            if original:
                print(f"      Text: {original.text[:150]}...")
                print(f"      Source: {original.source}")
        
        print("\n✓ Integration test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        raise
    
    finally:
        sql_client.close()
        cosmos_client.close()
        print("\nConnections closed.")


if __name__ == "__main__":
    main()