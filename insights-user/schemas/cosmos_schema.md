# Cosmos DB for PostgreSQL Schema

Database: `citus` (Azure Cosmos DB for PostgreSQL with pgvector extension)

## Tables

### embeddings

1536-dimensional text embeddings for semantic search and clustering.

| Column | Type | Description |
|--------|------|-------------|
| `feedback_id` | VARCHAR(255) | Primary key |
| `vector` | VECTOR(1536) | OpenAI text-embedding-3-small vector |
| `model` | VARCHAR(100) | Embedding model used |
| `source` | VARCHAR(50) | Feedback source: `review`, `return`, `chat` |
| `style` | VARCHAR(200) | Product style |
| `feedback_text` | TEXT | Original feedback text |
| `created_at` | TIMESTAMP | When embedding was created |

## Indexes

| Index | Type | Purpose |
|-------|------|---------|
| `embeddings_vector_idx` | IVFFlat (cosine) | Fast approximate nearest neighbor search |
| `embeddings_source_idx` | B-tree | Filter by source |

## Vector Operations

The `pgvector` extension provides these operators:

| Operator | Description |
|----------|-------------|
| `<=>` | Cosine distance (0 = identical, 2 = opposite) |
| `<->` | Euclidean (L2) distance |
| `<#>` | Inner product (negative, for max inner product search) |

### Example: Find Similar Feedback

```sql
-- Find 10 most similar feedback to a given vector
SELECT feedback_id, feedback_text, vector <=> '[0.1, 0.2, ...]'::vector AS distance
FROM embeddings
ORDER BY distance
LIMIT 10;
```

### Example: Filter by Source

```sql
-- Find similar reviews only
SELECT feedback_id, feedback_text, vector <=> %s::vector AS distance
FROM embeddings
WHERE source = 'review'
ORDER BY distance
LIMIT 10;
```

## Notes

- Vectors are 1536 dimensions (OpenAI text-embedding-3-small)
- The IVFFlat index uses 1000 lists for approximate search
- For exact search, use `SET LOCAL ivfflat.probes = 1000;` (slower)
- Cosine distance is preferred for text embeddings
