# Customer Insights Pipeline

A Python-based data pipeline for processing customer feedback using embeddings, clustering, and LLM-based tagging.

## Features

- **Embedding Generation**: Generate embeddings for customer feedback using OpenAI's text-embedding models
- **Vector Storage**: Store embeddings in PostgreSQL with pgvector extension
- **Clustering**: Group similar feedback using KMeans, DBSCAN, or Agglomerative clustering
- **Tagging**: Automatically tag feedback with predefined categories using LLM
- **SQL Integration**: Read feedback from SQL Server and store results

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Test Coverage

Current test coverage: 43%
- 37 tests passing
- Core components (schemas, embedder, llm_agent, clusterer) all tested
