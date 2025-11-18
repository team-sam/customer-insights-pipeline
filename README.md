# Customer Insights Pipeline

A Python-based data pipeline for processing customer feedback using embeddings, clustering, and LLM-based tagging.

## Features

- **Flexible Ingestion**: Process feedback over any date range with daily, custom, or full backfill options
- **Embedding Generation**: Generate embeddings for customer feedback using OpenAI's text-embedding models
- **Vector Storage**: Store embeddings in PostgreSQL with pgvector extension
- **Clustering**: Group similar feedback using KMeans, DBSCAN, or Agglomerative clustering
- **Tagging**: Automatically tag feedback with predefined categories using LLM
- **SQL Integration**: Read feedback from SQL Server and store results
- **Azure Integration**: Automated deployment to Azure Blob Storage for Azure Batch execution via Azure Data Factory

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Ingestion Pipeline

The ingestion pipeline processes customer feedback records, generating embeddings and tags. It supports flexible date range options:

**Daily ingestion (process feedback from the last N days):**
```bash
python -m src.pipelines.ingest --days-back 1
```

**Custom date range:**
```bash
python -m src.pipelines.ingest --start-date 2024-01-01 --end-date 2024-01-31
```

**Full backfill (process all records):**
```bash
python -m src.pipelines.ingest
```

**With batch size and record limits:**
```bash
python -m src.pipelines.ingest --days-back 7 --batch-size 50 --limit 1000
```

**Available options:**
- `--days-back N`: Process feedback from the last N days
- `--start-date YYYY-MM-DD`: Start date for processing (inclusive)
- `--end-date YYYY-MM-DD`: End date for processing (inclusive)
- `--batch-size N`: Number of records to process in each batch (default: from config)
- `--limit N`: Maximum number of records to process

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Azure Deployment

This repository includes GitHub Actions for automated deployment to Azure. Scripts are automatically uploaded to Azure Blob Storage where they can be executed by Azure Batch via Azure Data Factory.

For setup instructions, see [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).

## Test Coverage

Current test coverage: 43%
- 37 tests passing
- Core components (schemas, embedder, llm_agent, clusterer) all tested
