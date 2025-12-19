# Customer Insights Pipeline

A Python-based data pipeline for processing customer feedback using embeddings, clustering, and LLM-based tagging.

## Features

- **Flexible Ingestion**: Process feedback over any date range with daily, custom, or full backfill options
- **Embedding Generation**: Generate embeddings for customer feedback using OpenAI's text-embedding models
- **Vector Storage**: Store embeddings in PostgreSQL with pgvector extension
- **Advanced Clustering**: **UMAP + HDBSCAN with Recursive Clustering** - Discover hierarchical customer segments by "zooming in" on clusters
- **Tagging**: Automatically tag feedback with predefined categories using LLM
- **SQL Integration**: Read feedback from SQL Server and store results
- **Azure Integration**: Automated deployment to Azure Blob Storage for Azure Batch execution via Azure Data Factory

## Installation
```bash
pip install -r requirements.txt
```

### Additional Dependencies for Advanced Clustering

For UMAP + HDBSCAN recursive clustering:
```bash
pip install umap-learn hdbscan
```

For local visualization and analysis:
```bash
pip install matplotlib seaborn
```

## Configuration

The pipeline can be configured via environment variables or a `.env` file. Key configuration options:

### Performance Tuning

- `MAX_WORKERS` (default: 5): Number of concurrent threads for LLM tagging and embedding operations. Increase for higher throughput when processing large batches, but be mindful of API rate limits.
- `BATCH_SIZE` (default: 100): Number of records to process in each pipeline batch.

### Thread Safety & Rate Limiting

The pipeline uses multithreading to parallelize I/O-bound OpenAI API calls for both tagging and embedding operations. The OpenAI client is thread-safe, and no shared mutable state is accessed during parallel processing, ensuring safe concurrent execution. Both the `FeedbackTagger` and `Embedder` classes use ThreadPoolExecutor to process batches in parallel while maintaining result ordering.

**Automatic Rate Limit Handling**: All API calls include exponential backoff retry logic that automatically handles rate limit errors. If the OpenAI API returns a rate limit error, the system will:
- Retry up to 5 times with exponential backoff delays (1s, 2s, 4s, 8s, 16s)
- Log warnings when rate limits are hit
- Only fail after all retry attempts are exhausted

This ensures robust processing even when API rate limits are encountered during high-volume operations.

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

**Embeddings only (skip LLM tagging):**
```bash
python -m src.pipelines.ingest --days-back 1 --embeddings-only
```

**Resume failed ingestion (skip already embedded items):**
```bash
python -m src.pipelines.ingest --start-date 2024-01-01 --end-date 2024-01-31 --skip-embedded
```

**Available options:**
- `--days-back N`: Process feedback from the last N days
- `--start-date YYYY-MM-DD`: Start date for processing (inclusive)
- `--end-date YYYY-MM-DD`: End date for processing (inclusive)
- `--batch-size N`: Number of records to process in each batch (default: from config)
- `--limit N`: Maximum number of records to process
- `--embeddings-only`: Only generate embeddings without LLM tagging
- `--skip-embedded`: Skip items that have already been embedded (useful for resuming failed jobs)

#### Resumable Ingestion

The pipeline tracks which feedback items have been embedded in a SQL Server table (`customer_insights.embedded_items`). This enables resuming ingestion jobs that fail mid-run:

1. **Initial run fails partway through:**
   ```bash
   python -m src.pipelines.ingest --start-date 2024-01-01 --end-date 2024-01-31
   # Job fails after processing 1000 of 5000 records
   ```

2. **Resume from where it left off:**
   ```bash
   python -m src.pipelines.ingest --start-date 2024-01-01 --end-date 2024-01-31 --skip-embedded
   # Only processes the 4000 remaining records
   ```

The tracking table stores the `feedback_id` and `embedded_at` timestamp for each item successfully embedded to Cosmos DB.

### Clustering Pipeline

#### UMAP + HDBSCAN with Recursive Clustering

This approach implements the methodology described in [this article](https://link-to-article.com) for discovering hierarchical customer segments through dimensionality reduction and density-based clustering.

**How it works:**
1. **UMAP (Uniform Manifold Approximation and Projection)**: Reduces high-dimensional embeddings while preserving local and global structure
2. **HDBSCAN (Hierarchical Density-Based Spatial Clustering)**: Finds natural clusters without requiring predefined cluster counts
3. **Recursive "Zooming"**: Automatically identifies subclusters within parent clusters to reveal nuanced customer segments

**Production deployment (Azure Data Factory):**
```bash
python -m src.pipelines.umap_clustering --start-date 2024-01-01 --end-date 2024-01-31 --recursive-depth 2
```

**Local development with visualizations and analysis:**
```bash
python -m src.pipelines.umap_clustering --start-date 2024-01-01 --end-date 2024-01-31 --local --recursive-depth 2 --limit 5000
```

**Available options:**
- `--start-date YYYY-MM-DD`: Start date for clustering
- `--end-date YYYY-MM-DD`: End date for clustering
- `--limit N`: Maximum number of records to cluster
- `--recursive-depth N`: How many levels to recurse (1 = no recursion, 2-3 recommended)
- `--min-cluster-size N`: Minimum points for HDBSCAN cluster (default: 100)
- `--min-cluster-pct`: Minimum cluster size as percentage of data (default: 0.01)
- `--n-neighbors N`: UMAP n_neighbors parameter (default: 15)
- `--n-components N`: UMAP dimensionality (default: 2, use 2-3 for visualization)
- `--hdbscan-metric`: Distance metric for HDBSCAN (`euclidean` or `cosine`)
- `--local`: Enable local mode with visualizations and analysis outputs
- `--output-dir PATH`: Directory for local mode outputs (default: ./cluster_output)

**Local mode outputs:**

When `--local` is enabled, the pipeline generates:

1. **Cluster visualizations** (`clusters_*.png`): Scatter plots showing cluster structure and size distributions at each depth level
2. **Cluster analysis** (`cluster_analysis_*.csv`): Statistics and sample reviews for each cluster
3. **Detailed assignments** (`cluster_assignments_*.csv`): Complete mapping of feedback_id to hierarchical cluster labels
4. **Summary report** (`summary_report.txt`): Overall clustering statistics and top clusters

**Example workflow:**
```bash
# Step 1: Local analysis - explore clustering structure
python -m src.pipelines.umap_clustering \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --local \
  --recursive-depth 2 \
  --min-cluster-size 200 \
  --output-dir ./analysis_2024q1

# Step 2: Review outputs in ./analysis_2024q1/
# - Check cluster visualizations to validate structure
# - Examine cluster_analysis CSVs for segment characteristics
# - Review summary_report.txt for overall statistics

# Step 3: Production run - save results to database
python -m src.pipelines.umap_clustering \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --recursive-depth 2 \
  --min-cluster-size 200
```

**Understanding cluster labels:**

Clusters are labeled hierarchically:
- `root.0`, `root.1`, `root.2`: Top-level clusters
- `root.0.0`, `root.0.1`: Subclusters within `root.0`
- `root.0.0.noise`: Noise points that don't fit any subcluster

**Tuning recommendations:**

- **For broad segments**: Use `--recursive-depth 1` with larger `--min-cluster-size 1000`
- **For detailed segments**: Use `--recursive-depth 2-3` with smaller `--min-cluster-size 200-500`
- **For text embeddings**: Use `--hdbscan-metric cosine` if keeping high dimensions, or `euclidean` for 2D UMAP output
- **For large datasets**: Start with `--limit 10000` locally to validate parameters before full run

## Testing
```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Azure Deployment

This repository includes GitHub Actions for automated deployment to Azure. Scripts are automatically uploaded to Azure Blob Storage where they can be executed by Azure Batch via Azure Data Factory.

For setup instructions, see [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).

## Test Coverage

- Core components tested: schemas, embedder, llm_agent, ingestion pipeline
- Tests include multithreading and rate limit retry scenarios