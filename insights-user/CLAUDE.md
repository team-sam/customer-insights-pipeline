# Claude Code Context - Customer Insights Analyst Toolkit

This file provides context for Claude Code to assist users with analyzing customer feedback data.

## What This Toolkit Does

This is a **read-only** analyst toolkit for exploring customer feedback (reviews, returns, support chats) that has been processed through an ML pipeline. The data includes:

- **Feedback text** with ratings and metadata
- **Embeddings** (1536-dimensional vectors) for semantic similarity search
- **Clusters** (hierarchical groupings discovered by UMAP + HDBSCAN)
- **Tags** (LLM-assigned categories like "Waterproof Leak", "Too Narrow", etc.)

## Database Connections

### SQL Server (Vessi_DB)
Contains structured data: feedback, clusters, tags, product info.

```python
import pymssql
import os
from dotenv import load_dotenv

load_dotenv()

conn = pymssql.connect(
    server=os.getenv("SQL_SERVER_HOST"),
    port=int(os.getenv("SQL_SERVER_PORT", 1433)),
    database=os.getenv("SQL_SERVER_DATABASE"),
    user=os.getenv("SQL_SERVER_USERNAME"),
    password=os.getenv("SQL_SERVER_PASSWORD"),
)
```

### Cosmos DB (PostgreSQL + pgvector)
Contains vector embeddings for similarity search.

```python
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_PORT", 5432)),
    database=os.getenv("POSTGRES_DATABASE"),
    user=os.getenv("POSTGRES_USERNAME"),
    password=os.getenv("POSTGRES_PASSWORD"),
    sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
)
```

## Database Schema

### SQL Server Tables

#### customer_insights.feedback
| Column | Type | Description |
|--------|------|-------------|
| feedback_id | VARCHAR(255) | Primary key |
| feedback_text | TEXT | The actual feedback text |
| feedback_source | VARCHAR(50) | `review`, `return`, or `chat` |
| created_at | DATETIME | When submitted |
| sku | VARCHAR(100) | Product SKU |
| category | VARCHAR(100) | Product category |
| rating | INT | 1-5 rating (nullable) |
| cluster_id | VARCHAR(500) | Assigned cluster label |

#### customer_insights.clusters
| Column | Type | Description |
|--------|------|-------------|
| cluster_id | VARCHAR(500) | Primary key (hierarchical label) |
| cluster_description | TEXT | LLM-generated description of themes |
| record_count | INT | Number of items in cluster |
| style | VARCHAR(200) | Product style |
| source | VARCHAR(50) | Feedback source |
| cluster_depth | INT | Hierarchy depth (2=top-level, 3=sub, 4=sub-sub) |

#### customer_insights.tags
| Column | Type | Description |
|--------|------|-------------|
| feedback_id | VARCHAR(255) | Foreign key |
| tag_name | VARCHAR(100) | Category name (30 predefined) |
| confidence_score | FLOAT | 0.0-1.0 confidence |

#### dbo.inventory_info
| Column | Type | Description |
|--------|------|-------------|
| SKU | VARCHAR(100) | Product SKU |
| Style | VARCHAR(200) | Product style name |

### Cosmos DB Table (pgvector)

#### embeddings
| Column | Type | Description |
|--------|------|-------------|
| feedback_id | VARCHAR(255) | Primary key |
| vector | VECTOR(1536) | OpenAI embedding |
| source | VARCHAR(50) | `review`, `return`, `chat` |
| style | VARCHAR(200) | Product style |
| feedback_text | TEXT | Original text |

## Cluster Label Format

Clusters use hierarchical labels:
```
source_{source}.style_{style}.{level0}.{level1}...

Examples:
- source_review.style_Weekend.0           -> Top-level cluster
- source_review.style_Weekend.0.1         -> Sub-cluster
- source_review.style_Weekend.0.1.2       -> Sub-sub-cluster
- source_review.style_Weekend.noise       -> Unclustered items
```

## Tag Categories (30 total)

Common categories include:
- Waterproof Leak, Upper Knit Separation, Insole Issue, Glue Gap
- Sizes not standard, Toe Area too narrow/big, Shoe too narrow/wide
- No heel lock/heel slip, Lack of grip/traction, Lack of support
- Not breathable, Back Heel Rubbing, Smelly, Blisters
- Too Bulky, Too Heavy, Looks different than picture

## Common Query Patterns

### Count feedback by source
```sql
SELECT feedback_source, COUNT(*) as count
FROM customer_insights.feedback
GROUP BY feedback_source
```

### Get top clusters by size
```sql
SELECT cluster_id, cluster_description, record_count, style, source
FROM customer_insights.clusters
WHERE cluster_depth = 2  -- top-level only
ORDER BY record_count DESC
```

### Get feedback for a specific cluster
```sql
SELECT f.feedback_id, f.feedback_text, f.rating, f.created_at
FROM customer_insights.feedback f
WHERE f.cluster_id = 'source_review.style_Weekend.0'
```

### Get tag distribution
```sql
SELECT tag_name, COUNT(*) as count, AVG(confidence_score) as avg_confidence
FROM customer_insights.tags
GROUP BY tag_name
ORDER BY count DESC
```

### Find feedback by style and date range
```sql
SELECT f.feedback_id, f.feedback_text, f.rating, f.feedback_source
FROM customer_insights.feedback f
JOIN dbo.inventory_info i ON f.sku = i.SKU
WHERE i.Style = 'Weekend'
  AND f.created_at >= '2024-01-01'
  AND f.created_at < '2024-04-01'
```

### Similarity search (Cosmos DB)
```sql
-- Find 10 most similar to a given vector
SELECT feedback_id, feedback_text, vector <=> %s::vector AS distance
FROM embeddings
WHERE source = 'review'
ORDER BY distance
LIMIT 10;
```

## Existing Example Scripts

Users can run these directly:

1. **examples/read_clusters.py** - Analyze cluster results
   ```bash
   python examples/read_clusters.py --start-date 2024-01-01 --end-date 2024-03-31
   ```

2. **examples/read_feedback.py** - Query feedback with filters
   ```bash
   python examples/read_feedback.py --cluster-id "source_review.style_Weekend.0"
   ```

3. **examples/similarity_search.py** - Find similar feedback
   ```bash
   python examples/similarity_search.py --query "shoes are too narrow"
   python examples/similarity_search.py --feedback-id "abc123"
   ```

4. **examples/recluster.py** - Re-run clustering on a subset
   ```bash
   python examples/recluster.py --style "Weekend" --source "review"
   ```

5. **examples/query_examples.sql** - Reference SQL queries

## How to Help Users

When users ask questions, you can:

1. **Write and run Python scripts** to query the databases
2. **Use existing example scripts** when they fit the use case
3. **Generate reports** with pandas DataFrames
4. **Create visualizations** with matplotlib/seaborn
5. **Export data to CSV** for further analysis

### Example user requests and approaches:

- "What are the main complaints about Weekend shoes?"
  -> Query clusters for style=Weekend, show descriptions

- "Find reviews similar to 'my shoes leak in the rain'"
  -> Use similarity_search.py or write a script with OpenAI embeddings

- "How many people complained about sizing last month?"
  -> Query tags table for sizing-related tags with date filter

- "Show me the distribution of ratings by product style"
  -> Join feedback with inventory_info, group by style, visualize

- "Export all negative reviews for the Everyday style"
  -> Query feedback where rating <= 2 and style = 'Everyday', export to CSV

## Environment Setup

Users should have:
1. Anaconda installed
2. A `.env` file with credentials (copied from `config_template.env`)
3. Dependencies installed: `pip install -r requirements.txt`

## Important Notes

- This is **READ-ONLY** access - no data modification
- OpenAI API key required for text-based similarity search
- Embeddings use cosine distance (lower = more similar)
- Cluster depth 2 = broad themes, depth 3-4 = specific issues
- "noise" clusters contain outliers that didn't fit patterns
