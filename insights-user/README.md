# Customer Insights - Analyst Toolkit

A toolkit for exploring customer feedback data using Claude Code. Ask questions in plain English and get insights from reviews, returns, and support chats.

## Getting Started

### 1. Install Anaconda
Download and install [Anaconda](https://www.anaconda.com/download) if you don't have it.

### 2. Create the Conda Environment
Open Anaconda Prompt and run:
```bash
cd path/to/insights-user
conda env create -f environment.yml
```
This creates a conda environment called `insights-user` with Python 3.11 and all required dependencies.

### 3. Activate the Environment
```bash
conda activate insights-user
```

### 4. Configure Credentials
1. Copy `config_template.env` to `.env`:
   ```bash
   cp config_template.env .env
   ```
2. Fill in your credentials (ask your administrator if you don't have them)

### 5. Start Claude Code
Open this folder in Claude Code and start asking questions!

### Updating the Environment
If `environment.yml` has been updated, sync your environment:
```bash
conda env update -f environment.yml --prune
```

---

## Example Questions You Can Ask Claude Code

### Exploring Feedback Themes

> "What are the main complaints about Weekend shoes?"

> "Show me the biggest clusters of feedback for reviews"

> "What issues do customers mention most often about the Everyday style?"

> "Summarize the top 5 complaint themes across all products"

### Finding Specific Feedback

> "Find reviews where customers complain about shoes being too narrow"

> "Show me all feedback about waterproof issues from the last 3 months"

> "Get me 20 examples of negative reviews (rating 1-2) for the Cityscape style"

> "Find feedback similar to 'my shoes leak when it rains'"

### Analyzing Patterns

> "How many complaints are about sizing vs durability?"

> "What's the distribution of ratings by product style?"

> "Which product style has the most complaints about heel slip?"

> "Compare the top issues between reviews and returns"

### Working with Tags (Categories)

> "How many reviews are tagged as 'Waterproof Leak'?"

> "Show me the breakdown of all tag categories"

> "What percentage of Weekend shoe feedback mentions sizing issues?"

> "List feedback tagged as 'Too Narrow' with high confidence"

### Generating Reports

> "Create a summary report of all clusters for Q1 2024"

> "Export all negative reviews for Weekend shoes to a CSV"

> "Generate a chart showing complaints by category"

> "Build a report comparing feedback themes across all styles"

### Similarity Search

> "Find reviews similar to: 'the sole came unglued after 2 weeks'"

> "What other feedback is like this one: [paste feedback text]"

> "Search for feedback about arch support problems"

### Date-Based Analysis

> "What were the top complaints last month?"

> "Show me trends in waterproof complaints over the past year"

> "Compare feedback themes between 2023 and 2024"

---

## What Data Is Available

### Feedback Sources
- **Reviews** - Customer product reviews with ratings (1-5 stars)
- **Returns** - Reasons customers give when returning products
- **Chat** - Support chat conversations

### Product Styles
Products are grouped by style (e.g., Weekend, Everyday, Cityscape, etc.)

### Analysis Features

| Feature | Description |
|---------|-------------|
| **Clusters** | Feedback automatically grouped by theme (e.g., "Sizing Issues", "Waterproof Problems") |
| **Tags** | Each feedback categorized into 30 predefined issue types |
| **Similarity Search** | Find feedback similar to any text query |
| **Embeddings** | AI-generated vectors for semantic understanding |

### Tag Categories (30 total)
Feedback is automatically tagged with categories like:
- Waterproof Leak, Insole Issue, Glue Gap
- Sizes not standard, Too narrow/wide, Half size requests
- No heel lock, Lack of grip, Lack of support
- Not breathable, Back Heel Rubbing, Blisters
- Too Bulky, Too Heavy, Looks different than picture
- ...and more

---

## Understanding Clusters

Clusters are hierarchical groupings discovered by AI:

```
source_review.style_Weekend.0       <- Broad theme (e.g., "Sizing Issues")
source_review.style_Weekend.0.1     <- Sub-theme (e.g., "Too Narrow")
source_review.style_Weekend.0.1.2   <- Specific (e.g., "Narrow in Toe Box")
source_review.style_Weekend.noise   <- Outliers that don't fit a pattern
```

| Depth | What it means |
|-------|---------------|
| 2 | Broad theme |
| 3 | Specific sub-theme |
| 4+ | Very specific patterns |

---

## Running Scripts Directly

You can also run the example scripts directly:

### Read Clusters
```bash
python examples/read_clusters.py --start-date 2024-01-01 --end-date 2024-03-31
```

### Read Feedback
```bash
python examples/read_feedback.py --cluster-id "source_review.style_Weekend.0"
```

### Similarity Search
```bash
python examples/similarity_search.py --query "shoes are too narrow"
python examples/similarity_search.py --feedback-id "abc123"
```

### Re-cluster Data
```bash
python examples/recluster.py --style "Weekend" --source "review" --output ./my_clusters/
```

---

## Credentials

Your `.env` file needs these credentials:

```bash
# SQL Server (read-only)
SQL_SERVER_HOST=your-server.database.windows.net
SQL_SERVER_DATABASE=Vessi_DB
SQL_SERVER_USERNAME=readonly_user
SQL_SERVER_PASSWORD=...

# Cosmos DB (read-only)
POSTGRES_HOST=your-cosmos.postgres.cosmos.azure.com
POSTGRES_DATABASE=citus
POSTGRES_USERNAME=readonly_user
POSTGRES_PASSWORD=...
POSTGRES_SSLMODE=require

# OpenAI (for text-based similarity search)
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Contact your administrator for credentials.

---

## Need Help?

- Ask Claude Code: "What can I do with this toolkit?"
- Check `schemas/` folder for detailed database documentation
- See `examples/query_examples.sql` for common SQL queries
