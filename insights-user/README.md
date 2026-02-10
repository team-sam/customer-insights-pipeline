# Customer Insights - Analyst Toolkit

A toolkit for exploring Vessi customer feedback using Claude Code. Ask questions in plain English and get insights from reviews, returns, and support chats.

---

## Getting Started

### 1. Install Anaconda
Download and install [Anaconda](https://www.anaconda.com/download) if you don't have it.

### 2. Create the Environment
Open Anaconda Prompt and run:
```bash
cd path/to/insights-user
conda env create -f environment.yml
```

### 3. Activate the Environment
```bash
conda activate insights-user
```

### 4. Set Up Credentials
Copy `config_template.env` to `.env` and fill in your credentials. Contact your administrator if you don't have them.

### 5. Install Claude Code
Claude Code requires Node.js. Follow these steps in order:

**a) Install Node.js** (one-time, skip if you already have it):
- Download the **LTS** version from [https://nodejs.org](https://nodejs.org)
- Run the installer with default settings
- **Close and reopen** Anaconda Prompt after installing

**b) Install Claude Code** (one-time):
Open Anaconda Prompt and run:
```bash
npm install -g @anthropic-ai/claude-code
```

**c) Verify it works:**
```bash
claude --version
```
If you see a version number, you're good to go.

### 6. Launch Claude Code
Every time you want to use this toolkit, open Anaconda Prompt and run these three commands:
```bash
conda activate insights-user
cd path/to/insights-user
claude
```
**Important:** Always activate the conda environment first. This is what gives Claude access to the Python packages it needs to query the databases.

Before you ask your first question, **paste the context block below** so Claude knows how to work with the data.

---

## IMPORTANT: Give Claude Context First

Every time you start a new Claude Code session, **copy and paste the block below** as your first message. This tells Claude what tools are available and how to answer your questions well.

### Copy everything inside the box:

```
You are helping me analyze Vessi customer feedback data. I have two databases and I need you to pick the right approach depending on my question:

DATABASE 1 - SQL Server (structured data):
- Connection: use pymssql with credentials from .env file
- Tables: customer_insights.feedback (feedback_id, feedback_text, feedback_source, created_at, sku, category, rating, cluster_id), customer_insights.clusters (cluster_id, cluster_description, record_count, style, source, cluster_depth), customer_insights.tags (feedback_id, tag_name, confidence_score), dbo.inventory_info (SKU, Style)
- Use this for: counting, filtering, grouping, percentages, tag breakdowns, cluster browsing, date-range queries, ratings analysis

DATABASE 2 - Cosmos DB / PostgreSQL with pgvector (AI-powered search):
- Connection: use psycopg2 with credentials from .env file (POSTGRES_* vars), sslmode=require
- Table: embeddings (feedback_id, vector VECTOR(1536), source, style, feedback_text)
- Similarity search syntax: SELECT feedback_id, feedback_text, source, style, vector <=> %s::vector AS distance FROM embeddings ORDER BY distance LIMIT N
- Use this for: finding feedback semantically similar to a phrase or description, even if the exact words are different

OPENAI API (for converting text to searchable vectors):
- Use the openai library with OPENAI_API_KEY from .env
- Model: text-embedding-3-small (from OPENAI_EMBEDDING_MODEL in .env)
- When I ask to find feedback similar to a phrase, first embed my phrase using OpenAI, then search the vector database with cosine distance

HOW TO PICK THE RIGHT APPROACH:

1. STRUCTURED QUERY - If I ask for counts, percentages, rankings, comparisons, tag breakdowns, or anything with exact numbers, query SQL Server directly.

2. SIMILARITY SEARCH - If I describe a specific complaint or phrase and want to find matching feedback, embed my phrase with OpenAI and search the vector database. Return the actual feedback text and distance scores.

3. SEARCH + SUMMARIZE - If I ask you to summarize what customers are saying about a topic, do BOTH steps: first search the vector database to retrieve the most relevant feedback (get 30-50 results), then read through those results and write a summary highlighting the main themes, patterns, and include representative quotes.

4. CLUSTER BROWSING - If I ask about themes, clusters, or what the AI has already discovered, query the clusters table in SQL Server.

Always load credentials from the .env file using dotenv. Always show me the results clearly. If you export to CSV, tell me the file path.
```

After pasting that, you can ask questions normally and Claude will know what to do.

---

## How to Ask Questions

Here's a plain-English guide to getting the answers you need.

### Want exact numbers?
Just ask. Claude will count it up from the database.

> "How many reviews are tagged as 'Waterproof Leak'?"

> "What percentage of Weekend shoe returns mention sizing?"

> "Which style has the most 1-star reviews?"

> "Compare complaint counts between reviews and returns"

### Want to find feedback that sounds like something?
Describe the complaint in your own words. Claude will use AI to find the closest matching real feedback -- even if customers used different words.

> "Find feedback similar to 'my shoes started leaking after a month'"

> "Find reviews that talk about the shoes being too tight in the toe area"

> "Search for return reasons where people say the shoe fell apart"

> "Find 20 reviews most similar to 'I love the look but they hurt my feet'"

### Want a summary of what customers are saying?
Ask Claude to find relevant feedback AND summarize it. This gives you themes and patterns instead of just a list.

> "Find feedback about waterproof problems on Weekend shoes and summarize the main patterns"

> "What are customers saying about comfort issues with the Everyday style? Search for relevant feedback and summarize"

> "Search for reviews about heel rubbing and give me a summary of the problem -- which styles are most affected?"

> "Find feedback similar to 'the sole is slippery' and summarize what people are experiencing"

### Want to see what themes the AI already found?
Ask about clusters or themes. These are pre-built groupings with AI-written descriptions.

> "What are the main complaint themes for Weekend shoes?"

> "Show me the biggest feedback clusters for reviews"

> "List all clusters for Cityscape with their descriptions"

### Want a report or export?
Just say what you want and where.

> "Export all negative reviews for Weekend shoes to a CSV"

> "Create a chart showing complaints by category"

> "Build a report comparing the top issues across all styles"

---

## Tips for Better Answers

1. **Name the product** -- say "Weekend shoes" or "Cityscape style" not just "shoes"
2. **Name the source** if it matters -- "reviews", "returns", or "chat"
3. **Give a date range** when relevant -- "in Q1 2024" or "since January"
4. **Say how many results** you want -- "show me 20 examples" or "top 10"
5. **Combine steps** -- "Find reviews about heel pain for Weekend shoes and summarize the themes"

---

## What Data Is Available

### Feedback Sources
- **Reviews** -- Customer product reviews with 1-5 star ratings
- **Returns** -- Reasons customers give when returning products


### Product Styles
Products are grouped by style (e.g., Weekend, Everyday, Cityscape, Sunday Slipper, Alta High Top, Kids Weekend, etc.)

### Pre-Built Analysis

| Feature | What it is |
|---------|------------|
| **Clusters** | Feedback automatically grouped into themes (e.g., "Sizing Issues", "Waterproof Problems") with AI-written descriptions |
| **Tags** | Each piece of feedback categorized into one or more of 30 issue types |
| **Similarity Search** | Find feedback that means the same thing as any phrase you type |

### Tag Categories (30 total)
Every piece of feedback has been tagged with categories like:
- Waterproof Leak, Insole Issue, Glue Gap, Upper Knit Separation
- Sizes not standard, Too narrow/wide, Half size requests
- No heel lock, Lack of grip, Lack of support
- Not breathable, Back Heel Rubbing, Blisters, Smelly
- Too Bulky, Too Heavy, Looks different than picture
- ...and more

---

## Understanding Clusters

Clusters are groups of similar feedback discovered by AI. They're organized in layers:

```
Broad theme     -->  "Sizing Issues" (500 reviews)
  Sub-theme     -->    "Too Narrow" (200 reviews)
    Specific    -->      "Narrow in Toe Box" (80 reviews)
```

The clusters table has a description for each one explaining what customers in that group are talking about.

---

## Credentials

Your `.env` file needs these credentials (copy from `config_template.env`):

```bash
# SQL Server (read-only)
SQL_SERVER_HOST=...
SQL_SERVER_DATABASE=Vessi_DB
SQL_SERVER_USERNAME=...
SQL_SERVER_PASSWORD=...

# Cosmos DB (read-only)
POSTGRES_HOST=...
POSTGRES_DATABASE=citus
POSTGRES_USERNAME=...
POSTGRES_PASSWORD=...
POSTGRES_SSLMODE=require

# OpenAI (needed for similarity search)
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Contact your administrator for credentials.

---

## Need Help?

- Ask Claude: "What can I do with this toolkit?"
- Check `schemas/` folder for database details
- See `examples/query_examples.sql` for reference queries
