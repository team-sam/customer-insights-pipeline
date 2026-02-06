# SQL Server Schema

Database: `Vessi_DB`

## Tables

### customer_insights.feedback

Source feedback records with cluster assignments.

| Column | Type | Description |
|--------|------|-------------|
| `feedback_id` | VARCHAR(255) | Primary key |
| `feedback_text` | TEXT | Raw feedback text |
| `feedback_source` | VARCHAR(50) | Source: `review`, `return`, `chat` |
| `created_at` | DATETIME | When feedback was submitted |
| `sku` | VARCHAR(100) | Product SKU |
| `category` | VARCHAR(100) | Product category |
| `rating` | INT | Rating (1-5, nullable) |
| `cluster_id` | VARCHAR(500) | Assigned cluster label (nullable) |

### customer_insights.clusters

Cluster metadata and LLM-generated descriptions.

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | VARCHAR(500) | Primary key (hierarchical label) |
| `cluster_label` | VARCHAR(500) | Display label |
| `cluster_description` | TEXT | LLM-generated description of cluster themes |
| `sample_feedback_ids` | TEXT | Comma-separated sample feedback IDs |
| `record_count` | INT | Number of feedback items in cluster |
| `period_start` | DATETIME | Start of clustering period |
| `period_end` | DATETIME | End of clustering period |
| `created_at` | DATETIME | When cluster was created |
| `style` | VARCHAR(200) | Product style (extracted from label) |
| `source` | VARCHAR(50) | Feedback source (extracted from label) |
| `cluster_depth` | INT | Hierarchy depth (0=top-level) |

### customer_insights.tags

LLM-assigned category tags for feedback.

| Column | Type | Description |
|--------|------|-------------|
| `feedback_id` | VARCHAR(255) | Foreign key to feedback |
| `tag_name` | VARCHAR(100) | Category name |
| `confidence_score` | FLOAT | Confidence (0.0-1.0) |
| `created_at` | DATETIME | When tag was assigned |

**Primary Key:** (`feedback_id`, `tag_name`)

### customer_insights.embedded_items

Tracks which feedback has been embedded (internal use).

| Column | Type | Description |
|--------|------|-------------|
| `feedback_id` | VARCHAR(255) | Primary key |
| `embedded_at` | DATETIME | When embedding was created |

### dbo.inventory_info

Product information mapping.

| Column | Type | Description |
|--------|------|-------------|
| `SKU` | VARCHAR(100) | Product SKU |
| `Style` | VARCHAR(200) | Product style name |

## Tag Categories (30 total)

| Category | Description |
|----------|-------------|
| Waterproof Leak | Water penetration issues |
| Upper Knit Separation | Upper material coming apart |
| Insole Issue | Insole defects or discomfort |
| Inner Lining Rip | Internal lining damage |
| Glue Gap | Adhesive failures |
| Discolouration | Color changes/fading |
| Sizes not standard | General sizing inconsistency |
| Toe Area too narrow/big | Toe box fit issues |
| Instep too small/high | Instep fit problems |
| Shoe too narrow/wide | Width issues |
| Half size requests | Need for half sizes |
| No heel lock/heel slip | Heel fit problems |
| Lack of grip/traction | Sole grip issues |
| Squeaky sound | Noise during wear |
| Not breathable | Ventilation problems |
| Hard to put on/take off | Accessibility issues |
| Lack of support | Insufficient arch/foot support |
| Heel Cup - too big | Oversized heel cup |
| Smelly | Odor issues |
| Back Heel Rubbing | Friction causing irritation |
| Warping | Shape deformation |
| Stains | Marking/staining issues |
| Looks different than picture | Visual discrepancy |
| Blisters | Skin irritation |
| Too Bulky | Excessive volume |
| Too Heavy | Weight concerns |
