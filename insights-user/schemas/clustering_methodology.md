# Clustering Methodology

This document explains how customer feedback is clustered using UMAP + HDBSCAN with recursive hierarchical clustering.

## Overview

The pipeline uses a two-stage approach:
1. **UMAP** (Uniform Manifold Approximation and Projection) - Reduces 1536-dimensional embeddings to lower dimensions while preserving structure
2. **HDBSCAN** (Hierarchical Density-Based Spatial Clustering) - Finds natural clusters without requiring a predefined cluster count

Clustering is applied **recursively** to discover sub-themes within larger clusters.

## Data Partitioning

Before clustering, data is partitioned by:
1. **Source** - `review`, `return`, `chat`
2. **Style** - Product style (e.g., `Weekend`, `Everyday`, `Cityscape`)

Each source+style combination is clustered independently, resulting in labels like:
```
source_review.style_Weekend.0.1.2
```

## Cluster Label Format

```
source_{source}.style_{style}.{level0}.{level1}.{level2}...

Examples:
├── source_review.style_Weekend.0           # Top-level cluster 0
│   ├── source_review.style_Weekend.0.0     # Sub-cluster 0 of cluster 0
│   ├── source_review.style_Weekend.0.1     # Sub-cluster 1 of cluster 0
│   └── source_review.style_Weekend.0.noise # Points that didn't fit subclusters
├── source_review.style_Weekend.1           # Top-level cluster 1
└── source_review.style_Weekend.noise       # Top-level noise (unclustered)
```

**Special labels:**
- `.noise` - Points that HDBSCAN couldn't assign to any cluster
- `.unclustered` - Groups too small to cluster (< 50 points)
- `.failed` - Clustering failed for this subset

## Clustering Parameters

### Default Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recursive_depth` | 5 | Maximum levels of sub-clustering |
| `min_cluster_pct` | 0.02 (2%) | Minimum cluster size as % of data |
| `min_sample_pct` | 0.003 (0.3%) | HDBSCAN min_samples as % of data |
| `n_neighbors` | 15 | Base UMAP neighborhood size |
| `n_components` | 3 | UMAP output dimensions |
| `hdbscan_metric` | euclidean | Distance metric for HDBSCAN |

### Adaptive Parameter Calculation

Parameters are **automatically adjusted** based on cluster size and depth.

#### UMAP Parameters (Adaptive)

```
n_neighbors = (sqrt(n_points) / 2) * (0.85 ^ depth)

Bounds:
- Minimum: 2 * n_components (to ensure valid embedding)
- Maximum: min(30, 5% of dataset)
```

| Cluster Size | min_dist | spread |
|--------------|----------|--------|
| < 100 points | 0.0 | 1.0 |
| 100-500 points | 0.05 | 1.5 |
| > 500 points | 0.1 | 2.0 |

#### HDBSCAN Parameters (Adaptive)

```
min_cluster_size = (base_pct * 0.6^depth) * n_points

Bounds by size:
- < 100 points:   min=5,  max=n/5
- 100-500 points: min=8,  max=n/8
- 500-2000 points: min=10, max=n/10
- > 2000 points:  min=15, max=n/10

min_samples = max(3, min_cluster_size / 2)
```

## Recursion Control

The pipeline decides whether to recurse deeper based on:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Max depth reached | `depth >= recursive_depth` | Stop |
| Too few points | `< 50 * (1.3^depth)` | Stop |
| Low cluster persistence | Varies by depth (0.05 + 0.025*depth) | Stop |
| Over-fragmentation | `n_clusters > n_points/10` | Stop |
| Excessive noise | `noise_ratio > 50%` | Stop |

## Quality Metrics

The pipeline tracks these metrics for each clustering operation:

### UMAP Quality
| Metric | Description | Good Range |
|--------|-------------|------------|
| `trustworthiness` | How well local structure is preserved | > 0.8 |

### HDBSCAN Quality
| Metric | Description | Good Range |
|--------|-------------|------------|
| `relative_validity` | Overall cluster validity score | > 0 |
| `mean_persistence` | Average cluster stability | > 0.1 |
| `mean_probability` | Confidence in assignments | > 0.7 |
| `noise_ratio` | Fraction of unclustered points | < 0.3 |

## Example: How Clustering Progresses

```
Input: 5000 reviews for source=review, style=Weekend

Depth 0: UMAP + HDBSCAN on 5000 points
├── Cluster 0: 2100 points → recurse
├── Cluster 1: 1800 points → recurse
├── Cluster 2: 850 points → recurse
└── Noise: 250 points → stop

Depth 1: UMAP + HDBSCAN on Cluster 0 (2100 points)
├── Cluster 0.0: 900 points → recurse
├── Cluster 0.1: 750 points → recurse
├── Cluster 0.2: 350 points → stop (persistence too low)
└── Noise: 100 points → stop

Depth 2: UMAP + HDBSCAN on Cluster 0.0 (900 points)
├── Cluster 0.0.0: 400 points → stop (max useful depth)
├── Cluster 0.0.1: 350 points → stop
└── Noise: 150 points → stop

Final clusters for this partition:
- source_review.style_Weekend.0.0.0 (400 points)
- source_review.style_Weekend.0.0.1 (350 points)
- source_review.style_Weekend.0.0.noise (150 points)
- source_review.style_Weekend.0.1.* (subclusters...)
- ... etc
```

## Interpreting Results

### cluster_depth Column

| Depth | Meaning |
|-------|---------|
| 2 | Top-level theme (e.g., "Sizing Issues") |
| 3 | Sub-theme (e.g., "Too Narrow") |
| 4 | Specific issue (e.g., "Narrow in Toe Box") |
| 5+ | Very specific patterns |

### Noise Clusters

Noise points (`.noise` suffix) are feedback items that:
- Don't fit well into any dense cluster
- May be unique complaints or outliers
- Can still contain valuable insights (review manually)

### Recommended Analysis Approach

1. **Start with depth 2** - Understand major themes
2. **Drill into large clusters** - Look at depth 3-4 for details
3. **Check noise clusters** - May contain emerging issues
4. **Compare across styles** - Same issues may appear differently

## Running with Custom Parameters

```bash
# More aggressive sub-clustering
python -m src.pipelines.umap_clustering \
  --recursive-depth 4 \
  --min-cluster-pct 0.01 \
  --start-date 2024-01-01 \
  --end-date 2024-03-31

# Broader clusters (less fragmentation)
python -m src.pipelines.umap_clustering \
  --recursive-depth 2 \
  --min-cluster-pct 0.05 \
  --n-neighbors 30

# Local analysis with visualizations
python -m src.pipelines.umap_clustering \
  --local \
  --output-dir ./my_analysis \
  --recursive-depth 3
```
