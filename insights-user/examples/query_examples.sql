-- =============================================================================
-- Customer Insights - Common SQL Queries for Reporting
-- =============================================================================

-- -----------------------------------------------------------------------------
-- CLUSTER ANALYSIS
-- -----------------------------------------------------------------------------

-- Get all clusters with record counts (top-level only)
SELECT
    cluster_id,
    cluster_description,
    record_count,
    style,
    source,
    period_start,
    period_end
FROM customer_insights.clusters
WHERE cluster_depth = 0
ORDER BY record_count DESC;

-- Get cluster hierarchy for a specific style
SELECT
    cluster_id,
    cluster_depth,
    cluster_description,
    record_count
FROM customer_insights.clusters
WHERE style = 'Weekend'
ORDER BY cluster_depth, record_count DESC;

-- Get top 10 largest clusters across all styles
SELECT TOP 10
    cluster_id,
    cluster_description,
    record_count,
    style,
    source
FROM customer_insights.clusters
ORDER BY record_count DESC;

-- -----------------------------------------------------------------------------
-- FEEDBACK ANALYSIS
-- -----------------------------------------------------------------------------

-- Get feedback for a specific cluster
SELECT
    f.feedback_id,
    f.feedback_text,
    f.feedback_source,
    f.rating,
    f.created_at
FROM customer_insights.feedback f
WHERE f.cluster_id = 'source_review.style_Weekend.0.1'
ORDER BY f.created_at DESC;

-- Count feedback by source
SELECT
    feedback_source,
    COUNT(*) as count
FROM customer_insights.feedback
WHERE cluster_id IS NOT NULL
GROUP BY feedback_source
ORDER BY count DESC;

-- Get feedback with cluster info joined
SELECT
    f.feedback_id,
    f.feedback_text,
    f.feedback_source,
    f.rating,
    c.cluster_description,
    c.record_count
FROM customer_insights.feedback f
JOIN customer_insights.clusters c ON f.cluster_id = c.cluster_id
WHERE f.created_at >= '2024-01-01'
ORDER BY c.record_count DESC, f.created_at DESC;

-- -----------------------------------------------------------------------------
-- TAG ANALYSIS
-- -----------------------------------------------------------------------------

-- Get most common tags
SELECT
    tag_name,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence
FROM customer_insights.tags
GROUP BY tag_name
ORDER BY count DESC;

-- Get tags for a specific feedback item
SELECT
    t.tag_name,
    t.confidence_score,
    f.feedback_text
FROM customer_insights.tags t
JOIN customer_insights.feedback f ON t.feedback_id = f.feedback_id
WHERE t.feedback_id = 'your-feedback-id';

-- Get feedback tagged with a specific category
SELECT
    f.feedback_id,
    f.feedback_text,
    f.feedback_source,
    t.confidence_score
FROM customer_insights.tags t
JOIN customer_insights.feedback f ON t.feedback_id = f.feedback_id
WHERE t.tag_name = 'Waterproof Leak'
AND t.confidence_score >= 0.8
ORDER BY t.confidence_score DESC;

-- Cross-tabulate clusters and tags
SELECT
    c.cluster_id,
    c.cluster_description,
    t.tag_name,
    COUNT(*) as count
FROM customer_insights.feedback f
JOIN customer_insights.clusters c ON f.cluster_id = c.cluster_id
JOIN customer_insights.tags t ON f.feedback_id = t.feedback_id
WHERE c.cluster_depth = 0
GROUP BY c.cluster_id, c.cluster_description, t.tag_name
ORDER BY c.cluster_id, count DESC;

-- -----------------------------------------------------------------------------
-- PRODUCT ANALYSIS
-- -----------------------------------------------------------------------------

-- Get clusters by product style
SELECT
    i.Style,
    c.cluster_id,
    c.cluster_description,
    c.record_count
FROM customer_insights.clusters c
JOIN customer_insights.feedback f ON f.cluster_id = c.cluster_id
JOIN dbo.inventory_info i ON f.sku = i.SKU
WHERE c.cluster_depth = 0
GROUP BY i.Style, c.cluster_id, c.cluster_description, c.record_count
ORDER BY i.Style, c.record_count DESC;

-- Get top issues by product category
SELECT
    f.category,
    t.tag_name,
    COUNT(*) as count
FROM customer_insights.feedback f
JOIN customer_insights.tags t ON f.feedback_id = t.feedback_id
WHERE t.confidence_score >= 0.7
GROUP BY f.category, t.tag_name
ORDER BY f.category, count DESC;

-- -----------------------------------------------------------------------------
-- TIME-BASED ANALYSIS
-- -----------------------------------------------------------------------------

-- Monthly cluster trends
SELECT
    FORMAT(f.created_at, 'yyyy-MM') as month,
    c.cluster_description,
    COUNT(*) as count
FROM customer_insights.feedback f
JOIN customer_insights.clusters c ON f.cluster_id = c.cluster_id
WHERE c.cluster_depth = 0
GROUP BY FORMAT(f.created_at, 'yyyy-MM'), c.cluster_description
ORDER BY month, count DESC;

-- Weekly tag trends
SELECT
    DATEPART(year, f.created_at) as year,
    DATEPART(week, f.created_at) as week,
    t.tag_name,
    COUNT(*) as count
FROM customer_insights.feedback f
JOIN customer_insights.tags t ON f.feedback_id = t.feedback_id
GROUP BY DATEPART(year, f.created_at), DATEPART(week, f.created_at), t.tag_name
ORDER BY year, week, count DESC;
