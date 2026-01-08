-- scripts/init_db.sql
-- Database initialization script for breast cancer risk prediction system

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create users table (optional - for authenticated access)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255),
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Risk prediction results
    risk_score FLOAT NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_category VARCHAR(20) NOT NULL CHECK (risk_category IN ('low', 'moderate', 'high')),
    confidence_lower FLOAT CHECK (confidence_lower >= 0 AND confidence_lower <= 1),
    confidence_upper FLOAT CHECK (confidence_upper >= 0 AND confidence_upper <= 1),
    percentile INTEGER CHECK (percentile >= 0 AND percentile <= 100),
    
    -- Model information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'ga_optimized',
    
    -- Input features (stored as JSONB for flexibility)
    input_features JSONB NOT NULL,
    preprocessed_features JSONB,
    
    -- Feature importance and analysis
    feature_importance JSONB,
    top_risk_factors JSONB,
    
    -- Model comparison (if multiple models used)
    comparison_results JSONB,
    
    -- Performance metrics
    processing_time_ms FLOAT,
    inference_time_ms FLOAT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_predictions_created_at (created_at DESC),
    INDEX idx_predictions_user_id (user_id),
    INDEX idx_predictions_risk_category (risk_category),
    INDEX idx_predictions_model_version (model_version),
    INDEX idx_predictions_risk_score (risk_score)
);

-- Create model registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_stage VARCHAR(20) CHECK (model_stage IN ('None', 'Staging', 'Production', 'Archived')),
    mlflow_run_id VARCHAR(255),
    
    -- Model metadata
    algorithm VARCHAR(50),
    features_used JSONB,
    feature_count INTEGER,
    training_date TIMESTAMP WITH TIME ZONE,
    
    -- Performance metrics
    auc FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    
    -- Deployment info
    deployed_at TIMESTAMP WITH TIME ZONE,
    deployed_by VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(model_name, model_version),
    INDEX idx_model_registry_active (is_active),
    INDEX idx_model_registry_stage (model_stage)
);

-- Create feature distributions table for drift monitoring
CREATE TABLE IF NOT EXISTS feature_distributions (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    hour INTEGER DEFAULT 0,
    
    -- Distribution statistics
    mean FLOAT,
    std_dev FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    p25 FLOAT,
    p50 FLOAT,
    p75 FLOAT,
    
    -- Counts
    sample_count INTEGER,
    missing_count INTEGER DEFAULT 0,
    
    -- Drift metrics
    drift_score FLOAT,
    is_drifted BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(feature_name, date, hour),
    INDEX idx_feature_dist_date (date DESC),
    INDEX idx_feature_dist_drift (is_drifted)
);

-- Create analytics events table
CREATE TABLE IF NOT EXISTS analytics_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    prediction_id UUID,
    
    -- Event metadata
    ip_address INET,
    user_agent TEXT,
    referrer TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_analytics_event_type (event_type),
    INDEX idx_analytics_created_at (created_at DESC)
);

-- Create API keys table (for programmatic access)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    scopes JSONB DEFAULT '["predict"]',
    
    -- Usage tracking
    last_used_at TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    
    -- Lifecycle
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_api_keys_user (user_id),
    INDEX idx_api_keys_active (is_active)
);

-- Create performance monitoring table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- API metrics
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms FLOAT,
    
    -- Model metrics
    model_name VARCHAR(100),
    model_latency_ms FLOAT,
    batch_size INTEGER DEFAULT 1,
    
    -- System metrics
    cpu_usage_percent FLOAT,
    memory_usage_mb FLOAT,
    
    -- Error info
    error_type VARCHAR(255),
    error_message TEXT,
    
    INDEX idx_perf_metrics_timestamp (timestamp DESC),
    INDEX idx_perf_metrics_endpoint (endpoint)
);

-- Create materialized views for analytics

-- Daily prediction statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_prediction_stats AS
SELECT 
    DATE(created_at) as prediction_date,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(risk_score)::FLOAT as avg_risk_score,
    STDDEV(risk_score)::FLOAT as std_risk_score,
    
    -- Risk category distribution
    COUNT(*) FILTER (WHERE risk_category = 'low') as low_risk_count,
    COUNT(*) FILTER (WHERE risk_category = 'moderate') as moderate_risk_count,
    COUNT(*) FILTER (WHERE risk_category = 'high') as high_risk_count,
    
    -- Performance metrics
    AVG(processing_time_ms)::FLOAT as avg_processing_time,
    MAX(processing_time_ms)::FLOAT as max_processing_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms)::FLOAT as p95_processing_time,
    
    -- Model usage
    COUNT(*) FILTER (WHERE model_type = 'ga_optimized') as ga_model_count,
    COUNT(*) FILTER (WHERE model_type = 'baseline') as baseline_model_count

FROM predictions
GROUP BY DATE(created_at)
ORDER BY prediction_date DESC;

-- Create indexes on materialized view
CREATE INDEX idx_daily_stats_date ON daily_prediction_stats(prediction_date DESC);

-- Feature importance trends
CREATE MATERIALIZED VIEW IF NOT EXISTS feature_importance_trends AS
SELECT 
    feature_name,
    DATE(p.created_at) as date,
    AVG((feature_data->>'importance')::FLOAT) as avg_importance,
    COUNT(*) as sample_count
FROM predictions p,
     LATERAL jsonb_each(p.feature_importance) AS feature_data(feature_name, feature_data)
WHERE p.feature_importance IS NOT NULL
GROUP BY feature_name, DATE(p.created_at)
ORDER BY date DESC, avg_importance DESC;

-- Functions and triggers

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON model_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean old predictions
CREATE OR REPLACE FUNCTION cleanup_old_predictions(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM predictions
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep
    AND user_id IS NULL;  -- Only delete anonymous predictions
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Refresh materialized views function
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_prediction_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY feature_importance_trends;
END;
$$ LANGUAGE plpgsql;

-- Initial data

-- Create admin user (change password in production!)
INSERT INTO users (email, hashed_password, full_name, is_admin, is_active)
VALUES ('admin@breastcancerrisk.ai', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewYpawUi4rssmYXC', 'Admin User', TRUE, TRUE)
ON CONFLICT (email) DO NOTHING;

-- Create sample API key (for testing - remove in production!)
INSERT INTO api_keys (user_id, key_hash, name, scopes)
SELECT id, '$2b$12$SAMPLE_KEY_HASH_CHANGE_THIS', 'Test API Key', '["predict", "read"]'::jsonb
FROM users WHERE email = 'admin@breastcancerrisk.ai'
ON CONFLICT (key_hash) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO breast_cancer;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO breast_cancer;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO breast_cancer;