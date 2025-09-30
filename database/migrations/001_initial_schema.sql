-- 001_initial_schema.sql
-- Initial database schema for Attrahere ML Code Quality Platform
-- This migration creates the foundation tables for data persistence and analytics

-- Enable UUID extension for better primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Organizations table for multi-tenant support
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    
    -- Metadata
    plan_type VARCHAR(50) DEFAULT 'free', -- free, pro, enterprise
    max_analyses_per_month INTEGER DEFAULT 1000,
    
    CONSTRAINT organizations_slug_format CHECK (slug ~ '^[a-z0-9-]+$')
);

-- Users table for authentication and tracking
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(64) UNIQUE NOT NULL, -- SHA-256 hash of API key
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    
    -- Usage tracking
    total_analyses INTEGER DEFAULT 0,
    current_month_analyses INTEGER DEFAULT 0,
    
    CONSTRAINT users_email_format CHECK (email ~ '^[^@]+@[^@]+\.[^@]+$')
);

-- Analysis events table - the core data persistence table
CREATE TABLE analysis_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Request metadata
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    request_ip_hash VARCHAR(64), -- Hashed IP for privacy compliance
    user_agent_hash VARCHAR(64), -- Hashed user agent
    
    -- Code analysis data
    code_length INTEGER NOT NULL, -- Number of characters in analyzed code
    code_language VARCHAR(50) NOT NULL, -- Detected or specified language
    code_hash VARCHAR(64) NOT NULL, -- SHA-256 of the code for deduplication
    
    -- Analysis results
    patterns_detected JSONB NOT NULL DEFAULT '[]', -- Array of detected anti-patterns
    total_patterns_count INTEGER NOT NULL DEFAULT 0,
    analysis_confidence DECIMAL(3,2), -- 0.00 to 1.00
    
    -- Performance metrics
    processing_time_ms INTEGER NOT NULL,
    analyzer_version VARCHAR(50) NOT NULL, -- V3, V4, etc.
    
    -- Quality scores (for future analytics)
    code_quality_score DECIMAL(3,2), -- 0.00 to 1.00
    complexity_score DECIMAL(3,2),   -- 0.00 to 1.00
    
    CONSTRAINT analysis_events_confidence_range CHECK (analysis_confidence BETWEEN 0 AND 1),
    CONSTRAINT analysis_events_quality_score_range CHECK (code_quality_score BETWEEN 0 AND 1),
    CONSTRAINT analysis_events_complexity_score_range CHECK (complexity_score BETWEEN 0 AND 1),
    CONSTRAINT analysis_events_processing_time_positive CHECK (processing_time_ms > 0),
    CONSTRAINT analysis_events_code_length_positive CHECK (code_length > 0)
);

-- Pattern details table for granular anti-pattern tracking
CREATE TABLE pattern_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_event_id UUID NOT NULL REFERENCES analysis_events(id) ON DELETE CASCADE,
    
    -- Pattern identification
    pattern_type VARCHAR(100) NOT NULL, -- magic_number, god_class, etc.
    pattern_category VARCHAR(50) NOT NULL, -- code_smell, anti_pattern, etc.
    severity_level VARCHAR(20) NOT NULL, -- low, medium, high, critical
    
    -- Location in code
    line_number INTEGER,
    column_number INTEGER,
    code_snippet TEXT, -- The actual problematic code
    
    -- Detection details
    confidence_score DECIMAL(3,2) NOT NULL,
    suggested_fix TEXT, -- AI-generated suggestion for improvement
    
    -- Metadata
    detector_name VARCHAR(100) NOT NULL, -- Which specific detector found this
    detection_time_ms INTEGER NOT NULL,
    
    CONSTRAINT pattern_detections_confidence_range CHECK (confidence_score BETWEEN 0 AND 1),
    CONSTRAINT pattern_detections_severity_values CHECK (severity_level IN ('low', 'medium', 'high', 'critical'))
);

-- API usage tracking for rate limiting and billing
CREATE TABLE api_usage_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Request details
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    endpoint VARCHAR(255) NOT NULL,
    http_method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    
    -- Performance
    response_time_ms INTEGER NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    
    -- Rate limiting
    rate_limit_remaining INTEGER,
    rate_limit_reset_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance optimization
CREATE INDEX idx_analysis_events_user_timestamp ON analysis_events(user_id, timestamp DESC);
CREATE INDEX idx_analysis_events_org_timestamp ON analysis_events(organization_id, timestamp DESC);
CREATE INDEX idx_analysis_events_code_hash ON analysis_events(code_hash);
CREATE INDEX idx_analysis_events_analyzer_version ON analysis_events(analyzer_version);
CREATE INDEX idx_analysis_events_timestamp ON analysis_events(timestamp DESC);

CREATE INDEX idx_pattern_detections_analysis_event ON pattern_detections(analysis_event_id);
CREATE INDEX idx_pattern_detections_pattern_type ON pattern_detections(pattern_type);
CREATE INDEX idx_pattern_detections_severity ON pattern_detections(severity_level);

CREATE INDEX idx_api_usage_user_timestamp ON api_usage_events(user_id, timestamp DESC);
CREATE INDEX idx_api_usage_org_timestamp ON api_usage_events(organization_id, timestamp DESC);

CREATE INDEX idx_users_api_key_hash ON users(api_key_hash);
CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_organizations_slug ON organizations(slug);

-- Function to automatically update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default organization for single-tenant initial setup
INSERT INTO organizations (name, slug, plan_type, max_analyses_per_month) 
VALUES ('Default Organization', 'default', 'enterprise', 999999)
ON CONFLICT (slug) DO NOTHING;