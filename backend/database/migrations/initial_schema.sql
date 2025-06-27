-- backend/database/migrations/initial_schema.sql
--
-- This script sets up the complete database schema for the Prometheus Consciousness System.
-- It is designed to be executed by a PostgreSQL superuser or a script with equivalent permissions.
-- The script is idempotent; it can be run multiple times without error on the same database.

-- Enable required extensions if they don't already exist.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- IMPORTANT NOTE: PostgreSQL vector extension limitations
-- - HNSW and IVFFlat indexes support maximum 2000 dimensions
-- - Vectors with >2000 dimensions are stored but not indexed
-- - This affects dream_embedding (2048D) and unified_state (2048D)
-- - Text embeddings (768D) can be indexed normally

-- ====================================================================================
-- =============================== MEMORY SYSTEM TABLES ===============================
-- ====================================================================================

-- Working Memory: For transient, short-term data like the current focus of attention.
CREATE TABLE IF NOT EXISTS working_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    key VARCHAR(255) NOT NULL,
    value JSONB NOT NULL,
    ttl_seconds INTEGER DEFAULT 3600,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(session_id, key)
);

CREATE INDEX IF NOT EXISTS idx_wm_session_key ON working_memory(session_id, key);
CREATE INDEX IF NOT EXISTS idx_wm_last_accessed ON working_memory(last_accessed DESC);


-- Truth Memory: Stores verified facts and beliefs with associated confidence.
CREATE TABLE IF NOT EXISTS truths (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim TEXT NOT NULL UNIQUE,
    value VARCHAR(20) NOT NULL CHECK (value IN ('TRUE', 'FALSE', 'UNDETERMINED')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence JSONB,
    source VARCHAR(255),
    claim_embedding vector(768), -- Dimension must match the model in config (e.g., all-mpnet-base-v2)
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_truths_claim_full_text ON truths USING GIN(to_tsvector('english', claim));
CREATE INDEX IF NOT EXISTS idx_truths_value ON truths(value);
CREATE INDEX IF NOT EXISTS idx_truths_confidence ON truths(confidence DESC);
-- HNSW is generally a good choice for high-dimensional vector search.
CREATE INDEX IF NOT EXISTS idx_truths_embedding ON truths USING hnsw(claim_embedding vector_l2_ops);


-- Dream Memory: Stores the outputs of the autonomous dreaming process.
CREATE TABLE IF NOT EXISTS dream_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    symbols JSONB,
    emotions JSONB,
    coherence_score FLOAT CHECK (coherence_score >= 0 AND coherence_score <= 1),
    vividness_score FLOAT CHECK (vividness_score >= 0 AND vividness_score <= 1),
    dream_embedding vector(2048), -- A larger dimension for capturing abstract dream states.
    consciousness_depth FLOAT,
    dream_type VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    session_id VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_dreams_symbols ON dream_entries USING GIN(symbols);
CREATE INDEX IF NOT EXISTS idx_dreams_emotions ON dream_entries USING GIN(emotions);
-- Note: PostgreSQL vector extension has a 2000-dimension limit for indexes.
-- High-dimensional vectors (2048D) are stored but not indexed for now.


-- Contextual Memory (Legacy/Simple): Long-term storage of all interactions.
-- This table is kept for backward compatibility but will be superseded by the Hierarchical Memory.
CREATE TABLE IF NOT EXISTS contextual_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    unified_state vector(2048),
    text_content TEXT,
    text_embedding vector(768),
    mind_states JSONB,
    emotional_context JSONB,
    relevance_score FLOAT DEFAULT 1.0,
    quality_score FLOAT DEFAULT 0.5,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_context_session_time ON contextual_interactions(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_context_text_full_text ON contextual_interactions USING GIN(to_tsvector('english', text_content));
CREATE INDEX IF NOT EXISTS idx_context_text_embedding ON contextual_interactions USING hnsw(text_embedding vector_l2_ops);

-- ====================================================================================
-- =================== HIERARCHICAL MEMORY SYSTEM TABLES (NEW) ========================
-- ====================================================================================

-- Stores individual memories that are promoted from the active tier.
CREATE TABLE IF NOT EXISTS memory_nodes (
    node_id VARCHAR(64) PRIMARY KEY,
    tier VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    timestamp TIMESTAMPTZ NOT NULL,
    token_count INTEGER NOT NULL,
    importance_score FLOAT NOT NULL,
    access_frequency INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    chunk_id VARCHAR(64), -- Can be null for RECENT tier nodes
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nodes_tier_importance ON memory_nodes(tier, importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON memory_nodes(timestamp DESC);


-- Stores compressed chunks of older memories (Long-Term, Archive).
CREATE TABLE IF NOT EXISTS memory_chunks (
    chunk_id VARCHAR(64) PRIMARY KEY,
    tier VARCHAR(20) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    token_count INTEGER NOT NULL,
    compressed_data BYTEA NOT NULL,
    summary TEXT NOT NULL,
    keywords JSONB NOT NULL,
    embedding vector(768),
    compression_ratio FLOAT NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE memory_nodes DROP CONSTRAINT IF EXISTS memory_nodes_chunk_id_fkey;
ALTER TABLE memory_nodes ADD CONSTRAINT memory_nodes_chunk_id_fkey
    FOREIGN KEY (chunk_id) REFERENCES memory_chunks(chunk_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_tier_time ON memory_chunks(tier, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_keywords ON memory_chunks USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON memory_chunks USING hnsw(embedding vector_cosine_ops);


-- ====================================================================================
-- ========================== SYSTEM & OPERATIONAL TABLES =============================
-- ====================================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state VARCHAR(50) DEFAULT 'active' CHECK (state IN ('active', 'archived', 'expired')),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity DESC);


CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metric_type VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    component VARCHAR(100),
    session_id VARCHAR(255),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_type_component ON performance_metrics(metric_type, component);


-- ====================================================================================
-- ============================ FUNCTIONS AND TRIGGERS ================================
-- ====================================================================================

CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trg_update_truths_timestamp ON truths;
CREATE TRIGGER trg_update_truths_timestamp
BEFORE UPDATE ON truths
FOR EACH ROW
EXECUTE FUNCTION update_timestamp_column();


CREATE OR REPLACE FUNCTION track_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.access_count = OLD.access_count + 1;
    NEW.last_accessed = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trg_track_truths_access ON truths;
CREATE TRIGGER trg_track_truths_access
BEFORE UPDATE ON truths
FOR EACH ROW
WHEN (OLD.access_count IS DISTINCT FROM NEW.access_count)
EXECUTE FUNCTION track_access();

DROP TRIGGER IF EXISTS trg_track_context_access ON contextual_interactions;
CREATE TRIGGER trg_track_context_access
BEFORE UPDATE ON contextual_interactions
FOR EACH ROW
WHEN (OLD.access_count IS DISTINCT FROM NEW.access_count)
EXECUTE FUNCTION track_access();


-- ====================================================================================
-- ============================ USER ROLES & PERMISSIONS ==============================
-- ====================================================================================

DO
$$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'prometheus_app') THEN

      CREATE ROLE prometheus_app WITH LOGIN PASSWORD 'your_secure_password_here';
   END IF;
END
$$;

GRANT USAGE, CREATE ON SCHEMA public TO prometheus_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO prometheus_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO prometheus_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO prometheus_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO prometheus_app;