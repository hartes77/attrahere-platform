# database/models.py
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Numeric, 
    Text, ForeignKey, JSON, CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .config import Base

class Organization(Base):
    """Organization model for multi-tenant support."""
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Plan configuration
    plan_type = Column(String(50), default='free')
    max_analyses_per_month = Column(Integer, default=1000)
    
    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    analysis_events = relationship("AnalysisEvent", back_populates="organization", cascade="all, delete-orphan")
    api_usage_events = relationship("ApiUsageEvent", back_populates="organization", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint("slug ~ '^[a-z0-9-]+$'", name='organizations_slug_format'),
        Index('idx_organizations_slug', 'slug'),
    )

class User(Base):
    """User model for authentication and tracking."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    api_key_hash = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_active_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    
    # Usage tracking
    total_analyses = Column(Integer, default=0)
    current_month_analyses = Column(Integer, default=0)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    analysis_events = relationship("AnalysisEvent", back_populates="user", cascade="all, delete-orphan")
    api_usage_events = relationship("ApiUsageEvent", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint("email ~ '^[^@]+@[^@]+\\.[^@]+$'", name='users_email_format'),
        Index('idx_users_api_key_hash', 'api_key_hash'),
        Index('idx_users_organization', 'organization_id'),
    )

class AnalysisEvent(Base):
    """Core table for storing analysis events."""
    __tablename__ = "analysis_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False)
    
    # Request metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    request_ip_hash = Column(String(64))
    user_agent_hash = Column(String(64))
    
    # Code analysis data
    code_length = Column(Integer, nullable=False)
    code_language = Column(String(50), nullable=False)
    code_hash = Column(String(64), nullable=False)
    
    # Analysis results
    patterns_detected = Column(JSON, nullable=False, default=list)
    total_patterns_count = Column(Integer, nullable=False, default=0)
    analysis_confidence = Column(Numeric(3, 2))
    
    # Performance metrics
    processing_time_ms = Column(Integer, nullable=False)
    analyzer_version = Column(String(50), nullable=False)
    
    # Quality scores
    code_quality_score = Column(Numeric(3, 2))
    complexity_score = Column(Numeric(3, 2))
    
    # Relationships
    user = relationship("User", back_populates="analysis_events")
    organization = relationship("Organization", back_populates="analysis_events")
    pattern_detections = relationship("PatternDetection", back_populates="analysis_event", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint("analysis_confidence BETWEEN 0 AND 1", name='analysis_events_confidence_range'),
        CheckConstraint("code_quality_score BETWEEN 0 AND 1", name='analysis_events_quality_score_range'),
        CheckConstraint("complexity_score BETWEEN 0 AND 1", name='analysis_events_complexity_score_range'),
        CheckConstraint("processing_time_ms > 0", name='analysis_events_processing_time_positive'),
        CheckConstraint("code_length > 0", name='analysis_events_code_length_positive'),
        Index('idx_analysis_events_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_analysis_events_org_timestamp', 'organization_id', 'timestamp'),
        Index('idx_analysis_events_code_hash', 'code_hash'),
        Index('idx_analysis_events_analyzer_version', 'analyzer_version'),
        Index('idx_analysis_events_timestamp', 'timestamp'),
    )

class PatternDetection(Base):
    """Detailed pattern detection records."""
    __tablename__ = "pattern_detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_event_id = Column(UUID(as_uuid=True), ForeignKey('analysis_events.id'), nullable=False)
    
    # Pattern identification
    pattern_type = Column(String(100), nullable=False)
    pattern_category = Column(String(50), nullable=False)
    severity_level = Column(String(20), nullable=False)
    
    # Location in code
    line_number = Column(Integer)
    column_number = Column(Integer)
    code_snippet = Column(Text)
    
    # Detection details
    confidence_score = Column(Numeric(3, 2), nullable=False)
    suggested_fix = Column(Text)
    
    # Metadata
    detector_name = Column(String(100), nullable=False)
    detection_time_ms = Column(Integer, nullable=False)
    
    # Relationships
    analysis_event = relationship("AnalysisEvent", back_populates="pattern_detections")
    
    __table_args__ = (
        CheckConstraint("confidence_score BETWEEN 0 AND 1", name='pattern_detections_confidence_range'),
        CheckConstraint("severity_level IN ('low', 'medium', 'high', 'critical')", name='pattern_detections_severity_values'),
        Index('idx_pattern_detections_analysis_event', 'analysis_event_id'),
        Index('idx_pattern_detections_pattern_type', 'pattern_type'),
        Index('idx_pattern_detections_severity', 'severity_level'),
    )

class ApiUsageEvent(Base):
    """API usage tracking for rate limiting and billing."""
    __tablename__ = "api_usage_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False)
    
    # Request details
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    endpoint = Column(String(255), nullable=False)
    http_method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    
    # Performance
    response_time_ms = Column(Integer, nullable=False)
    request_size_bytes = Column(Integer)
    response_size_bytes = Column(Integer)
    
    # Rate limiting
    rate_limit_remaining = Column(Integer)
    rate_limit_reset_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="api_usage_events")
    organization = relationship("Organization", back_populates="api_usage_events")
    
    __table_args__ = (
        Index('idx_api_usage_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_api_usage_org_timestamp', 'organization_id', 'timestamp'),
    )