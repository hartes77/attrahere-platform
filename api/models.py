"""
API Data Models for Experimental ML Code Analysis Tool

This module contains basic Pydantic models for API request/response validation
and documentation. Contains experimental test set contamination detection patterns.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class PatternSeverity(str, Enum):
    """Severity levels for detected ML anti-patterns"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


class PatternCategory(str, Enum):
    """Categories of ML anti-patterns for organization"""
    DATA_LEAKAGE = "data_leakage"
    TEST_CONTAMINATION = "test_contamination"
    GPU_MEMORY = "gpu_memory"
    REPRODUCIBILITY = "reproducibility"
    PERFORMANCE = "performance"
    CODE_QUALITY = "code_quality"


# === Request Models ===

class CodeAnalysisRequest(BaseModel):
    """Request for experimental ML code analysis"""
    code: str = Field(
        ..., 
        description="Python source code to analyze (experimental)",
        example="""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
y = [0, 1, 0]

# This might be detected as potential data leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
"""
    )
    file_name: Optional[str] = Field(
        "example.py",
        description="Optional filename for analysis context"
    )


class CreateUserRequest(BaseModel):
    """Request to create a new user account"""
    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User full name")
    organization_slug: Optional[str] = Field(
        "default", 
        description="Organization identifier"
    )


# === Response Models ===

class MLAntiPatternResponse(BaseModel):
    """Detected ML anti-pattern (experimental detection)"""
    pattern_type: str = Field(
        ..., 
        description="Pattern identifier",
        example="preprocessing_before_split"
    )
    category: PatternCategory = Field(
        ...,
        description="Basic category of the pattern"
    )
    severity: PatternSeverity = Field(
        ...,
        description="Estimated severity level"
    )
    line_number: int = Field(
        ...,
        description="Line number where detected",
        example=15
    )
    column: int = Field(
        0,
        description="Column position (if available)"
    )
    message: str = Field(
        ...,
        description="Basic description of the issue",
        example="Preprocessing applied before train/test split"
    )
    explanation: str = Field(
        ...,
        description="Simple explanation of the potential problem",
        example="Applying preprocessing to entire dataset before splitting may cause data leakage."
    )
    suggested_fix: str = Field(
        ...,
        description="Basic suggestion for fixing",
        example="Move preprocessing after train_test_split"
    )
    confidence: float = Field(
        ...,
        description="Experimental confidence score (0.0 to 1.0)",
        example=0.80,
        ge=0.0,
        le=1.0
    )
    code_snippet: str = Field(
        ...,
        description="Code context where issue was found",
        example="scaler.fit_transform(X)"
    )
    fix_snippet: str = Field(
        "",
        description="Basic example of potential fix",
        example="""X_train, X_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)"""
    )
    references: List[str] = Field(
        [],
        description="Basic references (if any)",
        example=["https://scikit-learn.org/stable/common_pitfalls.html"]
    )


class CodeAnalysisResponse(BaseModel):
    """Basic analysis results (experimental)"""
    file_name: str = Field(..., description="Name of analyzed file")
    patterns: List[MLAntiPatternResponse] = Field(
        ...,
        description="List of detected patterns (experimental)"
    )
    analysis_summary: Dict[str, Any] = Field(
        ...,
        description="Basic analysis statistics",
        example={
            "total_patterns": 2,
            "critical_issues": 1,
            "high_issues": 1,
            "medium_issues": 0,
            "low_issues": 0,
            "processing_time_ms": 3,
            "lines_analyzed": 12,
            "detector_version": "experimental"
        }
    )


class CreateUserResponse(BaseModel):
    """Response with new user details"""
    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User name")
    api_key: str = Field(..., description="Generated API key for authentication")
    organization: str = Field(..., description="Organization name")


class UserStatsResponse(BaseModel):
    """User statistics and usage information"""
    user_id: str = Field(..., description="User identifier")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User name")
    organization: str = Field(..., description="Organization name")
    total_analyses: int = Field(..., description="Total analyses performed")
    current_month_analyses: int = Field(..., description="Analyses this month")
    max_monthly_analyses: int = Field(..., description="Monthly analysis limit")
    remaining_analyses: int = Field(..., description="Remaining analyses this month")


# === Pattern Type Definitions for Documentation ===

class TestContaminationPatterns(str, Enum):
    """Test Set Contamination Pattern Types (Sprint 3)"""
    TEST_SET_EXACT_DUPLICATES = "test_set_exact_duplicates"
    MISSING_DUPLICATE_CHECK = "missing_duplicate_check"
    CROSS_SPLIT_MERGE = "cross_split_merge"
    PREPROCESSING_BEFORE_SPLIT = "preprocessing_before_split"
    FEATURE_LEAKAGE = "feature_leakage"
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    INCORRECT_TEMPORAL_SPLIT = "incorrect_temporal_split"
    CV_PREPROCESSING_LEAKAGE = "cv_preprocessing_leakage"
    CV_FEATURE_SELECTION_LEAKAGE = "cv_feature_selection_leakage"


class DataLeakagePatterns(str, Enum):
    """Data Leakage Pattern Types"""
    DATA_LEAKAGE_PREPROCESSING = "data_leakage_preprocessing"
    DATA_LEAKAGE_TARGET = "data_leakage_target"
    DATA_LEAKAGE_TEMPORAL = "data_leakage_temporal"


class GPUMemoryPatterns(str, Enum):
    """GPU Memory Issue Pattern Types"""
    GPU_MEMORY_LEAK = "gpu_memory_leak"
    INEFFICIENT_GPU_USAGE = "inefficient_gpu_usage"
    MISSING_GPU_CLEANUP = "missing_gpu_cleanup"


class ReproducibilityPatterns(str, Enum):
    """Reproducibility Issue Pattern Types"""
    MISSING_RANDOM_SEED = "missing_random_seed"
    NON_DETERMINISTIC_OPERATION = "non_deterministic_operation"
    MISSING_VERSION_PINNING = "missing_version_pinning"


class PerformancePatterns(str, Enum):
    """Performance Issue Pattern Types"""
    INEFFICIENT_DATA_LOADING = "inefficient_data_loading"
    MAGIC_NUMBERS = "magic_numbers"
    HARDCODED_THRESHOLDS = "hardcoded_thresholds"


# === Error Models ===

class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")


class APIError(BaseModel):
    """Generic API error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class RateLimitError(BaseModel):
    """Rate limit exceeded error"""
    error: str = Field("rate_limit_exceeded", description="Error type")
    message: str = Field(..., description="Rate limit error message")
    retry_after: int = Field(..., description="Seconds until rate limit resets")
    limit: int = Field(..., description="Current rate limit")
    remaining: int = Field(0, description="Remaining requests")


# === Analytics Models ===

class AnalyticsResponse(BaseModel):
    """Analytics data response"""
    period: str = Field(..., description="Time period of the data")
    total_analyses: int = Field(..., description="Total analyses in period")
    unique_users: int = Field(..., description="Unique users in period") 
    top_patterns: List[Dict[str, Any]] = Field(
        ...,
        description="Most common patterns detected",
        example=[
            {"pattern_type": "test_set_exact_duplicates", "count": 145, "percentage": 23.4},
            {"pattern_type": "preprocessing_before_split", "count": 98, "percentage": 15.8}
        ]
    )
    pattern_trends: List[Dict[str, Any]] = Field(
        ...,
        description="Pattern detection trends over time"
    )


# === Health Check Models ===

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field("ok", description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Response timestamp")
    uptime_seconds: float = Field(..., description="Service uptime")
    database_status: str = Field(..., description="Database connection status")
    detector_status: Dict[str, str] = Field(
        ...,
        description="Status of each detector module",
        example={
            "test_contamination": "ok",
            "data_leakage": "ok", 
            "gpu_memory": "ok",
            "reproducibility": "ok"
        }
    )