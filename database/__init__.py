# database/__init__.py
"""
Database package for Attrahere ML Code Quality Platform.

This package provides database configuration, models, and utilities
for persisting analysis events and user data.
"""

from .config import get_database_session, init_database, close_database
from .models import Organization, User, AnalysisEvent, PatternDetection, ApiUsageEvent

__all__ = [
    "get_database_session",
    "init_database", 
    "close_database",
    "Organization",
    "User", 
    "AnalysisEvent",
    "PatternDetection",
    "ApiUsageEvent",
]