# analytics/__init__.py
"""
Analytics package for Attrahere ML Code Quality Platform.

This package provides data aggregation and query capabilities
for business intelligence and dashboard functionality.
"""

from .aggregation import AnalyticsAggregator, run_daily_aggregation, run_weekly_cleanup
from .queries import AnalyticsQueries, UsageMetrics, PatternTrend, PerformanceMetrics

__all__ = [
    "AnalyticsAggregator",
    "run_daily_aggregation", 
    "run_weekly_cleanup",
    "AnalyticsQueries",
    "UsageMetrics",
    "PatternTrend",
    "PerformanceMetrics",
]