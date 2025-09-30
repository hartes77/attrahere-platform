# analytics/queries.py
"""
Analytics query functions for dashboard and reporting.

This module provides high-level query functions to retrieve
pre-aggregated data for dashboards and business intelligence.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from dataclasses import dataclass

@dataclass
class UsageMetrics:
    """Data class for usage metrics."""
    total_analyses: int
    total_users: int
    total_patterns: int
    avg_quality_score: float
    avg_processing_time: float

@dataclass
class PatternTrend:
    """Data class for pattern trend data."""
    pattern_type: str
    total_occurrences: int
    avg_confidence: float
    severity_distribution: Dict[str, int]
    users_affected: int

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    avg_response_time: float
    p95_response_time: float
    error_rate: float
    total_requests: int
    peak_hour: int

class AnalyticsQueries:
    """Analytics query service for dashboard data."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_usage_metrics(
        self, 
        organization_id: Optional[str] = None,
        days: int = 30
    ) -> UsageMetrics:
        """Get usage metrics for the last N days."""
        
        start_date = datetime.now().date() - timedelta(days=days)
        
        query = """
            SELECT 
                SUM(total_analyses) as total_analyses,
                COUNT(DISTINCT user_id) as total_users,
                SUM(total_patterns_found) as total_patterns,
                AVG(avg_code_quality_score) as avg_quality_score,
                AVG(avg_processing_time_ms) as avg_processing_time
            FROM daily_usage_summary 
            WHERE date >= :start_date
        """
        
        params = {"start_date": start_date}
        
        if organization_id:
            query += " AND organization_id = :org_id"
            params["org_id"] = organization_id
        
        result = await self.session.execute(text(query), params)
        row = result.fetchone()
        
        return UsageMetrics(
            total_analyses=row.total_analyses or 0,
            total_users=row.total_users or 0,
            total_patterns=row.total_patterns or 0,
            avg_quality_score=float(row.avg_quality_score or 0),
            avg_processing_time=float(row.avg_processing_time or 0)
        )
    
    async def get_daily_usage_trend(
        self,
        organization_id: Optional[str] = None,
        days: int = 30
    ) -> List[Dict]:
        """Get daily usage trend for the last N days."""
        
        start_date = datetime.now().date() - timedelta(days=days)
        
        query = """
            SELECT 
                date,
                SUM(total_analyses) as analyses,
                COUNT(DISTINCT user_id) as active_users,
                SUM(total_patterns_found) as patterns_found,
                AVG(avg_code_quality_score) as avg_quality
            FROM daily_usage_summary 
            WHERE date >= :start_date
        """
        
        params = {"start_date": start_date}
        
        if organization_id:
            query += " AND organization_id = :org_id"
            params["org_id"] = organization_id
        
        query += " GROUP BY date ORDER BY date"
        
        result = await self.session.execute(text(query), params)
        
        return [
            {
                "date": row.date.isoformat(),
                "analyses": row.analyses or 0,
                "active_users": row.active_users or 0,
                "patterns_found": row.patterns_found or 0,
                "avg_quality": float(row.avg_quality or 0)
            }
            for row in result.fetchall()
        ]
    
    async def get_top_patterns(
        self,
        organization_id: Optional[str] = None,
        days: int = 30,
        limit: int = 10
    ) -> List[PatternTrend]:
        """Get top anti-patterns by occurrence."""
        
        start_date = datetime.now().date() - timedelta(days=days)
        
        # Get top patterns
        query = """
            SELECT 
                pattern_type,
                SUM(total_occurrences) as total_occurrences,
                AVG(avg_confidence_score) as avg_confidence,
                SUM(unique_users_affected) as users_affected
            FROM pattern_trends_summary
            WHERE date >= :start_date
        """
        
        params = {"start_date": start_date}
        
        if organization_id:
            query += " AND organization_id = :org_id"
            params["org_id"] = organization_id
        
        query += """
            GROUP BY pattern_type
            ORDER BY total_occurrences DESC
            LIMIT :limit
        """
        params["limit"] = limit
        
        result = await self.session.execute(text(query), params)
        patterns = []
        
        for row in result.fetchall():
            # Get severity distribution for this pattern
            severity_query = """
                SELECT 
                    severity_level,
                    SUM(total_occurrences) as count
                FROM pattern_trends_summary
                WHERE pattern_type = :pattern_type 
                AND date >= :start_date
            """
            
            severity_params = {
                "pattern_type": row.pattern_type,
                "start_date": start_date
            }
            
            if organization_id:
                severity_query += " AND organization_id = :org_id"
                severity_params["org_id"] = organization_id
            
            severity_query += " GROUP BY severity_level"
            
            severity_result = await self.session.execute(text(severity_query), severity_params)
            severity_dist = {
                sev_row.severity_level: sev_row.count 
                for sev_row in severity_result.fetchall()
            }
            
            patterns.append(PatternTrend(
                pattern_type=row.pattern_type,
                total_occurrences=row.total_occurrences or 0,
                avg_confidence=float(row.avg_confidence or 0),
                severity_distribution=severity_dist,
                users_affected=row.users_affected or 0
            ))
        
        return patterns
    
    async def get_user_leaderboard(
        self,
        organization_id: Optional[str] = None,
        days: int = 30,
        limit: int = 10
    ) -> List[Dict]:
        """Get most active users leaderboard."""
        
        start_date = datetime.now().date() - timedelta(days=days)
        
        query = """
            SELECT 
                u.name,
                u.email,
                SUM(dus.total_analyses) as total_analyses,
                SUM(dus.total_patterns_found) as patterns_found,
                AVG(dus.avg_code_quality_score) as avg_quality_score
            FROM daily_usage_summary dus
            JOIN users u ON dus.user_id = u.id
            WHERE dus.date >= :start_date
        """
        
        params = {"start_date": start_date}
        
        if organization_id:
            query += " AND dus.organization_id = :org_id"
            params["org_id"] = organization_id
        
        query += """
            GROUP BY u.id, u.name, u.email
            ORDER BY total_analyses DESC
            LIMIT :limit
        """
        params["limit"] = limit
        
        result = await self.session.execute(text(query), params)
        
        return [
            {
                "name": row.name,
                "email": row.email,
                "total_analyses": row.total_analyses or 0,
                "patterns_found": row.patterns_found or 0,
                "avg_quality_score": float(row.avg_quality_score or 0)
            }
            for row in result.fetchall()
        ]
    
    async def get_performance_metrics(self, days: int = 7) -> PerformanceMetrics:
        """Get performance metrics for the last N days."""
        
        start_date = datetime.now().date() - timedelta(days=days)
        
        query = """
            SELECT 
                AVG(avg_response_time_ms) as avg_response_time,
                AVG(p95_response_time_ms) as p95_response_time,
                AVG(error_rate) as error_rate,
                SUM(total_requests) as total_requests
            FROM performance_metrics_summary
            WHERE date >= :start_date
        """
        
        result = await self.session.execute(text(query), {"start_date": start_date})
        row = result.fetchone()
        
        # Get peak hour
        peak_query = """
            SELECT hour_of_day, AVG(total_requests) as avg_requests
            FROM performance_metrics_summary
            WHERE date >= :start_date
            GROUP BY hour_of_day
            ORDER BY avg_requests DESC
            LIMIT 1
        """
        
        peak_result = await self.session.execute(text(peak_query), {"start_date": start_date})
        peak_row = peak_result.fetchone()
        
        return PerformanceMetrics(
            avg_response_time=float(row.avg_response_time or 0),
            p95_response_time=float(row.p95_response_time or 0),
            error_rate=float(row.error_rate or 0),
            total_requests=row.total_requests or 0,
            peak_hour=peak_row.hour_of_day if peak_row else 0
        )
    
    async def get_code_quality_trends(
        self,
        organization_id: Optional[str] = None,
        days: int = 30
    ) -> List[Dict]:
        """Get code quality trends over time."""
        
        start_date = datetime.now().date() - timedelta(days=days)
        
        query = """
            SELECT 
                date,
                AVG(avg_code_quality_score) as avg_quality,
                AVG(avg_complexity_score) as avg_complexity,
                SUM(total_patterns_found)::float / NULLIF(SUM(total_analyses), 0) as patterns_per_analysis
            FROM daily_usage_summary
            WHERE date >= :start_date
        """
        
        params = {"start_date": start_date}
        
        if organization_id:
            query += " AND organization_id = :org_id"
            params["org_id"] = organization_id
        
        query += " GROUP BY date ORDER BY date"
        
        result = await self.session.execute(text(query), params)
        
        return [
            {
                "date": row.date.isoformat(),
                "avg_quality": float(row.avg_quality or 0),
                "avg_complexity": float(row.avg_complexity or 0),
                "patterns_per_analysis": float(row.patterns_per_analysis or 0)
            }
            for row in result.fetchall()
        ]
    
    async def get_organization_comparison(self, days: int = 30) -> List[Dict]:
        """Compare performance across organizations."""
        
        start_date = datetime.now().date() - timedelta(days=days)
        
        query = """
            SELECT 
                o.name as organization_name,
                SUM(oas.total_analyses) as total_analyses,
                SUM(oas.total_users_active) as active_users,
                AVG(oas.avg_code_quality_score) as avg_quality,
                SUM(oas.total_patterns_found) as total_patterns
            FROM organization_analytics_summary oas
            JOIN organizations o ON oas.organization_id = o.id
            WHERE oas.date >= :start_date
            GROUP BY o.id, o.name
            ORDER BY total_analyses DESC
        """
        
        result = await self.session.execute(text(query), {"start_date": start_date})
        
        return [
            {
                "organization": row.organization_name,
                "total_analyses": row.total_analyses or 0,
                "active_users": row.active_users or 0,
                "avg_quality": float(row.avg_quality or 0),
                "total_patterns": row.total_patterns or 0
            }
            for row in result.fetchall()
        ]