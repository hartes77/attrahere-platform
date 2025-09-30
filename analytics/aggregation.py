# analytics/aggregation.py
"""
Data aggregation scripts for analytics and reporting.

This module contains functions to pre-aggregate raw analysis events
into summary tables for fast reporting and dashboard queries.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, text, and_
from sqlalchemy.dialects.postgresql import insert

from database.config import AsyncSessionLocal
from database.models import (
    AnalysisEvent, PatternDetection, User, Organization,
    ApiUsageEvent
)

class AnalyticsAggregator:
    """Main class for data aggregation operations."""
    
    def __init__(self):
        self.session: Optional[AsyncSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = AsyncSessionLocal()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def create_summary_tables(self):
        """Create summary tables for pre-aggregated data."""
        
        # Daily usage summary table
        await self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_usage_summary (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                date DATE NOT NULL,
                organization_id UUID REFERENCES organizations(id),
                user_id UUID REFERENCES users(id),
                total_analyses INTEGER DEFAULT 0,
                total_patterns_found INTEGER DEFAULT 0,
                avg_processing_time_ms NUMERIC(10,2),
                avg_code_quality_score NUMERIC(3,2),
                avg_complexity_score NUMERIC(3,2),
                unique_pattern_types INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                UNIQUE(date, organization_id, user_id)
            );
        """))
        
        # Pattern trends summary table
        await self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS pattern_trends_summary (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                date DATE NOT NULL,
                pattern_type VARCHAR(100) NOT NULL,
                pattern_category VARCHAR(50) NOT NULL,
                severity_level VARCHAR(20) NOT NULL,
                organization_id UUID REFERENCES organizations(id),
                total_occurrences INTEGER DEFAULT 0,
                avg_confidence_score NUMERIC(3,2),
                unique_users_affected INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                UNIQUE(date, pattern_type, organization_id)
            );
        """))
        
        # Organization analytics summary
        await self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS organization_analytics_summary (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                date DATE NOT NULL,
                organization_id UUID REFERENCES organizations(id),
                total_analyses INTEGER DEFAULT 0,
                total_users_active INTEGER DEFAULT 0,
                total_patterns_found INTEGER DEFAULT 0,
                avg_code_quality_score NUMERIC(3,2),
                top_pattern_type VARCHAR(100),
                total_api_requests INTEGER DEFAULT 0,
                avg_response_time_ms NUMERIC(10,2),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                UNIQUE(date, organization_id)
            );
        """))
        
        # Performance metrics summary
        await self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS performance_metrics_summary (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                date DATE NOT NULL,
                hour_of_day INTEGER NOT NULL,
                total_requests INTEGER DEFAULT 0,
                avg_response_time_ms NUMERIC(10,2),
                max_response_time_ms INTEGER DEFAULT 0,
                error_rate NUMERIC(3,2) DEFAULT 0,
                p95_response_time_ms NUMERIC(10,2),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                UNIQUE(date, hour_of_day)
            );
        """))
        
        await self.session.commit()
        print("Summary tables created successfully")
    
    async def aggregate_daily_usage(self, target_date: datetime = None):
        """Aggregate daily usage statistics."""
        if target_date is None:
            target_date = datetime.now().date() - timedelta(days=1)  # Yesterday
        
        print(f"Aggregating daily usage for {target_date}")
        
        # Delete existing data for this date to avoid duplicates
        await self.session.execute(
            text("DELETE FROM daily_usage_summary WHERE date = :date"),
            {"date": target_date}
        )
        
        # Aggregate daily usage per user/organization
        result = await self.session.execute(text("""
            INSERT INTO daily_usage_summary (
                date, organization_id, user_id, total_analyses, 
                total_patterns_found, avg_processing_time_ms,
                avg_code_quality_score, avg_complexity_score, unique_pattern_types
            )
            SELECT 
                :target_date as date,
                ae.organization_id,
                ae.user_id,
                COUNT(*) as total_analyses,
                SUM(ae.total_patterns_count) as total_patterns_found,
                AVG(ae.processing_time_ms) as avg_processing_time_ms,
                AVG(ae.code_quality_score) as avg_code_quality_score,
                AVG(ae.complexity_score) as avg_complexity_score,
                COUNT(DISTINCT pd.pattern_type) as unique_pattern_types
            FROM analysis_events ae
            LEFT JOIN pattern_detections pd ON ae.id = pd.analysis_event_id
            WHERE DATE(ae.timestamp) = :target_date
            GROUP BY ae.organization_id, ae.user_id
        """), {"target_date": target_date})
        
        await self.session.commit()
        print(f"Daily usage aggregation completed for {target_date}")
    
    async def aggregate_pattern_trends(self, target_date: datetime = None):
        """Aggregate pattern trend statistics."""
        if target_date is None:
            target_date = datetime.now().date() - timedelta(days=1)
        
        print(f"Aggregating pattern trends for {target_date}")
        
        # Delete existing data
        await self.session.execute(
            text("DELETE FROM pattern_trends_summary WHERE date = :date"),
            {"date": target_date}
        )
        
        # Aggregate pattern trends
        await self.session.execute(text("""
            INSERT INTO pattern_trends_summary (
                date, pattern_type, pattern_category, severity_level,
                organization_id, total_occurrences, avg_confidence_score,
                unique_users_affected
            )
            SELECT 
                :target_date as date,
                pd.pattern_type,
                pd.pattern_category,
                pd.severity_level,
                ae.organization_id,
                COUNT(*) as total_occurrences,
                AVG(pd.confidence_score) as avg_confidence_score,
                COUNT(DISTINCT ae.user_id) as unique_users_affected
            FROM pattern_detections pd
            JOIN analysis_events ae ON pd.analysis_event_id = ae.id
            WHERE DATE(ae.timestamp) = :target_date
            GROUP BY ae.organization_id, pd.pattern_type, pd.pattern_category, pd.severity_level
        """), {"target_date": target_date})
        
        await self.session.commit()
        print(f"Pattern trends aggregation completed for {target_date}")
    
    async def aggregate_organization_analytics(self, target_date: datetime = None):
        """Aggregate organization-level analytics."""
        if target_date is None:
            target_date = datetime.now().date() - timedelta(days=1)
        
        print(f"Aggregating organization analytics for {target_date}")
        
        # Delete existing data
        await self.session.execute(
            text("DELETE FROM organization_analytics_summary WHERE date = :date"),
            {"date": target_date}
        )
        
        # Aggregate organization analytics
        await self.session.execute(text("""
            INSERT INTO organization_analytics_summary (
                date, organization_id, total_analyses, total_users_active,
                total_patterns_found, avg_code_quality_score, top_pattern_type,
                total_api_requests, avg_response_time_ms
            )
            SELECT 
                :target_date as date,
                ae.organization_id,
                COUNT(DISTINCT ae.id) as total_analyses,
                COUNT(DISTINCT ae.user_id) as total_users_active,
                SUM(ae.total_patterns_count) as total_patterns_found,
                AVG(ae.code_quality_score) as avg_code_quality_score,
                (
                    SELECT pd.pattern_type 
                    FROM pattern_detections pd 
                    JOIN analysis_events ae2 ON pd.analysis_event_id = ae2.id
                    WHERE ae2.organization_id = ae.organization_id 
                    AND DATE(ae2.timestamp) = :target_date
                    GROUP BY pd.pattern_type 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 1
                ) as top_pattern_type,
                (
                    SELECT COUNT(*) 
                    FROM api_usage_events aue 
                    WHERE aue.organization_id = ae.organization_id 
                    AND DATE(aue.timestamp) = :target_date
                ) as total_api_requests,
                (
                    SELECT AVG(aue.response_time_ms) 
                    FROM api_usage_events aue 
                    WHERE aue.organization_id = ae.organization_id 
                    AND DATE(aue.timestamp) = :target_date
                ) as avg_response_time_ms
            FROM analysis_events ae
            WHERE DATE(ae.timestamp) = :target_date
            GROUP BY ae.organization_id
        """), {"target_date": target_date})
        
        await self.session.commit()
        print(f"Organization analytics aggregation completed for {target_date}")
    
    async def aggregate_performance_metrics(self, target_date: datetime = None):
        """Aggregate performance metrics by hour."""
        if target_date is None:
            target_date = datetime.now().date() - timedelta(days=1)
        
        print(f"Aggregating performance metrics for {target_date}")
        
        # Delete existing data
        await self.session.execute(
            text("DELETE FROM performance_metrics_summary WHERE date = :date"),
            {"date": target_date}
        )
        
        # Aggregate performance metrics
        await self.session.execute(text("""
            INSERT INTO performance_metrics_summary (
                date, hour_of_day, total_requests, avg_response_time_ms,
                max_response_time_ms, error_rate, p95_response_time_ms
            )
            SELECT 
                :target_date as date,
                EXTRACT(HOUR FROM aue.timestamp) as hour_of_day,
                COUNT(*) as total_requests,
                AVG(aue.response_time_ms) as avg_response_time_ms,
                MAX(aue.response_time_ms) as max_response_time_ms,
                (COUNT(*) FILTER (WHERE aue.status_code >= 400))::numeric / COUNT(*)::numeric * 100 as error_rate,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY aue.response_time_ms) as p95_response_time_ms
            FROM api_usage_events aue
            WHERE DATE(aue.timestamp) = :target_date
            GROUP BY EXTRACT(HOUR FROM aue.timestamp)
            ORDER BY hour_of_day
        """), {"target_date": target_date})
        
        await self.session.commit()
        print(f"Performance metrics aggregation completed for {target_date}")
    
    async def run_full_aggregation(self, target_date: datetime = None):
        """Run all aggregation tasks for a given date."""
        print(f"Starting full aggregation process for {target_date or 'yesterday'}")
        
        try:
            await self.create_summary_tables()
            await self.aggregate_daily_usage(target_date)
            await self.aggregate_pattern_trends(target_date)
            await self.aggregate_organization_analytics(target_date)
            await self.aggregate_performance_metrics(target_date)
            
            print("Full aggregation process completed successfully")
            
        except Exception as e:
            print(f"Error during aggregation: {e}")
            await self.session.rollback()
            raise
    
    async def cleanup_old_raw_data(self, days_to_keep: int = 90):
        """Clean up old raw data to manage database size."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        print(f"Cleaning up raw data older than {cutoff_date}")
        
        # Delete old pattern detections (cascades via analysis_events)
        result = await self.session.execute(
            text("DELETE FROM analysis_events WHERE timestamp < :cutoff_date"),
            {"cutoff_date": cutoff_date}
        )
        
        deleted_count = result.rowcount
        await self.session.commit()
        
        print(f"Deleted {deleted_count} old analysis events")

# Standalone functions for AWS Lambda or scheduled tasks
async def run_daily_aggregation():
    """Entry point for daily aggregation (AWS Lambda, cron job, etc.)."""
    async with AnalyticsAggregator() as aggregator:
        await aggregator.run_full_aggregation()

async def run_weekly_cleanup():
    """Entry point for weekly cleanup (AWS Lambda, cron job, etc.)."""
    async with AnalyticsAggregator() as aggregator:
        await aggregator.cleanup_old_raw_data()

if __name__ == "__main__":
    # For local testing
    asyncio.run(run_daily_aggregation())