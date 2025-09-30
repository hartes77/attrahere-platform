# database/services.py
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .models import Organization, User, AnalysisEvent, PatternDetection, ApiUsageEvent

class DatabaseService:
    """Service class for database operations."""
    
    @staticmethod
    def hash_string(data: str) -> str:
        """Create SHA-256 hash of a string for privacy compliance."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    async def get_user_by_api_key_hash(session: AsyncSession, api_key_hash: str) -> Optional[User]:
        """Get user by API key hash."""
        result = await session.execute(
            select(User)
            .options(selectinload(User.organization))
            .where(User.api_key_hash == api_key_hash)
            .where(User.is_active == True)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_or_create_default_organization(session: AsyncSession) -> Organization:
        """Get or create default organization for single-tenant setup."""
        result = await session.execute(
            select(Organization).where(Organization.slug == 'default')
        )
        org = result.scalar_one_or_none()
        
        if not org:
            org = Organization(
                name='Default Organization',
                slug='default',
                plan_type='enterprise',
                max_analyses_per_month=999999
            )
            session.add(org)
            await session.commit()
            await session.refresh(org)
        
        return org
    
    @staticmethod
    async def save_analysis_event(
        session: AsyncSession,
        user_id: UUID,
        organization_id: UUID,
        code: str,
        code_language: str,
        analysis_result: Dict,
        processing_time_ms: int,
        analyzer_version: str,
        request_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AnalysisEvent:
        """Save analysis event to database."""
        
        # Hash sensitive data for privacy
        code_hash = DatabaseService.hash_string(code)
        request_ip_hash = DatabaseService.hash_string(request_ip) if request_ip else None
        user_agent_hash = DatabaseService.hash_string(user_agent) if user_agent else None
        
        # Extract patterns from analysis result
        patterns = analysis_result.get('patterns', [])
        total_patterns_count = len(patterns)
        
        # Calculate confidence and quality scores
        analysis_confidence = analysis_result.get('confidence', 0.95)
        code_quality_score = max(0.0, 1.0 - (total_patterns_count * 0.1))
        complexity_score = min(1.0, len(code) / 1000.0)  # Simple complexity metric
        
        # Create analysis event
        analysis_event = AnalysisEvent(
            user_id=user_id,
            organization_id=organization_id,
            request_ip_hash=request_ip_hash,
            user_agent_hash=user_agent_hash,
            code_length=len(code),
            code_language=code_language,
            code_hash=code_hash,
            patterns_detected=patterns,
            total_patterns_count=total_patterns_count,
            analysis_confidence=analysis_confidence,
            processing_time_ms=processing_time_ms,
            analyzer_version=analyzer_version,
            code_quality_score=code_quality_score,
            complexity_score=complexity_score
        )
        
        session.add(analysis_event)
        await session.flush()  # Get the ID
        
        # Create detailed pattern detections
        for pattern in patterns:
            pattern_detection = PatternDetection(
                analysis_event_id=analysis_event.id,
                pattern_type=pattern.get('type', 'unknown'),
                pattern_category='anti_pattern',  # Default category
                severity_level=pattern.get('severity', 'medium'),
                line_number=pattern.get('line', 0),
                column_number=pattern.get('column', 0),
                code_snippet=pattern.get('code_snippet', ''),
                confidence_score=pattern.get('confidence', 0.85),
                suggested_fix=pattern.get('suggested_fix', ''),
                detector_name=pattern.get('detector', 'ml_analyzer'),
                detection_time_ms=int(processing_time_ms / max(1, total_patterns_count))
            )
            session.add(pattern_detection)
        
        await session.commit()
        await session.refresh(analysis_event)
        
        # Update user statistics
        await DatabaseService.update_user_statistics(session, user_id)
        
        return analysis_event
    
    @staticmethod
    async def save_api_usage_event(
        session: AsyncSession,
        user_id: UUID,
        organization_id: UUID,
        endpoint: str,
        http_method: str,
        status_code: int,
        response_time_ms: int,
        request_size_bytes: Optional[int] = None,
        response_size_bytes: Optional[int] = None,
        rate_limit_remaining: Optional[int] = None
    ) -> ApiUsageEvent:
        """Save API usage event for monitoring and billing."""
        
        usage_event = ApiUsageEvent(
            user_id=user_id,
            organization_id=organization_id,
            endpoint=endpoint,
            http_method=http_method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            rate_limit_remaining=rate_limit_remaining
        )
        
        session.add(usage_event)
        await session.commit()
        
        return usage_event
    
    @staticmethod
    async def update_user_statistics(session: AsyncSession, user_id: UUID):
        """Update user analysis statistics."""
        user = await session.get(User, user_id)
        if user:
            # Update total analyses
            user.total_analyses += 1
            user.current_month_analyses += 1  # This should be reset monthly
            user.last_active_at = datetime.utcnow()
            
            await session.commit()
    
    @staticmethod
    async def get_user_monthly_usage(session: AsyncSession, user_id: UUID) -> int:
        """Get user's current month analysis count."""
        user = await session.get(User, user_id)
        return user.current_month_analyses if user else 0
    
    @staticmethod
    async def check_rate_limit(session: AsyncSession, user_id: UUID) -> tuple[bool, int]:
        """
        Check if user has exceeded rate limits.
        
        Returns:
            Tuple of (is_allowed: bool, remaining_requests: int)
        """
        user = await session.get(User, user_id)
        if not user:
            return False, 0
        
        # Load organization to check limits
        await session.refresh(user, ['organization'])
        
        max_monthly = user.organization.max_analyses_per_month
        current_usage = user.current_month_analyses
        
        remaining = max(0, max_monthly - current_usage)
        is_allowed = remaining > 0
        
        return is_allowed, remaining