# api/auth.py
import secrets
import hashlib
from typing import Optional
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_database_session, User
from database.services import DatabaseService

security = HTTPBearer()

class AuthenticationError(HTTPException):
    """Custom authentication error."""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class RateLimitError(HTTPException):
    """Custom rate limit error."""
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
        )

def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key and its hash.
    
    Returns:
        Tuple of (api_key: str, api_key_hash: str)
    """
    api_key = f"ak_{secrets.token_urlsafe(40)}"  # ak_ prefix for identification
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return api_key, api_key_hash

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_session: AsyncSession = Depends(get_database_session)
) -> User:
    """
    Dependency to get current authenticated user from API key.
    
    Args:
        credentials: Bearer token from Authorization header
        db_session: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        AuthenticationError: If authentication fails
    """
    if not credentials:
        raise AuthenticationError("Missing API key")
    
    api_key = credentials.credentials
    
    # Validate API key format
    if not api_key.startswith("ak_"):
        raise AuthenticationError("Invalid API key format")
    
    # Hash the provided API key
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Look up user by API key hash
    user = await DatabaseService.get_user_by_api_key_hash(db_session, api_key_hash)
    
    if not user:
        raise AuthenticationError("Invalid API key")
    
    if not user.is_active:
        raise AuthenticationError("Account deactivated")
    
    return user

async def check_rate_limits(
    user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_database_session)
) -> User:
    """
    Dependency to check rate limits for authenticated user.
    
    Args:
        user: Authenticated user
        db_session: Database session
        
    Returns:
        User: User object if within limits
        
    Raises:
        RateLimitError: If rate limit exceeded
    """
    is_allowed, remaining = await DatabaseService.check_rate_limit(db_session, user.id)
    
    if not is_allowed:
        raise RateLimitError(
            detail=f"Monthly analysis limit exceeded. Used: {user.current_month_analyses}, "
                   f"Limit: {user.organization.max_analyses_per_month}"
        )
    
    return user

# Helper functions for user management
async def create_new_user(
    db_session: AsyncSession,
    organization_id: str,
    email: str,
    name: str
) -> tuple[User, str]:
    """
    Create a new user with API key.
    
    Returns:
        Tuple of (User object, plain_text_api_key)
    """
    api_key, api_key_hash = generate_api_key()
    
    user = User(
        organization_id=organization_id,
        email=email,
        name=name,
        api_key_hash=api_key_hash
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user, api_key