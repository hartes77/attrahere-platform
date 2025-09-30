# database/config.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://attrahere_admin:password@localhost:5432/attrahere"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging in development
    future=True,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,  # Validate connections before use
    pool_recycle=3600,   # Recycle connections every hour
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for SQLAlchemy models
Base = declarative_base()

# Dependency to get database session
async def get_database_session():
    """
    Dependency function to provide database session to FastAPI endpoints.
    
    Yields:
        AsyncSession: Database session that automatically closes after use.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Initialize database connection
async def init_database():
    """
    Initialize database connection and create tables if they don't exist.
    This function should be called during application startup.
    """
    async with engine.begin() as conn:
        # Import models to ensure they're registered with Base
        from .models import Organization, User, AnalysisEvent, PatternDetection, ApiUsageEvent
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

# Close database connection
async def close_database():
    """
    Close database connection pool.
    This function should be called during application shutdown.
    """
    await engine.dispose()