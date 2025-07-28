from fastapi import Depends, HTTPException, status
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.core.predictive import PredictiveAnalytics
from app.utils.analyzer import ComplianceAnalyzer
from app.database.db.db_connection import SessionLocal
from app.config import settings
import os
from typing import AsyncGenerator, Generator, Type, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
import logging
from functools import lru_cache
import time
from prometheus_client import Counter, Histogram
from contextlib import asynccontextmanager
from app.utils.startup import analyzer

logger = logging.getLogger(__name__)

# Metrics
DEPENDENCY_ERRORS = Counter(
    'dependency_injection_errors_total',
    'Total number of dependency injection errors',
    ['dependency']
)

DEPENDENCY_LATENCY = Histogram(
    'dependency_injection_latency_seconds',
    'Dependency injection latency in seconds',
    ['dependency']
)

# Async Database dependency with monitoring
@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async database session dependency with monitoring and health checks.
    """
    start_time = time.time()
    try:
        # if not await check_db_health():
        #     logger.error("Database health check failed")
        #     raise HTTPException(
        #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        #         detail="Database is not available"
        #     )
            
        async with SessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
                
        DEPENDENCY_LATENCY.labels('database').observe(time.time() - start_time)
    except Exception as e:
        DEPENDENCY_ERRORS.labels('database').inc()
        logger.error(f"Database session error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error occurred"
        )

# Sync Database dependency with monitoring
@asynccontextmanager
async def get_sync_db():
    """
    Traditional sync database session with monitoring.
    """
    start_time = time.time()
    try:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
            DEPENDENCY_LATENCY.labels('database_sync').observe(time.time() - start_time)
    except Exception as e:
        DEPENDENCY_ERRORS.labels('database_sync').inc()
        logger.error(f"Sync database session error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error occurred"
        )

# Default to async DB
get_db = get_async_db

# Groq client dependency with caching
@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Cached Groq API client dependency"""
    start_time = time.time()
    try:
        api_key = os.getenv("GROQ_API_KEY", settings.GROQ_API_KEY)
        if not api_key:
            raise ValueError("Groq API key not found")
        client = Groq(api_key=api_key)
        DEPENDENCY_LATENCY.labels('groq').observe(time.time() - start_time)
        return client
    except Exception as e:
        DEPENDENCY_ERRORS.labels('groq').inc()
        logger.error(f"Groq client initialization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service unavailable"
        )

# Sentence Transformer model dependency with caching
@lru_cache(maxsize=1)
def get_sentence_model() -> SentenceTransformer:
    """Cached sentence embedding model dependency"""
    start_time = time.time()
    try:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        DEPENDENCY_LATENCY.labels('sentence_model').observe(time.time() - start_time)
        return model
    except Exception as e:
        DEPENDENCY_ERRORS.labels('sentence_model').inc()
        logger.error(f"Sentence model initialization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Text processing service unavailable"
        )

# Predictive Analytics dependency with caching
@lru_cache(maxsize=1)
def get_predictive_analytics(
    model: SentenceTransformer = Depends(get_sentence_model)
) -> PredictiveAnalytics:
    """Cached predictive analytics service dependency"""
    start_time = time.time()
    try:
        analytics = PredictiveAnalytics(model)
        DEPENDENCY_LATENCY.labels('predictive').observe(time.time() - start_time)
        return analytics
    except Exception as e:
        DEPENDENCY_ERRORS.labels('predictive').inc()
        logger.error(f"Predictive analytics initialization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics service unavailable"
        )

# Analyzer dependency with state validation
_analyzer_instance: Optional[ComplianceAnalyzer] = None

def get_analyzer(
    groq_client: Groq = Depends(get_groq_client),
    model: SentenceTransformer = Depends(get_sentence_model),
    predictive_analytics: PredictiveAnalytics = Depends(get_predictive_analytics)
) -> ComplianceAnalyzer:
    """Singleton compliance analyzer dependency with state validation"""
    global _analyzer_instance
    start_time = time.time()
    
    try:
        if _analyzer_instance is None:
            _analyzer_instance = ComplianceAnalyzer(groq_client, model, predictive_analytics)
            
        # Verify analyzer state
        if not hasattr(_analyzer_instance, 'knowledge_base') or \
           not _analyzer_instance.knowledge_base:
            logger.error("Invalid analyzer state detected")
            _analyzer_instance = None
            raise ValueError("Analyzer state invalid")
            
        DEPENDENCY_LATENCY.labels('analyzer').observe(time.time() - start_time)
        return _analyzer_instance
    except Exception as e:
        DEPENDENCY_ERRORS.labels('analyzer').inc()
        logger.error(f"Analyzer initialization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Compliance analyzer service unavailable"
        )
def get_analyzer_2():
    return analyzer

# Settings dependency with validation
@lru_cache(maxsize=1)
def get_settings() -> Dict[str, Any]:
    """Cached and validated settings dependency"""
    start_time = time.time()
    try:
        # Validate critical settings
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")
            
        # Return validated settings
        DEPENDENCY_LATENCY.labels('settings').observe(time.time() - start_time)
        return settings.dict()
    except Exception as e:
        DEPENDENCY_ERRORS.labels('settings').inc()
        logger.error(f"Settings validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration error"
        )