from fastapi import Depends, HTTPException, status, Request
from fastapi_limiter.depends import RateLimiter as FastAPIRateLimiter
from typing import Callable, Optional
import functools
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import logging

logger = logging.getLogger(__name__)

# In-memory rate limiting store
class InMemoryLimiter:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = InMemoryLimiter()
        return cls._instance
    
    def __init__(self):
        self.tokens = defaultdict(list)
        self.lock = threading.Lock()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_old_tokens())
    
    async def _cleanup_old_tokens(self):
        while True:
            await asyncio.sleep(60)  # Run every minute
            now = datetime.now()
            with self.lock:
                for key in list(self.tokens.keys()):
                    # Remove tokens older than 1 hour
                    self.tokens[key] = [t for t in self.tokens[key] if (now - t) < timedelta(hours=1)]
                    # Remove empty lists
                    if not self.tokens[key]:
                        del self.tokens[key]
    
    async def request(self, key: str, times: int, minutes: float):
        with self.lock:
            now = datetime.now()
            window = timedelta(minutes=minutes)
            
            # Remove tokens older than the window
            self.tokens[key] = [t for t in self.tokens[key] if (now - t) < window]
            
            # Check if limit is exceeded
            if len(self.tokens[key]) >= times:
                return False
            
            # Add token
            self.tokens[key].append(now)
            return True
            
    async def close(self):
        if hasattr(self, 'cleanup_task') and self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass


# Unified rate limiter that works with or without Redis
def RateLimiter(times: int = 10, minutes: float = 1.0):
    """
    Unified rate limiter that automatically selects between Redis and in-memory
    based on whether FastAPILimiter has been initialized.
    
    Args:
        times: Maximum number of requests allowed within the time window
        minutes: Time window in minutes
    """
    # Try to import FastAPILimiter for Redis-based limiting
    from fastapi_limiter import FastAPILimiter
    
    async def limiter(request: Request):
        # Check if Redis is available through FastAPILimiter
        if hasattr(FastAPILimiter, '_redis') and FastAPILimiter._redis is not None:
            # Use Redis-based limiter
            # We need to call the function directly, not pass it through Depends
            redis_limiter = FastAPIRateLimiter(times=times, minutes=minutes)
            await redis_limiter(request)
        else:
            # Use in-memory limiter
            client_ip = request.client.host
            key = f"{client_ip}:{request.url.path}"
            
            in_memory = InMemoryLimiter.get_instance()
            success = await in_memory.request(key, times, minutes)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests"
                )
    
    return limiter 