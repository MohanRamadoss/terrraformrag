from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
from functools import lru_cache
import logging
from .models import QueryRequest, TerraformCode, AppConfig

async def create_aiohttp_session():
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=aiohttp.TCPConnector(limit=20)
    )

@lru_cache(maxsize=100)
def get_cached_code(query_hash: str) -> Optional[TerraformCode]:
    # Implement caching logic
    pass

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection."""
    # Remove potentially dangerous characters
    forbidden_chars = [';', '&&', '||', '`', '$', '>', '<']
    for char in forbidden_chars:
        text = text.replace(char, '')
    return text.strip()

async def safe_chromadb_operation(operation: callable, *args, max_retries: int = 3):
    """Execute ChromaDB operations with retry logic."""
    for attempt in range(max_retries):
        try:
            return await operation(*args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

class APIRateLimiter:
    """Rate limiter for API calls."""
    def __init__(self, rate_limit: int, time_window: int):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = datetime.now()
            self.calls = [t for t in self.calls if (now - t).seconds < self.time_window]
            if len(self.calls) >= self.rate_limit:
                return False
            self.calls.append(now)
            return True
