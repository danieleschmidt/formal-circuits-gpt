"""Rate limiting utilities for API calls and resource management."""

import time
import threading
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens. Returns True if successful."""
        pass

    @abstractmethod
    def wait_for_token(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for tokens to become available. Returns True if successful."""
        pass


class TokenBucket(RateLimiter):
    """Token bucket rate limiter implementation."""

    def __init__(
        self, capacity: int, refill_rate: float, initial_tokens: Optional[int] = None
    ):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = threading.RLock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens immediately."""
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_for_token(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for tokens to become available."""
        start_time = time.time()

        while True:
            if self.acquire(tokens):
                return True

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False

            # Calculate wait time until next token
            with self._lock:
                if self.refill_rate > 0:
                    wait_time = min(0.1, tokens / self.refill_rate)
                else:
                    wait_time = 0.1

            time.sleep(wait_time)

    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill()
            return self.tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "available_tokens": self.get_available_tokens(),
            "utilization": 1.0 - (self.get_available_tokens() / self.capacity),
        }


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter."""

    def __init__(self, max_requests: int, window_size: float):
        """Initialize sliding window rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_size: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = []
        self._lock = threading.RLock()

    def _clean_old_requests(self):
        """Remove requests older than window size."""
        now = time.time()
        cutoff = now - self.window_size
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire permission immediately."""
        with self._lock:
            self._clean_old_requests()

            if len(self.requests) + tokens <= self.max_requests:
                now = time.time()
                for _ in range(tokens):
                    self.requests.append(now)
                return True
            return False

    def wait_for_token(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for permission to make request."""
        start_time = time.time()

        while True:
            if self.acquire(tokens):
                return True

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False

            # Wait a bit before trying again
            time.sleep(0.1)

    def get_current_requests(self) -> int:
        """Get current number of requests in window."""
        with self._lock:
            self._clean_old_requests()
            return len(self.requests)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_requests = self.get_current_requests()
        return {
            "max_requests": self.max_requests,
            "window_size": self.window_size,
            "current_requests": current_requests,
            "utilization": current_requests / self.max_requests,
        }


class RateLimiterManager:
    """Manages multiple rate limiters."""

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.RLock()

    def create_token_bucket(
        self,
        name: str,
        capacity: int,
        refill_rate: float,
        initial_tokens: Optional[int] = None,
    ) -> TokenBucket:
        """Create or get a token bucket rate limiter."""
        with self._lock:
            if name not in self._limiters:
                self._limiters[name] = TokenBucket(
                    capacity, refill_rate, initial_tokens
                )
            return self._limiters[name]

    def create_sliding_window(
        self, name: str, max_requests: int, window_size: float
    ) -> SlidingWindowRateLimiter:
        """Create or get a sliding window rate limiter."""
        with self._lock:
            if name not in self._limiters:
                self._limiters[name] = SlidingWindowRateLimiter(
                    max_requests, window_size
                )
            return self._limiters[name]

    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """Get rate limiter by name."""
        with self._lock:
            return self._limiters.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all rate limiters."""
        with self._lock:
            stats = {}
            for name, limiter in self._limiters.items():
                if hasattr(limiter, "get_stats"):
                    stats[name] = limiter.get_stats()
            return stats


# Global rate limiter manager
rate_limiter_manager = RateLimiterManager()
