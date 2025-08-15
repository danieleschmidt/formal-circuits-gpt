"""Retry policies for resilient operations."""

import time
import random
import asyncio
from typing import Callable, Any, Optional, List, Type
from abc import ABC, abstractmethod


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""

    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if we should retry based on attempt number and exception."""
        pass

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        pass


class ExponentialBackoff(RetryPolicy):
    """Exponential backoff with optional jitter."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """Initialize exponential backoff policy.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            multiplier: Backoff multiplier
            jitter: Whether to add random jitter
            retryable_exceptions: List of exceptions that should trigger retry
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Check if we should retry."""
        if attempt >= self.max_attempts:
            return False

        # Check if exception is retryable
        return any(
            isinstance(exception, exc_type) for exc_type in self.retryable_exceptions
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        delay = self.base_delay * (self.multiplier**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


class LinearBackoff(RetryPolicy):
    """Linear backoff retry policy."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        increment: float = 1.0,
        max_delay: float = 60.0,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.increment = increment
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions or [Exception]

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Check if we should retry."""
        if attempt >= self.max_attempts:
            return False
        return any(
            isinstance(exception, exc_type) for exc_type in self.retryable_exceptions
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with linear backoff."""
        delay = self.base_delay + (self.increment * attempt)
        return min(delay, self.max_delay)


class FixedDelay(RetryPolicy):
    """Fixed delay retry policy."""

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.retryable_exceptions = retryable_exceptions or [Exception]

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Check if we should retry."""
        if attempt >= self.max_attempts:
            return False
        return any(
            isinstance(exception, exc_type) for exc_type in self.retryable_exceptions
        )

    def get_delay(self, attempt: int) -> float:
        """Return fixed delay."""
        return self.delay


def retry_with_policy(policy: RetryPolicy):
    """Decorator for retrying functions with a given policy."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(attempt, e):
                        raise e

                    delay = policy.get_delay(attempt)
                    time.sleep(delay)
                    attempt += 1

        return wrapper

    return decorator


def async_retry_with_policy(policy: RetryPolicy):
    """Async decorator for retrying functions with a given policy."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None

            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(attempt, e):
                        raise e

                    delay = policy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    attempt += 1

        return wrapper

    return decorator


class RetryableOperation:
    """Class for executing operations with retry logic."""

    def __init__(self, policy: RetryPolicy):
        self.policy = policy

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        attempt = 0

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not self.policy.should_retry(attempt, e):
                    raise e

                delay = self.policy.get_delay(attempt)
                time.sleep(delay)
                attempt += 1

    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        attempt = 0

        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if not self.policy.should_retry(attempt, e):
                    raise e

                delay = self.policy.get_delay(attempt)
                await asyncio.sleep(delay)
                attempt += 1
