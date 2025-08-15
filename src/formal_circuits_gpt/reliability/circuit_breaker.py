"""Circuit breaker pattern implementation for fault tolerance."""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from datetime import datetime, timedelta


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Circuit breaker exception."""

    def __init__(self, message: str, state: CircuitBreakerState):
        super().__init__(message)
        self.state = state


class CircuitBreaker:
    """Circuit breaker for fault tolerance and fail-fast behavior."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "CircuitBreaker",
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            timeout: Time in seconds before trying again (half-open)
            expected_exception: Exception type that counts as failure
            name: Name for logging and identification
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitBreakerState.CLOSED
        self._lock = threading.RLock()

        # Statistics
        self._call_count = 0
        self._success_count = 0
        self._failure_count_total = 0

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def call_count(self) -> int:
        """Get total call count."""
        return self._call_count

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0 to 1.0)."""
        if self._call_count == 0:
            return 1.0
        return self._success_count / self._call_count

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset (move to half-open)."""
        return (
            self._state == CircuitBreakerState.OPEN
            and self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self.timeout
        )

    def _record_success(self):
        """Record successful operation."""
        with self._lock:
            self._failure_count = 0
            self._success_count += 1
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED

    def _record_failure(self):
        """Record failed operation."""
        with self._lock:
            self._failure_count += 1
            self._failure_count_total += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN

    def __call__(self, func: Callable) -> Callable:
        """Decorator version of circuit breaker."""

        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self._call_count += 1

            # Check if circuit is open
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN - "
                        f"too many failures ({self._failure_count}). "
                        f"Will retry after {self.timeout}s.",
                        CircuitBreakerState.OPEN,
                    )

        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise e

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "total_calls": self._call_count,
            "total_successes": self._success_count,
            "total_failures": self._failure_count_total,
            "success_rate": self.success_rate,
            "last_failure_time": self._last_failure_time,
        }


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""

    def __init__(self):
        self._breakers = {}
        self._lock = threading.RLock()

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    timeout=timeout,
                    expected_exception=expected_exception,
                    name=name,
                )
            return self._breakers[name]

    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_stats() for name, breaker in self._breakers.items()
            }


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()
