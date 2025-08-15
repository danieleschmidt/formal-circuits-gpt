"""Advanced metrics collection for system monitoring."""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json


@dataclass
class MetricPoint:
    """A single metric data point."""

    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status."""

    healthy: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    last_check: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of metric points to keep in history
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()

        # Health monitoring
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        self._last_health_status = HealthStatus(True, 1.0)

        # Performance tracking
        self._start_time = time.time()
        self._active_operations = 0

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Dict[str, str] = None
    ):
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            self._add_metric_point(name, self._counters[name], labels)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self._lock:
            self._gauges[name] = value
            self._add_metric_point(name, value, labels)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a value in a histogram."""
        with self._lock:
            self._histograms[name].append(value)
            # Keep only last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
            self._add_metric_point(name, value, labels)

    def time_operation(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimedOperation(self, name, labels)

    def _add_metric_point(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add a metric point to history."""
        point = MetricPoint(timestamp=time.time(), value=value, labels=labels or {})
        self._metrics[name].append(point)

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        with self._lock:
            return self._counters[name]

    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._gauges[name]

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = self._histograms[name]
            if not values:
                return {
                    "count": 0,
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0,
                }

            sorted_values = sorted(values)
            count = len(sorted_values)

            return {
                "count": count,
                "min": min(sorted_values),
                "max": max(sorted_values),
                "mean": sum(sorted_values) / count,
                "p50": self._percentile(sorted_values, 0.5),
                "p95": self._percentile(sorted_values, 0.95),
                "p99": self._percentile(sorted_values, 0.99),
            }

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        index = int(p * (len(sorted_values) - 1))
        return sorted_values[index]

    def get_metric_history(
        self, name: str, since: Optional[float] = None
    ) -> List[MetricPoint]:
        """Get metric history, optionally filtered by time."""
        with self._lock:
            points = list(self._metrics[name])
            if since is not None:
                points = [p for p in points if p.timestamp >= since]
            return points

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self._health_checks[name] = check_func

    def check_health(self) -> HealthStatus:
        """Run all health checks and return overall status."""
        issues = []
        total_checks = len(self._health_checks)
        passed_checks = 0

        for name, check_func in self._health_checks.items():
            try:
                if check_func():
                    passed_checks += 1
                else:
                    issues.append(f"Health check '{name}' failed")
            except Exception as e:
                issues.append(f"Health check '{name}' threw exception: {str(e)}")

        # Calculate health score
        if total_checks == 0:
            score = 1.0
        else:
            score = passed_checks / total_checks

        healthy = score >= 0.8 and len(issues) == 0

        self._last_health_status = HealthStatus(
            healthy=healthy, score=score, issues=issues, last_check=time.time()
        )

        return self._last_health_status

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        uptime = time.time() - self._start_time

        return {
            "uptime_seconds": uptime,
            "active_operations": self._active_operations,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name) for name in self._histograms
            },
            "health": {
                "healthy": self._last_health_status.healthy,
                "score": self._last_health_status.score,
                "issues": self._last_health_status.issues,
                "last_check": self._last_health_status.last_check,
            },
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Export counters
        for name, value in self._counters.items():
            lines.append(f"# TYPE {name}_total counter")
            lines.append(f"{name}_total {value}")

        # Export gauges
        for name, value in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Export histograms
        for name, stats in {
            name: self.get_histogram_stats(name) for name in self._histograms
        }.items():
            lines.append(f"# TYPE {name} histogram")
            for stat_name, stat_value in stats.items():
                lines.append(f"{name}_{stat_name} {stat_value}")

        return "\\n".join(lines)


class TimedOperation:
    """Context manager for timing operations."""

    def __init__(
        self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        with self.collector._lock:
            self.collector._active_operations += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_histogram(
                f"{self.name}_duration_ms", duration * 1000, self.labels
            )

        with self.collector._lock:
            self.collector._active_operations -= 1

        # Record success/failure
        if exc_type is None:
            self.collector.increment_counter(
                f"{self.name}_success_total", labels=self.labels
            )
        else:
            self.collector.increment_counter(
                f"{self.name}_error_total", labels=self.labels
            )


class SystemMetricsCollector:
    """Collects system-level metrics."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._collection_interval = 30.0  # seconds
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start collecting system metrics."""
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop collecting system metrics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _collect_loop(self):
        """Main collection loop."""
        while not self._stop_event.wait(self._collection_interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                # Log error but continue collecting
                pass

    def _collect_system_metrics(self):
        """Collect system metrics."""
        import psutil
        import os

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.collector.set_gauge("system_cpu_usage_percent", cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.collector.set_gauge("system_memory_usage_bytes", memory.used)
        self.collector.set_gauge("system_memory_usage_percent", memory.percent)

        # Disk usage
        disk = psutil.disk_usage("/")
        self.collector.set_gauge("system_disk_usage_bytes", disk.used)
        self.collector.set_gauge(
            "system_disk_usage_percent", (disk.used / disk.total) * 100
        )

        # Process-specific metrics
        process = psutil.Process(os.getpid())
        self.collector.set_gauge("process_memory_rss_bytes", process.memory_info().rss)
        self.collector.set_gauge("process_cpu_percent", process.cpu_percent())


# Global metrics collector instance
global_metrics = MetricsCollector()
system_metrics = SystemMetricsCollector(global_metrics)
