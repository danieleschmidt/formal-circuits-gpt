"""Metrics collection and monitoring."""

import time
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass 
class VerificationMetrics:
    """Verification operation metrics."""
    total_verifications: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    average_duration: float = 0.0
    total_duration: float = 0.0
    by_prover: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_status: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to keep metrics data
        """
        self.retention_hours = retention_hours
        self._metrics: List[MetricPoint] = []
        self._lock = threading.RLock()
        
        # Verification-specific metrics
        self.verification_metrics = VerificationMetrics()
        self._verification_durations: List[float] = []
        
        # System metrics
        self._start_time = datetime.now()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        with self._lock:
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self._metrics.append(metric)
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric.
        
        Args:
            name: Counter name
            tags: Optional tags
        """
        self.record_metric(name, 1.0, tags)
    
    def record_verification_start(self, prover: str, circuit_name: str = "unknown"):
        """Record start of verification operation.
        
        Args:
            prover: Theorem prover used
            circuit_name: Name of circuit being verified
        """
        self.increment_counter("verification_started", {
            "prover": prover,
            "circuit": circuit_name
        })
    
    def record_verification_complete(self, prover: str, status: str, 
                                   duration: float, circuit_name: str = "unknown",
                                   properties_count: int = 0):
        """Record completion of verification operation.
        
        Args:
            prover: Theorem prover used
            status: Verification status (VERIFIED, FAILED, ERROR)
            duration: Duration in seconds
            circuit_name: Name of circuit
            properties_count: Number of properties verified
        """
        with self._lock:
            # Update counters
            self.verification_metrics.total_verifications += 1
            self.verification_metrics.by_prover[prover] += 1
            self.verification_metrics.by_status[status] += 1
            
            if status == "VERIFIED":
                self.verification_metrics.successful_verifications += 1
            else:
                self.verification_metrics.failed_verifications += 1
            
            # Update duration metrics
            self._verification_durations.append(duration)
            self.verification_metrics.total_duration += duration
            self.verification_metrics.average_duration = (
                self.verification_metrics.total_duration / 
                self.verification_metrics.total_verifications
            )
        
        # Record individual metrics
        self.record_metric("verification_duration", duration, {
            "prover": prover,
            "status": status,
            "circuit": circuit_name
        })
        
        self.record_metric("properties_verified", properties_count, {
            "prover": prover,
            "circuit": circuit_name
        })
        
        self.increment_counter("verification_completed", {
            "prover": prover,
            "status": status,
            "circuit": circuit_name
        })
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit.
        
        Args:
            cache_type: Type of cache (memory, database, file)
        """
        self.increment_counter("cache_hit", {"cache_type": cache_type})
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss.
        
        Args:
            cache_type: Type of cache (memory, database, file)
        """
        self.increment_counter("cache_miss", {"cache_type": cache_type})
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            duration: Request duration in seconds
        """
        self.record_metric("api_request_duration", duration, {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code)
        })
        
        self.increment_counter("api_requests", {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code)
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            current_time = datetime.now()
            uptime = current_time - self._start_time
            
            # Calculate recent metrics (last hour)
            recent_cutoff = current_time - timedelta(hours=1)
            recent_metrics = [m for m in self._metrics if m.timestamp > recent_cutoff]
            
            # Count metrics by name
            metric_counts = Counter(m.name for m in recent_metrics)
            
            # Calculate success rate
            success_rate = 0.0
            if self.verification_metrics.total_verifications > 0:
                success_rate = (
                    self.verification_metrics.successful_verifications / 
                    self.verification_metrics.total_verifications * 100
                )
            
            return {
                "system": {
                    "uptime_seconds": uptime.total_seconds(),
                    "uptime_human": str(uptime),
                    "total_metrics": len(self._metrics),
                    "recent_metrics": len(recent_metrics)
                },
                "verification": {
                    "total_verifications": self.verification_metrics.total_verifications,
                    "successful_verifications": self.verification_metrics.successful_verifications,
                    "failed_verifications": self.verification_metrics.failed_verifications,
                    "success_rate_percent": success_rate,
                    "average_duration_seconds": self.verification_metrics.average_duration,
                    "by_prover": dict(self.verification_metrics.by_prover),
                    "by_status": dict(self.verification_metrics.by_status)
                },
                "recent_activity": dict(metric_counts),
                "performance": {
                    "min_duration": min(self._verification_durations) if self._verification_durations else 0,
                    "max_duration": max(self._verification_durations) if self._verification_durations else 0,
                    "p95_duration": self._calculate_percentile(self._verification_durations, 95),
                    "p99_duration": self._calculate_percentile(self._verification_durations, 99)
                }
            }
    
    def get_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        metrics_summary = self.get_metrics_summary()
        lines = []
        
        # System metrics
        lines.append(f"# HELP formal_circuits_uptime_seconds System uptime in seconds")
        lines.append(f"# TYPE formal_circuits_uptime_seconds gauge")
        lines.append(f"formal_circuits_uptime_seconds {metrics_summary['system']['uptime_seconds']}")
        
        # Verification metrics
        lines.append(f"# HELP formal_circuits_verifications_total Total number of verifications")
        lines.append(f"# TYPE formal_circuits_verifications_total counter")
        lines.append(f"formal_circuits_verifications_total {metrics_summary['verification']['total_verifications']}")
        
        lines.append(f"# HELP formal_circuits_verification_success_rate Success rate percentage")
        lines.append(f"# TYPE formal_circuits_verification_success_rate gauge")
        lines.append(f"formal_circuits_verification_success_rate {metrics_summary['verification']['success_rate_percent']}")
        
        lines.append(f"# HELP formal_circuits_verification_duration_seconds Average verification duration")
        lines.append(f"# TYPE formal_circuits_verification_duration_seconds gauge")
        lines.append(f"formal_circuits_verification_duration_seconds {metrics_summary['verification']['average_duration_seconds']}")
        
        # By prover metrics
        for prover, count in metrics_summary['verification']['by_prover'].items():
            lines.append(f"formal_circuits_verifications_by_prover{{prover=\"{prover}\"}} {count}")
        
        return "\n".join(lines) + "\n"
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values.
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _cleanup_worker(self):
        """Background worker to clean up old metrics."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._cleanup_old_metrics()
            except Exception:
                pass  # Ignore cleanup errors
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            self._metrics = [m for m in self._metrics if m.timestamp > cutoff_time]
            
            # Also trim verification durations to prevent memory growth
            if len(self._verification_durations) > 10000:
                self._verification_durations = self._verification_durations[-5000:]


# Global metrics collector instance
_global_metrics = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics