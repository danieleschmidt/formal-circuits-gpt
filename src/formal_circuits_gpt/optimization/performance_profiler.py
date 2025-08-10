"""Advanced performance profiling and optimization system."""

import time
import psutil
import threading
import functools
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    tokens_used: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    parallel_workers: Optional[int] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass 
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    active_threads: int


class PerformanceProfiler:
    """Advanced performance profiler with optimization recommendations."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize profiler.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.operation_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Profiling state
        self.enabled = True
        self.sampling_interval = 1.0  # seconds
        self.system_monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        # Performance thresholds for optimization
        self.thresholds = {
            "slow_operation_ms": 5000,
            "high_memory_mb": 1024,
            "high_cpu_percent": 80,
            "cache_miss_rate": 0.3
        }
        
        # Start system monitoring
        self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start background system metrics collection."""
        if self.system_monitor_thread is None or not self.system_monitor_thread.is_alive():
            self.stop_monitoring.clear()
            self.system_monitor_thread = threading.Thread(
                target=self._system_monitor_loop,
                daemon=True
            )
            self.system_monitor_thread.start()
    
    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self.stop_monitoring.set()
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=2.0)
    
    def _system_monitor_loop(self):
        """Background system monitoring loop."""
        last_disk_io = psutil.disk_io_counters()
        last_net_io = psutil.net_io_counters()
        
        while not self.stop_monitoring.wait(self.sampling_interval):
            try:
                # Get current system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                
                # Calculate rates
                disk_read_mb = (disk_io.read_bytes - last_disk_io.read_bytes) / 1024 / 1024
                disk_write_mb = (disk_io.write_bytes - last_disk_io.write_bytes) / 1024 / 1024
                net_sent_mb = (net_io.bytes_sent - last_net_io.bytes_sent) / 1024 / 1024  
                net_recv_mb = (net_io.bytes_recv - last_net_io.bytes_recv) / 1024 / 1024
                
                system_metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_mb=memory.available / 1024 / 1024,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_sent_mb=net_sent_mb,
                    network_recv_mb=net_recv_mb,
                    open_files=len(psutil.Process().open_files()),
                    active_threads=threading.active_count()
                )
                
                self.system_metrics_history.append(system_metrics)
                
                # Update for next iteration
                last_disk_io = disk_io
                last_net_io = net_io
                
            except Exception as e:
                # Continue monitoring even if some metrics fail
                print(f"System monitoring error: {e}")
    
    @contextmanager
    def profile_operation(self, operation: str, **extra_context):
        """Context manager for profiling operations.
        
        Args:
            operation: Name of the operation being profiled
            **extra_context: Additional context to include in metrics
        """
        if not self.enabled:
            yield
            return
        
        # Get initial system state
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()
            
            metrics = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=end_memory - start_memory,
                cpu_percent=cpu_percent,
                timestamp=start_time,
                **extra_context
            )
            
            # Store metrics
            self.record_metrics(metrics)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        self.operation_stats[metrics.operation].append(metrics)
        
        # Keep operation stats bounded
        if len(self.operation_stats[metrics.operation]) > 100:
            self.operation_stats[metrics.operation] = self.operation_stats[metrics.operation][-50:]
        
        # Check for performance issues
        self._check_performance_thresholds(metrics)
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check metrics against performance thresholds."""
        issues = []
        
        if metrics.duration_ms > self.thresholds["slow_operation_ms"]:
            issues.append(f"Slow operation: {metrics.operation} took {metrics.duration_ms:.1f}ms")
        
        if metrics.memory_mb > self.thresholds["high_memory_mb"]:
            issues.append(f"High memory usage: {metrics.operation} used {metrics.memory_mb:.1f}MB")
        
        if metrics.cpu_percent > self.thresholds["high_cpu_percent"]:
            issues.append(f"High CPU usage: {metrics.operation} used {metrics.cpu_percent:.1f}% CPU")
        
        if (metrics.cache_hits is not None and metrics.cache_misses is not None and 
            metrics.cache_hits + metrics.cache_misses > 0):
            miss_rate = metrics.cache_misses / (metrics.cache_hits + metrics.cache_misses)
            if miss_rate > self.thresholds["cache_miss_rate"]:
                issues.append(f"High cache miss rate: {miss_rate:.2%} for {metrics.operation}")
        
        if issues:
            print(f"Performance issues detected:\n" + "\n".join(f"  - {issue}" for issue in issues))
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation not in self.operation_stats:
            return {"error": f"No stats for operation: {operation}"}
        
        metrics_list = self.operation_stats[operation]
        durations = [m.duration_ms for m in metrics_list]
        memory_usage = [m.memory_mb for m in metrics_list]
        
        return {
            "operation": operation,
            "call_count": len(metrics_list),
            "duration_stats": {
                "min_ms": min(durations),
                "max_ms": max(durations), 
                "avg_ms": sum(durations) / len(durations),
                "total_ms": sum(durations)
            },
            "memory_stats": {
                "min_mb": min(memory_usage),
                "max_mb": max(memory_usage),
                "avg_mb": sum(memory_usage) / len(memory_usage),
                "total_mb": sum(memory_usage)
            },
            "recent_performance": [asdict(m) for m in metrics_list[-10:]]
        }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get system health and performance report."""
        if not self.system_metrics_history:
            return {"error": "No system metrics available"}
        
        recent_metrics = list(self.system_metrics_history)[-100:]
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        # Detect trends
        cpu_trend = "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
        memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
        
        return {
            "timestamp": time.time(),
            "system_health": {
                "cpu": {
                    "current_percent": cpu_values[-1],
                    "avg_percent": sum(cpu_values) / len(cpu_values),
                    "max_percent": max(cpu_values),
                    "trend": cpu_trend
                },
                "memory": {
                    "current_percent": memory_values[-1],
                    "avg_percent": sum(memory_values) / len(memory_values), 
                    "max_percent": max(memory_values),
                    "trend": memory_trend,
                    "available_mb": recent_metrics[-1].memory_available_mb
                },
                "threads": {
                    "active_count": recent_metrics[-1].active_threads,
                    "open_files": recent_metrics[-1].open_files
                }
            },
            "performance_summary": self._get_performance_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.metrics_history:
            return {}
        
        total_operations = len(self.metrics_history)
        operations_by_type = defaultdict(int)
        slow_operations = 0
        high_memory_operations = 0
        
        for metrics in self.metrics_history:
            operations_by_type[metrics.operation] += 1
            if metrics.duration_ms > self.thresholds["slow_operation_ms"]:
                slow_operations += 1
            if metrics.memory_mb > self.thresholds["high_memory_mb"]:
                high_memory_operations += 1
        
        return {
            "total_operations": total_operations,
            "operations_by_type": dict(operations_by_type),
            "performance_issues": {
                "slow_operations": slow_operations,
                "high_memory_operations": high_memory_operations,
                "slow_operation_rate": slow_operations / total_operations if total_operations > 0 else 0
            },
            "top_operations": sorted(
                operations_by_type.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        # Analyze operation patterns
        for operation, metrics_list in self.operation_stats.items():
            if len(metrics_list) < 5:  # Need enough samples
                continue
            
            durations = [m.duration_ms for m in metrics_list]
            avg_duration = sum(durations) / len(durations)
            
            if avg_duration > self.thresholds["slow_operation_ms"]:
                recommendations.append({
                    "type": "performance",
                    "operation": operation,
                    "issue": f"Slow operation (avg {avg_duration:.1f}ms)",
                    "suggestions": [
                        "Consider caching results",
                        "Optimize algorithm complexity", 
                        "Use parallel processing",
                        "Profile individual steps"
                    ]
                })
            
            # Cache analysis
            if metrics_list[-1].cache_hits is not None:
                cache_metrics = [m for m in metrics_list if m.cache_hits is not None]
                if cache_metrics:
                    total_hits = sum(m.cache_hits for m in cache_metrics)
                    total_misses = sum(m.cache_misses for m in cache_metrics)
                    if total_hits + total_misses > 0:
                        miss_rate = total_misses / (total_hits + total_misses)
                        if miss_rate > self.thresholds["cache_miss_rate"]:
                            recommendations.append({
                                "type": "caching", 
                                "operation": operation,
                                "issue": f"High cache miss rate ({miss_rate:.2%})",
                                "suggestions": [
                                    "Increase cache size",
                                    "Optimize cache key strategy",
                                    "Use predictive caching",
                                    "Review cache eviction policy"
                                ]
                            })
        
        # System-level recommendations
        if self.system_metrics_history:
            recent_metrics = list(self.system_metrics_history)[-50:]
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            
            if avg_cpu > 70:
                recommendations.append({
                    "type": "system",
                    "issue": f"High CPU utilization ({avg_cpu:.1f}%)",
                    "suggestions": [
                        "Scale horizontally with more workers",
                        "Optimize CPU-intensive operations", 
                        "Use async processing where possible",
                        "Consider upgrading hardware"
                    ]
                })
            
            if avg_memory > 80:
                recommendations.append({
                    "type": "system",
                    "issue": f"High memory utilization ({avg_memory:.1f}%)",
                    "suggestions": [
                        "Implement memory-efficient algorithms",
                        "Increase garbage collection frequency",
                        "Use memory profiling to find leaks",
                        "Consider memory scaling"
                    ]
                })
        
        return recommendations
    
    def export_metrics(self, filepath: Path):
        """Export metrics to file for analysis."""
        export_data = {
            "timestamp": time.time(),
            "performance_metrics": [asdict(m) for m in self.metrics_history],
            "system_metrics": [asdict(m) for m in self.system_metrics_history],
            "operation_stats": {op: [asdict(m) for m in metrics] for op, metrics in self.operation_stats.items()},
            "recommendations": self.get_optimization_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def profile_function(self, operation_name: str = None):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_operation(op_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# Global profiler instance
performance_profiler = PerformanceProfiler()


# Convenience decorator
def profile(operation_name: str = None):
    """Convenience decorator for profiling."""
    return performance_profiler.profile_function(operation_name)


# Context manager for easy profiling
@contextmanager
def profile_operation(operation: str, **context):
    """Convenience context manager for profiling."""
    with performance_profiler.profile_operation(operation, **context):
        yield