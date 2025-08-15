"""Advanced resource management and monitoring."""

import os
import time
import threading
import psutil
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from collections import deque
from enum import Enum
import warnings


class ResourceType(Enum):
    """Types of system resources."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class ResourceLimits:
    """Resource usage limits."""

    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_disk_percent: float = 90.0
    max_network_mbps: float = 1000.0
    max_gpu_memory_percent: float = 90.0
    max_open_files: int = 1000
    max_threads: int = 100


@dataclass
class ResourceUsage:
    """Current resource usage."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    network_sent_mbps: float = 0.0
    network_recv_mbps: float = 0.0
    gpu_memory_percent: float = 0.0
    open_files: int = 0
    thread_count: int = 0
    load_average: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ResourceMonitor:
    """Real-time resource monitoring."""

    def __init__(
        self,
        collection_interval: float = 1.0,
        history_size: int = 300,
        enable_gpu_monitoring: bool = True,
    ):
        """Initialize resource monitor.

        Args:
            collection_interval: Seconds between collections
            history_size: Number of historical samples to keep
            enable_gpu_monitoring: Whether to monitor GPU resources
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring

        self._usage_history: deque = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # Network monitoring state
        self._last_network_stats = None
        self._last_network_time = None

        # GPU monitoring
        self._gpu_available = False
        if enable_gpu_monitoring:
            self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check if GPU monitoring is available."""
        try:
            import GPUtil

            self._gpu_available = len(GPUtil.getGPUs()) > 0
        except ImportError:
            self._gpu_available = False

    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                usage = self._collect_current_usage()
                with self._lock:
                    self._usage_history.append(usage)
            except Exception as e:
                # Continue monitoring even if collection fails
                warnings.warn(f"Resource collection failed: {e}")

    def _collect_current_usage(self) -> ResourceUsage:
        """Collect current resource usage."""
        current_time = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_free_gb = disk.free / (1024 * 1024 * 1024)

        # Network usage
        network_sent_mbps, network_recv_mbps = self._calculate_network_rate()

        # Process-specific metrics
        process = psutil.Process(os.getpid())
        open_files = len(process.open_files())
        thread_count = process.num_threads()

        # Load average (Unix-like systems)
        try:
            load_average = os.getloadavg()[0]  # 1-minute load average
        except (AttributeError, OSError):
            load_average = 0.0

        # GPU usage
        gpu_memory_percent = 0.0
        if self._gpu_available:
            gpu_memory_percent = self._get_gpu_memory_usage()

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb,
            network_sent_mbps=network_sent_mbps,
            network_recv_mbps=network_recv_mbps,
            gpu_memory_percent=gpu_memory_percent,
            open_files=open_files,
            thread_count=thread_count,
            load_average=load_average,
            timestamp=current_time,
        )

    def _calculate_network_rate(self) -> tuple[float, float]:
        """Calculate network transfer rates in Mbps."""
        current_stats = psutil.net_io_counters()
        current_time = time.time()

        if self._last_network_stats is None or self._last_network_time is None:
            self._last_network_stats = current_stats
            self._last_network_time = current_time
            return 0.0, 0.0

        time_delta = current_time - self._last_network_time
        if time_delta == 0:
            return 0.0, 0.0

        sent_delta = current_stats.bytes_sent - self._last_network_stats.bytes_sent
        recv_delta = current_stats.bytes_recv - self._last_network_stats.bytes_recv

        sent_mbps = (sent_delta * 8) / (1024 * 1024 * time_delta)  # Convert to Mbps
        recv_mbps = (recv_delta * 8) / (1024 * 1024 * time_delta)

        self._last_network_stats = current_stats
        self._last_network_time = current_time

        return sent_mbps, recv_mbps

    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUtil * 100  # Convert to percentage
        except ImportError:
            pass
        return 0.0

    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        return self._collect_current_usage()

    def get_usage_history(
        self, duration_seconds: Optional[float] = None
    ) -> List[ResourceUsage]:
        """Get resource usage history."""
        with self._lock:
            history = list(self._usage_history)

        if duration_seconds is not None:
            cutoff_time = time.time() - duration_seconds
            history = [usage for usage in history if usage.timestamp >= cutoff_time]

        return history

    def get_average_usage(
        self, duration_seconds: Optional[float] = None
    ) -> ResourceUsage:
        """Get average resource usage over specified duration."""
        history = self.get_usage_history(duration_seconds)

        if not history:
            return ResourceUsage()

        # Calculate averages
        return ResourceUsage(
            cpu_percent=sum(u.cpu_percent for u in history) / len(history),
            memory_percent=sum(u.memory_percent for u in history) / len(history),
            memory_used_mb=sum(u.memory_used_mb for u in history) / len(history),
            memory_available_mb=sum(u.memory_available_mb for u in history)
            / len(history),
            disk_percent=sum(u.disk_percent for u in history) / len(history),
            disk_used_gb=sum(u.disk_used_gb for u in history) / len(history),
            disk_free_gb=sum(u.disk_free_gb for u in history) / len(history),
            network_sent_mbps=sum(u.network_sent_mbps for u in history) / len(history),
            network_recv_mbps=sum(u.network_recv_mbps for u in history) / len(history),
            gpu_memory_percent=sum(u.gpu_memory_percent for u in history)
            / len(history),
            open_files=int(sum(u.open_files for u in history) / len(history)),
            thread_count=int(sum(u.thread_count for u in history) / len(history)),
            load_average=sum(u.load_average for u in history) / len(history),
            timestamp=time.time(),
        )


class ResourceManager:
    """Advanced resource manager with automatic throttling and optimization."""

    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        monitor_interval: float = 1.0,
        enable_auto_throttling: bool = True,
    ):
        """Initialize resource manager.

        Args:
            limits: Resource usage limits
            monitor_interval: Resource monitoring interval
            enable_auto_throttling: Enable automatic throttling
        """
        self.limits = limits or ResourceLimits()
        self.enable_auto_throttling = enable_auto_throttling

        self.monitor = ResourceMonitor(collection_interval=monitor_interval)
        self._throttling_active = False
        self._throttling_factor = 1.0

        # Callbacks for resource events
        self._limit_callbacks: Dict[ResourceType, List[Callable]] = {
            resource_type: [] for resource_type in ResourceType
        }

        # Performance optimization state
        self._gc_frequency = 10  # Garbage collection frequency
        self._operation_count = 0

    def start(self):
        """Start resource monitoring and management."""
        self.monitor.start_monitoring()

    def stop(self):
        """Stop resource monitoring and management."""
        self.monitor.stop_monitoring()

    def register_limit_callback(
        self,
        resource_type: ResourceType,
        callback: Callable[[ResourceUsage, ResourceLimits], None],
    ):
        """Register callback for resource limit violations."""
        self._limit_callbacks[resource_type].append(callback)

    def check_resource_limits(self) -> Dict[ResourceType, bool]:
        """Check if resource usage exceeds limits."""
        current_usage = self.monitor.get_current_usage()
        violations = {}

        # Check CPU
        if current_usage.cpu_percent > self.limits.max_cpu_percent:
            violations[ResourceType.CPU] = True
            for callback in self._limit_callbacks[ResourceType.CPU]:
                try:
                    callback(current_usage, self.limits)
                except Exception:
                    pass

        # Check Memory
        if current_usage.memory_percent > self.limits.max_memory_percent:
            violations[ResourceType.MEMORY] = True
            for callback in self._limit_callbacks[ResourceType.MEMORY]:
                try:
                    callback(current_usage, self.limits)
                except Exception:
                    pass

        # Check Disk
        if current_usage.disk_percent > self.limits.max_disk_percent:
            violations[ResourceType.DISK] = True
            for callback in self._limit_callbacks[ResourceType.DISK]:
                try:
                    callback(current_usage, self.limits)
                except Exception:
                    pass

        # Check Network
        total_network = (
            current_usage.network_sent_mbps + current_usage.network_recv_mbps
        )
        if total_network > self.limits.max_network_mbps:
            violations[ResourceType.NETWORK] = True
            for callback in self._limit_callbacks[ResourceType.NETWORK]:
                try:
                    callback(current_usage, self.limits)
                except Exception:
                    pass

        # Check GPU
        if current_usage.gpu_memory_percent > self.limits.max_gpu_memory_percent:
            violations[ResourceType.GPU] = True
            for callback in self._limit_callbacks[ResourceType.GPU]:
                try:
                    callback(current_usage, self.limits)
                except Exception:
                    pass

        # Auto-throttling
        if self.enable_auto_throttling and violations:
            self._apply_throttling(violations, current_usage)

        return violations

    def _apply_throttling(
        self, violations: Dict[ResourceType, bool], usage: ResourceUsage
    ):
        """Apply automatic throttling based on resource violations."""
        if not violations:
            self._throttling_active = False
            self._throttling_factor = 1.0
            return

        self._throttling_active = True

        # Calculate throttling factor based on severity
        severity_factor = 1.0

        if ResourceType.CPU in violations:
            cpu_overage = usage.cpu_percent / self.limits.max_cpu_percent
            severity_factor = min(severity_factor, 1.0 / cpu_overage)

        if ResourceType.MEMORY in violations:
            memory_overage = usage.memory_percent / self.limits.max_memory_percent
            severity_factor = min(severity_factor, 1.0 / memory_overage)

        # Apply throttling (minimum 10% of original capacity)
        self._throttling_factor = max(0.1, severity_factor)

    def get_throttling_factor(self) -> float:
        """Get current throttling factor (0.0 to 1.0)."""
        return self._throttling_factor

    def is_throttling_active(self) -> bool:
        """Check if throttling is currently active."""
        return self._throttling_active

    def optimize_performance(self):
        """Perform performance optimizations."""
        self._operation_count += 1

        # Periodic garbage collection
        if self._operation_count % self._gc_frequency == 0:
            import gc

            gc.collect()

        # Memory pressure relief
        current_usage = self.monitor.get_current_usage()
        if current_usage.memory_percent > 70:
            # More frequent garbage collection under memory pressure
            self._gc_frequency = max(5, self._gc_frequency - 1)
        else:
            # Less frequent GC when memory is available
            self._gc_frequency = min(20, self._gc_frequency + 1)

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        current_usage = self.monitor.get_current_usage()
        average_usage = self.monitor.get_average_usage(
            duration_seconds=300
        )  # 5 minutes

        return {
            "current_usage": {
                "cpu_percent": current_usage.cpu_percent,
                "memory_percent": current_usage.memory_percent,
                "memory_used_mb": current_usage.memory_used_mb,
                "disk_percent": current_usage.disk_percent,
                "network_sent_mbps": current_usage.network_sent_mbps,
                "network_recv_mbps": current_usage.network_recv_mbps,
                "gpu_memory_percent": current_usage.gpu_memory_percent,
                "open_files": current_usage.open_files,
                "thread_count": current_usage.thread_count,
                "load_average": current_usage.load_average,
            },
            "average_usage_5min": {
                "cpu_percent": average_usage.cpu_percent,
                "memory_percent": average_usage.memory_percent,
                "memory_used_mb": average_usage.memory_used_mb,
                "disk_percent": average_usage.disk_percent,
                "network_sent_mbps": average_usage.network_sent_mbps,
                "network_recv_mbps": average_usage.network_recv_mbps,
                "gpu_memory_percent": average_usage.gpu_memory_percent,
                "load_average": average_usage.load_average,
            },
            "limits": {
                "max_cpu_percent": self.limits.max_cpu_percent,
                "max_memory_percent": self.limits.max_memory_percent,
                "max_disk_percent": self.limits.max_disk_percent,
                "max_network_mbps": self.limits.max_network_mbps,
                "max_gpu_memory_percent": self.limits.max_gpu_memory_percent,
            },
            "throttling": {
                "active": self._throttling_active,
                "factor": self._throttling_factor,
            },
            "optimization": {
                "gc_frequency": self._gc_frequency,
                "operation_count": self._operation_count,
            },
        }


# Global resource manager instance
global_resource_manager = ResourceManager()
