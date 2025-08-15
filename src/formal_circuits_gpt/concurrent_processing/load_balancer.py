"""Intelligent load balancing for verification tasks."""

import time
import heapq
import threading
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class WorkerStats:
    """Statistics for a worker."""

    worker_id: str
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time_ms: float = 0.0
    last_task_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    queue_size: int = 0
    weight: float = 1.0
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)

    @property
    def average_response_time_ms(self) -> float:
        """Calculate average response time."""
        if self.completed_tasks == 0:
            return 0.0
        return self.total_execution_time_ms / self.completed_tasks

    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 1.0
        return self.completed_tasks / total

    @property
    def load_score(self) -> float:
        """Calculate overall load score (lower is better)."""
        base_load = self.active_tasks + (self.queue_size * 0.5)
        response_penalty = min(self.average_response_time_ms / 1000, 10.0)  # Cap at 10s
        resource_penalty = (self.cpu_usage + self.memory_usage_mb / 1000) / 2
        health_penalty = 0 if self.is_healthy else 1000

        return base_load + response_penalty + resource_penalty + health_penalty


@dataclass
class TaskPriority:
    """Task with priority information."""

    task: Any
    priority: int
    arrival_time: float
    estimated_duration_ms: float = 1000.0

    def __lt__(self, other):
        """For heap ordering (higher priority first)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.arrival_time < other.arrival_time


class IntelligentLoadBalancer:
    """Intelligent load balancer for verification tasks."""

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
        health_check_interval: float = 30.0,
    ):
        """Initialize load balancer.

        Args:
            strategy: Load balancing strategy to use
            health_check_interval: Interval between health checks in seconds
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval

        self._workers: Dict[str, WorkerStats] = {}
        self._task_queue = []  # Priority queue
        self._round_robin_index = 0
        self._lock = threading.RLock()

        # Performance tracking
        self._response_times = defaultdict(lambda: deque(maxlen=100))
        self._throughput_history = deque(maxlen=60)  # Last 60 seconds
        self._last_throughput_check = time.time()
        self._tasks_in_last_second = 0

        # Health monitoring
        self._health_check_callbacks: Dict[str, Callable[[str], bool]] = {}
        self._last_health_check = time.time()

    def register_worker(
        self,
        worker_id: str,
        weight: float = 1.0,
        health_check_callback: Optional[Callable[[str], bool]] = None,
    ):
        """Register a worker with the load balancer."""
        with self._lock:
            self._workers[worker_id] = WorkerStats(worker_id=worker_id, weight=weight)

            if health_check_callback:
                self._health_check_callbacks[worker_id] = health_check_callback

    def unregister_worker(self, worker_id: str):
        """Unregister a worker from the load balancer."""
        with self._lock:
            self._workers.pop(worker_id, None)
            self._health_check_callbacks.pop(worker_id, None)

    def select_worker(
        self, task: Any = None, estimated_duration_ms: float = 1000.0
    ) -> Optional[str]:
        """Select the best worker for a task based on the current strategy."""
        with self._lock:
            available_workers = [
                worker_id
                for worker_id, stats in self._workers.items()
                if stats.is_healthy
            ]

            if not available_workers:
                return None

            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(available_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(available_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._select_weighted_round_robin(available_workers)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return self._select_by_response_time(available_workers)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
                return self._select_resource_aware(
                    available_workers, estimated_duration_ms
                )
            else:
                return available_workers[0]  # Fallback

    def _select_round_robin(self, workers: List[str]) -> str:
        """Round robin selection."""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker

    def _select_least_connections(self, workers: List[str]) -> str:
        """Select worker with least active connections."""
        return min(workers, key=lambda w: self._workers[w].active_tasks)

    def _select_weighted_round_robin(self, workers: List[str]) -> str:
        """Weighted round robin based on worker weights."""
        # Simple implementation: repeat workers based on their weight
        weighted_workers = []
        for worker in workers:
            weight = int(self._workers[worker].weight)
            weighted_workers.extend([worker] * max(1, weight))

        if weighted_workers:
            worker = weighted_workers[self._round_robin_index % len(weighted_workers)]
            self._round_robin_index += 1
            return worker
        return workers[0]

    def _select_by_response_time(self, workers: List[str]) -> str:
        """Select worker with best response time."""
        return min(workers, key=lambda w: self._workers[w].average_response_time_ms)

    def _select_resource_aware(
        self, workers: List[str], estimated_duration_ms: float
    ) -> str:
        """Select worker based on comprehensive resource analysis."""
        best_worker = None
        best_score = float("inf")

        for worker_id in workers:
            stats = self._workers[worker_id]

            # Calculate comprehensive score
            load_score = stats.load_score

            # Factor in estimated completion time
            estimated_completion = estimated_duration_ms + (
                stats.queue_size * stats.average_response_time_ms
            )
            time_penalty = estimated_completion / 10000  # Scale down

            # Success rate bonus
            success_bonus = (1.0 - stats.success_rate) * 5

            total_score = load_score + time_penalty + success_bonus

            if total_score < best_score:
                best_score = total_score
                best_worker = worker_id

        return best_worker or workers[0]

    def update_worker_stats(
        self,
        worker_id: str,
        active_tasks: Optional[int] = None,
        cpu_usage: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        queue_size: Optional[int] = None,
    ):
        """Update worker statistics."""
        with self._lock:
            if worker_id not in self._workers:
                return

            stats = self._workers[worker_id]

            if active_tasks is not None:
                stats.active_tasks = active_tasks
            if cpu_usage is not None:
                stats.cpu_usage = cpu_usage
            if memory_usage_mb is not None:
                stats.memory_usage_mb = memory_usage_mb
            if queue_size is not None:
                stats.queue_size = queue_size

    def record_task_completion(
        self, worker_id: str, execution_time_ms: float, success: bool
    ):
        """Record completion of a task."""
        with self._lock:
            if worker_id not in self._workers:
                return

            stats = self._workers[worker_id]

            if success:
                stats.completed_tasks += 1
                stats.total_execution_time_ms += execution_time_ms
            else:
                stats.failed_tasks += 1

            stats.last_task_time = time.time()

            # Update response time history
            self._response_times[worker_id].append(execution_time_ms)

            # Update throughput tracking
            self._tasks_in_last_second += 1

    def check_worker_health(self):
        """Check health of all workers."""
        current_time = time.time()

        if current_time - self._last_health_check < self.health_check_interval:
            return

        with self._lock:
            for worker_id, stats in self._workers.items():
                # Run custom health check if available
                if worker_id in self._health_check_callbacks:
                    try:
                        is_healthy = self._health_check_callbacks[worker_id](worker_id)
                        stats.is_healthy = is_healthy
                    except Exception:
                        stats.is_healthy = False
                else:
                    # Default health check based on response times and failures
                    recent_failures = stats.failed_tasks / max(
                        1, stats.completed_tasks + stats.failed_tasks
                    )
                    stats.is_healthy = (
                        recent_failures < 0.5  # Less than 50% failure rate
                        and stats.average_response_time_ms
                        < 30000  # Less than 30s response time
                        and (current_time - stats.last_task_time)
                        < 300  # Active within 5 minutes
                    )

                stats.last_health_check = current_time

        self._last_health_check = current_time

    def get_worker_stats(self, worker_id: str) -> Optional[WorkerStats]:
        """Get statistics for a specific worker."""
        with self._lock:
            return self._workers.get(worker_id)

    def get_all_worker_stats(self) -> Dict[str, WorkerStats]:
        """Get statistics for all workers."""
        with self._lock:
            return dict(self._workers)

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        with self._lock:
            # Calculate throughput
            current_time = time.time()
            if current_time - self._last_throughput_check >= 1.0:
                self._throughput_history.append(self._tasks_in_last_second)
                self._tasks_in_last_second = 0
                self._last_throughput_check = current_time

            avg_throughput = (
                statistics.mean(self._throughput_history)
                if self._throughput_history
                else 0
            )

            # Worker health summary
            healthy_workers = sum(
                1 for stats in self._workers.values() if stats.is_healthy
            )
            total_workers = len(self._workers)

            # Overall response time
            all_response_times = []
            for response_times in self._response_times.values():
                all_response_times.extend(response_times)

            avg_response_time = (
                statistics.mean(all_response_times) if all_response_times else 0
            )

            return {
                "strategy": self.strategy.value,
                "total_workers": total_workers,
                "healthy_workers": healthy_workers,
                "health_percentage": (healthy_workers / max(1, total_workers)) * 100,
                "average_throughput_per_second": avg_throughput,
                "average_response_time_ms": avg_response_time,
                "total_active_tasks": sum(
                    stats.active_tasks for stats in self._workers.values()
                ),
                "total_completed_tasks": sum(
                    stats.completed_tasks for stats in self._workers.values()
                ),
                "total_failed_tasks": sum(
                    stats.failed_tasks for stats in self._workers.values()
                ),
                "workers": {
                    worker_id: {
                        "active_tasks": stats.active_tasks,
                        "completed_tasks": stats.completed_tasks,
                        "failed_tasks": stats.failed_tasks,
                        "average_response_time_ms": stats.average_response_time_ms,
                        "success_rate": stats.success_rate,
                        "load_score": stats.load_score,
                        "is_healthy": stats.is_healthy,
                        "cpu_usage": stats.cpu_usage,
                        "memory_usage_mb": stats.memory_usage_mb,
                    }
                    for worker_id, stats in self._workers.items()
                },
            }


class TaskQueue:
    """Priority-based task queue."""

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self._queue = []
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def put(
        self,
        task: Any,
        priority: int = 0,
        estimated_duration_ms: float = 1000.0,
        timeout: Optional[float] = None,
    ) -> bool:
        """Add a task to the queue."""
        task_item = TaskPriority(
            task=task,
            priority=priority,
            arrival_time=time.time(),
            estimated_duration_ms=estimated_duration_ms,
        )

        with self._not_full:
            if self.max_size is not None:
                while len(self._queue) >= self.max_size:
                    if timeout == 0:
                        return False
                    if not self._not_full.wait(timeout):
                        return False

            heapq.heappush(self._queue, task_item)
            self._not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Get the highest priority task from the queue."""
        with self._not_empty:
            while not self._queue:
                if timeout == 0:
                    return None
                if not self._not_empty.wait(timeout):
                    return None

            task_item = heapq.heappop(self._queue)
            self._not_full.notify()
            return task_item.task

    def qsize(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)

    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0
