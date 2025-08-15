"""Distributed verification system with auto-scaling and global optimization."""

import asyncio
import aiohttp
import time
import json
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets
import hashlib

from .parallel_verifier import VerificationTask, VerificationResult, VerificationWorker
from ..monitoring.logger import get_logger
from ..cache.optimized_cache import OptimizedCacheManager


class NodeRole(Enum):
    """Roles in distributed system."""

    COORDINATOR = "coordinator"
    WORKER = "worker"
    CACHE_SERVER = "cache_server"
    LOAD_BALANCER = "load_balancer"


class NodeStatus(Enum):
    """Node status states."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class NodeInfo:
    """Information about a distributed node."""

    node_id: str
    role: NodeRole
    status: NodeStatus
    host: str
    port: int
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    max_capacity: int = 10
    last_heartbeat: float = 0.0
    performance_metrics: Dict[str, float] = None

    def __post_init__(self):
        self.performance_metrics = self.performance_metrics or {}
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


@dataclass
class DistributedTask:
    """Distributed verification task with routing info."""

    task: VerificationTask
    priority: int = 0
    assigned_node: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    affinity_requirements: List[str] = None

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        self.affinity_requirements = self.affinity_requirements or []


class DistributedVerifier:
    """Distributed verification system with intelligent load balancing."""

    def __init__(
        self,
        role: NodeRole,
        node_id: Optional[str] = None,
        host: str = "localhost",
        port: int = 8000,
        coordinator_url: Optional[str] = None,
        auto_scaling: bool = True,
        max_nodes: int = 100,
    ):
        """Initialize distributed verifier node.

        Args:
            role: Role of this node
            node_id: Unique node identifier
            host: Host address
            port: Port number
            coordinator_url: URL of coordinator node (for workers)
            auto_scaling: Enable automatic scaling
            max_nodes: Maximum number of nodes
        """
        self.role = role
        self.node_id = node_id or f"{role.value}_{uuid.uuid4().hex[:8]}"
        self.host = host
        self.port = port
        self.coordinator_url = coordinator_url
        self.auto_scaling = auto_scaling
        self.max_nodes = max_nodes

        self.logger = get_logger(f"distributed_{self.node_id}")

        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.my_info = NodeInfo(
            node_id=self.node_id,
            role=role,
            status=NodeStatus.INITIALIZING,
            host=host,
            port=port,
            capabilities=self._get_node_capabilities(),
        )

        # Task management
        self.task_queue: asyncio.Queue = None
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, VerificationResult] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}

        # Load balancing
        self.load_balancer = IntelligentLoadBalancer()
        self.task_scheduler = DistributedTaskScheduler()

        # Caching
        self.cache_manager = (
            OptimizedCacheManager()
            if role in [NodeRole.COORDINATOR, NodeRole.CACHE_SERVER]
            else None
        )
        self.global_cache_enabled = True

        # Performance tracking
        self.performance_tracker = PerformanceTracker(self.node_id)

        # Networking
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}

        # Auto-scaling
        self.scaling_policy = AutoScalingPolicy() if auto_scaling else None

        # Synchronization
        self.lock = threading.RLock()

        self.logger.info(
            f"Initialized distributed node {self.node_id} with role {role.value}"
        )

    async def start(self):
        """Start the distributed node."""
        try:
            self.task_queue = asyncio.Queue(maxsize=10000)
            self.session = aiohttp.ClientSession()

            if self.role == NodeRole.COORDINATOR:
                await self._start_coordinator()
            elif self.role == NodeRole.WORKER:
                await self._start_worker()
            elif self.role == NodeRole.CACHE_SERVER:
                await self._start_cache_server()
            elif self.role == NodeRole.LOAD_BALANCER:
                await self._start_load_balancer()

            self.my_info.status = NodeStatus.ACTIVE
            self.logger.info(f"Node {self.node_id} started successfully")

        except Exception as e:
            self.my_info.status = NodeStatus.FAILED
            self.logger.error(f"Failed to start node: {str(e)}")
            raise

    async def stop(self):
        """Stop the distributed node."""
        self.my_info.status = NodeStatus.MAINTENANCE

        if self.session:
            await self.session.close()

        # Close websocket connections
        for connection in self.websocket_connections.values():
            await connection.close()

        self.logger.info(f"Node {self.node_id} stopped")

    async def submit_distributed_task(
        self,
        task: VerificationTask,
        priority: int = 0,
        affinity: Optional[List[str]] = None,
    ) -> str:
        """Submit task for distributed processing."""

        distributed_task = DistributedTask(
            task=task, priority=priority, affinity_requirements=affinity or []
        )

        if self.role == NodeRole.COORDINATOR:
            return await self._schedule_task(distributed_task)
        else:
            # Forward to coordinator
            return await self._forward_to_coordinator(distributed_task)

    async def _start_coordinator(self):
        """Start coordinator node."""
        # Start web server for node management
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/submit_task", self._handle_task_submission)
        app.router.add_get("/nodes", self._handle_node_list)
        app.router.add_get("/stats", self._handle_stats_request)
        app.router.add_post("/register_node", self._handle_node_registration)
        app.router.add_ws("/ws", self._handle_websocket)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        # Start background tasks
        asyncio.create_task(self._coordinator_main_loop())
        asyncio.create_task(self._node_health_monitor())

        if self.auto_scaling:
            asyncio.create_task(self._auto_scaling_monitor())

        self.logger.info(f"Coordinator started on {self.host}:{self.port}")

    async def _start_worker(self):
        """Start worker node."""
        # Register with coordinator
        if self.coordinator_url:
            await self._register_with_coordinator()

        # Start worker loop
        asyncio.create_task(self._worker_main_loop())

        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())

        self.logger.info(f"Worker started, registered with coordinator")

    async def _start_cache_server(self):
        """Start dedicated cache server."""
        # Initialize cache with large capacity
        self.cache_manager = OptimizedCacheManager(max_size=10000, max_memory_mb=2048.0)

        # Start cache API server
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/cache/{key}", self._handle_cache_get)
        app.router.add_put("/cache/{key}", self._handle_cache_put)
        app.router.add_delete("/cache/{key}", self._handle_cache_delete)
        app.router.add_get("/cache_stats", self._handle_cache_stats)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        self.logger.info(f"Cache server started on {self.host}:{self.port}")

    async def _start_load_balancer(self):
        """Start load balancer node."""
        # Initialize load balancer
        self.load_balancer = IntelligentLoadBalancer()

        # Start load balancer API
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/balance", self._handle_load_balance)
        app.router.add_get("/health", self._handle_health_check)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        self.logger.info(f"Load balancer started on {self.host}:{self.port}")

    async def _schedule_task(self, distributed_task: DistributedTask) -> str:
        """Schedule task to appropriate worker."""

        # Find best node for task
        best_node = await self.load_balancer.select_best_node(
            self.nodes, distributed_task.task, distributed_task.affinity_requirements
        )

        if not best_node:
            self.logger.warning("No available nodes for task scheduling")
            raise RuntimeError("No available worker nodes")

        distributed_task.assigned_node = best_node.node_id
        self.active_tasks[distributed_task.task.task_id] = distributed_task

        # Send task to worker
        await self._send_task_to_worker(best_node, distributed_task)

        self.logger.debug(
            f"Scheduled task {distributed_task.task.task_id} to node {best_node.node_id}"
        )

        return distributed_task.task.task_id

    async def _send_task_to_worker(self, worker: NodeInfo, task: DistributedTask):
        """Send task to worker node."""

        url = f"http://{worker.host}:{worker.port}/execute_task"
        task_data = {"task": asdict(task.task), "distributed_info": asdict(task)}

        try:
            async with self.session.post(url, json=task_data) as response:
                if response.status == 200:
                    task.started_at = time.time()
                    self.logger.debug(
                        f"Task {task.task.task_id} sent to worker {worker.node_id}"
                    )
                else:
                    self.logger.error(
                        f"Failed to send task to worker: {response.status}"
                    )

        except Exception as e:
            self.logger.error(f"Error sending task to worker: {str(e)}")
            await self._handle_task_failure(task, str(e))

    async def _coordinator_main_loop(self):
        """Main coordinator processing loop."""

        while self.my_info.status == NodeStatus.ACTIVE:
            try:
                # Process completed tasks
                await self._process_completed_tasks()

                # Handle failed tasks
                await self._handle_failed_tasks()

                # Update performance metrics
                await self._update_performance_metrics()

                # Global cache optimization
                if self.cache_manager:
                    await self._optimize_global_cache()

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in coordinator loop: {str(e)}")
                await asyncio.sleep(5.0)

    async def _worker_main_loop(self):
        """Main worker processing loop."""

        worker = VerificationWorker(self.node_id, shared_cache=True)

        while self.my_info.status == NodeStatus.ACTIVE:
            try:
                # Get task from queue
                if not self.task_queue.empty():
                    distributed_task = await self.task_queue.get()

                    # Update status
                    self.my_info.status = NodeStatus.BUSY
                    self.my_info.current_load = self.task_queue.qsize()

                    # Process task
                    start_time = time.time()
                    result = worker.process_task(distributed_task.task)
                    execution_time = (time.time() - start_time) * 1000

                    # Update performance metrics
                    self.performance_tracker.record_task_completion(
                        execution_time, result.success
                    )

                    # Report result back to coordinator
                    await self._report_task_result(distributed_task, result)

                    # Update status
                    self.my_info.status = NodeStatus.ACTIVE
                    self.my_info.current_load = self.task_queue.qsize()

                else:
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in worker loop: {str(e)}")
                await asyncio.sleep(1.0)

    def _get_node_capabilities(self) -> Dict[str, Any]:
        """Get node capabilities."""
        import psutil

        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_gb": psutil.disk_usage("/").total / (1024**3),
            "supported_provers": ["isabelle", "coq"],
            "supported_models": ["gpt-4-turbo", "claude-3"],
            "max_concurrent_tasks": min(psutil.cpu_count() * 2, 16),
        }

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""

        if self.role != NodeRole.COORDINATOR:
            return {"error": "Only coordinator can provide cluster status"}

        total_capacity = sum(node.max_capacity for node in self.nodes.values())
        total_load = sum(node.current_load for node in self.nodes.values())
        active_nodes = len(
            [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
        )

        return {
            "cluster_size": len(self.nodes),
            "active_nodes": active_nodes,
            "total_capacity": total_capacity,
            "current_load": total_load,
            "utilization": total_load / max(total_capacity, 1),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "cache_stats": (
                self.cache_manager.get_stats() if self.cache_manager else None
            ),
            "performance_metrics": self.performance_tracker.get_global_metrics(),
        }


class IntelligentLoadBalancer:
    """AI-driven load balancing with predictive scheduling."""

    def __init__(self):
        self.logger = get_logger("load_balancer")
        self.scheduling_history: List[Dict[str, Any]] = []
        self.node_performance_models: Dict[str, Dict[str, float]] = {}

    async def select_best_node(
        self, nodes: Dict[str, NodeInfo], task: VerificationTask, affinity: List[str]
    ) -> Optional[NodeInfo]:
        """Select best node for task using AI-driven algorithm."""

        available_nodes = [
            node
            for node in nodes.values()
            if node.status == NodeStatus.ACTIVE
            and node.current_load < node.max_capacity
        ]

        if not available_nodes:
            return None

        # Score nodes based on multiple factors
        node_scores = []

        for node in available_nodes:
            score = await self._calculate_node_score(node, task, affinity)
            node_scores.append((node, score))

        # Select best node
        best_node = max(node_scores, key=lambda x: x[1])[0]

        # Record decision for learning
        self._record_scheduling_decision(best_node, task, node_scores)

        return best_node

    async def _calculate_node_score(
        self, node: NodeInfo, task: VerificationTask, affinity: List[str]
    ) -> float:
        """Calculate comprehensive node score."""

        score = 0.0

        # Load balancing factor (prefer less loaded nodes)
        load_factor = 1.0 - (node.current_load / max(node.max_capacity, 1))
        score += load_factor * 0.3

        # Performance factor (prefer historically fast nodes)
        if node.node_id in self.node_performance_models:
            perf_model = self.node_performance_models[node.node_id]
            expected_time = perf_model.get("avg_execution_time", 1000.0)
            perf_factor = 1000.0 / max(expected_time, 100.0)  # Normalize
            score += perf_factor * 0.25

        # Capability matching factor
        capability_score = self._calculate_capability_match(node, task)
        score += capability_score * 0.2

        # Affinity factor
        affinity_score = self._calculate_affinity_score(node, affinity)
        score += affinity_score * 0.15

        # Geographic/network proximity factor
        proximity_score = self._calculate_proximity_score(node)
        score += proximity_score * 0.1

        return score

    def _calculate_capability_match(
        self, node: NodeInfo, task: VerificationTask
    ) -> float:
        """Calculate how well node capabilities match task requirements."""

        score = 0.0

        # Prover support
        if task.prover in node.capabilities.get("supported_provers", []):
            score += 0.5

        # Model support
        if task.model in node.capabilities.get("supported_models", []):
            score += 0.3

        # Resource adequacy
        estimated_memory = len(task.hdl_code) * 0.001  # Rough estimate in GB
        available_memory = node.capabilities.get("memory_gb", 0)

        if available_memory >= estimated_memory * 2:  # 2x buffer
            score += 0.2

        return score

    def _calculate_affinity_score(self, node: NodeInfo, affinity: List[str]) -> float:
        """Calculate affinity score based on requirements."""

        if not affinity:
            return 0.5  # Neutral score

        score = 0.0
        for requirement in affinity:
            if requirement in node.capabilities.get("tags", []):
                score += 1.0
            elif requirement == node.node_id:
                score += 1.0
            elif requirement in [node.host, f"{node.host}:{node.port}"]:
                score += 0.8

        return min(1.0, score / len(affinity))

    def _calculate_proximity_score(self, node: NodeInfo) -> float:
        """Calculate network proximity score."""
        # Simplified - in real implementation would use latency measurements
        return 0.5  # Neutral score

    def _record_scheduling_decision(
        self,
        selected_node: NodeInfo,
        task: VerificationTask,
        all_scores: List[Tuple[NodeInfo, float]],
    ):
        """Record scheduling decision for learning."""

        decision_record = {
            "timestamp": time.time(),
            "selected_node": selected_node.node_id,
            "task_type": task.prover,
            "task_size": len(task.hdl_code),
            "all_scores": [(node.node_id, score) for node, score in all_scores],
        }

        self.scheduling_history.append(decision_record)

        # Keep only recent history
        cutoff_time = time.time() - 3600  # 1 hour
        self.scheduling_history = [
            record
            for record in self.scheduling_history
            if record["timestamp"] > cutoff_time
        ]


class DistributedTaskScheduler:
    """Advanced task scheduling with priority and deadline management."""

    def __init__(self):
        self.logger = get_logger("task_scheduler")
        self.priority_queues: Dict[int, List[DistributedTask]] = {}
        self.deadline_queue: List[DistributedTask] = []

    def add_task(self, task: DistributedTask):
        """Add task to appropriate queue."""

        # Add to priority queue
        if task.priority not in self.priority_queues:
            self.priority_queues[task.priority] = []

        self.priority_queues[task.priority].append(task)

        # Add to deadline queue if deadline exists
        if hasattr(task.task, "deadline") and task.task.deadline:
            self.deadline_queue.append(task)
            self.deadline_queue.sort(key=lambda t: t.task.deadline)

    def get_next_task(self) -> Optional[DistributedTask]:
        """Get next task based on scheduling policy."""

        # Check deadline queue first
        if self.deadline_queue:
            urgent_task = self.deadline_queue[0]
            deadline_urgency = time.time() + 300  # 5 minutes buffer

            if (
                hasattr(urgent_task.task, "deadline")
                and urgent_task.task.deadline < deadline_urgency
            ):
                self.deadline_queue.pop(0)

                # Remove from priority queue
                for priority, tasks in self.priority_queues.items():
                    if urgent_task in tasks:
                        tasks.remove(urgent_task)
                        break

                return urgent_task

        # Get highest priority task
        for priority in sorted(self.priority_queues.keys(), reverse=True):
            if self.priority_queues[priority]:
                task = self.priority_queues[priority].pop(0)

                # Remove from deadline queue
                if task in self.deadline_queue:
                    self.deadline_queue.remove(task)

                return task

        return None


class AutoScalingPolicy:
    """Auto-scaling policy with predictive scaling."""

    def __init__(self):
        self.logger = get_logger("auto_scaler")
        self.scaling_history: List[Dict[str, Any]] = []
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_action = 0.0

    def should_scale_up(self, cluster_utilization: float, queue_length: int) -> bool:
        """Determine if cluster should scale up."""

        current_time = time.time()

        # Check cooldown
        if current_time - self.last_scaling_action < self.cooldown_period:
            return False

        # Scale up conditions
        if cluster_utilization > self.scale_up_threshold:
            return True

        if queue_length > 50:  # Large queue backlog
            return True

        # Predictive scaling based on historical patterns
        if self._predict_future_load() > self.scale_up_threshold:
            return True

        return False

    def should_scale_down(self, cluster_utilization: float, idle_nodes: int) -> bool:
        """Determine if cluster should scale down."""

        current_time = time.time()

        # Check cooldown
        if current_time - self.last_scaling_action < self.cooldown_period:
            return False

        # Scale down conditions
        if cluster_utilization < self.scale_down_threshold and idle_nodes > 2:
            return True

        return False

    def _predict_future_load(self) -> float:
        """Predict future load based on historical patterns."""

        if len(self.scaling_history) < 10:
            return 0.5  # Default neutral prediction

        # Simple trend analysis
        recent_utilizations = [
            record["utilization"] for record in self.scaling_history[-10:]
        ]

        # Calculate trend
        if len(recent_utilizations) >= 2:
            trend = (recent_utilizations[-1] - recent_utilizations[0]) / len(
                recent_utilizations
            )
            predicted_utilization = (
                recent_utilizations[-1] + trend * 3
            )  # 3 periods ahead

            return min(1.0, max(0.0, predicted_utilization))

        return recent_utilizations[-1] if recent_utilizations else 0.5


class PerformanceTracker:
    """Performance tracking and optimization recommendations."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = get_logger(f"perf_tracker_{node_id}")

        self.metrics_history: List[Dict[str, Any]] = []
        self.task_completions: List[Dict[str, Any]] = []

    def record_task_completion(self, execution_time: float, success: bool):
        """Record task completion metrics."""

        record = {
            "timestamp": time.time(),
            "execution_time_ms": execution_time,
            "success": success,
            "node_id": self.node_id,
        }

        self.task_completions.append(record)

        # Keep only recent completions
        cutoff_time = time.time() - 3600  # 1 hour
        self.task_completions = [
            completion
            for completion in self.task_completions
            if completion["timestamp"] > cutoff_time
        ]

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global performance metrics."""

        if not self.task_completions:
            return {"error": "No task completion data available"}

        recent_completions = [
            c
            for c in self.task_completions
            if time.time() - c["timestamp"] < 1800  # 30 minutes
        ]

        if not recent_completions:
            return {"error": "No recent task completion data"}

        execution_times = [c["execution_time_ms"] for c in recent_completions]
        success_count = sum(1 for c in recent_completions if c["success"])

        return {
            "total_tasks": len(recent_completions),
            "success_rate": success_count / len(recent_completions),
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "min_execution_time_ms": min(execution_times),
            "max_execution_time_ms": max(execution_times),
            "throughput_per_hour": len(recent_completions) * 2,  # Scale to hourly
            "node_count": len(set(c["node_id"] for c in recent_completions)),
        }
