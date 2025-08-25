"""Tests for distributed verification system."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

from src.formal_circuits_gpt.concurrent_processing.distributed_verifier import (
    DistributedVerifier,
    NodeRole,
    NodeStatus,
    NodeInfo,
    DistributedTask,
    IntelligentLoadBalancer,
    AutoScalingPolicy,
    PerformanceTracker
)
from src.formal_circuits_gpt.concurrent_processing.parallel_verifier import VerificationTask


class TestDistributedVerifier:
    """Test distributed verification system."""
    
    @pytest.fixture
    def coordinator_node(self):
        """Create coordinator node for testing."""
        return DistributedVerifier(
            role=NodeRole.COORDINATOR,
            host="localhost",
            port=8000
        )
    
    @pytest.fixture 
    def worker_node(self):
        """Create worker node for testing."""
        return DistributedVerifier(
            role=NodeRole.WORKER,
            host="localhost", 
            port=8001,
            coordinator_url="http://localhost:8000"
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create sample verification task."""
        return VerificationTask(
            task_id="test_task_1",
            hdl_code="module test(); endmodule",
            prover="isabelle",
            model="gpt-4-turbo"
        )
    
    def test_node_initialization(self):
        """Test distributed node initialization."""
        node = DistributedVerifier(
            role=NodeRole.COORDINATOR,
            node_id="test_coordinator"
        )
        
        assert node.role == NodeRole.COORDINATOR
        assert node.node_id == "test_coordinator"
        assert node.my_info.status == NodeStatus.INITIALIZING
        assert node.my_info.node_id == "test_coordinator"
        assert isinstance(node.my_info.capabilities, dict)
    
    def test_node_capabilities_detection(self):
        """Test node capabilities detection."""
        node = DistributedVerifier(role=NodeRole.WORKER)
        capabilities = node._get_node_capabilities()
        
        required_keys = ["cpu_count", "memory_gb", "supported_provers", "supported_models"]
        for key in required_keys:
            assert key in capabilities
        
        assert isinstance(capabilities["cpu_count"], int)
        assert capabilities["cpu_count"] > 0
        assert isinstance(capabilities["memory_gb"], float)
        assert capabilities["memory_gb"] > 0
    
    @pytest.mark.asyncio
    async def test_coordinator_startup(self, coordinator_node):
        """Test coordinator node startup."""
        # Mock aiohttp components
        with patch('aiohttp.web.Application'), \
             patch('aiohttp.web.AppRunner') as mock_runner, \
             patch('aiohttp.web.TCPSite') as mock_site:
            
            mock_runner_instance = AsyncMock()
            mock_runner.return_value = mock_runner_instance
            
            mock_site_instance = AsyncMock()
            mock_site.return_value = mock_site_instance
            
            await coordinator_node._start_coordinator()
            
            assert coordinator_node.my_info.status == NodeStatus.ACTIVE
            mock_runner_instance.setup.assert_called_once()
            mock_site_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_worker_startup(self, worker_node):
        """Test worker node startup."""
        with patch.object(worker_node, '_register_with_coordinator', new_callable=AsyncMock) as mock_register, \
             patch('asyncio.create_task') as mock_create_task:
            
            await worker_node._start_worker()
            
            assert worker_node.my_info.status == NodeStatus.ACTIVE
            mock_register.assert_called_once()
            assert mock_create_task.call_count >= 2  # worker loop + heartbeat
    
    @pytest.mark.asyncio
    async def test_task_submission(self, coordinator_node, sample_task):
        """Test distributed task submission."""
        distributed_task = DistributedTask(task=sample_task, priority=1)
        
        with patch.object(coordinator_node, '_schedule_task', new_callable=AsyncMock) as mock_schedule:
            mock_schedule.return_value = sample_task.task_id
            
            result = await coordinator_node.submit_distributed_task(
                sample_task, priority=1
            )
            
            assert result == sample_task.task_id
            mock_schedule.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_task_scheduling(self, coordinator_node, sample_task):
        """Test task scheduling to workers."""
        # Create mock worker node
        worker_info = NodeInfo(
            node_id="worker_1",
            role=NodeRole.WORKER,
            status=NodeStatus.ACTIVE,
            host="localhost",
            port=8001,
            capabilities={"supported_provers": ["isabelle"]},
            max_capacity=10,
            current_load=2.0
        )
        
        coordinator_node.nodes["worker_1"] = worker_info
        distributed_task = DistributedTask(task=sample_task)
        
        with patch.object(coordinator_node.load_balancer, 'select_best_node', new_callable=AsyncMock) as mock_select, \
             patch.object(coordinator_node, '_send_task_to_worker', new_callable=AsyncMock) as mock_send:
            
            mock_select.return_value = worker_info
            
            task_id = await coordinator_node._schedule_task(distributed_task)
            
            assert task_id == sample_task.task_id
            assert distributed_task.assigned_node == "worker_1"
            assert sample_task.task_id in coordinator_node.active_tasks
            mock_send.assert_called_once_with(worker_info, distributed_task)
    
    @pytest.mark.asyncio
    async def test_task_worker_communication(self, coordinator_node):
        """Test communication between coordinator and worker."""
        worker_info = NodeInfo(
            node_id="worker_1",
            role=NodeRole.WORKER,
            status=NodeStatus.ACTIVE,
            host="localhost",
            port=8001,
            capabilities={}
        )
        
        task = DistributedTask(task=VerificationTask(
            task_id="comm_test",
            hdl_code="test code"
        ))
        
        # Mock HTTP session
        mock_response = AsyncMock()
        mock_response.status = 200
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        coordinator_node.session = mock_session
        
        await coordinator_node._send_task_to_worker(worker_info, task)
        
        assert task.started_at is not None
        mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cluster_status(self, coordinator_node):
        """Test cluster status reporting."""
        # Add mock nodes
        coordinator_node.nodes = {
            "worker_1": NodeInfo(
                node_id="worker_1",
                role=NodeRole.WORKER,
                status=NodeStatus.ACTIVE,
                host="localhost",
                port=8001,
                capabilities={},
                max_capacity=10,
                current_load=3.0
            ),
            "worker_2": NodeInfo(
                node_id="worker_2", 
                role=NodeRole.WORKER,
                status=NodeStatus.BUSY,
                host="localhost",
                port=8002,
                capabilities={},
                max_capacity=8,
                current_load=7.0
            )
        }
        
        # Add mock tasks
        coordinator_node.active_tasks = {"task_1": Mock(), "task_2": Mock()}
        coordinator_node.completed_tasks = {"task_3": Mock()}
        coordinator_node.failed_tasks = {"task_4": Mock()}
        
        status = await coordinator_node.get_cluster_status()
        
        assert status["cluster_size"] == 2
        assert status["active_nodes"] == 1  # Only worker_1 is ACTIVE
        assert status["total_capacity"] == 18
        assert status["current_load"] == 10.0
        assert status["utilization"] == 10.0 / 18
        assert status["active_tasks"] == 2
        assert status["completed_tasks"] == 1
        assert status["failed_tasks"] == 1


class TestIntelligentLoadBalancer:
    """Test intelligent load balancing."""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer for testing."""
        return IntelligentLoadBalancer()
    
    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes for load balancing."""
        return {
            "worker_1": NodeInfo(
                node_id="worker_1",
                role=NodeRole.WORKER,
                status=NodeStatus.ACTIVE,
                host="host1",
                port=8001,
                capabilities={
                    "supported_provers": ["isabelle", "coq"],
                    "supported_models": ["gpt-4-turbo"],
                    "memory_gb": 16.0,
                    "cpu_count": 8
                },
                max_capacity=10,
                current_load=2.0
            ),
            "worker_2": NodeInfo(
                node_id="worker_2",
                role=NodeRole.WORKER, 
                status=NodeStatus.ACTIVE,
                host="host2",
                port=8002,
                capabilities={
                    "supported_provers": ["coq"],
                    "supported_models": ["gpt-4-turbo", "claude-3"],
                    "memory_gb": 32.0,
                    "cpu_count": 16
                },
                max_capacity=15,
                current_load=8.0
            ),
            "worker_3": NodeInfo(
                node_id="worker_3",
                role=NodeRole.WORKER,
                status=NodeStatus.OVERLOADED,  # Should be filtered out
                host="host3",
                port=8003,
                capabilities={},
                max_capacity=5,
                current_load=6.0  # Over capacity
            )
        }
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for load balancing."""
        return VerificationTask(
            task_id="lb_test",
            hdl_code="test code",
            prover="isabelle",
            model="gpt-4-turbo"
        )
    
    @pytest.mark.asyncio
    async def test_node_selection(self, load_balancer, sample_nodes, sample_task):
        """Test intelligent node selection."""
        best_node = await load_balancer.select_best_node(
            sample_nodes, sample_task, []
        )
        
        # Should select an active node that supports the required prover
        assert best_node is not None
        assert best_node.status == NodeStatus.ACTIVE
        assert sample_task.prover in best_node.capabilities.get("supported_provers", [])
        
        # Should not select overloaded node
        assert best_node.node_id != "worker_3"
    
    @pytest.mark.asyncio
    async def test_node_scoring(self, load_balancer, sample_nodes, sample_task):
        """Test node scoring algorithm."""
        worker_1 = sample_nodes["worker_1"]
        worker_2 = sample_nodes["worker_2"]
        
        score_1 = await load_balancer._calculate_node_score(worker_1, sample_task, [])
        score_2 = await load_balancer._calculate_node_score(worker_2, sample_task, [])
        
        # Both should have valid scores
        assert 0 <= score_1 <= 5  # Max possible score
        assert 0 <= score_2 <= 5
        
        # Worker 1 should score higher due to lower load and isabelle support
        assert score_1 > score_2
    
    def test_capability_matching(self, load_balancer, sample_nodes, sample_task):
        """Test capability matching scoring."""
        worker_1 = sample_nodes["worker_1"]  # Supports isabelle
        worker_2 = sample_nodes["worker_2"]  # Doesn't support isabelle
        
        score_1 = load_balancer._calculate_capability_match(worker_1, sample_task)
        score_2 = load_balancer._calculate_capability_match(worker_2, sample_task)
        
        # Worker 1 should score higher due to prover support
        assert score_1 > score_2
    
    def test_affinity_scoring(self, load_balancer, sample_nodes):
        """Test affinity scoring."""
        worker_1 = sample_nodes["worker_1"]
        
        # Test node ID affinity
        affinity_score_1 = load_balancer._calculate_affinity_score(worker_1, ["worker_1"])
        assert affinity_score_1 == 1.0
        
        # Test host affinity
        affinity_score_2 = load_balancer._calculate_affinity_score(worker_1, ["host1"])
        assert affinity_score_2 > 0.5
        
        # Test no affinity
        affinity_score_3 = load_balancer._calculate_affinity_score(worker_1, [])
        assert affinity_score_3 == 0.5
    
    def test_scheduling_history(self, load_balancer, sample_nodes, sample_task):
        """Test scheduling decision recording."""
        initial_history_size = len(load_balancer.scheduling_history)
        
        # Record a decision
        all_scores = [(node, 0.5) for node in sample_nodes.values()]
        load_balancer._record_scheduling_decision(
            sample_nodes["worker_1"], sample_task, all_scores
        )
        
        assert len(load_balancer.scheduling_history) == initial_history_size + 1
        
        latest_record = load_balancer.scheduling_history[-1]
        assert latest_record["selected_node"] == "worker_1"
        assert latest_record["task_type"] == sample_task.prover
        assert "all_scores" in latest_record


class TestAutoScalingPolicy:
    """Test auto-scaling policy."""
    
    @pytest.fixture
    def scaling_policy(self):
        """Create auto-scaling policy for testing."""
        return AutoScalingPolicy()
    
    def test_scale_up_conditions(self, scaling_policy):
        """Test scale-up decision logic."""
        # High utilization should trigger scale up
        assert scaling_policy.should_scale_up(0.9, 10)
        
        # Large queue should trigger scale up
        assert scaling_policy.should_scale_up(0.5, 100)
        
        # Low utilization and small queue should not scale up
        assert not scaling_policy.should_scale_up(0.3, 5)
    
    def test_scale_down_conditions(self, scaling_policy):
        """Test scale-down decision logic."""
        # Low utilization with idle nodes should trigger scale down
        assert scaling_policy.should_scale_down(0.2, 5)
        
        # High utilization should not scale down
        assert not scaling_policy.should_scale_down(0.8, 10)
        
        # Low utilization but few idle nodes should not scale down
        assert not scaling_policy.should_scale_down(0.2, 1)
    
    def test_cooldown_period(self, scaling_policy):
        """Test scaling cooldown period."""
        # Simulate recent scaling action
        scaling_policy.last_scaling_action = time.time()
        
        # Should not scale during cooldown
        assert not scaling_policy.should_scale_up(0.9, 100)
        assert not scaling_policy.should_scale_down(0.1, 10)
        
        # Simulate old scaling action
        scaling_policy.last_scaling_action = time.time() - 400  # Beyond cooldown
        
        # Should allow scaling after cooldown
        assert scaling_policy.should_scale_up(0.9, 100)
    
    def test_load_prediction(self, scaling_policy):
        """Test future load prediction."""
        # Add some history
        scaling_policy.scaling_history = [
            {"utilization": 0.3, "timestamp": time.time() - 100},
            {"utilization": 0.4, "timestamp": time.time() - 80},
            {"utilization": 0.5, "timestamp": time.time() - 60},
            {"utilization": 0.6, "timestamp": time.time() - 40},
            {"utilization": 0.7, "timestamp": time.time() - 20}
        ]
        
        predicted_load = scaling_policy._predict_future_load()
        
        # Should predict increasing load
        assert 0.0 <= predicted_load <= 1.0
        assert predicted_load > 0.5  # Upward trend


class TestPerformanceTracker:
    """Test performance tracking."""
    
    @pytest.fixture
    def tracker(self):
        """Create performance tracker for testing."""
        return PerformanceTracker("test_node")
    
    def test_task_completion_recording(self, tracker):
        """Test task completion recording."""
        initial_count = len(tracker.task_completions)
        
        tracker.record_task_completion(150.0, True)
        tracker.record_task_completion(200.0, False)
        
        assert len(tracker.task_completions) == initial_count + 2
        
        # Check recorded data
        latest_completion = tracker.task_completions[-1]
        assert latest_completion["execution_time_ms"] == 200.0
        assert latest_completion["success"] == False
        assert latest_completion["node_id"] == "test_node"
    
    def test_global_metrics_calculation(self, tracker):
        """Test global metrics calculation."""
        # Add test data
        tracker.task_completions = [
            {
                "timestamp": time.time(),
                "execution_time_ms": 100.0,
                "success": True,
                "node_id": "test_node"
            },
            {
                "timestamp": time.time(),
                "execution_time_ms": 200.0,
                "success": True,
                "node_id": "test_node"
            },
            {
                "timestamp": time.time(),
                "execution_time_ms": 150.0,
                "success": False,
                "node_id": "test_node"
            }
        ]
        
        metrics = tracker.get_global_metrics()
        
        assert metrics["total_tasks"] == 3
        assert metrics["success_rate"] == 2/3  # 2 out of 3 successful
        assert metrics["avg_execution_time_ms"] == 150.0  # Average of 100, 200, 150
        assert metrics["min_execution_time_ms"] == 100.0
        assert metrics["max_execution_time_ms"] == 200.0
        assert metrics["node_count"] == 1
    
    def test_metrics_with_no_data(self, tracker):
        """Test metrics calculation with no data."""
        metrics = tracker.get_global_metrics()
        
        assert "error" in metrics
        assert "No task completion data available" in metrics["error"]
    
    def test_old_data_filtering(self, tracker):
        """Test filtering of old completion data."""
        # Add old data
        old_completion = {
            "timestamp": time.time() - 7200,  # 2 hours ago
            "execution_time_ms": 100.0,
            "success": True,
            "node_id": "test_node"
        }
        
        recent_completion = {
            "timestamp": time.time(),
            "execution_time_ms": 200.0,
            "success": True,
            "node_id": "test_node"
        }
        
        tracker.task_completions = [old_completion, recent_completion]
        
        metrics = tracker.get_global_metrics()
        
        # Should only consider recent data
        assert metrics["total_tasks"] == 1
        assert metrics["avg_execution_time_ms"] == 200.0


class TestDistributedTaskManagement:
    """Test distributed task management."""
    
    def test_distributed_task_creation(self, sample_task):
        """Test distributed task wrapper creation."""
        distributed_task = DistributedTask(
            task=sample_task,
            priority=5,
            affinity_requirements=["fast_nodes"]
        )
        
        assert distributed_task.task == sample_task
        assert distributed_task.priority == 5
        assert "fast_nodes" in distributed_task.affinity_requirements
        assert distributed_task.retry_count == 0
        assert distributed_task.created_at > 0
    
    def test_node_info_structure(self):
        """Test node info data structure."""
        node = NodeInfo(
            node_id="test_node",
            role=NodeRole.WORKER,
            status=NodeStatus.ACTIVE,
            host="testhost",
            port=9000,
            capabilities={"test": "value"}
        )
        
        assert node.node_id == "test_node"
        assert node.role == NodeRole.WORKER
        assert node.status == NodeStatus.ACTIVE
        assert node.current_load == 0.0  # Default
        assert node.max_capacity == 10  # Default
        assert node.last_heartbeat > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_task_submission(self, coordinator_node, sample_task):
        """Test error handling during task submission."""
        # Mock load balancer to return no nodes
        with patch.object(coordinator_node.load_balancer, 'select_best_node', new_callable=AsyncMock) as mock_select:
            mock_select.return_value = None
            
            distributed_task = DistributedTask(task=sample_task)
            
            with pytest.raises(RuntimeError, match="No available worker nodes"):
                await coordinator_node._schedule_task(distributed_task)
    
    @pytest.mark.asyncio
    async def test_node_health_monitoring(self, coordinator_node):
        """Test node health monitoring."""
        # Add a node with old heartbeat
        old_node = NodeInfo(
            node_id="old_worker",
            role=NodeRole.WORKER,
            status=NodeStatus.ACTIVE,
            host="localhost",
            port=8001,
            capabilities={}
        )
        old_node.last_heartbeat = time.time() - 3600  # 1 hour ago
        
        coordinator_node.nodes["old_worker"] = old_node
        
        # Health monitoring should detect and handle stale nodes
        # This would typically be done in the background monitoring loop
        cutoff_time = time.time() - 300  # 5 minutes
        
        stale_nodes = [
            node_id for node_id, node in coordinator_node.nodes.items()
            if node.last_heartbeat < cutoff_time
        ]
        
        assert "old_worker" in stale_nodes