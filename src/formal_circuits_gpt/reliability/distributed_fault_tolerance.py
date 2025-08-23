"""
Distributed Fault Tolerance System

Advanced fault tolerance mechanisms for distributed formal verification
including Byzantine fault tolerance, consensus protocols, and self-healing capabilities.
"""

import asyncio
import json
import time
import uuid
import hashlib
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

from ..monitoring.logger import get_logger
from ..core import CircuitVerifier, ProofResult


class NodeStatus(Enum):
    """Status of a verification node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    SUSPICIOUS = "suspicious"


class ConsensusAlgorithm(Enum):
    """Consensus algorithms for distributed verification."""
    RAFT = "raft"
    PBFT = "pbft"  # Practical Byzantine Fault Tolerance
    POW = "proof_of_work"
    POS = "proof_of_stake"


@dataclass
class VerificationNode:
    """Represents a node in the distributed verification system."""
    node_id: str
    endpoint: str
    status: NodeStatus
    last_heartbeat: float
    reputation_score: float
    processing_power: float
    verification_count: int
    success_rate: float
    byzantine_evidence: List[str]
    role: str = "worker"  # leader, worker, validator


@dataclass
class DistributedTask:
    """Represents a distributed verification task."""
    task_id: str
    circuit_hash: str
    properties: List[str]
    assigned_nodes: List[str]
    redundancy_factor: int
    deadline: float
    priority: int
    consensus_threshold: float
    results: Dict[str, Any]
    status: str = "pending"


@dataclass
class ConsensusResult:
    """Result of consensus protocol execution."""
    task_id: str
    agreed_result: Optional[ProofResult]
    confidence_score: float
    participating_nodes: List[str]
    byzantine_nodes: List[str]
    execution_time: float
    consensus_rounds: int


class DistributedFaultTolerance:
    """
    Advanced distributed fault tolerance system for formal verification
    with Byzantine fault tolerance and self-healing capabilities.
    """

    def __init__(
        self,
        node_id: str,
        consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.PBFT,
        byzantine_tolerance: float = 0.33,
        healing_enabled: bool = True
    ):
        self.node_id = node_id
        self.consensus_algorithm = consensus_algorithm
        self.byzantine_tolerance = byzantine_tolerance
        self.healing_enabled = healing_enabled
        
        self.logger = get_logger("distributed_fault_tolerance")
        
        # Distributed system state
        self.nodes: Dict[str, VerificationNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.consensus_cache: Dict[str, ConsensusResult] = {}
        
        # Fault detection and recovery
        self.fault_detector = FaultDetector(self)
        self.recovery_manager = RecoveryManager(self)
        self.byzantine_detector = ByzantineDetector(self)
        
        # Performance monitoring
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_consensus": 0,
            "byzantine_attacks_detected": 0,
            "nodes_recovered": 0,
            "average_consensus_time": 0.0
        }
        
        # Consensus parameters
        self.consensus_timeout = 300.0  # 5 minutes
        self.min_nodes_for_consensus = 3
        self.max_consensus_rounds = 10
        
        self.logger.info(f"Initialized distributed fault tolerance system (node: {node_id})")

    async def register_node(
        self, 
        node_id: str, 
        endpoint: str, 
        processing_power: float = 1.0
    ) -> bool:
        """Register a new verification node in the distributed system."""
        try:
            node = VerificationNode(
                node_id=node_id,
                endpoint=endpoint,
                status=NodeStatus.HEALTHY,
                last_heartbeat=time.time(),
                reputation_score=1.0,
                processing_power=processing_power,
                verification_count=0,
                success_rate=1.0,
                byzantine_evidence=[]
            )
            
            self.nodes[node_id] = node
            
            # Send welcome message and sync state
            await self._sync_node_state(node_id)
            
            self.logger.info(f"Node {node_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node_id}: {e}")
            return False

    async def submit_distributed_verification(
        self,
        circuit_code: str,
        properties: List[str],
        redundancy_factor: int = 3,
        priority: int = 1,
        deadline_minutes: int = 60
    ) -> str:
        """Submit a verification task for distributed processing."""
        task_id = str(uuid.uuid4())
        circuit_hash = hashlib.sha256(circuit_code.encode()).hexdigest()
        
        # Select nodes for task assignment
        assigned_nodes = await self._select_nodes_for_task(
            redundancy_factor, properties, priority
        )
        
        if len(assigned_nodes) < self.min_nodes_for_consensus:
            raise RuntimeError(f"Insufficient nodes for consensus (need {self.min_nodes_for_consensus}, have {len(assigned_nodes)})")
        
        task = DistributedTask(
            task_id=task_id,
            circuit_hash=circuit_hash,
            properties=properties,
            assigned_nodes=assigned_nodes,
            redundancy_factor=redundancy_factor,
            deadline=time.time() + (deadline_minutes * 60),
            priority=priority,
            consensus_threshold=0.67,  # 2/3 majority
            results={}
        )
        
        self.tasks[task_id] = task
        
        # Distribute task to selected nodes
        await self._distribute_task(task, circuit_code)
        
        self.logger.info(f"Distributed verification task {task_id} submitted to {len(assigned_nodes)} nodes")
        return task_id

    async def wait_for_consensus(
        self, 
        task_id: str, 
        timeout: Optional[float] = None
    ) -> ConsensusResult:
        """Wait for consensus on a distributed verification task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        timeout = timeout or (task.deadline - time.time())
        
        start_time = time.time()
        consensus_round = 0
        
        self.logger.info(f"Starting consensus for task {task_id}")
        
        while consensus_round < self.max_consensus_rounds and (time.time() - start_time) < timeout:
            consensus_round += 1
            
            # Collect results from nodes
            await self._collect_node_results(task)
            
            # Detect Byzantine behavior
            byzantine_nodes = await self._detect_byzantine_behavior(task)
            
            # Run consensus algorithm
            consensus_result = await self._run_consensus_algorithm(
                task, byzantine_nodes, consensus_round
            )
            
            if consensus_result.agreed_result is not None:
                # Consensus reached
                self.consensus_cache[task_id] = consensus_result
                self._update_performance_metrics(consensus_result)
                
                self.logger.info(f"Consensus reached for task {task_id} in round {consensus_round}")
                return consensus_result
            
            # If no consensus, wait and retry
            await asyncio.sleep(5.0)
        
        # Consensus failed
        self.logger.warning(f"Consensus failed for task {task_id} after {consensus_round} rounds")
        
        return ConsensusResult(
            task_id=task_id,
            agreed_result=None,
            confidence_score=0.0,
            participating_nodes=task.assigned_nodes,
            byzantine_nodes=[],
            execution_time=time.time() - start_time,
            consensus_rounds=consensus_round
        )

    async def _select_nodes_for_task(
        self, 
        redundancy_factor: int, 
        properties: List[str], 
        priority: int
    ) -> List[str]:
        """Select optimal nodes for task execution based on capabilities and reputation."""
        
        # Filter healthy nodes
        healthy_nodes = [
            node for node in self.nodes.values() 
            if node.status == NodeStatus.HEALTHY and node.reputation_score > 0.5
        ]
        
        if len(healthy_nodes) < redundancy_factor:
            # Include degraded nodes if necessary
            degraded_nodes = [
                node for node in self.nodes.values()
                if node.status == NodeStatus.DEGRADED and node.reputation_score > 0.3
            ]
            healthy_nodes.extend(degraded_nodes)
        
        # Score nodes based on multiple factors
        node_scores = []
        for node in healthy_nodes:
            score = self._calculate_node_score(node, properties, priority)
            node_scores.append((node.node_id, score))
        
        # Sort by score and select top nodes
        node_scores.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = [node_id for node_id, _ in node_scores[:redundancy_factor * 2]]  # Select extra for redundancy
        
        return selected_nodes[:min(len(selected_nodes), redundancy_factor + 2)]

    def _calculate_node_score(
        self, 
        node: VerificationNode, 
        properties: List[str], 
        priority: int
    ) -> float:
        """Calculate suitability score for a node for given task."""
        score = 0.0
        
        # Base reputation score (0.0 - 1.0)
        score += node.reputation_score * 0.3
        
        # Success rate (0.0 - 1.0)
        score += node.success_rate * 0.25
        
        # Processing power (relative)
        score += min(node.processing_power / 2.0, 1.0) * 0.2
        
        # Availability (based on recent heartbeat)
        time_since_heartbeat = time.time() - node.last_heartbeat
        availability = max(0.0, 1.0 - (time_since_heartbeat / 300.0))  # 5 min threshold
        score += availability * 0.15
        
        # Experience with similar tasks
        if node.verification_count > 0:
            experience_factor = min(node.verification_count / 100.0, 1.0)
            score += experience_factor * 0.1
        
        return score

    async def _distribute_task(self, task: DistributedTask, circuit_code: str):
        """Distribute verification task to assigned nodes."""
        distribution_tasks = []
        
        for node_id in task.assigned_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                distribution_task = self._send_task_to_node(
                    node, task, circuit_code
                )
                distribution_tasks.append(distribution_task)
        
        # Send tasks in parallel
        results = await asyncio.gather(*distribution_tasks, return_exceptions=True)
        
        # Check for failed distributions
        failed_nodes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                node_id = task.assigned_nodes[i]
                failed_nodes.append(node_id)
                self.logger.warning(f"Failed to send task to node {node_id}: {result}")
        
        # Remove failed nodes from assignment
        for node_id in failed_nodes:
            if node_id in task.assigned_nodes:
                task.assigned_nodes.remove(node_id)
                if node_id in self.nodes:
                    self.nodes[node_id].status = NodeStatus.DEGRADED

    async def _send_task_to_node(
        self, 
        node: VerificationNode, 
        task: DistributedTask, 
        circuit_code: str
    ):
        """Send verification task to a specific node."""
        try:
            # In a real implementation, this would use network communication
            # For simulation, we'll use local processing
            
            task_payload = {
                "task_id": task.task_id,
                "circuit_code": circuit_code,
                "properties": task.properties,
                "deadline": task.deadline,
                "priority": task.priority
            }
            
            # Simulate network delay
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Update node status
            node.last_heartbeat = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send task to node {node.node_id}: {e}")
            raise

    async def _collect_node_results(self, task: DistributedTask):
        """Collect verification results from assigned nodes."""
        collection_tasks = []
        
        for node_id in task.assigned_nodes:
            if node_id in self.nodes:
                collection_task = self._collect_result_from_node(node_id, task)
                collection_tasks.append(collection_task)
        
        # Collect results in parallel
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Process collected results
        for i, result in enumerate(results):
            node_id = task.assigned_nodes[i]
            if isinstance(result, dict) and not isinstance(result, Exception):
                task.results[node_id] = result
            elif isinstance(result, Exception):
                self.logger.warning(f"Failed to collect result from node {node_id}: {result}")

    async def _collect_result_from_node(
        self, 
        node_id: str, 
        task: DistributedTask
    ) -> Dict[str, Any]:
        """Collect verification result from a specific node."""
        try:
            node = self.nodes[node_id]
            
            # Simulate verification execution
            await asyncio.sleep(random.uniform(1.0, 5.0))
            
            # Simulate different result scenarios
            success_probability = node.success_rate * (1.0 - len(node.byzantine_evidence) * 0.1)
            
            if random.random() < success_probability:
                # Successful verification
                status = "VERIFIED" if random.random() > 0.3 else "FAILED"
                result = {
                    "node_id": node_id,
                    "status": status,
                    "proof_hash": hashlib.sha256(f"{task.task_id}_{node_id}_{status}".encode()).hexdigest(),
                    "execution_time": random.uniform(10.0, 60.0),
                    "confidence": random.uniform(0.8, 1.0),
                    "timestamp": time.time()
                }
                
                # Update node statistics
                node.verification_count += 1
                node.last_heartbeat = time.time()
                
                return result
            else:
                # Failed verification or timeout
                raise TimeoutError(f"Node {node_id} failed to complete verification")
                
        except Exception as e:
            self.logger.error(f"Failed to collect result from node {node_id}: {e}")
            # Mark node as potentially problematic
            if node_id in self.nodes:
                self.nodes[node_id].status = NodeStatus.DEGRADED
            raise

    async def _detect_byzantine_behavior(self, task: DistributedTask) -> List[str]:
        """Detect Byzantine behavior among participating nodes."""
        byzantine_nodes = []
        
        if len(task.results) < 2:
            return byzantine_nodes
        
        # Analyze result consistency
        result_groups = {}
        
        for node_id, result in task.results.items():
            result_key = (result.get("status", ""), result.get("proof_hash", ""))
            if result_key not in result_groups:
                result_groups[result_key] = []
            result_groups[result_key].append(node_id)
        
        # Identify outliers
        if len(result_groups) > 1:
            group_sizes = [(len(nodes), result_key, nodes) for result_key, nodes in result_groups.items()]
            group_sizes.sort(reverse=True)
            
            # Majority group
            majority_size, majority_key, majority_nodes = group_sizes[0]
            
            # Check for suspicious minorities
            for size, key, nodes in group_sizes[1:]:
                if size < majority_size * 0.5:  # Suspicious if less than half of majority
                    for node_id in nodes:
                        byzantine_nodes.append(node_id)
                        if node_id in self.nodes:
                            self.nodes[node_id].byzantine_evidence.append(
                                f"Inconsistent result in task {task.task_id}"
                            )
        
        # Additional Byzantine detection heuristics
        for node_id, result in task.results.items():
            # Check for suspiciously fast completion
            execution_time = result.get("execution_time", 0)
            if execution_time < 1.0:  # Too fast to be realistic
                if node_id not in byzantine_nodes:
                    byzantine_nodes.append(node_id)
                    if node_id in self.nodes:
                        self.nodes[node_id].byzantine_evidence.append(
                            f"Suspiciously fast execution in task {task.task_id}"
                        )
            
            # Check for repeated identical results (possible replay attack)
            proof_hash = result.get("proof_hash", "")
            identical_count = sum(1 for r in task.results.values() if r.get("proof_hash") == proof_hash)
            if identical_count > 1 and proof_hash:
                if node_id not in byzantine_nodes:
                    byzantine_nodes.append(node_id)
                    if node_id in self.nodes:
                        self.nodes[node_id].byzantine_evidence.append(
                            f"Duplicate proof hash in task {task.task_id}"
                        )
        
        if byzantine_nodes:
            self.logger.warning(f"Detected Byzantine behavior from nodes: {byzantine_nodes}")
            self.performance_metrics["byzantine_attacks_detected"] += len(byzantine_nodes)
        
        return byzantine_nodes

    async def _run_consensus_algorithm(
        self, 
        task: DistributedTask, 
        byzantine_nodes: List[str], 
        round_number: int
    ) -> ConsensusResult:
        """Run the consensus algorithm to agree on verification result."""
        
        if self.consensus_algorithm == ConsensusAlgorithm.PBFT:
            return await self._run_pbft_consensus(task, byzantine_nodes, round_number)
        elif self.consensus_algorithm == ConsensusAlgorithm.RAFT:
            return await self._run_raft_consensus(task, byzantine_nodes, round_number)
        else:
            # Default simple majority voting
            return await self._run_majority_consensus(task, byzantine_nodes, round_number)

    async def _run_pbft_consensus(
        self, 
        task: DistributedTask, 
        byzantine_nodes: List[str], 
        round_number: int
    ) -> ConsensusResult:
        """Run Practical Byzantine Fault Tolerance consensus."""
        start_time = time.time()
        
        # Filter out Byzantine nodes
        honest_results = {
            node_id: result for node_id, result in task.results.items()
            if node_id not in byzantine_nodes
        }
        
        if len(honest_results) < self.min_nodes_for_consensus:
            return ConsensusResult(
                task_id=task.task_id,
                agreed_result=None,
                confidence_score=0.0,
                participating_nodes=list(honest_results.keys()),
                byzantine_nodes=byzantine_nodes,
                execution_time=time.time() - start_time,
                consensus_rounds=round_number
            )
        
        # PBFT requires 3f+1 nodes to tolerate f Byzantine nodes
        max_byzantine = len(honest_results) // 3
        
        if len(byzantine_nodes) > max_byzantine:
            self.logger.error(f"Too many Byzantine nodes detected: {len(byzantine_nodes)} > {max_byzantine}")
            return ConsensusResult(
                task_id=task.task_id,
                agreed_result=None,
                confidence_score=0.0,
                participating_nodes=list(honest_results.keys()),
                byzantine_nodes=byzantine_nodes,
                execution_time=time.time() - start_time,
                consensus_rounds=round_number
            )
        
        # Count votes for each result
        vote_counts = {}
        for node_id, result in honest_results.items():
            vote_key = result.get("status", "UNKNOWN")
            if vote_key not in vote_counts:
                vote_counts[vote_key] = []
            vote_counts[vote_key].append((node_id, result))
        
        # Find result with 2f+1 majority (where f is max Byzantine nodes)
        required_votes = 2 * max_byzantine + 1
        
        for vote_key, votes in vote_counts.items():
            if len(votes) >= required_votes:
                # Consensus reached
                consensus_confidence = len(votes) / len(honest_results)
                
                # Create agreed result from majority vote
                agreed_result = self._create_consensus_result(votes, task.task_id)
                
                return ConsensusResult(
                    task_id=task.task_id,
                    agreed_result=agreed_result,
                    confidence_score=consensus_confidence,
                    participating_nodes=list(honest_results.keys()),
                    byzantine_nodes=byzantine_nodes,
                    execution_time=time.time() - start_time,
                    consensus_rounds=round_number
                )
        
        # No consensus reached
        return ConsensusResult(
            task_id=task.task_id,
            agreed_result=None,
            confidence_score=0.0,
            participating_nodes=list(honest_results.keys()),
            byzantine_nodes=byzantine_nodes,
            execution_time=time.time() - start_time,
            consensus_rounds=round_number
        )

    async def _run_majority_consensus(
        self, 
        task: DistributedTask, 
        byzantine_nodes: List[str], 
        round_number: int
    ) -> ConsensusResult:
        """Run simple majority voting consensus."""
        start_time = time.time()
        
        # Filter out Byzantine nodes
        honest_results = {
            node_id: result for node_id, result in task.results.items()
            if node_id not in byzantine_nodes
        }
        
        if len(honest_results) < self.min_nodes_for_consensus:
            return ConsensusResult(
                task_id=task.task_id,
                agreed_result=None,
                confidence_score=0.0,
                participating_nodes=list(honest_results.keys()),
                byzantine_nodes=byzantine_nodes,
                execution_time=time.time() - start_time,
                consensus_rounds=round_number
            )
        
        # Count votes
        vote_counts = {}
        for node_id, result in honest_results.items():
            vote_key = result.get("status", "UNKNOWN")
            if vote_key not in vote_counts:
                vote_counts[vote_key] = []
            vote_counts[vote_key].append((node_id, result))
        
        # Find majority
        majority_threshold = len(honest_results) / 2
        
        for vote_key, votes in vote_counts.items():
            if len(votes) > majority_threshold:
                # Majority reached
                consensus_confidence = len(votes) / len(honest_results)
                
                agreed_result = self._create_consensus_result(votes, task.task_id)
                
                return ConsensusResult(
                    task_id=task.task_id,
                    agreed_result=agreed_result,
                    confidence_score=consensus_confidence,
                    participating_nodes=list(honest_results.keys()),
                    byzantine_nodes=byzantine_nodes,
                    execution_time=time.time() - start_time,
                    consensus_rounds=round_number
                )
        
        # No majority
        return ConsensusResult(
            task_id=task.task_id,
            agreed_result=None,
            confidence_score=0.0,
            participating_nodes=list(honest_results.keys()),
            byzantine_nodes=byzantine_nodes,
            execution_time=time.time() - start_time,
            consensus_rounds=round_number
        )

    def _create_consensus_result(
        self, 
        votes: List[Tuple[str, Dict[str, Any]]], 
        task_id: str
    ) -> ProofResult:
        """Create a consensus ProofResult from majority votes."""
        # Use the first vote as template and aggregate information
        _, template_result = votes[0]
        
        status = template_result.get("status", "UNKNOWN")
        confidence_scores = [vote[1].get("confidence", 0.8) for vote in votes]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Create a synthetic proof result
        # In a real implementation, this would be more sophisticated
        return ProofResult(
            status=status,
            proof_code=f"Consensus proof for task {task_id}",
            errors=[],
            properties_verified=[],
            verification_id=task_id,
            duration_ms=0.0,
            refinement_attempts=0
        )

    async def _sync_node_state(self, node_id: str):
        """Synchronize state with a newly registered node."""
        try:
            # Send current system state to the new node
            state_sync = {
                "system_info": {
                    "total_nodes": len(self.nodes),
                    "active_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
                    "consensus_algorithm": self.consensus_algorithm.value
                },
                "node_registry": {
                    nid: {
                        "status": node.status.value,
                        "reputation": node.reputation_score
                    } for nid, node in self.nodes.items() if nid != node_id
                }
            }
            
            # In a real implementation, this would be sent over network
            self.logger.debug(f"State synchronized with node {node_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to sync state with node {node_id}: {e}")

    def _update_performance_metrics(self, consensus_result: ConsensusResult):
        """Update system performance metrics."""
        self.performance_metrics["total_tasks"] += 1
        
        if consensus_result.agreed_result is not None:
            self.performance_metrics["successful_consensus"] += 1
        
        # Update average consensus time
        old_avg = self.performance_metrics["average_consensus_time"]
        new_avg = (old_avg * (self.performance_metrics["total_tasks"] - 1) + 
                  consensus_result.execution_time) / self.performance_metrics["total_tasks"]
        self.performance_metrics["average_consensus_time"] = new_avg
        
        # Update node reputations based on participation
        for node_id in consensus_result.participating_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if consensus_result.agreed_result is not None:
                    # Reward successful participation
                    node.reputation_score = min(1.0, node.reputation_score + 0.01)
                else:
                    # Small penalty for failed consensus
                    node.reputation_score = max(0.0, node.reputation_score - 0.005)
        
        # Penalize Byzantine nodes
        for node_id in consensus_result.byzantine_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.reputation_score = max(0.0, node.reputation_score - 0.1)
                node.status = NodeStatus.SUSPICIOUS

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "system_status": "healthy",
            "total_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.HEALTHY]),
            "degraded_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.DEGRADED]),
            "failed_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED]),
            "suspicious_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.SUSPICIOUS]),
            "active_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "performance_metrics": self.performance_metrics.copy(),
            "byzantine_tolerance_met": True
        }
        
        # Check Byzantine tolerance
        healthy_count = health_status["healthy_nodes"]
        total_count = health_status["total_nodes"]
        
        if total_count > 0:
            byzantine_ratio = (health_status["suspicious_nodes"] + health_status["failed_nodes"]) / total_count
            health_status["byzantine_tolerance_met"] = byzantine_ratio <= self.byzantine_tolerance
            
            if byzantine_ratio > self.byzantine_tolerance:
                health_status["system_status"] = "degraded"
        
        # Check if minimum nodes for consensus are available
        if healthy_count < self.min_nodes_for_consensus:
            health_status["system_status"] = "critical"
        
        return health_status

    async def self_heal(self):
        """Perform self-healing operations."""
        if not self.healing_enabled:
            return
        
        self.logger.info("Starting self-healing operations")
        
        # Identify nodes that need recovery
        nodes_to_recover = [
            node_id for node_id, node in self.nodes.items()
            if node.status in [NodeStatus.DEGRADED, NodeStatus.FAILED]
        ]
        
        recovery_tasks = []
        for node_id in nodes_to_recover:
            recovery_task = self.recovery_manager.attempt_node_recovery(node_id)
            recovery_tasks.append(recovery_task)
        
        if recovery_tasks:
            recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
            
            successful_recoveries = sum(1 for result in recovery_results if result is True)
            self.performance_metrics["nodes_recovered"] += successful_recoveries
            
            self.logger.info(f"Self-healing completed: {successful_recoveries}/{len(recovery_tasks)} nodes recovered")

    def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        return {
            "system_id": self.node_id,
            "consensus_algorithm": self.consensus_algorithm.value,
            "byzantine_tolerance": self.byzantine_tolerance,
            "nodes": {
                node_id: {
                    "status": node.status.value,
                    "reputation_score": node.reputation_score,
                    "verification_count": node.verification_count,
                    "success_rate": node.success_rate,
                    "byzantine_evidence_count": len(node.byzantine_evidence)
                } for node_id, node in self.nodes.items()
            },
            "performance_metrics": self.performance_metrics,
            "active_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "consensus_cache_size": len(self.consensus_cache)
        }


class FaultDetector:
    """Detects faults in the distributed system."""
    
    def __init__(self, fault_tolerance_system):
        self.system = fault_tolerance_system
        self.logger = get_logger("fault_detector")
    
    async def monitor_nodes(self):
        """Continuously monitor node health."""
        while True:
            try:
                current_time = time.time()
                
                for node_id, node in self.system.nodes.items():
                    # Check heartbeat timeout
                    if current_time - node.last_heartbeat > 300:  # 5 minutes
                        if node.status == NodeStatus.HEALTHY:
                            node.status = NodeStatus.DEGRADED
                            self.logger.warning(f"Node {node_id} marked as degraded due to heartbeat timeout")
                    
                    # Check reputation threshold
                    if node.reputation_score < 0.3:
                        if node.status != NodeStatus.FAILED:
                            node.status = NodeStatus.FAILED
                            self.logger.warning(f"Node {node_id} marked as failed due to low reputation")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in node monitoring: {e}")
                await asyncio.sleep(60)


class RecoveryManager:
    """Manages recovery of failed nodes."""
    
    def __init__(self, fault_tolerance_system):
        self.system = fault_tolerance_system
        self.logger = get_logger("recovery_manager")
    
    async def attempt_node_recovery(self, node_id: str) -> bool:
        """Attempt to recover a degraded or failed node."""
        try:
            if node_id not in self.system.nodes:
                return False
            
            node = self.system.nodes[node_id]
            
            self.logger.info(f"Attempting recovery of node {node_id}")
            
            # Simulate recovery process
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Recovery success probability based on node state
            success_prob = 0.8 if node.status == NodeStatus.DEGRADED else 0.3
            
            if random.random() < success_prob:
                # Recovery successful
                node.status = NodeStatus.HEALTHY
                node.reputation_score = min(1.0, node.reputation_score + 0.1)
                node.last_heartbeat = time.time()
                node.byzantine_evidence.clear()  # Clear evidence after recovery
                
                self.logger.info(f"Node {node_id} recovered successfully")
                return True
            else:
                # Recovery failed
                self.logger.warning(f"Recovery failed for node {node_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during recovery of node {node_id}: {e}")
            return False


class ByzantineDetector:
    """Detects Byzantine behavior patterns."""
    
    def __init__(self, fault_tolerance_system):
        self.system = fault_tolerance_system
        self.logger = get_logger("byzantine_detector")
        self.behavior_patterns = {}
    
    def analyze_node_behavior(self, node_id: str, task_results: List[Dict[str, Any]]):
        """Analyze historical behavior patterns for Byzantine detection."""
        if node_id not in self.behavior_patterns:
            self.behavior_patterns[node_id] = {
                "result_consistency": [],
                "timing_patterns": [],
                "cooperation_score": 1.0
            }
        
        patterns = self.behavior_patterns[node_id]
        
        # Analyze result consistency
        for result in task_results:
            patterns["result_consistency"].append(result.get("status", "UNKNOWN"))
            patterns["timing_patterns"].append(result.get("execution_time", 0))
        
        # Keep only recent history
        if len(patterns["result_consistency"]) > 100:
            patterns["result_consistency"] = patterns["result_consistency"][-100:]
            patterns["timing_patterns"] = patterns["timing_patterns"][-100:]
        
        # Detect suspicious patterns
        self._detect_suspicious_patterns(node_id, patterns)
    
    def _detect_suspicious_patterns(self, node_id: str, patterns: Dict[str, Any]):
        """Detect suspicious behavior patterns."""
        suspicion_score = 0.0
        
        # Check for excessive failures
        if len(patterns["result_consistency"]) >= 10:
            failure_rate = patterns["result_consistency"].count("FAILED") / len(patterns["result_consistency"])
            if failure_rate > 0.5:
                suspicion_score += 0.3
        
        # Check for timing anomalies
        if len(patterns["timing_patterns"]) >= 10:
            avg_time = sum(patterns["timing_patterns"]) / len(patterns["timing_patterns"])
            extremely_fast = sum(1 for t in patterns["timing_patterns"] if t < avg_time * 0.1)
            if extremely_fast / len(patterns["timing_patterns"]) > 0.2:
                suspicion_score += 0.4
        
        # Update cooperation score
        patterns["cooperation_score"] = max(0.0, patterns["cooperation_score"] - suspicion_score)
        
        # Mark as suspicious if score drops too low
        if patterns["cooperation_score"] < 0.3 and node_id in self.system.nodes:
            self.system.nodes[node_id].status = NodeStatus.SUSPICIOUS
            self.logger.warning(f"Node {node_id} marked as suspicious due to behavior analysis")