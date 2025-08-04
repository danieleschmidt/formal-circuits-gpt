"""Parallel verification system for high-throughput processing."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass
import threading
import multiprocessing
from pathlib import Path

from ..core import CircuitVerifier, ProofResult
from ..monitoring.logger import get_logger
from ..cache.optimized_cache import OptimizedCacheManager


@dataclass
class VerificationTask:
    """Individual verification task."""
    task_id: str
    hdl_code: str
    properties: Optional[List[str]] = None
    prover: str = "isabelle"
    model: str = "gpt-4-turbo"
    temperature: float = 0.1
    timeout: int = 300
    priority: int = 0  # Higher numbers = higher priority
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}


@dataclass
class VerificationResult:
    """Result of parallel verification."""
    task_id: str
    proof_result: Optional[ProofResult]
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}


class VerificationWorker:
    """Individual verification worker."""
    
    def __init__(self, worker_id: str, shared_cache: bool = True):
        self.worker_id = worker_id
        self.logger = get_logger(f"worker_{worker_id}")
        
        # Initialize worker-specific verifier
        self.verifier = CircuitVerifier(strict_mode=False, debug_mode=False)
        
        # Shared cache if enabled
        if shared_cache:
            self.cache_manager = OptimizedCacheManager()
        else:
            self.cache_manager = None
        
        self.tasks_completed = 0
        self.total_execution_time = 0.0
        
        self.logger.info(f"Verification worker {worker_id} initialized")
    
    def process_task(self, task: VerificationTask) -> VerificationResult:
        """Process a single verification task."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing task {task.task_id}")
            
            # Check cache first if available
            cached_result = None
            if self.cache_manager:
                cached_result = self.cache_manager.get_proof_cache(
                    task.hdl_code, task.prover, task.model, task.properties or []
                )
            
            if cached_result:
                self.logger.debug(f"Task {task.task_id} served from cache")
                execution_time = (time.time() - start_time) * 1000
                
                return VerificationResult(
                    task_id=task.task_id,
                    proof_result=cached_result,
                    success=True,
                    execution_time_ms=execution_time,
                    worker_id=self.worker_id,
                    metadata={"cached": True}
                )
            
            # Perform verification
            verifier = CircuitVerifier(
                prover=task.prover,
                model=task.model,
                temperature=task.temperature,
                strict_mode=False  # Workers use relaxed validation for performance
            )
            
            result = verifier.verify(
                hdl_code=task.hdl_code,
                properties=task.properties,
                timeout=task.timeout
            )
            
            # Cache result if successful and cache is available
            if self.cache_manager and result.status == "VERIFIED":
                self.cache_manager.put_proof_cache(
                    task.hdl_code, task.prover, task.model, 
                    task.properties or [], result
                )
            
            execution_time = (time.time() - start_time) * 1000
            self.tasks_completed += 1
            self.total_execution_time += execution_time
            
            self.logger.debug(f"Task {task.task_id} completed successfully in {execution_time:.2f}ms")
            
            return VerificationResult(
                task_id=task.task_id,
                proof_result=result,
                success=True,
                execution_time_ms=execution_time,
                worker_id=self.worker_id,
                metadata={"cached": False}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            return VerificationResult(
                task_id=task.task_id,
                proof_result=None,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                worker_id=self.worker_id
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        avg_time = self.total_execution_time / self.tasks_completed if self.tasks_completed > 0 else 0
        
        return {
            "worker_id": self.worker_id,
            "tasks_completed": self.tasks_completed,
            "total_execution_time_ms": self.total_execution_time,
            "average_execution_time_ms": avg_time
        }


class ParallelVerifier:
    """High-performance parallel verification system."""
    
    def __init__(
        self,
        max_workers: int = None,
        use_process_pool: bool = False,
        shared_cache: bool = True,
        task_queue_size: int = 1000
    ):
        """Initialize parallel verifier.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_process_pool: Use process pool instead of thread pool
            shared_cache: Enable shared caching between workers
            task_queue_size: Maximum size of task queue
        """
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.use_process_pool = use_process_pool
        self.shared_cache = shared_cache
        self.task_queue_size = task_queue_size
        
        self.logger = get_logger("parallel_verifier")
        
        # Initialize executor
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            self.logger.info(f"Initialized with {self.max_workers} processes")
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self.logger.info(f"Initialized with {self.max_workers} threads")
        
        # Task management
        self.task_queue: List[VerificationTask] = []
        self.active_tasks: Dict[str, Future] = {}
        self.completed_results: Dict[str, VerificationResult] = {}
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Global cache if enabled
        if shared_cache:
            self.cache_manager = OptimizedCacheManager()
        else:
            self.cache_manager = None
    
    def submit_task(self, task: VerificationTask) -> str:
        """Submit a verification task for parallel processing."""
        with self.lock:
            if len(self.task_queue) >= self.task_queue_size:
                raise RuntimeError("Task queue is full")
            
            self.task_queue.append(task)
            self.stats["tasks_submitted"] += 1
            
            self.logger.debug(f"Task {task.task_id} submitted to queue")
            
            return task.task_id
    
    def submit_batch(self, tasks: List[VerificationTask]) -> List[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []
        
        with self.lock:
            for task in tasks:
                if len(self.task_queue) >= self.task_queue_size:
                    break
                
                self.task_queue.append(task)
                task_ids.append(task.task_id)
                self.stats["tasks_submitted"] += 1
        
        self.logger.info(f"Submitted batch of {len(task_ids)} tasks")
        return task_ids
    
    def process_queue(self, max_concurrent: Optional[int] = None) -> None:
        """Process all tasks in queue with limited concurrency."""
        if max_concurrent is None:
            max_concurrent = self.max_workers
        
        with self.lock:
            # Sort by priority (higher priority first)
            self.task_queue.sort(key=lambda t: t.priority, reverse=True)
            
            # Submit tasks to executor
            while self.task_queue and len(self.active_tasks) < max_concurrent:
                task = self.task_queue.pop(0)
                
                if self.use_process_pool:
                    # For process pool, we need to serialize the work
                    future = self.executor.submit(self._process_task_wrapper, task)
                else:
                    # For thread pool, we can pass the task directly
                    future = self.executor.submit(self._process_task_in_thread, task)
                
                self.active_tasks[task.task_id] = future
                
                self.logger.debug(f"Started processing task {task.task_id}")
    
    def _process_task_wrapper(self, task: VerificationTask) -> VerificationResult:
        """Wrapper for process pool execution."""
        # Create worker in new process
        worker = VerificationWorker(f"proc_{multiprocessing.current_process().pid}")
        return worker.process_task(task)
    
    def _process_task_in_thread(self, task: VerificationTask) -> VerificationResult:
        """Process task in thread pool."""
        worker_id = f"thread_{threading.current_thread().ident}"
        
        # Create or reuse worker
        if not hasattr(threading.current_thread(), 'verification_worker'):
            threading.current_thread().verification_worker = VerificationWorker(
                worker_id, shared_cache=self.shared_cache
            )
        
        worker = threading.current_thread().verification_worker
        return worker.process_task(task)
    
    def get_completed_results(self, timeout: float = 0.1) -> Dict[str, VerificationResult]:
        """Get completed verification results."""
        completed = {}
        
        with self.lock:
            # Check for completed tasks
            done_tasks = []
            
            for task_id, future in self.active_tasks.items():
                if future.done():
                    try:
                        result = future.result(timeout=timeout)
                        completed[task_id] = result
                        self.completed_results[task_id] = result
                        
                        # Update statistics
                        self.stats["tasks_completed"] += 1
                        self.stats["total_execution_time"] += result.execution_time_ms
                        
                        if not result.success:
                            self.stats["tasks_failed"] += 1
                        
                        if result.metadata.get("cached", False):
                            self.stats["cache_hits"] += 1
                        else:
                            self.stats["cache_misses"] += 1
                        
                        done_tasks.append(task_id)
                        
                    except Exception as e:
                        self.logger.error(f"Task {task_id} failed with exception: {str(e)}")
                        
                        # Create error result
                        error_result = VerificationResult(
                            task_id=task_id,
                            proof_result=None,
                            success=False,
                            error_message=str(e)
                        )
                        
                        completed[task_id] = error_result
                        self.completed_results[task_id] = error_result
                        self.stats["tasks_failed"] += 1
                        done_tasks.append(task_id)
            
            # Remove completed tasks from active list
            for task_id in done_tasks:
                del self.active_tasks[task_id]
        
        return completed
    
    def wait_for_completion(self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None) -> Dict[str, VerificationResult]:
        """Wait for specific tasks or all tasks to complete."""
        start_time = time.time()
        results = {}
        
        if task_ids is None:
            # Wait for all active tasks
            target_tasks = set(self.active_tasks.keys())
        else:
            target_tasks = set(task_ids)
        
        while target_tasks:
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Timeout waiting for tasks: {target_tasks}")
                break
            
            # Get completed results
            completed = self.get_completed_results()
            
            # Update results and remove completed tasks
            for task_id, result in completed.items():
                if task_id in target_tasks:
                    results[task_id] = result
                    target_tasks.remove(task_id)
            
            # Process more tasks if queue not empty
            self.process_queue()
            
            # Small delay to prevent busy waiting
            time.sleep(0.01)
        
        return results
    
    def verify_batch(
        self,
        hdl_codes: List[str],
        properties_list: Optional[List[List[str]]] = None,
        prover: str = "isabelle",
        model: str = "gpt-4-turbo",
        temperature: float = 0.1,
        timeout: int = 300,
        max_concurrent: Optional[int] = None
    ) -> List[VerificationResult]:
        """Verify multiple HDL codes in parallel."""
        
        # Create tasks
        tasks = []
        for i, hdl_code in enumerate(hdl_codes):
            properties = properties_list[i] if properties_list and i < len(properties_list) else None
            
            task = VerificationTask(
                task_id=f"batch_{int(time.time())}_{i}",
                hdl_code=hdl_code,
                properties=properties,
                prover=prover,
                model=model,
                temperature=temperature,
                timeout=timeout
            )
            tasks.append(task)
        
        # Submit and process
        task_ids = self.submit_batch(tasks)
        self.process_queue(max_concurrent)
        
        # Wait for completion
        results_dict = self.wait_for_completion(task_ids)
        
        # Return results in order
        results = []
        for task_id in task_ids:
            if task_id in results_dict:
                results.append(results_dict[task_id])
            else:
                # Create timeout result
                results.append(VerificationResult(
                    task_id=task_id,
                    proof_result=None,
                    success=False,
                    error_message="Task timeout or not completed"
                ))
        
        return results
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and processing status."""
        with self.lock:
            return {
                "queued_tasks": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_results),
                "max_workers": self.max_workers,
                "executor_type": "process" if self.use_process_pool else "thread"
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.lock:
            total_tasks = self.stats["tasks_completed"] + self.stats["tasks_failed"]
            success_rate = self.stats["tasks_completed"] / total_tasks if total_tasks > 0 else 0
            avg_time = self.stats["total_execution_time"] / self.stats["tasks_completed"] if self.stats["tasks_completed"] > 0 else 0
            
            cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
            cache_hit_rate = self.stats["cache_hits"] / cache_total if cache_total > 0 else 0
            
            stats = self.stats.copy()
            stats.update({
                "success_rate": success_rate,
                "average_execution_time_ms": avg_time,
                "cache_hit_rate": cache_hit_rate,
                "throughput_tasks_per_second": self.stats["tasks_completed"] / (self.stats["total_execution_time"] / 1000) if self.stats["total_execution_time"] > 0 else 0
            })
            
            return stats
    
    def shutdown(self, wait: bool = True):
        """Shutdown the parallel verifier."""
        self.logger.info("Shutting down parallel verifier")
        
        if wait:
            # Wait for active tasks to complete
            self.wait_for_completion()
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        self.logger.info("Parallel verifier shutdown complete")