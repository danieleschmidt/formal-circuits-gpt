"""Performance Profiler for Quality Gates."""

import asyncio
import time
import gc
import sys
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Performance monitoring will be limited.")
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import logging
import statistics

from .monitoring.logger import get_logger


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    
    name: str
    value: float
    unit: str
    timestamp: float
    context: Dict[str, Any] = None


@dataclass
class PerformanceProfile:
    """Performance profile for a code execution."""
    
    operation_name: str
    duration_ms: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    disk_io_mb: float
    network_io_mb: float
    gc_collections: int
    metrics: List[PerformanceMetric]
    success: bool
    error: str = None


@dataclass
class BenchmarkResult:
    """Result of performance benchmark."""
    
    benchmark_name: str
    iterations: int
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    ops_per_second: float
    memory_usage_mb: float
    profiles: List[PerformanceProfile]


class PerformanceProfiler:
    """Advanced performance profiler for quality gates."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.logger = get_logger("performance_profiler")
        self.active_profiles: Dict[str, Dict] = {}
        self.baseline_metrics: Dict[str, float] = {}
        
        # Performance thresholds
        self.thresholds = {
            "max_execution_time_ms": 5000,
            "max_memory_usage_mb": 500,
            "max_cpu_percent": 80,
            "min_ops_per_second": 10
        }

    @asynccontextmanager
    async def profile(self, operation_name: str):
        """Context manager for profiling code execution."""
        profile_id = f"{operation_name}_{time.time()}"
        
        # Initialize tracking
        process = None
        initial_memory = 0
        initial_cpu_times = None
        initial_io = None
        
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                initial_cpu_times = process.cpu_times()
                initial_io = process.io_counters() if hasattr(process, 'io_counters') else None
            except Exception:
                pass
        gc_before = gc.get_count()
        
        start_time = time.perf_counter()
        metrics = []
        
        self.active_profiles[profile_id] = {
            "start_time": start_time,
            "initial_memory": initial_memory,
            "process": process,
            "metrics": metrics
        }
        
        success = False
        error = None
        
        try:
            yield self._create_metric_collector(profile_id)
            success = True
            
        except Exception as e:
            error = str(e)
            self.logger.error(f"Error during profiling {operation_name}: {e}")
            raise
            
        finally:
            # Calculate final metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            try:
                final_memory = initial_memory
                memory_delta = 0
                cpu_percent = 0
                disk_io_mb = 0
                network_io_mb = 0
                
                if HAS_PSUTIL and process:
                    try:
                        final_memory = process.memory_info().rss / 1024 / 1024
                        memory_delta = final_memory - initial_memory
                        
                        # CPU usage (approximate)
                        if initial_cpu_times:
                            final_cpu_times = process.cpu_times()
                            cpu_time_delta = (final_cpu_times.user + final_cpu_times.system) - (initial_cpu_times.user + initial_cpu_times.system)
                            cpu_percent = (cpu_time_delta / (duration_ms / 1000)) * 100 if duration_ms > 0 else 0
                        
                        # I/O metrics
                        if initial_io and hasattr(process, 'io_counters'):
                            try:
                                final_io = process.io_counters()
                                disk_io_mb = (final_io.read_bytes + final_io.write_bytes - initial_io.read_bytes - initial_io.write_bytes) / 1024 / 1024
                            except:
                                pass
                    except Exception:
                        pass
                
                # Garbage collection
                gc_after = gc.get_count()
                gc_collections = sum(gc_after) - sum(gc_before)
                
                profile = PerformanceProfile(
                    operation_name=operation_name,
                    duration_ms=duration_ms,
                    memory_peak_mb=max(final_memory, initial_memory),
                    memory_delta_mb=memory_delta,
                    cpu_percent=min(cpu_percent, 100),  # Cap at 100%
                    disk_io_mb=disk_io_mb,
                    network_io_mb=network_io_mb,
                    gc_collections=gc_collections,
                    metrics=metrics,
                    success=success,
                    error=error
                )
                
                # Log performance data
                self.logger.info(
                    f"Profile {operation_name}: {duration_ms:.1f}ms, "
                    f"{memory_delta:+.1f}MB memory, {cpu_percent:.1f}% CPU"
                )
                
                # Save profile
                await self._save_profile(profile)
                
            except Exception as e:
                self.logger.error(f"Error finalizing profile for {operation_name}: {e}")
            
            finally:
                # Cleanup
                if profile_id in self.active_profiles:
                    del self.active_profiles[profile_id]

    def _create_metric_collector(self, profile_id: str):
        """Create a metric collector for the active profile."""
        def collect_metric(name: str, value: float, unit: str, context: Dict[str, Any] = None):
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=time.time(),
                context=context or {}
            )
            
            if profile_id in self.active_profiles:
                self.active_profiles[profile_id]["metrics"].append(metric)
        
        return collect_metric

    async def benchmark(
        self, 
        operation: Callable,
        name: str,
        iterations: int = 10,
        warmup_iterations: int = 3
    ) -> BenchmarkResult:
        """Benchmark an operation with multiple iterations."""
        self.logger.info(f"Starting benchmark '{name}' with {iterations} iterations")
        
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(operation):
                    await operation()
                else:
                    operation()
            except Exception:
                pass  # Ignore warmup errors
        
        # Benchmark runs
        times = []
        profiles = []
        total_memory = 0
        
        for i in range(iterations):
            try:
                async with self.profile(f"{name}_iter_{i}") as collect_metric:
                    start_time = time.perf_counter()
                    
                    if asyncio.iscoroutinefunction(operation):
                        await operation()
                    else:
                        operation()
                    
                    end_time = time.perf_counter()
                    iteration_time = (end_time - start_time) * 1000
                    times.append(iteration_time)
                    
                    # Collect memory usage
                    memory_mb = 0
                    if HAS_PSUTIL:
                        try:
                            process = psutil.Process()
                            memory_mb = process.memory_info().rss / 1024 / 1024
                        except Exception:
                            pass
                    total_memory += memory_mb
                    
                    collect_metric("iteration_time", iteration_time, "ms")
                    collect_metric("memory_usage", memory_mb, "MB")
                    
            except Exception as e:
                self.logger.error(f"Benchmark iteration {i} failed: {e}")
                times.append(float('inf'))  # Mark as failed
        
        # Filter out failed iterations
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            raise RuntimeError(f"All benchmark iterations failed for '{name}'")
        
        # Calculate statistics
        min_time = min(valid_times)
        max_time = max(valid_times)
        avg_time = statistics.mean(valid_times)
        median_time = statistics.median(valid_times)
        std_dev = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
        
        # Operations per second
        ops_per_second = 1000 / avg_time if avg_time > 0 else 0
        
        # Average memory usage
        avg_memory = total_memory / len(valid_times) if valid_times else 0
        
        result = BenchmarkResult(
            benchmark_name=name,
            iterations=len(valid_times),
            min_time_ms=min_time,
            max_time_ms=max_time,
            avg_time_ms=avg_time,
            median_time_ms=median_time,
            std_dev_ms=std_dev,
            ops_per_second=ops_per_second,
            memory_usage_mb=avg_memory,
            profiles=profiles
        )
        
        self.logger.info(
            f"Benchmark '{name}' completed: {avg_time:.1f}ms avg, "
            f"{ops_per_second:.1f} ops/sec, {avg_memory:.1f}MB memory"
        )
        
        await self._save_benchmark(result)
        return result

    async def profile_quality_gates(self) -> Dict[str, Any]:
        """Profile the performance of quality gate operations."""
        from .progressive_quality_gates import ProgressiveQualityGates
        
        results = {}
        gates = ProgressiveQualityGates(self.project_root)
        
        # Profile individual gates
        gate_methods = [
            "_gate_functionality",
            "_gate_basic_tests", 
            "_gate_syntax_check",
            "_gate_dependency_check",
            "_gate_structure_validation"
        ]
        
        for gate_method in gate_methods:
            if hasattr(gates, gate_method):
                try:
                    method = getattr(gates, gate_method)
                    
                    benchmark_result = await self.benchmark(
                        operation=method,
                        name=gate_method,
                        iterations=5,
                        warmup_iterations=1
                    )
                    
                    results[gate_method] = {
                        "avg_time_ms": benchmark_result.avg_time_ms,
                        "memory_usage_mb": benchmark_result.memory_usage_mb,
                        "ops_per_second": benchmark_result.ops_per_second,
                        "passed_threshold": benchmark_result.avg_time_ms < self.thresholds["max_execution_time_ms"]
                    }
                    
                except Exception as e:
                    self.logger.error(f"Failed to profile {gate_method}: {e}")
                    results[gate_method] = {"error": str(e)}
        
        # Profile complete generation execution
        try:
            async def run_gen1():
                return await gates.run_generation_gates("gen1")
            
            gen1_benchmark = await self.benchmark(
                operation=run_gen1,
                name="complete_gen1_execution",
                iterations=3,
                warmup_iterations=1
            )
            
            results["complete_generation"] = {
                "avg_time_ms": gen1_benchmark.avg_time_ms,
                "memory_usage_mb": gen1_benchmark.memory_usage_mb,
                "ops_per_second": gen1_benchmark.ops_per_second
            }
            
        except Exception as e:
            self.logger.error(f"Failed to profile complete generation: {e}")
            results["complete_generation"] = {"error": str(e)}
        
        return results

    async def detect_performance_issues(self, profiles: List[PerformanceProfile]) -> List[Dict[str, Any]]:
        """Detect performance issues from profiles."""
        issues = []
        
        for profile in profiles:
            # Check execution time
            if profile.duration_ms > self.thresholds["max_execution_time_ms"]:
                issues.append({
                    "type": "slow_execution",
                    "operation": profile.operation_name,
                    "severity": "high" if profile.duration_ms > self.thresholds["max_execution_time_ms"] * 2 else "medium",
                    "value": profile.duration_ms,
                    "threshold": self.thresholds["max_execution_time_ms"],
                    "recommendation": "Optimize algorithm or add caching"
                })
            
            # Check memory usage
            if profile.memory_peak_mb > self.thresholds["max_memory_usage_mb"]:
                issues.append({
                    "type": "high_memory_usage",
                    "operation": profile.operation_name,
                    "severity": "medium",
                    "value": profile.memory_peak_mb,
                    "threshold": self.thresholds["max_memory_usage_mb"],
                    "recommendation": "Reduce memory footprint or implement streaming"
                })
            
            # Check CPU usage
            if profile.cpu_percent > self.thresholds["max_cpu_percent"]:
                issues.append({
                    "type": "high_cpu_usage",
                    "operation": profile.operation_name,
                    "severity": "low",
                    "value": profile.cpu_percent,
                    "threshold": self.thresholds["max_cpu_percent"],
                    "recommendation": "Consider parallelization or algorithm optimization"
                })
            
            # Check for excessive garbage collection
            if profile.gc_collections > 10:
                issues.append({
                    "type": "excessive_gc",
                    "operation": profile.operation_name,
                    "severity": "low",
                    "value": profile.gc_collections,
                    "threshold": 10,
                    "recommendation": "Reduce object creation or optimize memory usage"
                })
        
        return issues

    async def _save_profile(self, profile: PerformanceProfile):
        """Save performance profile to file."""
        try:
            profiles_dir = self.project_root / "reports" / "performance"
            profiles_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"profile_{profile.operation_name}_{timestamp}.json"
            profile_file = profiles_dir / filename
            
            profile_data = asdict(profile)
            
            import json
            with open(profile_file, "w") as f:
                json.dump(profile_data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save profile: {e}")

    async def _save_benchmark(self, benchmark: BenchmarkResult):
        """Save benchmark result to file."""
        try:
            benchmarks_dir = self.project_root / "reports" / "benchmarks"
            benchmarks_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"benchmark_{benchmark.benchmark_name}_{timestamp}.json"
            benchmark_file = benchmarks_dir / filename
            
            benchmark_data = asdict(benchmark)
            
            import json
            with open(benchmark_file, "w") as f:
                json.dump(benchmark_data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save benchmark: {e}")

    async def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        # Run performance analysis
        gate_performance = await self.profile_quality_gates()
        
        # Load recent profiles
        profiles_dir = self.project_root / "reports" / "performance"
        recent_profiles = []
        
        if profiles_dir.exists():
            profile_files = sorted(
                profiles_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:10]
            
            for profile_file in profile_files:
                try:
                    import json
                    with open(profile_file) as f:
                        profile_data = json.load(f)
                        recent_profiles.append(profile_data)
                except Exception:
                    continue
        
        report = f"""
# Performance Analysis Report

## Quality Gates Performance

"""
        
        for gate_name, metrics in gate_performance.items():
            if "error" not in metrics:
                status = "✅ PASS" if metrics.get("passed_threshold", False) else "⚠️ SLOW"
                report += f"""
### {gate_name}
- **Status**: {status}
- **Average Time**: {metrics['avg_time_ms']:.1f}ms
- **Memory Usage**: {metrics['memory_usage_mb']:.1f}MB
- **Operations/Second**: {metrics['ops_per_second']:.1f}

"""
            else:
                report += f"""
### {gate_name}
- **Status**: ❌ ERROR
- **Error**: {metrics['error']}

"""
        
        # Performance summary
        valid_metrics = [m for m in gate_performance.values() if "error" not in m]
        if valid_metrics:
            avg_time = statistics.mean([m['avg_time_ms'] for m in valid_metrics])
            avg_memory = statistics.mean([m['memory_usage_mb'] for m in valid_metrics])
            
            report += f"""
## Summary
- **Average Gate Execution Time**: {avg_time:.1f}ms
- **Average Memory Usage**: {avg_memory:.1f}MB
- **Total Gates Analyzed**: {len(valid_metrics)}

"""
        
        # Recent performance trends
        if recent_profiles:
            report += """
## Recent Performance Trends

| Operation | Duration (ms) | Memory (MB) | Success |
|-----------|---------------|-------------|---------|
"""
            for profile in recent_profiles[:5]:
                success_icon = "✅" if profile.get("success", False) else "❌"
                report += f"| {profile.get('operation_name', 'Unknown')} | {profile.get('duration_ms', 0):.1f} | {profile.get('memory_peak_mb', 0):.1f} | {success_icon} |\n"
        
        return report

    def set_performance_baseline(self, baseline_metrics: Dict[str, float]):
        """Set performance baseline for comparison."""
        self.baseline_metrics = baseline_metrics.copy()
        self.logger.info(f"Set performance baseline with {len(baseline_metrics)} metrics")

    def compare_to_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current performance to baseline."""
        comparison = {
            "improved": [],
            "degraded": [],
            "unchanged": [],
            "baseline_coverage": 0.0
        }
        
        if not self.baseline_metrics:
            return comparison
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]
                change_percent = ((current_value - baseline_value) / baseline_value * 100) if baseline_value != 0 else 0
                
                if abs(change_percent) < 5:  # Within 5% tolerance
                    comparison["unchanged"].append({
                        "metric": metric_name,
                        "change_percent": change_percent
                    })
                elif change_percent < 0:  # Performance improved (assuming lower is better for time)
                    comparison["improved"].append({
                        "metric": metric_name,
                        "change_percent": change_percent
                    })
                else:  # Performance degraded
                    comparison["degraded"].append({
                        "metric": metric_name,
                        "change_percent": change_percent
                    })
        
        comparison["baseline_coverage"] = len([m for m in current_metrics if m in self.baseline_metrics]) / len(self.baseline_metrics) * 100
        
        return comparison


async def main():
    """Main function for performance profiling."""
    import sys
    
    profiler = PerformanceProfiler()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "profile":
            # Profile quality gates
            results = await profiler.profile_quality_gates()
            print("Quality Gates Performance:")
            for gate, metrics in results.items():
                if "error" not in metrics:
                    print(f"  {gate}: {metrics['avg_time_ms']:.1f}ms")
                else:
                    print(f"  {gate}: ERROR - {metrics['error']}")
        
        elif command == "report":
            # Generate performance report
            report = await profiler.generate_performance_report()
            print(report)
        
        else:
            print("Usage: python performance_profiler.py [profile|report]")
    
    else:
        print("Performance Profiler")
        print("Available commands: profile, report")


if __name__ == "__main__":
    asyncio.run(main())