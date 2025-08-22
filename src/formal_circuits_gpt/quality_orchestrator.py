"""Quality Gate Orchestrator for Autonomous SDLC."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

from .progressive_quality_gates import ProgressiveQualityGates, QualityReport
from .monitoring.logger import get_logger
from .monitoring.metrics import MetricsCollector
from .reliability.circuit_breaker import CircuitBreaker
from .reliability.retry_policy import ExponentialBackoff, RetryableOperation


@dataclass
class QualityConfiguration:
    """Configuration for quality gate execution."""
    
    parallel_execution: bool = True
    max_concurrent_gates: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    enable_detailed_logging: bool = True


@dataclass
class ProgressiveQualityResult:
    """Result of progressive quality execution."""
    
    generation: str
    overall_success: bool
    total_score: float
    generation_reports: List[QualityReport]
    execution_time_ms: float
    errors: List[str]
    recommendations: List[str]


class QualityOrchestrator:
    """Orchestrates progressive quality gates with reliability patterns."""
    
    def __init__(self, project_root: Path = None, config: QualityConfiguration = None):
        self.project_root = project_root or Path.cwd()
        self.config = config or QualityConfiguration()
        self.logger = get_logger("quality_orchestrator")
        self.metrics = MetricsCollector()
        
        # Reliability components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            timeout=30.0,
            expected_exception=Exception
        )
        retry_policy = ExponentialBackoff(
            max_attempts=self.config.retry_attempts,
            base_delay=1.0,
            max_delay=10.0,
            multiplier=2.0
        )
        self.retry_operation = RetryableOperation(retry_policy)
        
        self.quality_gates = ProgressiveQualityGates(self.project_root)

    async def execute_progressive_quality_gates(
        self, 
        generations: List[str] = None
    ) -> ProgressiveQualityResult:
        """Execute quality gates across multiple generations."""
        start_time = time.time()
        generations = generations or ["gen1", "gen2", "gen3"]
        
        self.logger.info(f"Starting progressive quality gate execution for {generations}")
        
        generation_reports = []
        errors = []
        recommendations = []
        overall_success = True
        
        for generation in generations:
            try:
                self.logger.info(f"Executing {generation} quality gates")
                
                # Execute with circuit breaker protection
                report = await self._execute_generation_with_protection(generation)
                generation_reports.append(report)
                
                # Collect recommendations
                for gate in report.gates:
                    recommendations.extend(gate.recommendations)
                
                # Check if generation passed
                if not report.overall_passed:
                    overall_success = False
                    self.logger.warning(f"{generation} quality gates failed")
                    
                    # For Generation 1, we can continue with warnings
                    if generation == "gen1" and report.overall_score >= 70.0:
                        self.logger.info(f"{generation} partial pass - continuing")
                        continue
                    
                    # For later generations, failure is critical
                    if generation in ["gen2", "gen3"]:
                        errors.append(f"{generation} quality gates failed")
                        break
                
                self.logger.info(f"{generation} quality gates passed")
                
            except Exception as e:
                error_msg = f"Failed to execute {generation} quality gates: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                overall_success = False
                break
        
        # Calculate overall metrics
        total_score = 0.0
        if generation_reports:
            total_score = sum(r.overall_score for r in generation_reports) / len(generation_reports)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        result = ProgressiveQualityResult(
            generation="+".join(generations),
            overall_success=overall_success,
            total_score=total_score,
            generation_reports=generation_reports,
            execution_time_ms=execution_time_ms,
            errors=errors,
            recommendations=list(set(recommendations))  # Deduplicate
        )
        
        # Save comprehensive report
        await self._save_progressive_report(result)
        
        # Log final results
        status = "SUCCESS" if overall_success else "FAILED"
        self.logger.info(
            f"Progressive quality gates {status}: "
            f"Score {total_score:.1f}/100.0, "
            f"Duration {execution_time_ms:.0f}ms"
        )
        
        return result

    async def _execute_generation_with_protection(self, generation: str) -> QualityReport:
        """Execute generation with circuit breaker and retry protection."""
        
        async def execute_generation():
            return await self.quality_gates.run_generation_gates(generation)
        
        # Wrap with circuit breaker
        protected_execution = self.circuit_breaker(execute_generation)
        
        # Wrap with retry policy
        return await self.retry_operation.execute_async(protected_execution)

    async def validate_environment(self) -> Dict[str, Any]:
        """Validate execution environment before running quality gates."""
        validation_results = {
            "python_version": None,
            "dependencies_available": [],
            "missing_dependencies": [],
            "project_structure_valid": False,
            "test_framework_available": False,
            "linting_tools_available": False,
        }
        
        try:
            # Check Python version
            import sys
            validation_results["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Check critical dependencies
            critical_deps = ["pytest", "flake8", "openai", "anthropic", "pydantic"]
            for dep in critical_deps:
                try:
                    __import__(dep)
                    validation_results["dependencies_available"].append(dep)
                except ImportError:
                    validation_results["missing_dependencies"].append(dep)
            
            # Check project structure
            required_paths = [
                self.project_root / "src",
                self.project_root / "tests",
                self.project_root / "pyproject.toml"
            ]
            validation_results["project_structure_valid"] = all(p.exists() for p in required_paths)
            
            # Check test framework
            validation_results["test_framework_available"] = "pytest" in validation_results["dependencies_available"]
            
            # Check linting tools
            validation_results["linting_tools_available"] = "flake8" in validation_results["dependencies_available"]
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results

    async def get_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quality metrics."""
        try:
            # Get latest quality reports
            reports_dir = self.project_root / "reports" / "quality_gates"
            recent_reports = []
            
            if reports_dir.exists():
                report_files = sorted(reports_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]
                
                for report_file in report_files:
                    try:
                        with open(report_file) as f:
                            report_data = json.load(f)
                            recent_reports.append(report_data)
                    except Exception:
                        continue
            
            # Calculate trend metrics
            if recent_reports:
                scores = [r.get("overall_score", 0) for r in recent_reports]
                avg_score = sum(scores) / len(scores)
                trend = "improving" if len(scores) > 1 and scores[0] > scores[-1] else "stable"
                
                # Gate success rates
                gate_stats = {}
                for report in recent_reports:
                    for gate in report.get("gates", []):
                        gate_name = gate["name"]
                        if gate_name not in gate_stats:
                            gate_stats[gate_name] = {"total": 0, "passed": 0}
                        gate_stats[gate_name]["total"] += 1
                        if gate["passed"]:
                            gate_stats[gate_name]["passed"] += 1
                
                # Calculate success rates
                for gate_name in gate_stats:
                    stats = gate_stats[gate_name]
                    stats["success_rate"] = (stats["passed"] / stats["total"]) * 100.0
                
                return {
                    "average_score": avg_score,
                    "trend": trend,
                    "total_reports": len(recent_reports),
                    "gate_statistics": gate_stats,
                    "last_execution": recent_reports[0].get("timestamp") if recent_reports else None
                }
            
            return {"message": "No quality reports available"}
            
        except Exception as e:
            self.logger.error(f"Failed to get quality metrics: {e}")
            return {"error": str(e)}

    async def _save_progressive_report(self, result: ProgressiveQualityResult):
        """Save comprehensive progressive quality report."""
        try:
            reports_dir = self.project_root / "reports" / "progressive_quality"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"progressive_quality_report_{timestamp}.json"
            report_file = reports_dir / filename
            
            # Convert to serializable format
            report_data = asdict(result)
            
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Progressive quality report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save progressive quality report: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of quality gate system."""
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "checks": {}
        }
        
        try:
            # Check circuit breaker status
            health_status["checks"]["circuit_breaker"] = {
                "status": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count
            }
            
            # Check project structure
            structure_check = await self.validate_environment()
            health_status["checks"]["environment"] = {
                "status": "healthy" if structure_check["project_structure_valid"] else "degraded",
                "details": structure_check
            }
            
            # Check reports directory
            reports_dir = self.project_root / "reports"
            health_status["checks"]["reports_system"] = {
                "status": "healthy" if reports_dir.exists() else "degraded",
                "writable": reports_dir.exists() and reports_dir.is_dir()
            }
            
            # Overall status
            failed_checks = [
                check for check_name, check in health_status["checks"].items()
                if check.get("status") != "healthy"
            ]
            
            if failed_checks:
                health_status["status"] = "degraded"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status


async def main():
    """Main function for standalone execution."""
    import sys
    
    config = QualityConfiguration(
        parallel_execution=True,
        max_concurrent_gates=3,
        enable_detailed_logging=True
    )
    
    orchestrator = QualityOrchestrator(config=config)
    
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        # Health check mode
        health = await orchestrator.health_check()
        print(json.dumps(health, indent=2))
        sys.exit(0 if health["status"] == "healthy" else 1)
    
    # Execute progressive quality gates
    generations = sys.argv[1:] if len(sys.argv) > 1 else ["gen1", "gen2", "gen3"]
    
    result = await orchestrator.execute_progressive_quality_gates(generations)
    
    # Print summary
    print(f"\nPROGRESSIVE QUALITY GATES REPORT")
    print("=" * 60)
    print(f"Status: {'SUCCESS' if result.overall_success else 'FAILED'}")
    print(f"Overall Score: {result.total_score:.1f}/100.0")
    print(f"Duration: {result.execution_time_ms:.0f}ms")
    print(f"Generations: {result.generation}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  • {error}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations[:10]:  # Limit output
            print(f"  • {rec}")
    
    # Exit with appropriate code
    sys.exit(0 if result.overall_success else 1)


if __name__ == "__main__":
    asyncio.run(main())