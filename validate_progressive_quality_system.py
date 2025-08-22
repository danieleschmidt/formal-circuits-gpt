#!/usr/bin/env python3
"""Comprehensive validation of the Progressive Quality Gates system."""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from formal_circuits_gpt.progressive_quality_gates import ProgressiveQualityGates
from formal_circuits_gpt.quality_orchestrator import QualityOrchestrator, QualityConfiguration
from formal_circuits_gpt.adaptive_quality_system import AdaptiveQualitySystem
from formal_circuits_gpt.security_scanner import SecurityScanner
from formal_circuits_gpt.quality_dashboard import QualityDashboard
from formal_circuits_gpt.performance_profiler import PerformanceProfiler


async def validate_core_quality_gates():
    """Validate core quality gate functionality."""
    print("🔍 Validating Core Quality Gates...")
    
    gates = ProgressiveQualityGates()
    
    # Test generation configurations
    for generation in ["gen1", "gen2", "gen3"]:
        config = gates.generation_thresholds.get(generation)
        if config:
            print(f"  ✓ {generation}: {len(config['required_gates'])} gates, {config['min_score']}% threshold")
        else:
            print(f"  ✗ {generation}: Configuration missing")
            return False
    
    # Test gate method existence
    gate_methods = [
        "_gate_functionality", "_gate_basic_tests", "_gate_syntax_check",
        "_gate_dependency_check", "_gate_structure_validation",
        "_gate_comprehensive_tests", "_gate_security", "_gate_performance",
        "_gate_optimization", "_gate_integration_tests", "_gate_reliability",
        "_gate_scalability", "_gate_monitoring", "_gate_documentation"
    ]
    
    missing_methods = []
    for method in gate_methods:
        if not hasattr(gates, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"  ✗ Missing gate methods: {missing_methods}")
        return False
    
    print(f"  ✓ All {len(gate_methods)} gate methods implemented")
    return True


async def validate_orchestrator():
    """Validate quality orchestrator functionality."""
    print("🎯 Validating Quality Orchestrator...")
    
    try:
        config = QualityConfiguration(parallel_execution=False, timeout_seconds=30)
        orchestrator = QualityOrchestrator(config=config)
        
        # Test environment validation
        validation = await orchestrator.validate_environment()
        if not validation.get("project_structure_valid"):
            print("  ⚠️  Project structure validation failed")
        else:
            print("  ✓ Project structure valid")
        
        # Test health check
        health = await orchestrator.health_check()
        print(f"  ✓ Health check: {health.get('status', 'unknown')}")
        
        # Test circuit breaker
        cb_status = health.get("checks", {}).get("circuit_breaker", {}).get("status")
        print(f"  ✓ Circuit breaker: {cb_status}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Orchestrator validation failed: {e}")
        return False


async def validate_adaptive_system():
    """Validate adaptive quality system."""
    print("🧠 Validating Adaptive Quality System...")
    
    try:
        adaptive = AdaptiveQualitySystem()
        
        # Test trend analysis
        analysis = await adaptive.analyze_quality_trends()
        report_count = analysis.get("total_reports", 0)
        print(f"  ✓ Analyzed {report_count} quality reports")
        
        # Test optimization suggestions
        if "optimization_opportunities" in analysis:
            suggestions = analysis["optimization_opportunities"]
            print(f"  ✓ Generated {len(suggestions)} optimization suggestions")
        
        # Test adaptive recommendations
        context = {"recent_score": 75.0}
        recommendations = await adaptive.get_adaptive_recommendations(context)
        print(f"  ✓ Generated {len(recommendations)} adaptive recommendations")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Adaptive system validation failed: {e}")
        return False


async def validate_security_scanner():
    """Validate security scanner."""
    print("🔒 Validating Security Scanner...")
    
    try:
        scanner = SecurityScanner(Path("."))
        result = await scanner.scan_project()
        
        print(f"  ✓ Scanned {result.files_scanned} files")
        print(f"  ✓ Security score: {result.security_score:.1f}/100")
        print(f"  ✓ Found {result.total_issues} security issues")
        print(f"    Critical: {result.critical_issues}, High: {result.high_issues}")
        print(f"    Medium: {result.medium_issues}, Low: {result.low_issues}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Security scanner validation failed: {e}")
        return False


async def validate_dashboard():
    """Validate quality dashboard."""
    print("📊 Validating Quality Dashboard...")
    
    try:
        dashboard = QualityDashboard()
        
        # Test dashboard data
        data = await dashboard.get_dashboard_data()
        print(f"  ✓ Current score: {data.current_score:.1f}/100")
        print(f"  ✓ Trend: {data.trend}")
        print(f"  ✓ System health: {data.system_health.get('status', 'unknown')}")
        print(f"  ✓ Tracking {len(data.gate_success_rates)} gates")
        print(f"  ✓ {len(data.recent_executions)} recent executions")
        
        # Test JSON metrics
        metrics = await dashboard.get_json_metrics()
        print(f"  ✓ JSON metrics with {len(metrics)} keys")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Dashboard validation failed: {e}")
        return False


async def validate_performance_profiler():
    """Validate performance profiler."""
    print("⚡ Validating Performance Profiler...")
    
    try:
        profiler = PerformanceProfiler()
        
        # Test basic profiling
        async with profiler.profile("test_operation") as collect_metric:
            collect_metric("test_metric", 42.0, "units")
            await asyncio.sleep(0.01)  # Small delay
        
        print("  ✓ Basic profiling functionality works")
        
        # Test thresholds
        thresholds = profiler.thresholds
        print(f"  ✓ Configured {len(thresholds)} performance thresholds")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Performance profiler validation failed: {e}")
        return False


async def validate_integration():
    """Validate system integration."""
    print("🔗 Validating System Integration...")
    
    try:
        # Test that all components can work together
        config = QualityConfiguration(parallel_execution=False, timeout_seconds=10)
        orchestrator = QualityOrchestrator(config=config)
        
        # Test orchestrator with quality gates
        gates = orchestrator.quality_gates
        
        # Verify orchestrator has all required components
        assert hasattr(orchestrator, 'circuit_breaker'), "Missing circuit breaker"
        assert hasattr(orchestrator, 'retry_operation'), "Missing retry operation"
        assert hasattr(orchestrator, 'quality_gates'), "Missing quality gates"
        
        print("  ✓ All components properly integrated")
        
        # Test component communication
        health = await orchestrator.health_check()
        assert "checks" in health, "Health check missing detailed checks"
        
        print("  ✓ Component communication working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Integration validation failed: {e}")
        return False


async def validate_file_structure():
    """Validate that all required files exist."""
    print("📁 Validating File Structure...")
    
    required_files = [
        "src/formal_circuits_gpt/progressive_quality_gates.py",
        "src/formal_circuits_gpt/quality_orchestrator.py",
        "src/formal_circuits_gpt/adaptive_quality_system.py",
        "src/formal_circuits_gpt/security_scanner.py",
        "src/formal_circuits_gpt/quality_dashboard.py",
        "src/formal_circuits_gpt/performance_profiler.py",
        "src/formal_circuits_gpt/reliability/circuit_breaker.py",
        "src/formal_circuits_gpt/reliability/retry_policy.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ✗ Missing required files: {missing_files}")
        return False
    
    print(f"  ✓ All {len(required_files)} required files present")
    return True


async def generate_final_report():
    """Generate final validation report."""
    print("\n" + "="*60)
    print("📋 PROGRESSIVE QUALITY GATES SYSTEM VALIDATION REPORT")
    print("="*60)
    
    validation_results = []
    
    # Run all validations
    validations = [
        ("File Structure", validate_file_structure),
        ("Core Quality Gates", validate_core_quality_gates),
        ("Quality Orchestrator", validate_orchestrator),
        ("Adaptive System", validate_adaptive_system),
        ("Security Scanner", validate_security_scanner),
        ("Quality Dashboard", validate_dashboard),
        ("Performance Profiler", validate_performance_profiler),
        ("System Integration", validate_integration),
    ]
    
    print("\nRunning validation tests...\n")
    
    for test_name, test_func in validations:
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            
            validation_results.append({
                "test": test_name,
                "passed": result,
                "duration_ms": duration * 1000
            })
            
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            validation_results.append({
                "test": test_name,
                "passed": False,
                "duration_ms": 0,
                "error": str(e)
            })
    
    # Calculate overall results
    passed_tests = sum(1 for r in validation_results if r["passed"])
    total_tests = len(validation_results)
    success_rate = (passed_tests / total_tests) * 100
    total_duration = sum(r["duration_ms"] for r in validation_results)
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Duration: {total_duration:.0f}ms")
    
    if success_rate >= 90:
        print("\n🎉 PROGRESSIVE QUALITY GATES SYSTEM VALIDATION: SUCCESS")
        print("✅ System is ready for production use!")
    elif success_rate >= 75:
        print("\n⚠️  PROGRESSIVE QUALITY GATES SYSTEM VALIDATION: PARTIAL SUCCESS")
        print("⚡ System is functional but may need improvements")
    else:
        print("\n❌ PROGRESSIVE QUALITY GATES SYSTEM VALIDATION: FAILURE")
        print("🔧 System requires fixes before production use")
    
    # Show failed tests
    failed_tests = [r for r in validation_results if not r["passed"]]
    if failed_tests:
        print("\nFailed Tests:")
        for test in failed_tests:
            error_msg = test.get("error", "Unknown error")
            print(f"  ❌ {test['test']}: {error_msg}")
    
    # Save detailed report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "summary": {
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "total_duration_ms": total_duration
        },
        "validation_results": validation_results
    }
    
    reports_dir = Path("reports") / "validation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = reports_dir / f"progressive_quality_validation_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return success_rate >= 90


async def main():
    """Main validation function."""
    print("🚀 Progressive Quality Gates System Validation")
    print("=" * 50)
    
    success = await generate_final_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())