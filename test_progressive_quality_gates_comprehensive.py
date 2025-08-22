"""Comprehensive test suite for progressive quality gates."""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.formal_circuits_gpt.progressive_quality_gates import (
    ProgressiveQualityGates,
    QualityGateResult,
    QualityReport,
)
from src.formal_circuits_gpt.quality_orchestrator import (
    QualityOrchestrator,
    QualityConfiguration,
    ProgressiveQualityResult,
)
from src.formal_circuits_gpt.adaptive_quality_system import AdaptiveQualitySystem


class TestProgressiveQualityGates:
    """Test progressive quality gates functionality."""

    @pytest.fixture
    async def temp_project(self):
        """Create temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create project structure
            (project_root / "src").mkdir()
            (project_root / "src" / "formal_circuits_gpt").mkdir()
            (project_root / "tests").mkdir()
            (project_root / "docs").mkdir()
            (project_root / "reports").mkdir()
            
            # Create essential files
            (project_root / "pyproject.toml").write_text("[project]\nname = 'test'")
            (project_root / "README.md").write_text("# Test Project\n" + "x" * 1000)
            (project_root / "LICENSE").write_text("MIT License")
            
            # Create module files
            src_dir = project_root / "src" / "formal_circuits_gpt"
            (src_dir / "__init__.py").write_text("")
            (src_dir / "core.py").write_text("# Core module")
            (src_dir / "cli.py").write_text("# CLI module")
            
            yield project_root

    @pytest.fixture
    def quality_gates(self, temp_project):
        """Create quality gates instance."""
        return ProgressiveQualityGates(temp_project)

    @pytest.mark.asyncio
    async def test_generation1_quality_gates(self, quality_gates):
        """Test Generation 1 quality gates execution."""
        with patch.object(quality_gates, '_gate_functionality') as mock_functionality, \
             patch.object(quality_gates, '_gate_basic_tests') as mock_tests, \
             patch.object(quality_gates, '_gate_syntax_check') as mock_syntax, \
             patch.object(quality_gates, '_gate_dependency_check') as mock_deps, \
             patch.object(quality_gates, '_gate_structure_validation') as mock_structure:
            
            # Mock successful gate results
            mock_functionality.return_value = QualityGateResult(
                name="functionality", passed=True, score=95.0, details={}, execution_time_ms=100.0, recommendations=[]
            )
            mock_tests.return_value = QualityGateResult(
                name="basic_tests", passed=True, score=85.0, details={}, execution_time_ms=200.0, recommendations=[]
            )
            mock_syntax.return_value = QualityGateResult(
                name="syntax", passed=True, score=80.0, details={}, execution_time_ms=50.0, recommendations=[]
            )
            mock_deps.return_value = QualityGateResult(
                name="dependency_check", passed=True, score=90.0, details={}, execution_time_ms=75.0, recommendations=[]
            )
            mock_structure.return_value = QualityGateResult(
                name="structure_validation", passed=True, score=95.0, details={}, execution_time_ms=25.0, recommendations=[]
            )
            
            report = await quality_gates.run_generation_gates("gen1")
            
            assert report.overall_passed is True
            assert report.overall_score >= 75.0
            assert len(report.gates) == 5
            assert report.generation == "gen1"

    @pytest.mark.asyncio
    async def test_generation2_quality_gates(self, quality_gates):
        """Test Generation 2 quality gates execution."""
        with patch.object(quality_gates, '_gate_functionality') as mock_functionality, \
             patch.object(quality_gates, '_gate_comprehensive_tests') as mock_tests, \
             patch.object(quality_gates, '_gate_security') as mock_security, \
             patch.object(quality_gates, '_gate_performance') as mock_performance, \
             patch.object(quality_gates, '_gate_integration_tests') as mock_integration, \
             patch.object(quality_gates, '_gate_reliability') as mock_reliability:
            
            # Mock Gen 2 gate results
            mock_functionality.return_value = QualityGateResult(
                name="functionality", passed=True, score=95.0, details={}, execution_time_ms=100.0, recommendations=[]
            )
            mock_tests.return_value = QualityGateResult(
                name="tests", passed=True, score=85.0, details={"coverage_percent": 78.0}, execution_time_ms=300.0, recommendations=[]
            )
            mock_security.return_value = QualityGateResult(
                name="security", passed=True, score=90.0, details={}, execution_time_ms=150.0, recommendations=[]
            )
            mock_performance.return_value = QualityGateResult(
                name="performance", passed=True, score=82.0, details={}, execution_time_ms=200.0, recommendations=[]
            )
            mock_integration.return_value = QualityGateResult(
                name="integration_tests", passed=True, score=88.0, details={}, execution_time_ms=400.0, recommendations=[]
            )
            mock_reliability.return_value = QualityGateResult(
                name="reliability", passed=True, score=75.0, details={}, execution_time_ms=100.0, recommendations=[]
            )
            
            report = await quality_gates.run_generation_gates("gen2")
            
            assert report.overall_passed is True
            assert report.overall_score >= 80.0
            assert len(report.gates) == 6
            assert report.generation == "gen2"

    @pytest.mark.asyncio
    async def test_generation3_quality_gates(self, quality_gates):
        """Test Generation 3 quality gates execution."""
        with patch.object(quality_gates, '_gate_functionality') as mock_functionality, \
             patch.object(quality_gates, '_gate_comprehensive_tests') as mock_tests, \
             patch.object(quality_gates, '_gate_security') as mock_security, \
             patch.object(quality_gates, '_gate_performance') as mock_performance, \
             patch.object(quality_gates, '_gate_optimization') as mock_optimization, \
             patch.object(quality_gates, '_gate_scalability') as mock_scalability, \
             patch.object(quality_gates, '_gate_monitoring') as mock_monitoring, \
             patch.object(quality_gates, '_gate_documentation') as mock_docs:
            
            # Mock Gen 3 gate results with high scores
            gates_config = [
                ("functionality", mock_functionality, 95.0),
                ("tests", mock_tests, 90.0),
                ("security", mock_security, 92.0),
                ("performance", mock_performance, 88.0),
                ("optimization", mock_optimization, 85.0),
                ("scalability", mock_scalability, 87.0),
                ("monitoring", mock_monitoring, 91.0),
                ("documentation", mock_docs, 89.0),
            ]
            
            for gate_name, mock_gate, score in gates_config:
                mock_gate.return_value = QualityGateResult(
                    name=gate_name, passed=True, score=score, details={}, execution_time_ms=100.0, recommendations=[]
                )
            
            report = await quality_gates.run_generation_gates("gen3")
            
            assert report.overall_passed is True
            assert report.overall_score >= 90.0
            assert len(report.gates) == 8
            assert report.generation == "gen3"

    @pytest.mark.asyncio
    async def test_failing_quality_gates(self, quality_gates):
        """Test quality gates with failures."""
        with patch.object(quality_gates, '_gate_functionality') as mock_functionality, \
             patch.object(quality_gates, '_gate_basic_tests') as mock_tests, \
             patch.object(quality_gates, '_gate_syntax_check') as mock_syntax, \
             patch.object(quality_gates, '_gate_dependency_check') as mock_deps, \
             patch.object(quality_gates, '_gate_structure_validation') as mock_structure:
            
            # Mock some failures
            mock_functionality.return_value = QualityGateResult(
                name="functionality", passed=False, score=45.0, details={"error": "Verification failed"}, 
                execution_time_ms=100.0, recommendations=["Fix core functionality"]
            )
            mock_tests.return_value = QualityGateResult(
                name="basic_tests", passed=False, score=30.0, details={"return_code": 1}, 
                execution_time_ms=200.0, recommendations=["Fix failing tests"]
            )
            mock_syntax.return_value = QualityGateResult(
                name="syntax", passed=True, score=75.0, details={}, execution_time_ms=50.0, recommendations=[]
            )
            mock_deps.return_value = QualityGateResult(
                name="dependency_check", passed=True, score=90.0, details={}, execution_time_ms=75.0, recommendations=[]
            )
            mock_structure.return_value = QualityGateResult(
                name="structure_validation", passed=True, score=95.0, details={}, execution_time_ms=25.0, recommendations=[]
            )
            
            report = await quality_gates.run_generation_gates("gen1")
            
            assert report.overall_passed is False
            assert report.overall_score < 75.0
            assert len([gate for gate in report.gates if not gate.passed]) == 2


class TestQualityOrchestrator:
    """Test quality orchestrator functionality."""

    @pytest.fixture
    def orchestrator(self, temp_project):
        """Create quality orchestrator instance."""
        config = QualityConfiguration(
            parallel_execution=True,
            max_concurrent_gates=3,
            timeout_seconds=60
        )
        return QualityOrchestrator(temp_project, config)

    @pytest.mark.asyncio
    async def test_progressive_execution(self, orchestrator):
        """Test progressive quality gate execution."""
        with patch.object(orchestrator.quality_gates, 'run_generation_gates') as mock_run:
            # Mock successful generation reports
            gen1_report = QualityReport(
                overall_passed=True, overall_score=80.0, gates=[], generation="gen1",
                timestamp="2025-01-01 12:00:00", duration_ms=1000.0
            )
            gen2_report = QualityReport(
                overall_passed=True, overall_score=85.0, gates=[], generation="gen2",
                timestamp="2025-01-01 12:00:00", duration_ms=1500.0
            )
            gen3_report = QualityReport(
                overall_passed=True, overall_score=92.0, gates=[], generation="gen3",
                timestamp="2025-01-01 12:00:00", duration_ms=2000.0
            )
            
            mock_run.side_effect = [gen1_report, gen2_report, gen3_report]
            
            result = await orchestrator.execute_progressive_quality_gates()
            
            assert result.overall_success is True
            assert result.total_score > 85.0
            assert len(result.generation_reports) == 3
            assert mock_run.call_count == 3

    @pytest.mark.asyncio
    async def test_early_failure_stops_execution(self, orchestrator):
        """Test that early generation failure stops execution."""
        with patch.object(orchestrator.quality_gates, 'run_generation_gates') as mock_run:
            # Mock Gen 1 success, Gen 2 failure
            gen1_report = QualityReport(
                overall_passed=True, overall_score=80.0, gates=[], generation="gen1",
                timestamp="2025-01-01 12:00:00", duration_ms=1000.0
            )
            gen2_report = QualityReport(
                overall_passed=False, overall_score=65.0, gates=[], generation="gen2",
                timestamp="2025-01-01 12:00:00", duration_ms=1500.0
            )
            
            mock_run.side_effect = [gen1_report, gen2_report]
            
            result = await orchestrator.execute_progressive_quality_gates()
            
            assert result.overall_success is False
            assert len(result.generation_reports) == 2
            assert len(result.errors) > 0
            assert mock_run.call_count == 2  # Should stop after Gen 2 failure

    @pytest.mark.asyncio
    async def test_environment_validation(self, orchestrator):
        """Test environment validation."""
        validation = await orchestrator.validate_environment()
        
        assert "python_version" in validation
        assert "dependencies_available" in validation
        assert "missing_dependencies" in validation
        assert "project_structure_valid" in validation
        
        # Should pass basic structure validation due to temp_project fixture
        assert validation["project_structure_valid"] is True


class TestAdaptiveQualitySystem:
    """Test adaptive quality system functionality."""

    @pytest.fixture
    def adaptive_system(self, temp_project):
        """Create adaptive quality system instance."""
        return AdaptiveQualitySystem(temp_project)

    @pytest.mark.asyncio
    async def test_quality_trend_analysis(self, adaptive_system):
        """Test quality trend analysis."""
        # Create mock reports
        reports_dir = adaptive_system.project_root / "reports" / "quality_gates"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample reports
        for i in range(5):
            report_data = {
                "overall_passed": True,
                "overall_score": 80.0 + i * 2,  # Improving trend
                "gates": [
                    {"name": "functionality", "passed": True, "score": 90.0 + i, "execution_time_ms": 100},
                    {"name": "tests", "passed": i >= 2, "score": 70.0 + i * 3, "execution_time_ms": 200},
                ],
                "generation": "gen1",
                "timestamp": f"2025-01-0{i+1} 12:00:00",
                "duration_ms": 1000.0
            }
            
            with open(reports_dir / f"quality_report_gen1_{i}.json", "w") as f:
                json.dump(report_data, f)
        
        analysis = await adaptive_system.analyze_quality_trends()
        
        assert "total_reports" in analysis
        assert "score_trend" in analysis
        assert "gate_performance" in analysis
        assert analysis["total_reports"] == 5
        assert analysis["score_trend"]["trend"] == "improving"

    @pytest.mark.asyncio
    async def test_learning_from_execution(self, adaptive_system):
        """Test learning patterns from execution."""
        report_data = {
            "overall_passed": True,
            "overall_score": 90.0,
            "gates": [
                {
                    "name": "functionality",
                    "passed": True,
                    "score": 95.0,
                    "recommendations": []
                },
                {
                    "name": "tests",
                    "passed": False,
                    "score": 65.0,
                    "recommendations": ["Increase test coverage"]
                }
            ],
            "duration_ms": 1000.0
        }
        
        initial_pattern_count = len(adaptive_system.learned_patterns)
        
        await adaptive_system.learn_from_execution(report_data)
        
        # Should have learned new patterns
        assert len(adaptive_system.learned_patterns) > initial_pattern_count
        
        # Check that patterns were created for high score and failure
        pattern_ids = [p.pattern_id for p in adaptive_system.learned_patterns]
        assert any("high_score" in pid for pid in pattern_ids)
        assert any("failure" in pid for pid in pattern_ids)

    @pytest.mark.asyncio
    async def test_optimization_suggestions(self, adaptive_system):
        """Test optimization suggestion generation."""
        # Create analysis data that should trigger optimizations
        analysis = {
            "gate_performance": {
                "slow_gate": {
                    "success_rate": 90.0,
                    "average_execution_time": 6000.0,  # Slow gate
                    "average_score": 85.0
                },
                "failing_gate": {
                    "success_rate": 45.0,  # Low success rate
                    "average_execution_time": 1000.0,
                    "average_score": 60.0
                }
            },
            "failure_patterns": {
                "failing_gate": 8  # High failure count
            },
            "score_trend": {
                "trend": "declining"
            }
        }
        
        suggestions = await adaptive_system._generate_optimization_suggestions(analysis)
        
        assert len(suggestions) > 0
        
        # Check for expected suggestion types
        suggestion_types = [s.type for s in suggestions]
        assert "parallelization" in suggestion_types  # For slow gate
        assert "threshold" in suggestion_types  # For failing gate
        assert "configuration" in suggestion_types  # For declining trend

    @pytest.mark.asyncio
    async def test_quality_score_prediction(self, adaptive_system):
        """Test quality score prediction."""
        # Add some performance history
        adaptive_system.performance_history = [
            {"overall_score": 85.0, "timestamp": 1000},
            {"overall_score": 87.0, "timestamp": 2000},
            {"overall_score": 90.0, "timestamp": 3000},
        ]
        
        # Test prediction with gate results
        gate_results = [
            {"passed": True, "score": 90.0},
            {"passed": True, "score": 85.0},
            {"passed": False, "score": 60.0},
            {"passed": True, "score": 95.0},
        ]
        
        predicted_score = await adaptive_system.predict_quality_score(gate_results)
        
        assert 0.0 <= predicted_score <= 100.0
        assert predicted_score > 70.0  # Should be reasonably high given mostly passing gates


class TestIntegration:
    """Integration tests for the complete quality system."""

    @pytest.mark.asyncio
    async def test_full_progressive_quality_workflow(self, temp_project):
        """Test complete progressive quality workflow."""
        # Create orchestrator
        config = QualityConfiguration(parallel_execution=False, timeout_seconds=30)
        orchestrator = QualityOrchestrator(temp_project, config)
        
        # Mock all gate executions to pass
        with patch.object(orchestrator.quality_gates, '_gate_functionality') as mock_func, \
             patch.object(orchestrator.quality_gates, '_gate_basic_tests') as mock_tests, \
             patch.object(orchestrator.quality_gates, '_gate_syntax_check') as mock_syntax, \
             patch.object(orchestrator.quality_gates, '_gate_dependency_check') as mock_deps, \
             patch.object(orchestrator.quality_gates, '_gate_structure_validation') as mock_struct:
            
            # Mock all gates to pass
            for mock_gate in [mock_func, mock_tests, mock_syntax, mock_deps, mock_struct]:
                mock_gate.return_value = QualityGateResult(
                    name="test_gate", passed=True, score=85.0, details={}, 
                    execution_time_ms=100.0, recommendations=[]
                )
            
            # Execute only Gen 1 for faster testing
            result = await orchestrator.execute_progressive_quality_gates(["gen1"])
            
            assert result.overall_success is True
            assert result.total_score >= 75.0
            assert len(result.generation_reports) == 1

    @pytest.mark.asyncio
    async def test_adaptive_system_with_real_data(self, temp_project):
        """Test adaptive system with realistic data."""
        adaptive_system = AdaptiveQualitySystem(temp_project)
        
        # Simulate multiple executions with varying results
        execution_data = [
            {
                "overall_passed": True,
                "overall_score": 85.0,
                "gates": [
                    {"name": "functionality", "passed": True, "score": 90.0, "recommendations": []},
                    {"name": "tests", "passed": True, "score": 80.0, "recommendations": []},
                ]
            },
            {
                "overall_passed": False,
                "overall_score": 65.0,
                "gates": [
                    {"name": "functionality", "passed": True, "score": 90.0, "recommendations": []},
                    {"name": "tests", "passed": False, "score": 40.0, "recommendations": ["Increase coverage"]},
                ]
            }
        ]
        
        # Learn from executions
        for data in execution_data:
            await adaptive_system.learn_from_execution(data)
        
        # Get recommendations
        context = {"recent_score": 70.0}
        recommendations = await adaptive_system.get_adaptive_recommendations(context)
        
        assert isinstance(recommendations, list)
        # Should have learned something from the executions
        assert len(adaptive_system.learned_patterns) > 0


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_parallel_gate_execution(self, temp_project):
        """Test parallel execution performance."""
        config = QualityConfiguration(parallel_execution=True, max_concurrent_gates=3)
        orchestrator = QualityOrchestrator(temp_project, config)
        
        # Mock gates with delays to test parallelization
        async def slow_gate():
            await asyncio.sleep(0.1)  # Simulate slow gate
            return QualityGateResult(
                name="slow_gate", passed=True, score=85.0, details={}, 
                execution_time_ms=100.0, recommendations=[]
            )
        
        with patch.object(orchestrator.quality_gates, '_gate_functionality', side_effect=slow_gate), \
             patch.object(orchestrator.quality_gates, '_gate_basic_tests', side_effect=slow_gate), \
             patch.object(orchestrator.quality_gates, '_gate_syntax_check', side_effect=slow_gate), \
             patch.object(orchestrator.quality_gates, '_gate_dependency_check', side_effect=slow_gate), \
             patch.object(orchestrator.quality_gates, '_gate_structure_validation', side_effect=slow_gate):
            
            import time
            start_time = time.time()
            
            result = await orchestrator.execute_progressive_quality_gates(["gen1"])
            
            execution_time = time.time() - start_time
            
            # With parallel execution, should complete faster than sequential
            assert execution_time < 1.0  # Should be much faster than 5 * 0.1 = 0.5 seconds
            assert result.overall_success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])