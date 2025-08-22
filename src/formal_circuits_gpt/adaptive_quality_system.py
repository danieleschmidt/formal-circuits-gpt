"""Adaptive Quality System with ML-based Optimization."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import statistics
from collections import defaultdict

from .quality_orchestrator import QualityOrchestrator, QualityConfiguration
from .monitoring.logger import get_logger
from .monitoring.metrics import MetricsCollector


@dataclass
class QualityPattern:
    """Represents a quality pattern learned from historical data."""
    
    pattern_id: str
    condition: str
    recommendation: str
    success_rate: float
    confidence: float
    usage_count: int


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion based on analysis."""
    
    type: str  # "threshold", "gate_order", "parallelization", "configuration"
    target: str
    current_value: Any
    suggested_value: Any
    expected_improvement: float
    confidence: float
    reasoning: str


class AdaptiveQualitySystem:
    """ML-enhanced adaptive quality system."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.logger = get_logger("adaptive_quality")
        self.metrics = MetricsCollector()
        self.orchestrator = QualityOrchestrator(self.project_root)
        
        # Learning storage
        self.patterns_file = self.project_root / "reports" / "quality_patterns.json"
        self.learned_patterns: List[QualityPattern] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Load existing patterns
        asyncio.create_task(self._load_patterns())

    async def _load_patterns(self):
        """Load learned patterns from storage."""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file) as f:
                    data = json.load(f)
                    self.learned_patterns = [
                        QualityPattern(**pattern) for pattern in data.get("patterns", [])
                    ]
                    self.performance_history = data.get("performance_history", [])
                self.logger.info(f"Loaded {len(self.learned_patterns)} quality patterns")
        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}")
            self.learned_patterns = []
            self.performance_history = []

    async def _save_patterns(self):
        """Save learned patterns to storage."""
        try:
            self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "patterns": [asdict(pattern) for pattern in self.learned_patterns],
                "performance_history": self.performance_history[-100:],  # Keep last 100 entries
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S UTC")
            }
            
            with open(self.patterns_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save patterns: {e}")

    async def analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends and patterns."""
        try:
            reports_dir = self.project_root / "reports" / "quality_gates"
            if not reports_dir.exists():
                return {"error": "No quality reports found"}
            
            # Get recent reports
            report_files = sorted(
                reports_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:50]  # Analyze last 50 reports
            
            analysis = {
                "total_reports": len(report_files),
                "score_trend": {},
                "gate_performance": {},
                "failure_patterns": {},
                "optimization_opportunities": []
            }
            
            scores = []
            gate_stats = defaultdict(list)
            failure_patterns = defaultdict(int)
            
            for report_file in report_files:
                try:
                    with open(report_file) as f:
                        report = json.load(f)
                    
                    # Collect scores
                    score = report.get("overall_score", 0)
                    scores.append(score)
                    
                    # Analyze gate performance
                    for gate in report.get("gates", []):
                        gate_name = gate["name"]
                        gate_stats[gate_name].append({
                            "passed": gate["passed"],
                            "score": gate["score"],
                            "execution_time": gate.get("execution_time_ms", 0)
                        })
                        
                        # Track failure patterns
                        if not gate["passed"]:
                            failure_patterns[gate_name] += 1
                
                except Exception:
                    continue
            
            # Calculate trends
            if scores:
                analysis["score_trend"] = {
                    "average": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "trend": self._calculate_trend(scores),
                    "stability": self._calculate_stability(scores)
                }
            
            # Gate performance analysis
            for gate_name, stats in gate_stats.items():
                if stats:
                    success_rate = sum(1 for s in stats if s["passed"]) / len(stats) * 100
                    avg_score = statistics.mean([s["score"] for s in stats])
                    avg_time = statistics.mean([s["execution_time"] for s in stats])
                    
                    analysis["gate_performance"][gate_name] = {
                        "success_rate": success_rate,
                        "average_score": avg_score,
                        "average_execution_time": avg_time,
                        "total_executions": len(stats)
                    }
            
            # Failure pattern analysis
            analysis["failure_patterns"] = dict(failure_patterns)
            
            # Generate optimization suggestions
            analysis["optimization_opportunities"] = await self._generate_optimization_suggestions(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze quality trends: {e}")
            return {"error": str(e)}

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction from scores."""
        if len(scores) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        recent = scores[:len(scores)//3] if len(scores) >= 6 else scores[:2]
        older = scores[-len(scores)//3:] if len(scores) >= 6 else scores[-2:]
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "declining"
        else:
            return "stable"

    def _calculate_stability(self, scores: List[float]) -> str:
        """Calculate stability rating."""
        if len(scores) < 3:
            return "insufficient_data"
        
        std_dev = statistics.stdev(scores)
        mean_score = statistics.mean(scores)
        
        cv = (std_dev / mean_score) * 100 if mean_score > 0 else 100
        
        if cv < 5:
            return "very_stable"
        elif cv < 10:
            return "stable"
        elif cv < 20:
            return "moderately_stable"
        else:
            return "unstable"

    async def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        try:
            gate_performance = analysis.get("gate_performance", {})
            failure_patterns = analysis.get("failure_patterns", {})
            score_trend = analysis.get("score_trend", {})
            
            # Suggest threshold adjustments for consistently failing gates
            for gate_name, failure_count in failure_patterns.items():
                if failure_count > 5:  # Frequently failing
                    gate_stats = gate_performance.get(gate_name, {})
                    if gate_stats.get("success_rate", 100) < 70:
                        suggestions.append(OptimizationSuggestion(
                            type="threshold",
                            target=gate_name,
                            current_value="current_threshold",
                            suggested_value="lower_threshold",
                            expected_improvement=15.0,
                            confidence=0.8,
                            reasoning=f"Gate {gate_name} has {failure_count} failures with {gate_stats.get('success_rate', 0):.1f}% success rate"
                        ))
            
            # Suggest parallelization for slow gates
            for gate_name, stats in gate_performance.items():
                avg_time = stats.get("average_execution_time", 0)
                if avg_time > 5000:  # Slower than 5 seconds
                    suggestions.append(OptimizationSuggestion(
                        type="parallelization",
                        target=gate_name,
                        current_value=f"{avg_time:.0f}ms",
                        suggested_value="parallel_execution",
                        expected_improvement=30.0,
                        confidence=0.7,
                        reasoning=f"Gate {gate_name} takes {avg_time:.0f}ms on average"
                    ))
            
            # Suggest configuration changes for declining trends
            if score_trend.get("trend") == "declining":
                suggestions.append(OptimizationSuggestion(
                    type="configuration",
                    target="quality_thresholds",
                    current_value="current_config",
                    suggested_value="adaptive_thresholds",
                    expected_improvement=10.0,
                    confidence=0.6,
                    reasoning="Overall quality score is declining, consider adaptive thresholds"
                ))
            
            # Suggest gate reordering for efficiency
            slow_gates = [
                name for name, stats in gate_performance.items()
                if stats.get("average_execution_time", 0) > 1000
            ]
            if len(slow_gates) > 2:
                suggestions.append(OptimizationSuggestion(
                    type="gate_order",
                    target="execution_order",
                    current_value="current_order",
                    suggested_value="optimized_order",
                    expected_improvement=20.0,
                    confidence=0.7,
                    reasoning=f"Reorder gates to run fast gates first: {len(slow_gates)} slow gates detected"
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization suggestions: {e}")
        
        return suggestions

    async def learn_from_execution(self, report_data: Dict[str, Any]):
        """Learn patterns from quality gate execution."""
        try:
            # Extract patterns from successful and failed executions
            overall_passed = report_data.get("overall_passed", False)
            overall_score = report_data.get("overall_score", 0)
            gates = report_data.get("gates", [])
            
            # Learn gate-specific patterns
            for gate in gates:
                gate_name = gate["name"]
                gate_passed = gate["passed"]
                gate_score = gate["score"]
                
                # Create pattern conditions
                if gate_passed and gate_score > 90:
                    pattern_id = f"high_score_{gate_name}"
                    condition = f"{gate_name}_score > 90"
                    recommendation = f"Continue current approach for {gate_name}"
                    
                    await self._update_pattern(pattern_id, condition, recommendation, True)
                
                elif not gate_passed:
                    pattern_id = f"failure_{gate_name}"
                    condition = f"{gate_name}_failed"
                    
                    # Extract recommendations from gate details
                    recommendations = gate.get("recommendations", [])
                    if recommendations:
                        recommendation = recommendations[0]
                        await self._update_pattern(pattern_id, condition, recommendation, False)
            
            # Learn overall execution patterns
            if overall_passed and overall_score > 85:
                pattern_id = "high_overall_score"
                condition = "overall_score > 85"
                recommendation = "Maintain current quality practices"
                await self._update_pattern(pattern_id, condition, recommendation, True)
            
            # Store performance data
            performance_entry = {
                "timestamp": time.time(),
                "overall_score": overall_score,
                "overall_passed": overall_passed,
                "duration_ms": report_data.get("duration_ms", 0),
                "gate_count": len(gates)
            }
            
            self.performance_history.append(performance_entry)
            
            # Save patterns
            await self._save_patterns()
            
        except Exception as e:
            self.logger.error(f"Failed to learn from execution: {e}")

    async def _update_pattern(self, pattern_id: str, condition: str, recommendation: str, success: bool):
        """Update or create a quality pattern."""
        # Find existing pattern
        existing_pattern = None
        for pattern in self.learned_patterns:
            if pattern.pattern_id == pattern_id:
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.usage_count += 1
            
            # Update success rate using exponential moving average
            alpha = 0.1
            new_success = 1.0 if success else 0.0
            existing_pattern.success_rate = (
                alpha * new_success + (1 - alpha) * existing_pattern.success_rate
            )
            
            # Update confidence based on usage
            existing_pattern.confidence = min(0.95, existing_pattern.usage_count / 20.0)
            
        else:
            # Create new pattern
            new_pattern = QualityPattern(
                pattern_id=pattern_id,
                condition=condition,
                recommendation=recommendation,
                success_rate=1.0 if success else 0.0,
                confidence=0.1,
                usage_count=1
            )
            self.learned_patterns.append(new_pattern)

    async def get_adaptive_recommendations(self, current_context: Dict[str, Any]) -> List[str]:
        """Get adaptive recommendations based on learned patterns."""
        recommendations = []
        
        try:
            # Get recent analysis
            analysis = await self.analyze_quality_trends()
            
            # Apply learned patterns
            for pattern in self.learned_patterns:
                if pattern.confidence > 0.5 and pattern.success_rate > 0.7:
                    # Simple pattern matching (in a real system, this would be more sophisticated)
                    if self._pattern_matches(pattern, current_context, analysis):
                        recommendations.append(pattern.recommendation)
            
            # Add optimization suggestions
            optimization_suggestions = analysis.get("optimization_opportunities", [])
            for suggestion in optimization_suggestions[:3]:  # Top 3 suggestions
                if suggestion.confidence > 0.6:
                    recommendations.append(suggestion.reasoning)
            
            # Deduplicate and prioritize
            unique_recommendations = list(set(recommendations))
            
            return unique_recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get adaptive recommendations: {e}")
            return []

    def _pattern_matches(self, pattern: QualityPattern, context: Dict[str, Any], analysis: Dict[str, Any]) -> bool:
        """Check if a pattern matches the current context."""
        try:
            # Simple pattern matching - in production this would use ML models
            condition = pattern.condition.lower()
            
            if "overall_score" in condition:
                current_score = context.get("recent_score", analysis.get("score_trend", {}).get("average", 0))
                if "> 85" in condition and current_score > 85:
                    return True
                elif "> 90" in condition and current_score > 90:
                    return True
            
            if "_failed" in condition:
                gate_name = condition.replace("_failed", "")
                failure_patterns = analysis.get("failure_patterns", {})
                return gate_name in failure_patterns and failure_patterns[gate_name] > 0
            
            return False
            
        except Exception:
            return False

    async def auto_optimize_configuration(self) -> Dict[str, Any]:
        """Automatically optimize quality gate configuration."""
        try:
            analysis = await self.analyze_quality_trends()
            optimization_suggestions = analysis.get("optimization_opportunities", [])
            
            optimization_results = {
                "optimizations_applied": [],
                "configuration_changes": {},
                "expected_improvements": {}
            }
            
            # Apply high-confidence optimizations automatically
            for suggestion in optimization_suggestions:
                if suggestion.confidence > 0.8 and suggestion.expected_improvement > 15:
                    
                    if suggestion.type == "threshold":
                        # Adjust thresholds for failing gates
                        optimization_results["configuration_changes"][suggestion.target] = {
                            "old_threshold": suggestion.current_value,
                            "new_threshold": suggestion.suggested_value,
                            "reason": suggestion.reasoning
                        }
                        optimization_results["optimizations_applied"].append(f"Adjusted threshold for {suggestion.target}")
                    
                    elif suggestion.type == "parallelization":
                        # Enable parallel execution for slow gates
                        optimization_results["configuration_changes"]["parallel_gates"] = {
                            "enabled": True,
                            "target_gates": [suggestion.target]
                        }
                        optimization_results["optimizations_applied"].append(f"Enabled parallel execution for {suggestion.target}")
                    
                    optimization_results["expected_improvements"][suggestion.target] = suggestion.expected_improvement
            
            # Save optimization results
            optimization_file = self.project_root / "reports" / "auto_optimizations.json"
            optimization_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(optimization_file, "w") as f:
                json.dump(optimization_results, f, indent=2)
            
            self.logger.info(f"Applied {len(optimization_results['optimizations_applied'])} automatic optimizations")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Failed to auto-optimize configuration: {e}")
            return {"error": str(e)}

    async def predict_quality_score(self, gate_results: List[Dict[str, Any]]) -> float:
        """Predict overall quality score based on gate results."""
        try:
            if not self.performance_history:
                return 0.0
            
            # Simple prediction based on historical patterns
            # In production, this would use a trained ML model
            
            passed_gates = sum(1 for gate in gate_results if gate.get("passed", False))
            total_gates = len(gate_results)
            avg_score = sum(gate.get("score", 0) for gate in gate_results) / total_gates if total_gates > 0 else 0
            
            # Weight based on historical performance
            base_score = (passed_gates / total_gates) * 100 if total_gates > 0 else 0
            weighted_score = (base_score * 0.7) + (avg_score * 0.3)
            
            # Adjust based on historical trends
            recent_scores = [entry["overall_score"] for entry in self.performance_history[-10:]]
            if recent_scores:
                historical_avg = statistics.mean(recent_scores)
                # Blend with historical average
                predicted_score = (weighted_score * 0.8) + (historical_avg * 0.2)
            else:
                predicted_score = weighted_score
            
            return max(0.0, min(100.0, predicted_score))
            
        except Exception as e:
            self.logger.error(f"Failed to predict quality score: {e}")
            return 0.0


async def main():
    """Main function for adaptive quality system."""
    import sys
    
    adaptive_system = AdaptiveQualitySystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "analyze":
            analysis = await adaptive_system.analyze_quality_trends()
            print(json.dumps(analysis, indent=2))
        
        elif command == "optimize":
            results = await adaptive_system.auto_optimize_configuration()
            print(json.dumps(results, indent=2))
        
        elif command == "recommendations":
            context = {"recent_score": 75.0}  # Example context
            recommendations = await adaptive_system.get_adaptive_recommendations(context)
            print("Adaptive Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        else:
            print("Usage: python adaptive_quality_system.py [analyze|optimize|recommendations]")
    
    else:
        print("Adaptive Quality System")
        print("Available commands: analyze, optimize, recommendations")


if __name__ == "__main__":
    asyncio.run(main())