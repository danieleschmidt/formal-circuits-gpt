"""Quality Dashboard for Real-time Monitoring."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from .monitoring.logger import get_logger
from .monitoring.metrics import MetricsCollector
from .quality_orchestrator import QualityOrchestrator


@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""
    
    current_score: float
    trend: str  # "improving", "stable", "declining"
    gate_success_rates: Dict[str, float]
    recent_executions: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: str


class QualityDashboard:
    """Real-time quality metrics dashboard."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.logger = get_logger("quality_dashboard")
        self.metrics = MetricsCollector()
        self.orchestrator = QualityOrchestrator(self.project_root)

    async def get_dashboard_data(self) -> DashboardMetrics:
        """Get comprehensive dashboard data."""
        try:
            # Get quality metrics
            quality_metrics = await self.orchestrator.get_quality_metrics()
            
            # Get system health
            health_data = await self.orchestrator.health_check()
            
            # Get performance metrics
            performance_data = await self._get_performance_metrics()
            
            # Get recent executions
            recent_executions = await self._get_recent_executions()
            
            # Extract gate success rates
            gate_success_rates = {}
            if "gate_statistics" in quality_metrics:
                for gate_name, stats in quality_metrics["gate_statistics"].items():
                    gate_success_rates[gate_name] = stats.get("success_rate", 0.0)
            
            return DashboardMetrics(
                current_score=quality_metrics.get("average_score", 0.0),
                trend=quality_metrics.get("trend", "unknown"),
                gate_success_rates=gate_success_rates,
                recent_executions=recent_executions,
                system_health=health_data,
                performance_metrics=performance_data,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return DashboardMetrics(
                current_score=0.0,
                trend="unknown",
                gate_success_rates={},
                recent_executions=[],
                system_health={"status": "error", "error": str(e)},
                performance_metrics={},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC")
            )

    async def _get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        try:
            reports_dir = self.project_root / "reports" / "quality_gates"
            if not reports_dir.exists():
                return {}
            
            # Get recent reports for performance analysis
            report_files = sorted(
                reports_dir.glob("*.json"), 
                key=lambda x: x.stat().st_mtime, 
                reverse=True
            )[:10]
            
            execution_times = []
            gate_times = {}
            
            for report_file in report_files:
                try:
                    with open(report_file) as f:
                        report_data = json.load(f)
                        
                    execution_times.append(report_data.get("duration_ms", 0))
                    
                    for gate in report_data.get("gates", []):
                        gate_name = gate["name"]
                        gate_time = gate.get("execution_time_ms", 0)
                        
                        if gate_name not in gate_times:
                            gate_times[gate_name] = []
                        gate_times[gate_name].append(gate_time)
                        
                except Exception:
                    continue
            
            metrics = {}
            
            if execution_times:
                metrics["avg_execution_time_ms"] = sum(execution_times) / len(execution_times)
                metrics["max_execution_time_ms"] = max(execution_times)
                metrics["min_execution_time_ms"] = min(execution_times)
            
            # Gate-specific performance
            for gate_name, times in gate_times.items():
                if times:
                    metrics[f"avg_{gate_name}_time_ms"] = sum(times) / len(times)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}

    async def _get_recent_executions(self) -> List[Dict[str, Any]]:
        """Get recent quality gate executions."""
        try:
            reports_dir = self.project_root / "reports" / "quality_gates"
            if not reports_dir.exists():
                return []
            
            # Get recent reports
            report_files = sorted(
                reports_dir.glob("*.json"), 
                key=lambda x: x.stat().st_mtime, 
                reverse=True
            )[:5]
            
            executions = []
            
            for report_file in report_files:
                try:
                    with open(report_file) as f:
                        report_data = json.load(f)
                    
                    execution = {
                        "timestamp": report_data.get("timestamp", "unknown"),
                        "generation": report_data.get("generation", "unknown"),
                        "overall_passed": report_data.get("overall_passed", False),
                        "overall_score": report_data.get("overall_score", 0.0),
                        "duration_ms": report_data.get("duration_ms", 0),
                        "gates_passed": sum(1 for gate in report_data.get("gates", []) if gate.get("passed", False)),
                        "total_gates": len(report_data.get("gates", []))
                    }
                    
                    executions.append(execution)
                    
                except Exception:
                    continue
            
            return executions
            
        except Exception as e:
            self.logger.error(f"Failed to get recent executions: {e}")
            return []

    async def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        data = await self.get_dashboard_data()
        
        # Create basic HTML dashboard
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Gates Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-error {{ color: #e74c3c; }}
        .progress-bar {{ width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #3498db; transition: width 0.3s; }}
        .gate-list {{ list-style: none; padding: 0; }}
        .gate-item {{ padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 4px; }}
        .execution-list {{ max-height: 300px; overflow-y: auto; }}
        .execution-item {{ padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 4px; border-left: 4px solid #3498db; }}
        .timestamp {{ font-size: 0.8em; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Quality Gates Dashboard</h1>
            <p>Last updated: {data.timestamp}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Quality Score</div>
                <div class="metric-value">{data.current_score:.1f}/100</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {data.current_score}%"></div>
                </div>
                <p class="metric-label">Trend: {data.trend}</p>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">System Health</div>
                <div class="metric-value status-{self._get_status_class(data.system_health.get('status', 'unknown'))}">{data.system_health.get('status', 'unknown').upper()}</div>
                <div class="metric-label">Circuit Breaker: {data.system_health.get('checks', {}).get('circuit_breaker', {}).get('status', 'unknown')}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Performance</div>
                <div class="metric-value">{data.performance_metrics.get('avg_execution_time_ms', 0):.0f}ms</div>
                <div class="metric-label">Average Execution Time</div>
            </div>
        </div>
        
        <div class="metrics-grid" style="margin-top: 20px;">
            <div class="metric-card">
                <h3>Gate Success Rates</h3>
                <ul class="gate-list">
        """
        
        for gate_name, success_rate in data.gate_success_rates.items():
            status_class = self._get_status_class_from_rate(success_rate)
            html += f"""
                    <li class="gate-item">
                        <strong>{gate_name}</strong>: 
                        <span class="status-{status_class}">{success_rate:.1f}%</span>
                        <div class="progress-bar" style="margin-top: 5px;">
                            <div class="progress-fill" style="width: {success_rate}%"></div>
                        </div>
                    </li>
            """
        
        html += """
                </ul>
            </div>
            
            <div class="metric-card">
                <h3>Recent Executions</h3>
                <div class="execution-list">
        """
        
        for execution in data.recent_executions:
            status_class = "good" if execution["overall_passed"] else "error"
            html += f"""
                    <div class="execution-item">
                        <div><strong>{execution['generation']}</strong> - 
                        <span class="status-{status_class}">{'PASS' if execution['overall_passed'] else 'FAIL'}</span></div>
                        <div>Score: {execution['overall_score']:.1f}/100, 
                        Gates: {execution['gates_passed']}/{execution['total_gates']}</div>
                        <div class="timestamp">{execution['timestamp']} ({execution['duration_ms']:.0f}ms)</div>
                    </div>
            """
        
        html += """
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            window.location.reload();
        }, 30000);
    </script>
</body>
</html>
        """
        
        return html

    def _get_status_class(self, status: str) -> str:
        """Get CSS class for status."""
        if status in ["healthy", "good"]:
            return "good"
        elif status in ["degraded", "warning"]:
            return "warning"
        else:
            return "error"

    def _get_status_class_from_rate(self, rate: float) -> str:
        """Get CSS class from success rate."""
        if rate >= 80:
            return "good"
        elif rate >= 60:
            return "warning"
        else:
            return "error"

    async def save_dashboard(self, filename: str = "quality_dashboard.html") -> Path:
        """Save dashboard to HTML file."""
        try:
            html_content = await self.generate_dashboard_html()
            
            dashboard_file = self.project_root / "reports" / filename
            dashboard_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dashboard_file, "w") as f:
                f.write(html_content)
            
            self.logger.info(f"Dashboard saved to {dashboard_file}")
            return dashboard_file
            
        except Exception as e:
            self.logger.error(f"Failed to save dashboard: {e}")
            raise

    async def get_json_metrics(self) -> Dict[str, Any]:
        """Get metrics in JSON format for API consumption."""
        data = await self.get_dashboard_data()
        
        return {
            "current_score": data.current_score,
            "trend": data.trend,
            "gate_success_rates": data.gate_success_rates,
            "system_health": data.system_health,
            "performance_metrics": data.performance_metrics,
            "recent_executions": data.recent_executions,
            "timestamp": data.timestamp
        }


async def main():
    """Main function for standalone execution."""
    import sys
    
    dashboard = QualityDashboard()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "json":
            # Output JSON metrics
            metrics = await dashboard.get_json_metrics()
            print(json.dumps(metrics, indent=2))
        elif sys.argv[1] == "save":
            # Save HTML dashboard
            dashboard_file = await dashboard.save_dashboard()
            print(f"Dashboard saved to: {dashboard_file}")
        else:
            print("Usage: python quality_dashboard.py [json|save]")
    else:
        # Print basic metrics
        data = await dashboard.get_dashboard_data()
        print(f"Quality Score: {data.current_score:.1f}/100")
        print(f"Trend: {data.trend}")
        print(f"System Health: {data.system_health.get('status', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())