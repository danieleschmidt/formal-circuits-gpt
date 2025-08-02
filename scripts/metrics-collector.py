#!/usr/bin/env python3
"""
Metrics collection script for formal-circuits-gpt

This script collects various metrics about the project and updates
the project-metrics.json file with current values.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class MetricsCollector:
    """Collects and updates project metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics_file = project_root / ".github" / "project-metrics.json"
        self.metrics_data = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics data."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metrics(self) -> None:
        """Save metrics data to file."""
        self.metrics_data["last_updated"] = datetime.now().isoformat() + "Z"
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
    
    def _run_command(self, command: str) -> Optional[str]:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception as e:
            print(f"Error running command '{command}': {e}")
            return None
    
    def collect_code_quality_metrics(self) -> None:
        """Collect code quality metrics."""
        print("Collecting code quality metrics...")
        
        # Test coverage
        coverage_result = self._run_command("python -m pytest --cov=formal_circuits_gpt --cov-report=json")
        if coverage_result is not None:
            try:
                # Try to read coverage.json if it exists
                coverage_file = self.project_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                        coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
                        self._update_metric("metrics.code_quality.test_coverage.current", coverage_percent)
            except Exception as e:
                print(f"Error parsing coverage data: {e}")
        
        # Code complexity (using radon if available)
        complexity_result = self._run_command("radon cc src/ -a -j")
        if complexity_result:
            try:
                complexity_data = json.loads(complexity_result)
                # Calculate average complexity
                total_complexity = 0
                total_functions = 0
                for file_data in complexity_data.values():
                    for item in file_data:
                        if isinstance(item, dict) and 'complexity' in item:
                            total_complexity += item['complexity']
                            total_functions += 1
                
                avg_complexity = total_complexity / total_functions if total_functions > 0 else 0
                self._update_metric("metrics.code_quality.code_complexity.current", avg_complexity)
            except Exception as e:
                print(f"Error parsing complexity data: {e}")
        
        # Security vulnerabilities
        safety_result = self._run_command("safety check --json")
        if safety_result:
            try:
                safety_data = json.loads(safety_result)
                vulnerability_count = len(safety_data)
                self._update_metric("metrics.code_quality.security_vulnerabilities.current", vulnerability_count)
            except Exception as e:
                print(f"Error parsing safety data: {e}")
    
    def collect_git_metrics(self) -> None:
        """Collect git-based development velocity metrics."""
        print("Collecting git metrics...")
        
        # Commits per week (last 4 weeks)
        commits_result = self._run_command("git log --since='4 weeks ago' --oneline | wc -l")
        if commits_result:
            try:
                commits_count = int(commits_result) / 4  # Average per week
                self._update_metric("metrics.development_velocity.commits_per_week.current", commits_count)
            except ValueError:
                pass
        
        # Get GitHub metrics if token is available
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            self._collect_github_metrics(github_token)
    
    def _collect_github_metrics(self, token: str) -> None:
        """Collect metrics from GitHub API."""
        print("Collecting GitHub metrics...")
        
        # Extract repo info from git remote
        remote_url = self._run_command("git remote get-url origin")
        if not remote_url:
            return
        
        # Parse GitHub repo from URL
        if "github.com" in remote_url:
            repo_path = remote_url.split("github.com/")[-1].replace(".git", "")
            owner, repo = repo_path.split("/")
            
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            try:
                # Repository data
                repo_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}",
                    headers=headers
                )
                if repo_response.status_code == 200:
                    repo_data = repo_response.json()
                    stars = repo_data.get("stargazers_count", 0)
                    self._update_metric("metrics.user_engagement.github_stars.current", stars)
                
                # Pull requests (last 30 days)
                pr_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/pulls",
                    headers=headers,
                    params={"state": "all", "since": (datetime.now() - datetime.timedelta(days=30)).isoformat()}
                )
                if pr_response.status_code == 200:
                    prs = pr_response.json()
                    pr_count = len(prs) / 4  # Average per week
                    self._update_metric("metrics.development_velocity.pull_requests_per_week.current", pr_count)
                
                # Contributors
                contributors_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contributors",
                    headers=headers
                )
                if contributors_response.status_code == 200:
                    contributors = contributors_response.json()
                    contributor_count = len(contributors)
                    self._update_metric("metrics.business.community_contributions.current", contributor_count)
                    
            except Exception as e:
                print(f"Error collecting GitHub metrics: {e}")
    
    def collect_performance_metrics(self) -> None:
        """Collect performance metrics from test runs."""
        print("Collecting performance metrics...")
        
        # Run benchmark tests if available
        benchmark_result = self._run_command("python -m pytest tests/benchmarks/ --benchmark-json=benchmark.json")
        if benchmark_result is not None:
            try:
                benchmark_file = self.project_root / "benchmark.json"
                if benchmark_file.exists():
                    with open(benchmark_file, 'r') as f:
                        benchmark_data = json.load(f)
                        
                    # Extract relevant metrics
                    benchmarks = benchmark_data.get("benchmarks", [])
                    if benchmarks:
                        avg_time = sum(b.get("stats", {}).get("mean", 0) for b in benchmarks) / len(benchmarks)
                        self._update_metric("metrics.performance.average_verification_time.current", avg_time)
                        
                        # Memory usage (if available in benchmark data)
                        memory_usage = max(b.get("stats", {}).get("max_memory", 0) for b in benchmarks)
                        if memory_usage > 0:
                            self._update_metric("metrics.performance.memory_usage.current", memory_usage / 1024 / 1024)  # Convert to MB
                            
            except Exception as e:
                print(f"Error parsing benchmark data: {e}")
    
    def collect_documentation_metrics(self) -> None:
        """Collect documentation coverage metrics."""
        print("Collecting documentation metrics...")
        
        # Count documented functions vs total functions
        python_files = list(self.project_root.glob("src/**/*.py"))
        total_functions = 0
        documented_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Simple regex-based counting (could be improved with AST parsing)
                import re
                functions = re.findall(r'def\s+\w+\(', content)
                docstrings = re.findall(r'def\s+\w+\([^)]*\):[^"]*"""', content, re.DOTALL)
                
                total_functions += len(functions)
                documented_functions += len(docstrings)
                
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
        
        if total_functions > 0:
            doc_coverage = (documented_functions / total_functions) * 100
            self._update_metric("metrics.business.documentation_coverage.current", doc_coverage)
    
    def _update_metric(self, path: str, value: Any) -> None:
        """Update a metric value using dot notation path."""
        keys = path.split('.')
        current = self.metrics_data
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
    
    def calculate_trends(self) -> None:
        """Calculate trends based on historical data."""
        print("Calculating trends...")
        
        # This is a simplified trend calculation
        # In a real implementation, you'd store historical data and calculate proper trends
        
        def update_trend(metric_path: str) -> None:
            keys = metric_path.split('.')
            current = self.metrics_data
            
            for key in keys:
                if key in current:
                    current = current[key]
                else:
                    return
            
            if isinstance(current, dict) and 'current' in current and 'target' in current:
                current_val = current['current']
                target_val = current['target']
                
                if current_val > target_val * 0.9:
                    current['trend'] = 'improving'
                elif current_val < target_val * 0.7:
                    current['trend'] = 'declining'
                else:
                    current['trend'] = 'stable'
        
        # Update trends for key metrics
        trend_metrics = [
            'metrics.code_quality.test_coverage',
            'metrics.development_velocity.commits_per_week',
            'metrics.performance.average_verification_time',
            'metrics.user_engagement.github_stars'
        ]
        
        for metric in trend_metrics:
            update_trend(metric)
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of current metrics."""
        metrics = self.metrics_data.get('metrics', {})
        
        report = f"""
# Metrics Summary Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Code Quality
- Test Coverage: {metrics.get('code_quality', {}).get('test_coverage', {}).get('current', 'N/A')}%
- Security Vulnerabilities: {metrics.get('code_quality', {}).get('security_vulnerabilities', {}).get('current', 'N/A')}
- Code Complexity: {metrics.get('code_quality', {}).get('code_complexity', {}).get('current', 'N/A')}

## Development Velocity
- Commits per Week: {metrics.get('development_velocity', {}).get('commits_per_week', {}).get('current', 'N/A')}
- Pull Requests per Week: {metrics.get('development_velocity', {}).get('pull_requests_per_week', {}).get('current', 'N/A')}

## Performance
- Average Verification Time: {metrics.get('performance', {}).get('average_verification_time', {}).get('current', 'N/A')}s
- Memory Usage: {metrics.get('performance', {}).get('memory_usage', {}).get('current', 'N/A')}MB

## User Engagement
- GitHub Stars: {metrics.get('user_engagement', {}).get('github_stars', {}).get('current', 'N/A')}
- Documentation Coverage: {metrics.get('business', {}).get('documentation_coverage', {}).get('current', 'N/A')}%

## Next Actions
- Review metrics against targets
- Address any declining trends
- Update targets if necessary
        """
        
        return report.strip()
    
    def run_collection(self) -> None:
        """Run the complete metrics collection process."""
        print("Starting metrics collection...")
        
        try:
            self.collect_code_quality_metrics()
            self.collect_git_metrics()
            self.collect_performance_metrics()
            self.collect_documentation_metrics()
            self.calculate_trends()
            
            # Save updated metrics
            self._save_metrics()
            
            # Generate and save report
            report = self.generate_summary_report()
            report_file = self.project_root / "metrics-report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"Metrics collection completed successfully!")
            print(f"Updated metrics saved to: {self.metrics_file}")
            print(f"Summary report saved to: {report_file}")
            
        except Exception as e:
            print(f"Error during metrics collection: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect project metrics for formal-circuits-gpt")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Path to project root directory"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report without collecting new metrics"
    )
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.project_root)
    
    if args.report_only:
        report = collector.generate_summary_report()
        print(report)
    else:
        collector.run_collection()


if __name__ == "__main__":
    main()