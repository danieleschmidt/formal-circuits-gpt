#!/usr/bin/env python3
"""
Comprehensive quality gates script for formal-circuits-gpt.
Implements automated testing, security scanning, performance benchmarking, and compliance checks.
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import shutil


class QualityGateResult:
    """Result of a quality gate check."""
    
    def __init__(self, name: str, passed: bool, score: float, details: Dict[str, Any] = None, 
                 execution_time: float = 0.0, warnings: List[str] = None, errors: List[str] = None):
        self.name = name
        self.passed = passed
        self.score = score
        self.details = details or {}
        self.execution_time = execution_time
        self.warnings = warnings or []
        self.errors = errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "execution_time_seconds": self.execution_time,
            "details": self.details,
            "warnings": self.warnings,
            "errors": self.errors
        }


class QualityGateRunner:
    """Comprehensive quality gate runner."""
    
    def __init__(self, project_root: Path, verbose: bool = False, fail_fast: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.results: List[QualityGateResult] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   capture_output: bool = True, timeout: int = 300) -> Tuple[int, str, str]:
        """Run command and capture output."""
        try:
            if cwd is None:
                cwd = self.project_root
            
            self.log(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", f"Command failed: {str(e)}"
    
    def check_dependencies(self) -> QualityGateResult:
        """Check that all dependencies are available."""
        start_time = time.time()
        
        dependencies = {
            "python3": ["python3", "--version"],
            "pip": ["pip", "--version"],
            "pytest": ["pytest", "--version"],
            "mypy": ["mypy", "--version"],
            "black": ["black", "--version"],
            "flake8": ["flake8", "--version"],
            "isort": ["isort", "--version"],
            "safety": ["safety", "--version"]
        }
        
        missing = []
        versions = {}
        
        for dep, cmd in dependencies.items():
            returncode, stdout, stderr = self.run_command(cmd, capture_output=True, timeout=30)
            
            if returncode == 0:
                versions[dep] = stdout.strip().split('\n')[0]
                self.log(f"✓ {dep}: {versions[dep]}")
            else:
                missing.append(dep)
                self.log(f"✗ {dep}: Not found or failed", "WARNING")
        
        passed = len(missing) == 0
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="dependency_check",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "required_dependencies": list(dependencies.keys()),
                "found_dependencies": list(versions.keys()),
                "missing_dependencies": missing,
                "versions": versions
            },
            execution_time=execution_time,
            errors=[f"Missing dependency: {dep}" for dep in missing]
        )
    
    def run_unit_tests(self) -> QualityGateResult:
        """Run unit tests with coverage."""
        start_time = time.time()
        
        # Check if tests directory exists
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            return QualityGateResult(
                name="unit_tests",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                errors=["Tests directory not found"]
            )
        
        # Run pytest with coverage
        cmd = [
            "python3", "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--cov=src/formal_circuits_gpt",
            "--cov-report=term-missing",
            "--cov-report=json",
            "--junitxml=test-results.xml"
        ]
        
        returncode, stdout, stderr = self.run_command(cmd, timeout=600)  # 10 minutes
        execution_time = time.time() - start_time
        
        # Parse coverage results
        coverage_data = {}
        try:
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
        except:
            pass
        
        # Parse test results
        test_stats = self._parse_pytest_output(stdout)
        coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0.0)
        
        # Quality thresholds
        min_coverage = 85.0
        max_failures = 0
        
        passed = (returncode == 0 and 
                 test_stats.get('failed', 1) <= max_failures and
                 coverage_percent >= min_coverage)
        
        score = min(1.0, coverage_percent / 100.0 * 0.7 + 
                   (1.0 - min(test_stats.get('failed', 0), 10) / 10.0) * 0.3)
        
        errors = []
        warnings = []
        
        if returncode != 0:
            errors.append(f"Tests failed with return code {returncode}")
        
        if test_stats.get('failed', 0) > max_failures:
            errors.append(f"Too many test failures: {test_stats.get('failed')}")
        
        if coverage_percent < min_coverage:
            warnings.append(f"Coverage below threshold: {coverage_percent:.1f}% < {min_coverage}%")
        
        return QualityGateResult(
            name="unit_tests",
            passed=passed,
            score=score,
            details={
                "test_stats": test_stats,
                "coverage_percent": coverage_percent,
                "min_coverage_threshold": min_coverage,
                "coverage_data": coverage_data.get('totals', {}),
                "stdout": stdout[-2000:] if len(stdout) > 2000 else stdout  # Last 2000 chars
            },
            execution_time=execution_time,
            warnings=warnings,
            errors=errors
        )
    
    def run_type_checking(self) -> QualityGateResult:
        """Run mypy type checking."""
        start_time = time.time()
        
        cmd = ["python3", "-m", "mypy", "src/formal_circuits_gpt", "--json-report", "mypy-report.json"]
        
        returncode, stdout, stderr = self.run_command(cmd, timeout=300)
        execution_time = time.time() - start_time
        
        # Parse mypy results
        errors = []
        warnings = []
        error_count = 0
        
        if returncode != 0:
            # Count errors from output
            for line in stdout.split('\n') + stderr.split('\n'):
                if ": error:" in line:
                    error_count += 1
                    errors.append(line.strip())
                elif ": warning:" in line:
                    warnings.append(line.strip())
        
        passed = returncode == 0
        score = 1.0 if passed else max(0.0, 1.0 - error_count / 50.0)  # Deduct based on error count
        
        return QualityGateResult(
            name="type_checking",
            passed=passed,
            score=score,
            details={
                "error_count": error_count,
                "mypy_output": stdout[-1000:] if len(stdout) > 1000 else stdout
            },
            execution_time=execution_time,
            warnings=warnings[:10],  # Limit warnings
            errors=errors[:10]  # Limit errors
        )
    
    def run_code_formatting(self) -> QualityGateResult:
        """Check code formatting with black and isort."""
        start_time = time.time()
        
        # Check black formatting
        black_cmd = ["python3", "-m", "black", "--check", "--diff", "src/", "tests/"]
        black_returncode, black_stdout, black_stderr = self.run_command(black_cmd, timeout=120)
        
        # Check isort formatting
        isort_cmd = ["python3", "-m", "isort", "--check-only", "--diff", "src/", "tests/"]
        isort_returncode, isort_stdout, isort_stderr = self.run_command(isort_cmd, timeout=120)
        
        execution_time = time.time() - start_time
        
        black_passed = black_returncode == 0
        isort_passed = isort_returncode == 0
        passed = black_passed and isort_passed
        
        errors = []
        if not black_passed:
            errors.append("Black formatting issues found")
        if not isort_passed:
            errors.append("Import sorting issues found")
        
        score = (0.5 if black_passed else 0.0) + (0.5 if isort_passed else 0.0)
        
        return QualityGateResult(
            name="code_formatting",
            passed=passed,
            score=score,
            details={
                "black_passed": black_passed,
                "isort_passed": isort_passed,
                "black_output": black_stdout[-500:] if len(black_stdout) > 500 else black_stdout,
                "isort_output": isort_stdout[-500:] if len(isort_stdout) > 500 else isort_stdout
            },
            execution_time=execution_time,
            errors=errors
        )
    
    def run_linting(self) -> QualityGateResult:
        """Run flake8 linting."""
        start_time = time.time()
        
        cmd = ["python3", "-m", "flake8", "src/", "tests/", "--statistics", "--count"]
        
        returncode, stdout, stderr = self.run_command(cmd, timeout=180)
        execution_time = time.time() - start_time
        
        # Parse flake8 output
        error_count = 0
        warnings = []
        
        lines = stdout.split('\n') + stderr.split('\n')
        for line in lines:
            if line.strip() and ':' in line and any(level in line for level in ['E', 'W', 'F', 'C']):
                if line.count(':') >= 3:  # file:line:col:code message format
                    error_count += 1
                    if len(warnings) < 20:  # Limit warnings reported
                        warnings.append(line.strip())
        
        passed = returncode == 0 and error_count == 0
        score = max(0.0, 1.0 - error_count / 100.0)  # Deduct based on error count
        
        errors = []
        if not passed:
            errors.append(f"Flake8 found {error_count} issues")
        
        return QualityGateResult(
            name="linting",
            passed=passed,
            score=score,
            details={
                "error_count": error_count,
                "flake8_output": stdout[-1000:] if len(stdout) > 1000 else stdout
            },
            execution_time=execution_time,
            warnings=warnings,
            errors=errors
        )
    
    def run_security_check(self) -> QualityGateResult:
        """Run security checks with safety and bandit."""
        start_time = time.time()
        
        errors = []
        warnings = []
        security_score = 1.0
        
        # Check for requirements file
        req_files = ["requirements.txt", "pyproject.toml"]
        req_file = None
        for rf in req_files:
            if (self.project_root / rf).exists():
                req_file = rf
                break
        
        # Run safety check
        safety_passed = True
        if req_file:
            if req_file.endswith('.txt'):
                safety_cmd = ["python3", "-m", "safety", "check", "--file", req_file, "--json"]
            else:
                safety_cmd = ["python3", "-m", "safety", "check", "--json"]
            
            safety_returncode, safety_stdout, safety_stderr = self.run_command(
                safety_cmd, timeout=120
            )
            
            if safety_returncode != 0:
                safety_passed = False
                try:
                    safety_results = json.loads(safety_stdout) if safety_stdout else []
                    for vuln in safety_results:
                        errors.append(f"Security vulnerability: {vuln.get('vulnerability', 'Unknown')}")
                        security_score *= 0.8
                except:
                    errors.append("Safety check failed")
                    security_score *= 0.9
        
        # Run bandit check (if available)
        bandit_passed = True
        bandit_cmd = ["python3", "-m", "bandit", "-r", "src/", "-f", "json"]
        bandit_returncode, bandit_stdout, bandit_stderr = self.run_command(
            bandit_cmd, timeout=120
        )
        
        if bandit_returncode != 0 and "No module named 'bandit'" not in bandit_stderr:
            bandit_passed = False
            try:
                bandit_results = json.loads(bandit_stdout) if bandit_stdout else {}
                issues = bandit_results.get('results', [])
                for issue in issues[:5]:  # Limit to first 5 issues
                    warnings.append(f"Security issue: {issue.get('issue_text', 'Unknown')}")
                if len(issues) > 5:
                    security_score *= max(0.3, 1.0 - len(issues) / 20.0)
            except:
                warnings.append("Bandit check failed to parse results")
        
        execution_time = time.time() - start_time
        passed = safety_passed and bandit_passed and security_score >= 0.8
        
        return QualityGateResult(
            name="security_check",
            passed=passed,
            score=security_score,
            details={
                "safety_passed": safety_passed,
                "bandit_passed": bandit_passed,
                "requirements_file": req_file
            },
            execution_time=execution_time,
            warnings=warnings,
            errors=errors
        )
    
    def run_performance_benchmark(self) -> QualityGateResult:
        """Run performance benchmarks."""
        start_time = time.time()
        
        # Look for benchmark tests
        benchmark_files = list((self.project_root / "tests").glob("**/test_performance.py"))
        benchmark_files.extend(list((self.project_root / "tests").glob("**/benchmark*.py")))
        
        if not benchmark_files:
            return QualityGateResult(
                name="performance_benchmark",
                passed=True,
                score=0.5,  # Neutral score - no benchmarks to run
                details={"message": "No benchmark tests found"},
                execution_time=time.time() - start_time,
                warnings=["No performance benchmarks found"]
            )
        
        # Run benchmarks
        passed = True
        benchmark_results = {}
        
        for benchmark_file in benchmark_files:
            cmd = ["python3", "-m", "pytest", str(benchmark_file), "-v", "--tb=short"]
            returncode, stdout, stderr = self.run_command(cmd, timeout=300)
            
            if returncode != 0:
                passed = False
            
            # Parse benchmark results (simplified)
            benchmark_results[benchmark_file.name] = {
                "passed": returncode == 0,
                "output": stdout[-500:] if len(stdout) > 500 else stdout
            }
        
        execution_time = time.time() - start_time
        score = 1.0 if passed else 0.5
        
        errors = []
        if not passed:
            failed_benchmarks = [name for name, result in benchmark_results.items() if not result["passed"]]
            errors.append(f"Failed benchmarks: {', '.join(failed_benchmarks)}")
        
        return QualityGateResult(
            name="performance_benchmark",
            passed=passed,
            score=score,
            details={
                "benchmark_files": [f.name for f in benchmark_files],
                "results": benchmark_results
            },
            execution_time=execution_time,
            errors=errors
        )
    
    def check_documentation(self) -> QualityGateResult:
        """Check documentation completeness."""
        start_time = time.time()
        
        required_docs = [
            "README.md",
            "CONTRIBUTING.md", 
            "LICENSE",
            "CHANGELOG.md"
        ]
        
        docs_dir = self.project_root / "docs"
        
        missing_docs = []
        found_docs = []
        
        for doc in required_docs:
            if (self.project_root / doc).exists():
                found_docs.append(doc)
            else:
                missing_docs.append(doc)
        
        # Check for additional documentation
        additional_docs = []
        if docs_dir.exists():
            for doc_file in docs_dir.rglob("*.md"):
                additional_docs.append(str(doc_file.relative_to(self.project_root)))
        
        # Check README quality
        readme_score = 0.5
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()
            readme_length = len(readme_content)
            
            # Basic quality checks
            if readme_length > 1000:  # Reasonable length
                readme_score += 0.1
            if "## Installation" in readme_content:
                readme_score += 0.1
            if "## Usage" in readme_content:
                readme_score += 0.1
            if "## Contributing" in readme_content:
                readme_score += 0.1
            if "## License" in readme_content:
                readme_score += 0.1
            
            readme_score = min(1.0, readme_score)
        
        execution_time = time.time() - start_time
        completeness_score = len(found_docs) / len(required_docs)
        overall_score = (completeness_score * 0.7) + (readme_score * 0.3)
        
        passed = len(missing_docs) <= 1  # Allow one missing doc
        
        warnings = []
        if missing_docs:
            warnings.extend([f"Missing documentation: {doc}" for doc in missing_docs])
        
        return QualityGateResult(
            name="documentation_check",
            passed=passed,
            score=overall_score,
            details={
                "required_docs": required_docs,
                "found_docs": found_docs,
                "missing_docs": missing_docs,
                "additional_docs": additional_docs,
                "readme_score": readme_score,
                "completeness_score": completeness_score
            },
            execution_time=execution_time,
            warnings=warnings
        )
    
    def run_integration_tests(self) -> QualityGateResult:
        """Run integration tests."""
        start_time = time.time()
        
        # Look for integration tests
        integration_patterns = [
            "tests/integration/",
            "tests/e2e/",
            "tests/**/test_*integration*.py",
            "tests/**/test_*e2e*.py"
        ]
        
        integration_files = []
        for pattern in integration_patterns:
            if "/" in pattern:
                test_dir = self.project_root / pattern
                if test_dir.exists():
                    integration_files.extend(list(test_dir.glob("test_*.py")))
            else:
                integration_files.extend(list(self.project_root.glob(pattern)))
        
        if not integration_files:
            return QualityGateResult(
                name="integration_tests",
                passed=True,
                score=0.5,  # Neutral score
                details={"message": "No integration tests found"},
                execution_time=time.time() - start_time,
                warnings=["No integration tests found"]
            )
        
        # Run integration tests
        cmd = ["python3", "-m", "pytest"] + [str(f) for f in integration_files] + ["-v", "--tb=short"]
        returncode, stdout, stderr = self.run_command(cmd, timeout=600)  # 10 minutes
        
        execution_time = time.time() - start_time
        passed = returncode == 0
        
        test_stats = self._parse_pytest_output(stdout)
        score = 1.0 if passed else max(0.0, 1.0 - test_stats.get('failed', 10) / 10.0)
        
        errors = []
        if not passed:
            errors.append(f"Integration tests failed: {test_stats.get('failed', 'unknown')} failures")
        
        return QualityGateResult(
            name="integration_tests",
            passed=passed,
            score=score,
            details={
                "integration_files": [f.name for f in integration_files],
                "test_stats": test_stats,
                "output": stdout[-1000:] if len(stdout) > 1000 else stdout
            },
            execution_time=execution_time,
            errors=errors
        )
    
    def _parse_pytest_output(self, output: str) -> Dict[str, int]:
        """Parse pytest output to extract test statistics."""
        stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        
        # Look for summary line like "5 passed, 2 failed, 1 skipped"
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and ('failed' in line or 'skipped' in line or 'error' in line):
                # Parse the summary line
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i + 1 < len(parts):
                        count = int(part)
                        next_word = parts[i + 1]
                        if 'passed' in next_word:
                            stats['passed'] = count
                        elif 'failed' in next_word:
                            stats['failed'] = count
                        elif 'skipped' in next_word:
                            stats['skipped'] = count
                        elif 'error' in next_word:
                            stats['errors'] = count
                break
        
        return stats
    
    def run_all_gates(self) -> List[QualityGateResult]:
        """Run all quality gates."""
        gates = [
            ("Dependency Check", self.check_dependencies),
            ("Unit Tests", self.run_unit_tests),
            ("Type Checking", self.run_type_checking),
            ("Code Formatting", self.run_code_formatting),
            ("Linting", self.run_linting),
            ("Security Check", self.run_security_check),
            ("Performance Benchmark", self.run_performance_benchmark),
            ("Documentation Check", self.check_documentation),
            ("Integration Tests", self.run_integration_tests)
        ]
        
        self.results = []
        
        for gate_name, gate_function in gates:
            self.log(f"Running {gate_name}...")
            
            try:
                result = gate_function()
                self.results.append(result)
                
                status = "✓ PASSED" if result.passed else "✗ FAILED"
                score = f"({result.score:.2f})"
                time_taken = f"[{result.execution_time:.2f}s]"
                
                self.log(f"{gate_name}: {status} {score} {time_taken}")
                
                if result.errors:
                    for error in result.errors[:3]:  # Show first 3 errors
                        self.log(f"  ERROR: {error}", "ERROR")
                
                if result.warnings:
                    for warning in result.warnings[:2]:  # Show first 2 warnings
                        self.log(f"  WARNING: {warning}", "WARNING")
                
                if self.fail_fast and not result.passed:
                    self.log(f"Failing fast due to {gate_name} failure", "ERROR")
                    break
                    
            except Exception as e:
                self.log(f"{gate_name} failed with exception: {str(e)}", "ERROR")
                
                result = QualityGateResult(
                    name=gate_name.lower().replace(" ", "_"),
                    passed=False,
                    score=0.0,
                    errors=[f"Gate execution failed: {str(e)}"]
                )
                self.results.append(result)
                
                if self.fail_fast:
                    break
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.results:
            return {"error": "No quality gate results available"}
        
        total_score = sum(result.score for result in self.results)
        max_score = len(self.results)
        overall_score = total_score / max_score if max_score > 0 else 0.0
        
        passed_gates = [r for r in self.results if r.passed]
        failed_gates = [r for r in self.results if not r.passed]
        
        total_execution_time = sum(result.execution_time for result in self.results)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "overall_score": overall_score,
            "total_score": total_score,
            "max_score": max_score,
            "overall_passed": len(failed_gates) == 0,
            "gates_passed": len(passed_gates),
            "gates_failed": len(failed_gates),
            "total_gates": len(self.results),
            "total_execution_time": total_execution_time,
            "results": [result.to_dict() for result in self.results],
            "summary": {
                "passed_gates": [r.name for r in passed_gates],
                "failed_gates": [r.name for r in failed_gates],
                "critical_issues": [
                    error for result in self.results 
                    for error in result.errors
                    if any(keyword in error.lower() for keyword in ['security', 'critical', 'fatal'])
                ]
            }
        }
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run quality gates for formal-circuits-gpt")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first failure")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output report file (JSON)")
    parser.add_argument("--gates", nargs="+",
                       choices=["deps", "tests", "types", "format", "lint", "security", 
                               "perf", "docs", "integration"],
                       help="Run specific gates only")
    
    args = parser.parse_args()
    
    # Validate project root
    if not args.project_root.exists():
        print(f"Error: Project root does not exist: {args.project_root}")
        sys.exit(1)
    
    if not (args.project_root / "src").exists():
        print(f"Error: Source directory not found in: {args.project_root}")
        sys.exit(1)
    
    runner = QualityGateRunner(
        project_root=args.project_root,
        verbose=args.verbose,
        fail_fast=args.fail_fast
    )
    
    print(f"Running quality gates for formal-circuits-gpt")
    print(f"Project root: {args.project_root.absolute()}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run quality gates
    start_time = time.time()
    results = runner.run_all_gates()
    total_time = time.time() - start_time
    
    # Generate report
    report = runner.generate_report()
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Overall Score: {report['overall_score']:.2f} ({report['total_score']:.1f}/{report['max_score']})")
    print(f"Gates Passed: {report['gates_passed']}/{report['total_gates']}")
    print(f"Execution Time: {total_time:.2f}s")
    print(f"Status: {'✓ PASSED' if report['overall_passed'] else '✗ FAILED'}")
    
    if report['summary']['failed_gates']:
        print(f"\nFailed Gates: {', '.join(report['summary']['failed_gates'])}")
    
    if report['summary']['critical_issues']:
        print(f"\nCritical Issues:")
        for issue in report['summary']['critical_issues'][:5]:
            print(f"  - {issue}")
    
    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_passed'] else 1)


if __name__ == "__main__":
    main()