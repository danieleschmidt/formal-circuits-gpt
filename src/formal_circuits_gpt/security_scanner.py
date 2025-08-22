"""Advanced Security Scanner for Quality Gates."""

import asyncio
import ast
import re
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass
import logging

from .monitoring.logger import get_logger


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    
    severity: str  # "critical", "high", "medium", "low", "info"
    category: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: str = None  # Common Weakness Enumeration ID


@dataclass
class SecurityScanResult:
    """Result of security scanning."""
    
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[SecurityIssue]
    scan_duration_ms: float
    files_scanned: int
    security_score: float  # 0-100


class SecurityScanner:
    """Advanced security scanner for Python code."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root) if isinstance(project_root, str) else project_root
        self.logger = get_logger("security_scanner")
        
        # Security patterns to detect
        self.security_patterns = {
            "hardcoded_secrets": [
                (r'password\s*=\s*["\'][^"\']+["\']', "high", "CWE-798"),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "high", "CWE-798"),
                (r'secret\s*=\s*["\'][^"\']+["\']', "high", "CWE-798"),
                (r'token\s*=\s*["\'][^"\']+["\']', "medium", "CWE-798"),
                (r'["\'][a-zA-Z0-9]{32,}["\']', "medium", "CWE-798"),  # Long strings that might be secrets
            ],
            "sql_injection": [
                (r'\.execute\s*\(\s*["\'].*%.*["\']', "high", "CWE-89"),
                (r'\.execute\s*\(\s*f["\'].*\{.*\}.*["\']', "high", "CWE-89"),
                (r'SELECT.*\+.*FROM', "medium", "CWE-89"),
            ],
            "command_injection": [
                (r'os\.system\s*\(', "high", "CWE-78"),
                (r'subprocess\.call\s*\(', "medium", "CWE-78"),
                (r'subprocess\.run\s*\([^)]*shell=True', "high", "CWE-78"),
                (r'eval\s*\(', "critical", "CWE-95"),
                (r'exec\s*\(', "critical", "CWE-95"),
            ],
            "path_traversal": [
                (r'open\s*\([^)]*\.\./.*[^)]*\)', "high", "CWE-22"),
                (r'Path\s*\([^)]*\.\./.*[^)]*\)', "medium", "CWE-22"),
            ],
            "insecure_random": [
                (r'random\.random\(\)', "low", "CWE-330"),
                (r'random\.randint\(', "low", "CWE-330"),
            ],
            "debug_code": [
                (r'print\s*\([^)]*password[^)]*\)', "medium", "CWE-532"),
                (r'print\s*\([^)]*secret[^)]*\)', "medium", "CWE-532"),
                (r'logging.*DEBUG.*password', "medium", "CWE-532"),
            ],
            "unsafe_deserialization": [
                (r'pickle\.loads?\s*\(', "high", "CWE-502"),
                (r'yaml\.load\s*\([^)]*Loader[^)]*\)', "medium", "CWE-502"),
            ]
        }
        
        # File extensions to scan
        self.scan_extensions = {".py", ".pyx", ".pyi"}

    async def scan_project(self) -> SecurityScanResult:
        """Perform comprehensive security scan of the project."""
        start_time = asyncio.get_event_loop().time()
        
        issues = []
        files_scanned = 0
        
        # Scan Python files
        for file_path in self._get_python_files():
            try:
                file_issues = await self._scan_file(file_path)
                issues.extend(file_issues)
                files_scanned += 1
            except Exception as e:
                self.logger.error(f"Failed to scan {file_path}: {e}")
        
        # Additional security checks
        dependency_issues = await self._check_dependencies()
        issues.extend(dependency_issues)
        
        config_issues = await self._check_configuration()
        issues.extend(config_issues)
        
        # Calculate metrics
        severity_counts = {
            "critical": len([i for i in issues if i.severity == "critical"]),
            "high": len([i for i in issues if i.severity == "high"]),
            "medium": len([i for i in issues if i.severity == "medium"]),
            "low": len([i for i in issues if i.severity == "low"]),
        }
        
        # Calculate security score (0-100)
        security_score = self._calculate_security_score(severity_counts, files_scanned)
        
        scan_duration = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return SecurityScanResult(
            total_issues=len(issues),
            critical_issues=severity_counts["critical"],
            high_issues=severity_counts["high"],
            medium_issues=severity_counts["medium"],
            low_issues=severity_counts["low"],
            issues=issues,
            scan_duration_ms=scan_duration,
            files_scanned=files_scanned,
            security_score=security_score
        )

    def _get_python_files(self) -> List[Path]:
        """Get all Python files to scan."""
        python_files = []
        
        for ext in self.scan_extensions:
            python_files.extend(self.project_root.rglob(f"*{ext}"))
        
        # Filter out common directories to skip
        skip_dirs = {".git", "__pycache__", ".pytest_cache", ".tox", "venv", ".venv", "node_modules"}
        
        filtered_files = []
        for file_path in python_files:
            if not any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                filtered_files.append(file_path)
        
        return filtered_files

    async def _scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a single file for security issues."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            
            # Pattern-based scanning
            for category, patterns in self.security_patterns.items():
                for pattern, severity, cwe_id in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        line_content = lines[line_number - 1] if line_number <= len(lines) else ""
                        
                        issue = SecurityIssue(
                            severity=severity,
                            category=category,
                            description=f"Potential {category.replace('_', ' ')} detected",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_number,
                            code_snippet=line_content.strip(),
                            recommendation=self._get_recommendation(category),
                            cwe_id=cwe_id
                        )
                        issues.append(issue)
            
            # AST-based analysis for more sophisticated checks
            ast_issues = await self._analyze_ast(file_path, content)
            issues.extend(ast_issues)
            
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
        
        return issues

    async def _analyze_ast(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Perform AST-based security analysis."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # Check for eval/exec
                        if func_name in ["eval", "exec"]:
                            issue = SecurityIssue(
                                severity="critical",
                                category="code_injection",
                                description=f"Use of dangerous function: {func_name}",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=node.lineno,
                                code_snippet=ast.get_source_segment(content, node) or "",
                                recommendation="Replace with safer alternatives",
                                cwe_id="CWE-95"
                            )
                            issues.append(issue)
                
                # Check for insecure imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["pickle", "cPickle"]:
                            issue = SecurityIssue(
                                severity="medium",
                                category="insecure_deserialization",
                                description=f"Import of potentially unsafe module: {alias.name}",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=node.lineno,
                                code_snippet=f"import {alias.name}",
                                recommendation="Use safer serialization methods like json",
                                cwe_id="CWE-502"
                            )
                            issues.append(issue)
        
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            self.logger.error(f"AST analysis error for {file_path}: {e}")
        
        return issues

    async def _check_dependencies(self) -> List[SecurityIssue]:
        """Check for known vulnerable dependencies."""
        issues = []
        
        try:
            # Check if safety is available for vulnerability scanning
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Safety found no vulnerabilities
                pass
            else:
                # Parse safety output for vulnerabilities
                try:
                    import json
                    vulnerabilities = json.loads(result.stdout)
                    
                    for vuln in vulnerabilities:
                        issue = SecurityIssue(
                            severity="high",
                            category="vulnerable_dependency",
                            description=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                            file_path="requirements.txt",
                            line_number=1,
                            code_snippet=vuln.get('vulnerability', ''),
                            recommendation="Update to secure version",
                            cwe_id="CWE-1104"
                        )
                        issues.append(issue)
                        
                except json.JSONDecodeError:
                    pass
                    
        except subprocess.TimeoutExpired:
            issues.append(SecurityIssue(
                severity="low",
                category="scan_timeout",
                description="Dependency vulnerability scan timed out",
                file_path="",
                line_number=0,
                code_snippet="",
                recommendation="Install 'safety' tool for dependency scanning"
            ))
        except FileNotFoundError:
            # Safety tool not installed
            issues.append(SecurityIssue(
                severity="info",
                category="missing_tool",
                description="Safety tool not available for dependency scanning",
                file_path="",
                line_number=0,
                code_snippet="",
                recommendation="Install 'safety' tool: pip install safety"
            ))
        
        return issues

    async def _check_configuration(self) -> List[SecurityIssue]:
        """Check configuration files for security issues."""
        issues = []
        
        # Check for insecure configurations
        config_files = [
            self.project_root / "pyproject.toml",
            self.project_root / "setup.py",
            self.project_root / ".env",
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    content = config_file.read_text()
                    
                    # Check for secrets in config
                    if re.search(r'password\s*=\s*["\'][^"\']{3,}["\']', content, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity="high",
                            category="config_secrets",
                            description="Potential password in configuration file",
                            file_path=str(config_file.relative_to(self.project_root)),
                            line_number=1,
                            code_snippet="",
                            recommendation="Use environment variables for secrets",
                            cwe_id="CWE-798"
                        ))
                    
                    # Check for debug mode in production configs
                    if re.search(r'debug\s*=\s*true', content, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity="medium",
                            category="debug_enabled",
                            description="Debug mode enabled in configuration",
                            file_path=str(config_file.relative_to(self.project_root)),
                            line_number=1,
                            code_snippet="",
                            recommendation="Disable debug mode in production",
                            cwe_id="CWE-489"
                        ))
                        
                except Exception as e:
                    self.logger.error(f"Error checking config file {config_file}: {e}")
        
        return issues

    def _get_recommendation(self, category: str) -> str:
        """Get security recommendation for a category."""
        recommendations = {
            "hardcoded_secrets": "Use environment variables or secure secret management",
            "sql_injection": "Use parameterized queries or ORM methods",
            "command_injection": "Validate input and avoid shell=True",
            "path_traversal": "Validate and sanitize file paths",
            "insecure_random": "Use secrets module for cryptographic purposes",
            "debug_code": "Remove debug statements containing sensitive data",
            "unsafe_deserialization": "Use safe serialization formats like JSON"
        }
        return recommendations.get(category, "Review and fix security issue")

    def _calculate_security_score(self, severity_counts: Dict[str, int], files_scanned: int) -> float:
        """Calculate overall security score (0-100)."""
        if files_scanned == 0:
            return 0.0
        
        # Weight different severity levels
        weights = {
            "critical": 25,
            "high": 10,
            "medium": 5,
            "low": 1
        }
        
        total_penalty = sum(count * weight for severity, count in severity_counts.items() for weight in [weights.get(severity, 0)])
        
        # Base score starts at 100, subtract penalties
        base_score = 100.0
        penalty_per_file = total_penalty / files_scanned if files_scanned > 0 else total_penalty
        
        # Cap penalty to avoid negative scores
        final_score = max(0.0, base_score - penalty_per_file)
        
        return final_score

    async def generate_security_report(self, scan_result: SecurityScanResult) -> str:
        """Generate a detailed security report."""
        report = f"""
# Security Scan Report

## Summary
- **Security Score**: {scan_result.security_score:.1f}/100
- **Total Issues**: {scan_result.total_issues}
- **Files Scanned**: {scan_result.files_scanned}
- **Scan Duration**: {scan_result.scan_duration_ms:.0f}ms

## Issue Breakdown
- **Critical**: {scan_result.critical_issues}
- **High**: {scan_result.high_issues}
- **Medium**: {scan_result.medium_issues}
- **Low**: {scan_result.low_issues}

## Detailed Issues

"""
        
        # Group issues by severity
        for severity in ["critical", "high", "medium", "low"]:
            severity_issues = [issue for issue in scan_result.issues if issue.severity == severity]
            
            if severity_issues:
                report += f"### {severity.upper()} Severity Issues\n\n"
                
                for issue in severity_issues:
                    report += f"""
**{issue.description}**
- File: {issue.file_path}:{issue.line_number}
- Category: {issue.category}
- CWE: {issue.cwe_id or 'N/A'}
- Code: `{issue.code_snippet}`
- Recommendation: {issue.recommendation}

"""
        
        return report


async def main():
    """Main function for standalone security scanning."""
    import sys
    
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    scanner = SecurityScanner(project_root)
    result = await scanner.scan_project()
    
    print(f"Security Scan Results")
    print(f"====================")
    print(f"Security Score: {result.security_score:.1f}/100")
    print(f"Total Issues: {result.total_issues}")
    print(f"Critical: {result.critical_issues}, High: {result.high_issues}, Medium: {result.medium_issues}, Low: {result.low_issues}")
    print(f"Files Scanned: {result.files_scanned}")
    print(f"Duration: {result.scan_duration_ms:.0f}ms")
    
    if result.issues:
        print("\nTop Issues:")
        for issue in sorted(result.issues, key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.severity], reverse=True)[:5]:
            print(f"  [{issue.severity.upper()}] {issue.description} ({issue.file_path}:{issue.line_number})")


if __name__ == "__main__":
    asyncio.run(main())