#!/usr/bin/env python3
"""
Autonomous Value Discovery Script
Discovers and scores development tasks for prioritized execution.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class ValueDiscoveryEngine:
    """Engine for discovering and scoring development tasks."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.config = self._load_config()
        self.metrics = self._load_metrics()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        config_path = self.repo_path / ".terragon" / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing value metrics."""
        metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}
    
    def discover_tasks(self) -> List[Dict[str, Any]]:
        """Discover potential development tasks."""
        tasks = []
        
        # Git history analysis
        tasks.extend(self._analyze_git_history())
        
        # Static code analysis
        tasks.extend(self._analyze_code_quality())
        
        # Security analysis
        tasks.extend(self._analyze_security())
        
        # Dependency analysis
        tasks.extend(self._analyze_dependencies())
        
        # Documentation analysis
        tasks.extend(self._analyze_documentation())
        
        return tasks
    
    def _analyze_git_history(self) -> List[Dict[str, Any]]:
        """Analyze git history for improvement opportunities."""
        tasks = []
        
        try:
            # Find TODO/FIXME comments
            result = subprocess.run(
                ["git", "grep", "-n", "-i", "todo\\|fixme\\|hack\\|temporary"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        tasks.append({
                            "id": f"todo-{len(tasks)+1}",
                            "title": f"Address TODO/FIXME: {line.split(':', 2)[-1].strip()[:50]}",
                            "category": "technical_debt",
                            "source": "git_history",
                            "location": line.split(':', 2)[:2],
                            "effort_hours": 2,
                            "impact": 5,
                            "confidence": 8,
                            "ease": 7
                        })
        except subprocess.SubprocessError:
            pass
        
        return tasks
    
    def _analyze_code_quality(self) -> List[Dict[str, Any]]:
        """Analyze code quality metrics."""
        tasks = []
        
        # Check for high complexity functions
        try:
            result = subprocess.run(
                ["python", "-m", "radon", "cc", "src/", "--min=C"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0 and result.stdout.strip():
                tasks.append({
                    "id": "complexity-001",
                    "title": "Reduce cyclomatic complexity in identified functions",
                    "category": "code_quality",
                    "source": "static_analysis",
                    "effort_hours": 4,
                    "impact": 7,
                    "confidence": 8,
                    "ease": 6
                })
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return tasks
    
    def _analyze_security(self) -> List[Dict[str, Any]]:
        """Analyze security vulnerabilities."""
        tasks = []
        
        # Run bandit security analysis
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode != 0:  # Bandit found issues
                tasks.append({
                    "id": "security-001", 
                    "title": "Fix security vulnerabilities identified by bandit",
                    "category": "security",
                    "source": "security_scan",
                    "effort_hours": 3,
                    "impact": 9,
                    "confidence": 9,
                    "ease": 8,
                    "security_boost": True
                })
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return tasks
    
    def _analyze_dependencies(self) -> List[Dict[str, Any]]:
        """Analyze dependency vulnerabilities and updates."""
        tasks = []
        
        # Check for dependency vulnerabilities
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode != 0:  # Vulnerabilities found
                tasks.append({
                    "id": "deps-001",
                    "title": "Update vulnerable dependencies",
                    "category": "security", 
                    "source": "dependency_scan",
                    "effort_hours": 2,
                    "impact": 8,
                    "confidence": 9,
                    "ease": 9,
                    "security_boost": True
                })
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return tasks
    
    def _analyze_documentation(self) -> List[Dict[str, Any]]:
        """Analyze documentation coverage and quality."""
        tasks = []
        
        # Check for missing docstrings
        python_files = list(self.repo_path.glob("src/**/*.py"))
        missing_docs = 0
        
        for py_file in python_files:
            with open(py_file, 'r') as f:
                content = f.read()
                if 'def ' in content and '"""' not in content:
                    missing_docs += 1
        
        if missing_docs > 0:
            tasks.append({
                "id": "docs-001",
                "title": f"Add docstrings to {missing_docs} functions/classes",
                "category": "documentation",
                "source": "documentation_analysis", 
                "effort_hours": missing_docs * 0.5,
                "impact": 6,
                "confidence": 9,
                "ease": 8
            })
        
        return tasks
    
    def score_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score tasks using WSJF + ICE + Technical Debt model."""
        weights = self.config.get("scoring", {}).get("weights", {}).get("developing", {
            "wsjf": 0.5, "ice": 0.2, "technicalDebt": 0.2, "security": 0.1
        })
        
        for task in tasks:
            # ICE Score
            ice_score = task["impact"] * task["confidence"] * task["ease"]
            
            # WSJF Components
            cost_of_delay = task["impact"] * 10  # Business value
            job_size = task["effort_hours"]
            wsjf_score = cost_of_delay / job_size if job_size > 0 else 0
            
            # Technical Debt Score
            debt_score = 50 if task["category"] == "technical_debt" else 20
            
            # Composite Score
            composite = (
                weights["wsjf"] * wsjf_score +
                weights["ice"] * ice_score / 100 +  # Normalize ICE
                weights["technicalDebt"] * debt_score +
                weights["security"] * (30 if task["category"] == "security" else 0)
            )
            
            # Apply security boost
            if task.get("security_boost", False):
                composite *= 2.0
            
            task["scores"] = {
                "ice": ice_score,
                "wsjf": wsjf_score,
                "technical_debt": debt_score,
                "composite": round(composite, 1)
            }
        
        # Sort by composite score
        return sorted(tasks, key=lambda x: x["scores"]["composite"], reverse=True)
    
    def update_backlog(self, tasks: List[Dict[str, Any]]) -> None:
        """Update the BACKLOG.md file with discovered tasks."""
        backlog_path = self.repo_path / "BACKLOG.md"
        
        # Generate backlog content
        content = f"""# 📊 Autonomous Value Backlog

**Repository**: formal-circuits-gpt  
**Maturity Level**: Developing (35% SDLC)  
**Last Updated**: {datetime.now().isoformat()}Z  
**Next Execution**: Immediate (on PR merge)  

## 🎯 Next Best Value Item
"""
        
        if tasks:
            top_task = tasks[0]
            content += f"""**[{top_task['id'].upper()}] {top_task['title']}**
- **Composite Score**: {top_task['scores']['composite']}
- **WSJF**: {top_task['scores']['wsjf']:.1f} | **ICE**: {top_task['scores']['ice']} | **Tech Debt**: {top_task['scores']['technical_debt']}
- **Estimated Effort**: {top_task['effort_hours']} hours
- **Category**: {top_task['category'].title()}

"""
        
        content += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, task in enumerate(tasks[:10], 1):
            content += f"| {i} | {task['id'].upper()} | {task['title'][:40]}... | {task['scores']['composite']} | {task['category'].title()} | {task['effort_hours']} |\n"
        
        content += f"""

## 📈 Value Metrics
- **Items Discovered**: {len(tasks)}
- **Average Composite Score**: {sum(t['scores']['composite'] for t in tasks) / len(tasks):.1f if tasks else 0}
- **Security Items**: {len([t for t in tasks if t['category'] == 'security'])}
- **Technical Debt Items**: {len([t for t in tasks if t['category'] == 'technical_debt'])}

---

*This backlog is continuously updated through autonomous value discovery.*
"""
        
        with open(backlog_path, 'w') as f:
            f.write(content)
    
    def run_discovery(self) -> Dict[str, Any]:
        """Run complete value discovery cycle."""
        print("🔍 Starting autonomous value discovery...")
        
        # Discover tasks
        tasks = self.discover_tasks()
        print(f"📋 Discovered {len(tasks)} potential tasks")
        
        # Score tasks
        scored_tasks = self.score_tasks(tasks)
        print(f"📊 Scored and prioritized {len(scored_tasks)} tasks")
        
        # Update backlog
        self.update_backlog(scored_tasks)
        print("📝 Updated BACKLOG.md")
        
        # Update metrics
        self.metrics["backlog_metrics"]["totalItems"] = len(scored_tasks)
        self.metrics["last_discovery"] = datetime.now().isoformat() + "Z"
        
        metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return {
            "tasks_discovered": len(tasks),
            "tasks_scored": len(scored_tasks),
            "top_score": scored_tasks[0]["scores"]["composite"] if scored_tasks else 0,
            "next_task": scored_tasks[0] if scored_tasks else None
        }


if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    results = engine.run_discovery()
    print(f"✅ Discovery complete: {results}")