"""Health check system for formal-circuits-gpt components."""

import os
import shutil
import time
from typing import Dict, List, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from ..provers import IsabelleInterface, CoqInterface
from ..llm.llm_client import LLMManager


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    component: str
    status: HealthStatus
    message: str
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.metadata = self.metadata or {}


class HealthChecker:
    """System health checker for all components."""

    def __init__(self):
        """Initialize health checker."""
        self.checks = {
            "system": self._check_system_resources,
            "provers": self._check_theorem_provers,
            "llm": self._check_llm_connectivity,
            "dependencies": self._check_dependencies,
            "storage": self._check_storage,
        }

    def check_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}

        for check_name, check_func in self.checks.items():
            try:
                start_time = time.time()
                results[check_name] = check_func()
                results[check_name].response_time_ms = (time.time() - start_time) * 1000
            except Exception as e:
                results[check_name] = HealthCheck(
                    component=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    response_time_ms=0.0,
                )

        return results

    def check_component(self, component: str) -> HealthCheck:
        """Run health check for specific component."""
        if component not in self.checks:
            return HealthCheck(
                component=component,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown component: {component}",
            )

        try:
            start_time = time.time()
            result = self.checks[component]()
            result.response_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return HealthCheck(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
            )

    def check_health(self) -> Dict[str, Any]:
        """Main health check method for compatibility."""
        results = self.check_all()
        overall_status, overall_message = self.get_overall_status(results)
        
        return {
            "status": overall_status.value,
            "message": overall_message,
            "components": {name: {
                "status": check.status.value,
                "message": check.message,
                "response_time_ms": check.response_time_ms,
                "metadata": check.metadata
            } for name, check in results.items()}
        }

    def get_overall_status(
        self, results: Dict[str, HealthCheck]
    ) -> Tuple[HealthStatus, str]:
        """Get overall system health status."""
        unhealthy_count = sum(
            1 for r in results.values() if r.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for r in results.values() if r.status == HealthStatus.DEGRADED
        )

        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY, f"{unhealthy_count} components unhealthy"
        elif degraded_count > 0:
            return HealthStatus.DEGRADED, f"{degraded_count} components degraded"
        else:
            return HealthStatus.HEALTHY, "All components healthy"

    def _check_system_resources(self) -> HealthCheck:
        """Check system resource availability."""
        try:
            import psutil

            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Check disk usage
            disk = psutil.disk_usage("/")
            disk_usage = disk.percent

            # Determine status
            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%"
            elif cpu_usage > 70 or memory_usage > 70 or disk_usage > 80:
                status = HealthStatus.DEGRADED
                message = f"Moderate resource usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resource usage normal - CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%"

            return HealthCheck(
                component="system",
                status=status,
                message=message,
                metadata={
                    "cpu_usage_percent": cpu_usage,
                    "memory_usage_percent": memory_usage,
                    "disk_usage_percent": disk_usage,
                    "memory_available_gb": memory.available / (1024**3),
                },
            )

        except ImportError:
            # psutil not available, basic check
            return HealthCheck(
                component="system",
                status=HealthStatus.DEGRADED,
                message="Resource monitoring unavailable (psutil not installed)",
            )
        except Exception as e:
            return HealthCheck(
                component="system",
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {str(e)}",
            )

    def _check_theorem_provers(self) -> HealthCheck:
        """Check theorem prover availability."""
        try:
            isabelle = IsabelleInterface()
            coq = CoqInterface()

            isabelle_ok = isabelle.check_installation()
            coq_ok = coq.check_installation()

            metadata = {"isabelle_available": isabelle_ok, "coq_available": coq_ok}

            if isabelle_ok:
                metadata["isabelle_version"] = isabelle.get_version()
            if coq_ok:
                metadata["coq_version"] = coq.get_version()

            if isabelle_ok and coq_ok:
                status = HealthStatus.HEALTHY
                message = "All theorem provers available"
            elif isabelle_ok or coq_ok:
                status = HealthStatus.DEGRADED
                available = "Isabelle" if isabelle_ok else "Coq"
                message = f"Only {available} available"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No theorem provers available"

            return HealthCheck(
                component="provers", status=status, message=message, metadata=metadata
            )

        except Exception as e:
            return HealthCheck(
                component="provers",
                status=HealthStatus.UNHEALTHY,
                message=f"Prover check failed: {str(e)}",
            )

    def _check_llm_connectivity(self) -> HealthCheck:
        """Check LLM service connectivity."""
        try:
            llm_manager = LLMManager.create_default()

            if not llm_manager.clients:
                return HealthCheck(
                    component="llm",
                    status=HealthStatus.UNHEALTHY,
                    message="No LLM clients configured",
                )

            # Test connectivity with a simple prompt
            test_prompt = "Hello, respond with 'OK' if you can read this."

            available_clients = []
            for client_name, client in llm_manager.clients.items():
                try:
                    # Quick connectivity test (with short timeout)
                    response = client.generate_sync(
                        test_prompt, max_tokens=10, temperature=0.0
                    )
                    if response and response.content:
                        available_clients.append(client_name)
                except:
                    pass  # Client failed, continue checking others

            if len(available_clients) == len(llm_manager.clients):
                status = HealthStatus.HEALTHY
                message = f"All LLM clients available: {', '.join(available_clients)}"
            elif available_clients:
                status = HealthStatus.DEGRADED
                message = f"Some LLM clients available: {', '.join(available_clients)}"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No LLM clients responding"

            return HealthCheck(
                component="llm",
                status=status,
                message=message,
                metadata={
                    "configured_clients": list(llm_manager.clients.keys()),
                    "available_clients": available_clients,
                    "default_client": llm_manager.default_client,
                },
            )

        except Exception as e:
            return HealthCheck(
                component="llm",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM connectivity check failed: {str(e)}",
            )

    def _check_dependencies(self) -> HealthCheck:
        """Check critical dependencies."""
        required_packages = ["click", "rich", "pydantic", "networkx", "jsonschema"]

        optional_packages = ["openai", "anthropic", "psutil"]

        missing_required = []
        missing_optional = []

        # Check required packages
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)

        # Check optional packages
        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(package)

        if missing_required:
            status = HealthStatus.UNHEALTHY
            message = f"Missing required packages: {', '.join(missing_required)}"
        elif missing_optional:
            status = HealthStatus.DEGRADED
            message = f"Missing optional packages: {', '.join(missing_optional)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All dependencies available"

        return HealthCheck(
            component="dependencies",
            status=status,
            message=message,
            metadata={
                "missing_required": missing_required,
                "missing_optional": missing_optional,
            },
        )

    def _check_storage(self) -> HealthCheck:
        """Check storage and temporary directory access."""
        try:
            import tempfile

            # Check temp directory access
            temp_dir = tempfile.gettempdir()

            # Test write access
            test_file = os.path.join(temp_dir, "formal_circuits_gpt_health_check")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                temp_writable = True
            except:
                temp_writable = False

            # Check working directory access
            working_dir = "/tmp/formal_circuits_gpt"
            try:
                os.makedirs(working_dir, exist_ok=True)
                test_file = os.path.join(working_dir, "health_check")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                working_writable = True
            except:
                working_writable = False

            if temp_writable and working_writable:
                status = HealthStatus.HEALTHY
                message = "Storage access OK"
            elif temp_writable or working_writable:
                status = HealthStatus.DEGRADED
                message = "Limited storage access"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Storage access failed"

            return HealthCheck(
                component="storage",
                status=status,
                message=message,
                metadata={
                    "temp_directory": temp_dir,
                    "temp_writable": temp_writable,
                    "working_directory": working_dir,
                    "working_writable": working_writable,
                },
            )

        except Exception as e:
            return HealthCheck(
                component="storage",
                status=HealthStatus.UNHEALTHY,
                message=f"Storage check failed: {str(e)}",
            )
