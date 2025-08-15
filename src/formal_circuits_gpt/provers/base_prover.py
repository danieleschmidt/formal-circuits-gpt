"""Base class for theorem prover interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProverResult:
    """Result from theorem prover execution."""

    success: bool
    output: str
    errors: List[str]
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.metadata = self.metadata or {}


class BaseProver(ABC):
    """Abstract base class for theorem prover interfaces."""

    def __init__(self, prover_path: Optional[str] = None, timeout: int = 300):
        """Initialize prover interface.

        Args:
            prover_path: Path to prover executable
            timeout: Timeout for proof verification in seconds
        """
        self.prover_path = prover_path
        self.timeout = timeout
        self.working_directory = "/tmp/formal_circuits_gpt"

    @abstractmethod
    def verify_proof(self, proof_content: str) -> ProverResult:
        """Verify a proof using the theorem prover.

        Args:
            proof_content: Complete proof code

        Returns:
            ProverResult with verification status and details
        """
        pass

    @abstractmethod
    def check_installation(self) -> bool:
        """Check if the theorem prover is properly installed."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get the version of the theorem prover."""
        pass

    def _create_temp_file(self, content: str, extension: str) -> str:
        """Create temporary file with proof content."""
        import tempfile
        import os

        # Ensure working directory exists
        os.makedirs(self.working_directory, exist_ok=True)

        # Create temporary file
        fd, temp_path = tempfile.mkstemp(
            suffix=extension, dir=self.working_directory, text=True
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
        except:
            os.close(fd)
            raise

        return temp_path

    def _cleanup_temp_file(self, file_path: str) -> None:
        """Remove temporary file."""
        import os

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass  # Ignore cleanup errors

    def _run_command(self, command: List[str], input_data: str = None) -> tuple:
        """Run command and return (stdout, stderr, returncode)."""
        import subprocess
        import time

        start_time = time.time()

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_directory,
            )

            stdout, stderr = process.communicate(input=input_data, timeout=self.timeout)

            execution_time = time.time() - start_time

            return stdout, stderr, process.returncode, execution_time

        except subprocess.TimeoutExpired:
            process.kill()
            execution_time = time.time() - start_time
            return "", f"Timeout after {self.timeout} seconds", -1, execution_time

        except Exception as e:
            execution_time = time.time() - start_time
            return "", f"Command execution failed: {str(e)}", -1, execution_time
