"""Coq theorem prover interface."""

import os
import re
import shutil
from typing import List, Optional
from .base_prover import BaseProver, ProverResult


class CoqInterface(BaseProver):
    """Interface for Coq theorem prover."""

    def __init__(self, coq_path: Optional[str] = None, timeout: int = 300):
        """Initialize Coq interface.

        Args:
            coq_path: Path to coqc compiler (auto-detected if None)
            timeout: Timeout for proof verification in seconds
        """
        if coq_path is None:
            coq_path = self._find_coq()

        super().__init__(coq_path, timeout)
        self.coq_extension = ".v"

    def verify_proof(self, proof_content: str) -> ProverResult:
        """Verify Coq vernacular file.

        Args:
            proof_content: Complete Coq vernacular content

        Returns:
            ProverResult with verification status
        """
        if not self.check_installation():
            return ProverResult(
                success=False,
                output="",
                errors=["Coq not found or not properly installed"],
            )

        # Create temporary Coq file
        coq_file = None
        try:
            coq_file = self._create_temp_file(proof_content, self.coq_extension)

            # Run coqc to compile the file
            command = [self.prover_path, "-q", coq_file]

            stdout, stderr, returncode, exec_time = self._run_command(command)

            # Parse output to determine success
            success = returncode == 0 and not stderr.strip()
            errors = self._parse_errors(stderr) if not success else []

            return ProverResult(
                success=success,
                output=stdout,
                errors=errors,
                execution_time=exec_time,
                metadata={"return_code": returncode, "file": coq_file},
            )

        except Exception as e:
            return ProverResult(
                success=False, output="", errors=[f"Coq verification failed: {str(e)}"]
            )

        finally:
            if coq_file:
                self._cleanup_temp_file(coq_file)
                # Also cleanup compiled files
                self._cleanup_temp_file(coq_file + "o")  # .vo file
                self._cleanup_temp_file(coq_file + "ok")  # .vok file
                self._cleanup_temp_file(coq_file + "os")  # .vos file

    def check_installation(self) -> bool:
        """Check if Coq is properly installed."""
        if not self.prover_path:
            return False

        try:
            stdout, stderr, returncode, _ = self._run_command([self.prover_path, "-v"])
            return returncode == 0 and "Coq" in stdout
        except:
            return False

    def get_version(self) -> str:
        """Get Coq version."""
        try:
            stdout, stderr, returncode, _ = self._run_command([self.prover_path, "-v"])
            if returncode == 0:
                # Extract version from output
                version_match = re.search(r"version\s+(\d+\.\d+(?:\.\d+)?)", stdout)
                if version_match:
                    return version_match.group(1)
                return stdout.strip().split("\n")[0]
            return "Unknown"
        except:
            return "Unknown"

    def _find_coq(self) -> Optional[str]:
        """Find Coq compiler in system PATH."""
        # Common Coq executable names
        coq_names = ["coqc", "coq"]

        for name in coq_names:
            path = shutil.which(name)
            if path:
                return path

        # Check common installation directories
        common_paths = [
            "/usr/local/bin/coqc",
            "/opt/coq/bin/coqc",
            "/usr/bin/coqc",
            os.path.expanduser("~/.local/bin/coqc"),
            os.path.expanduser("~/coq/bin/coqc"),
        ]

        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path

        return None

    def _parse_errors(self, stderr: str) -> List[str]:
        """Parse Coq error messages."""
        errors = []

        if not stderr.strip():
            return errors

        # Split by lines and process
        lines = stderr.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Common Coq error patterns
            if any(
                pattern in line.lower()
                for pattern in [
                    "error:",
                    "failed",
                    "exception",
                    "syntax error",
                    "type error",
                    "unbound",
                    "not found",
                    "anomaly",
                ]
            ):
                errors.append(line)
            elif line.startswith("File"):
                # File location information
                errors.append(line)

        # If no specific errors found but stderr has content, include it
        if not errors and stderr.strip():
            errors.append(stderr.strip())

        return errors

    def verify_interactive(self, proof_content: str) -> ProverResult:
        """Verify proof using interactive Coq (coqtop)."""
        coqtop_path = shutil.which("coqtop")
        if not coqtop_path:
            return ProverResult(
                success=False,
                output="",
                errors=["coqtop not found - cannot run interactive verification"],
            )

        try:
            # Run coqtop with the proof content as input
            command = [coqtop_path, "-q", "-batch"]

            stdout, stderr, returncode, exec_time = self._run_command(
                command, input_data=proof_content
            )

            # Interactive mode success is harder to detect
            success = returncode == 0 and "error" not in stderr.lower()
            errors = self._parse_errors(stderr) if not success else []

            return ProverResult(
                success=success,
                output=stdout,
                errors=errors,
                execution_time=exec_time,
                metadata={"mode": "interactive", "return_code": returncode},
            )

        except Exception as e:
            return ProverResult(
                success=False,
                output="",
                errors=[f"Interactive verification failed: {str(e)}"],
            )

    def check_dependencies(self, proof_content: str) -> List[str]:
        """Check and list required Coq dependencies."""
        dependencies = []

        # Extract Require statements
        require_pattern = re.compile(r"Require\s+(?:Import\s+)?(.+?)\.", re.MULTILINE)
        matches = require_pattern.findall(proof_content)

        for match in matches:
            # Split multiple imports
            imports = [imp.strip() for imp in match.split()]
            dependencies.extend(imports)

        return list(set(dependencies))  # Remove duplicates

    def create_makefile(self, coq_files: List[str], output_dir: str) -> str:
        """Create Makefile for Coq project."""
        makefile_content = (
            """# Coq Makefile generated by formal-circuits-gpt
        
COQC = coqc
COQDEP = coqdep

COQFILES = """
            + " ".join(coq_files)
            + """

VOFILES = $(COQFILES:.v=.vo)

all: $(VOFILES)

%.vo: %.v
\t$(COQC) -q $<

clean:
\trm -f *.vo *.vok *.vos *.glob

depend:
\t$(COQDEP) $(COQFILES) > .depend

include .depend

.PHONY: all clean depend
"""
        )

        makefile_path = os.path.join(output_dir, "Makefile")
        with open(makefile_path, "w", encoding="utf-8") as f:
            f.write(makefile_content)

        return makefile_path
