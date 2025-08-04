"""Isabelle/HOL theorem prover interface."""

import os
import re
import shutil
from typing import List, Optional
from .base_prover import BaseProver, ProverResult


class IsabelleInterface(BaseProver):
    """Interface for Isabelle/HOL theorem prover."""
    
    def __init__(self, isabelle_path: Optional[str] = None, timeout: int = 300):
        """Initialize Isabelle interface.
        
        Args:
            isabelle_path: Path to Isabelle executable (auto-detected if None)
            timeout: Timeout for proof verification in seconds
        """
        if isabelle_path is None:
            isabelle_path = self._find_isabelle()
        
        super().__init__(isabelle_path, timeout)
        self.theory_extension = ".thy"
    
    def verify_proof(self, proof_content: str) -> ProverResult:
        """Verify Isabelle theory file.
        
        Args:
            proof_content: Complete Isabelle theory content
            
        Returns:
            ProverResult with verification status
        """
        if not self.check_installation():
            return ProverResult(
                success=False,
                output="",
                errors=["Isabelle not found or not properly installed"]
            )
        
        # Create temporary theory file
        theory_file = None
        try:
            theory_file = self._create_temp_file(proof_content, self.theory_extension)
            
            # Extract theory name from content
            theory_name = self._extract_theory_name(proof_content)
            if not theory_name:
                return ProverResult(
                    success=False,
                    output="",
                    errors=["Could not extract theory name from proof content"]
                )
            
            # Run Isabelle to check the theory
            command = [
                self.prover_path,
                "build",
                "-D", os.path.dirname(theory_file),
                "-v"
            ]
            
            stdout, stderr, returncode, exec_time = self._run_command(command)
            
            # Parse output to determine success
            success = returncode == 0 and "error" not in stderr.lower()
            errors = self._parse_errors(stderr) if not success else []
            
            return ProverResult(
                success=success,
                output=stdout,
                errors=errors,
                execution_time=exec_time,
                metadata={"theory_name": theory_name, "return_code": returncode}
            )
            
        except Exception as e:
            return ProverResult(
                success=False,
                output="",
                errors=[f"Isabelle verification failed: {str(e)}"]
            )
        
        finally:
            if theory_file:
                self._cleanup_temp_file(theory_file)
    
    def check_installation(self) -> bool:
        """Check if Isabelle is properly installed."""
        if not self.prover_path:
            return False
        
        try:
            stdout, stderr, returncode, _ = self._run_command([self.prover_path, "version"])
            return returncode == 0 and "Isabelle" in stdout
        except:
            return False
    
    def get_version(self) -> str:
        """Get Isabelle version."""
        try:
            stdout, stderr, returncode, _ = self._run_command([self.prover_path, "version"])
            if returncode == 0:
                # Extract version from output
                version_match = re.search(r"Isabelle(\d+(?:-\d+)?)", stdout)
                if version_match:
                    return version_match.group(1)
                return stdout.strip()
            return "Unknown"
        except:
            return "Unknown"
    
    def _find_isabelle(self) -> Optional[str]:
        """Find Isabelle executable in system PATH."""
        # Common Isabelle executable names
        isabelle_names = ["isabelle", "Isabelle"]
        
        for name in isabelle_names:
            path = shutil.which(name)
            if path:
                return path
        
        # Check common installation directories
        common_paths = [
            "/usr/local/bin/isabelle",
            "/opt/Isabelle/bin/isabelle",
            "/usr/bin/isabelle",
            os.path.expanduser("~/Isabelle/bin/isabelle")
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def _extract_theory_name(self, proof_content: str) -> Optional[str]:
        """Extract theory name from Isabelle theory content."""
        # Look for theory declaration
        theory_match = re.search(r"theory\s+(\w+)", proof_content, re.IGNORECASE)
        if theory_match:
            return theory_match.group(1)
        return None
    
    def _parse_errors(self, stderr: str) -> List[str]:
        """Parse Isabelle error messages."""
        errors = []
        
        # Split by lines and look for error patterns
        lines = stderr.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Common Isabelle error patterns
            if any(pattern in line.lower() for pattern in [
                'error:', 'failed', 'exception', 'parse error', 
                'type error', 'proof failed', 'unfinished'
            ]):
                errors.append(line)
        
        # If no specific errors found but stderr has content, include it
        if not errors and stderr.strip():
            errors.append(stderr.strip())
        
        return errors
    
    def create_root_file(self, theory_name: str, output_dir: str) -> str:
        """Create ROOT file for Isabelle session."""
        root_content = f"""session {theory_name} = HOL +
  options [document = pdf, document_output = "output"]
  theories
    {theory_name}
"""
        
        root_path = os.path.join(output_dir, "ROOT")
        with open(root_path, 'w', encoding='utf-8') as f:
            f.write(root_content)
        
        return root_path
    
    def verify_with_build_session(self, proof_content: str, session_name: str = "Circuit_Verification") -> ProverResult:
        """Verify proof using Isabelle build session."""
        import tempfile
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory(dir=self.working_directory) as temp_dir:
            try:
                # Extract theory name
                theory_name = self._extract_theory_name(proof_content) or session_name
                
                # Write theory file
                theory_file = os.path.join(temp_dir, f"{theory_name}.thy")
                with open(theory_file, 'w', encoding='utf-8') as f:
                    f.write(proof_content)
                
                # Create ROOT file
                self.create_root_file(theory_name, temp_dir)
                
                # Run Isabelle build
                command = [
                    self.prover_path,
                    "build",
                    "-D", temp_dir,
                    "-v"
                ]
                
                stdout, stderr, returncode, exec_time = self._run_command(command)
                
                success = returncode == 0 and "error" not in stderr.lower()
                errors = self._parse_errors(stderr) if not success else []
                
                return ProverResult(
                    success=success,
                    output=stdout,
                    errors=errors,
                    execution_time=exec_time,
                    metadata={"theory_name": theory_name, "session": session_name}
                )
                
            except Exception as e:
                return ProverResult(
                    success=False,
                    output="",
                    errors=[f"Session build failed: {str(e)}"]
                )