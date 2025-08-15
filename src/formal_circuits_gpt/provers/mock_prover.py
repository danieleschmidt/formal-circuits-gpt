"""Mock theorem prover for testing and offline use."""

from typing import List
from .base_prover import BaseProver, ProverResult


class MockProver(BaseProver):
    """Mock theorem prover that always succeeds for testing."""

    def __init__(self):
        """Initialize mock prover."""
        super().__init__(prover_path="mock", timeout=1)

    def verify_proof(self, proof_content: str) -> ProverResult:
        """Always return successful verification for testing.

        Args:
            proof_content: Proof content (ignored)

        Returns:
            ProverResult indicating success
        """
        return ProverResult(
            success=True,
            output="Mock proof verification successful",
            errors=[],
            execution_time=0.1,
            metadata={"provider": "mock", "test_mode": True},
        )

    def check_installation(self) -> bool:
        """Mock installation check - always returns True."""
        return True

    def get_version(self) -> str:
        """Get mock prover version."""
        return "MockProver 1.0.0"
