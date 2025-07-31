"""Custom exceptions for formal-circuits-gpt."""


class FormalCircuitsGPTError(Exception):
    """Base exception for all formal-circuits-gpt errors."""
    pass


class VerificationError(FormalCircuitsGPTError):
    """Raised when circuit verification fails."""
    pass


class ProofFailure(VerificationError):
    """Raised when proof generation or validation fails."""
    
    def __init__(self, message: str, failed_goal: str = None, counterexample: str = None):
        super().__init__(message)
        self.failed_goal = failed_goal
        self.counterexample = counterexample


class ParsingError(FormalCircuitsGPTError):
    """Raised when HDL parsing fails."""
    pass


class TranslationError(FormalCircuitsGPTError):
    """Raised when HDL to formal language translation fails."""
    pass


class ProverError(FormalCircuitsGPTError):
    """Raised when theorem prover interaction fails."""
    pass