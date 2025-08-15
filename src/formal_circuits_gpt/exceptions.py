"""Custom exceptions for formal-circuits-gpt with enhanced error handling."""

from typing import Dict, Any, Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    PARSING = "parsing"
    VERIFICATION = "verification"
    PROVER = "prover"
    LLM = "llm"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    FILESYSTEM = "filesystem"


class FormalCircuitsGPTError(Exception):
    """Enhanced base exception for all formal-circuits-gpt errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: str = None,
        context: Dict[str, Any] = None,
        suggestions: List[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code
        self.context = context or {}
        self.suggestions = suggestions or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value if self.category else None,
            "severity": self.severity.value,
            "error_code": self.error_code,
            "context": self.context,
            "suggestions": self.suggestions,
        }

    def get_user_friendly_message(self) -> str:
        """Get a user-friendly error message with suggestions."""
        msg = f"{self.message}"
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"\n  {i}. {suggestion}"
        return msg


class VerificationError(FormalCircuitsGPTError):
    """Raised when circuit verification fails."""

    def __init__(
        self,
        message: str,
        circuit_name: str = None,
        properties_failed: List[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.VERIFICATION, **kwargs)
        self.circuit_name = circuit_name
        self.properties_failed = properties_failed or []

        # Add circuit-specific context
        self.context.update(
            {"circuit_name": circuit_name, "properties_failed": properties_failed}
        )


class ProofFailure(VerificationError):
    """Raised when proof generation or validation fails."""

    def __init__(
        self,
        message: str,
        failed_goal: str = None,
        counterexample: str = None,
        proof_attempt: str = None,
        refinement_attempts: int = 0,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.failed_goal = failed_goal
        self.counterexample = counterexample
        self.proof_attempt = proof_attempt
        self.refinement_attempts = refinement_attempts

        # Add proof-specific context
        self.context.update(
            {
                "failed_goal": failed_goal,
                "counterexample": counterexample,
                "refinement_attempts": refinement_attempts,
            }
        )

        # Add helpful suggestions
        if refinement_attempts < 3:
            self.suggestions.append("Try increasing the number of refinement rounds")
        if counterexample:
            self.suggestions.append(
                "Analyze the counterexample to understand the failure"
            )
        self.suggestions.append(
            "Consider simplifying the property or breaking it into smaller parts"
        )


class ParsingError(FormalCircuitsGPTError):
    """Enhanced HDL parsing error with detailed diagnostics."""

    def __init__(
        self,
        message: str,
        hdl_type: str = None,
        line_number: int = None,
        column: int = None,
        source_snippet: str = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.PARSING, **kwargs)
        self.hdl_type = hdl_type
        self.line_number = line_number
        self.column = column
        self.source_snippet = source_snippet

        # Add parsing context
        self.context.update(
            {
                "hdl_type": hdl_type,
                "line_number": line_number,
                "column": column,
                "source_snippet": source_snippet,
            }
        )

        # Add helpful suggestions
        if hdl_type:
            self.suggestions.append(f"Verify {hdl_type.upper()} syntax compliance")
        if line_number:
            self.suggestions.append(f"Check syntax around line {line_number}")
        self.suggestions.extend(
            [
                "Ensure all modules are properly closed with 'endmodule'",
                "Check for missing semicolons or mismatched parentheses",
            ]
        )


class TranslationError(FormalCircuitsGPTError):
    """Raised when HDL to formal language translation fails."""

    def __init__(
        self,
        message: str,
        target_language: str = None,
        unsupported_constructs: List[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.VERIFICATION, **kwargs)
        self.target_language = target_language
        self.unsupported_constructs = unsupported_constructs or []

        self.context.update(
            {
                "target_language": target_language,
                "unsupported_constructs": unsupported_constructs,
            }
        )

        if unsupported_constructs:
            self.suggestions.append(
                f"Consider rewriting using supported {target_language} constructs"
            )
        self.suggestions.append("Try using a different theorem prover")


class ProverError(FormalCircuitsGPTError):
    """Enhanced theorem prover interaction error."""

    def __init__(
        self,
        message: str,
        prover_name: str = None,
        prover_output: str = None,
        command: str = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.PROVER, **kwargs)
        self.prover_name = prover_name
        self.prover_output = prover_output
        self.command = command

        self.context.update(
            {
                "prover_name": prover_name,
                "prover_output": prover_output,
                "command": command,
            }
        )

        self.suggestions.extend(
            [
                (
                    f"Ensure {prover_name} is properly installed and in PATH"
                    if prover_name
                    else "Check prover installation"
                ),
                "Verify prover version compatibility",
                "Try using the mock prover for testing",
            ]
        )


class LLMError(FormalCircuitsGPTError):
    """LLM interaction and API errors."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        model: str = None,
        api_error_code: str = None,
        tokens_used: int = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.LLM, **kwargs)
        self.provider = provider
        self.model = model
        self.api_error_code = api_error_code
        self.tokens_used = tokens_used

        self.context.update(
            {
                "provider": provider,
                "model": model,
                "api_error_code": api_error_code,
                "tokens_used": tokens_used,
            }
        )

        # Provider-specific suggestions
        if "rate" in message.lower():
            self.suggestions.append("Wait before retrying due to rate limiting")
        if "api_key" in message.lower():
            self.suggestions.append("Check API key configuration")
        if provider:
            self.suggestions.append(f"Try switching to a different {provider} model")
        self.suggestions.append("Enable mock LLM client for testing")


class SecurityError(FormalCircuitsGPTError):
    """Security validation and sanitization errors."""

    def __init__(
        self,
        message: str,
        violation_type: str = None,
        dangerous_patterns: List[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        self.violation_type = violation_type
        self.dangerous_patterns = dangerous_patterns or []

        self.context.update(
            {"violation_type": violation_type, "dangerous_patterns": dangerous_patterns}
        )

        self.suggestions.extend(
            [
                "Review input for potentially dangerous patterns",
                "Use strict validation mode",
                "Sanitize input before processing",
            ]
        )


class ConfigurationError(FormalCircuitsGPTError):
    """Configuration and setup errors."""

    def __init__(
        self,
        message: str,
        config_file: str = None,
        missing_keys: List[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)
        self.config_file = config_file
        self.missing_keys = missing_keys or []

        self.context.update({"config_file": config_file, "missing_keys": missing_keys})

        if missing_keys:
            self.suggestions.append(
                f"Add missing configuration keys: {', '.join(missing_keys)}"
            )
        if config_file:
            self.suggestions.append(f"Check configuration file: {config_file}")
        self.suggestions.append(
            "Run setup verification with 'formal-circuits-gpt check-setup'"
        )


# Convenience functions for error handling
def handle_verification_error(func):
    """Decorator to handle verification errors gracefully."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FormalCircuitsGPTError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            # Wrap unexpected errors
            raise VerificationError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__, "args": str(args)},
                suggestions=["Enable debug mode for more details", "Report this issue"],
            ) from e

    return wrapper


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """Create standardized error context dictionary."""
    context = {
        "operation": operation,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    context.update(kwargs)
    return context
