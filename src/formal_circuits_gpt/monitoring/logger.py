"""Enhanced logging system for formal-circuits-gpt."""

import os
import sys
import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dataclasses import dataclass, asdict


@dataclass
class LogContext:
    """Structured logging context."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    circuit_name: Optional[str] = None
    verification_id: Optional[str] = None
    prover: Optional[str] = None
    model: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": getattr(record, "module", record.filename),
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context if available
        if hasattr(record, "context") and record.context:
            log_entry["context"] = asdict(record.context)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
                "context",
            ]:
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class SecurityFilter(logging.Filter):
    """Filter to prevent logging of sensitive information."""

    SENSITIVE_PATTERNS = [
        "api_key",
        "password",
        "token",
        "secret",
        "auth",
        "key=",
        "Authorization:",
        "Bearer ",
        "X-API-Key:",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out sensitive information from logs."""
        message = record.getMessage().lower()

        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.lower() in message:
                # Replace sensitive data with placeholder
                original_msg = record.getMessage()
                record.msg = "[REDACTED: Potentially sensitive data]"
                record.args = ()
                return True

        return True


class EnhancedLogger:
    """Enhanced logger with structured logging and security features."""

    def __init__(self, name: str, log_dir: Optional[str] = None):
        """Initialize enhanced logger."""
        self.name = name
        self.logger = logging.getLogger(name)
        self.context = LogContext()

        # Set up log directory
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Configure logger if not already configured
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Set up log handlers."""
        self.logger.setLevel(logging.DEBUG)

        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(SecurityFilter())

        # File handler for general logs
        file_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        file_handler.addFilter(SecurityFilter())

        # Error file handler
        error_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        error_handler.addFilter(SecurityFilter())

        # Performance log handler
        perf_handler = TimedRotatingFileHandler(
            self.log_dir / f"{self.name}_performance.log",
            when="midnight",
            interval=1,
            backupCount=30,
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(perf_handler)

    def set_context(self, **kwargs):
        """Set logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

    def clear_context(self):
        """Clear logging context."""
        self.context = LogContext()

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context."""
        extra = kwargs.copy()
        extra["context"] = self.context
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def log_verification_start(self, circuit_name: str, prover: str, model: str):
        """Log verification start."""
        self.set_context(
            circuit_name=circuit_name,
            prover=prover,
            model=model,
            verification_id=f"{circuit_name}_{int(time.time())}",
        )
        self.info(
            f"Starting verification of {circuit_name} using {prover} with {model}",
            event_type="verification_start",
        )

    def log_verification_end(
        self, status: str, duration_ms: float, properties_count: int
    ):
        """Log verification end."""
        self.info(
            f"Verification completed with status {status}",
            event_type="verification_end",
            status=status,
            duration_ms=duration_ms,
            properties_verified=properties_count,
        )

    def log_performance(self, operation: str, duration_ms: float, **metrics):
        """Log performance metrics."""
        self.info(
            f"Performance: {operation} took {duration_ms:.2f}ms",
            event_type="performance",
            operation=operation,
            duration_ms=duration_ms,
            **metrics,
        )

    def log_security_event(
        self, event_type: str, details: str, severity: str = "warning"
    ):
        """Log security-related events."""
        log_method = getattr(self, severity.lower(), self.warning)
        log_method(
            f"Security event: {event_type} - {details}",
            event_type="security",
            security_event_type=event_type,
        )

    def log_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
    ):
        """Log API request."""
        self.info(
            f"API {method} {endpoint} -> {status_code} ({duration_ms:.2f}ms)",
            event_type="api_request",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
        )

    def log_llm_interaction(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        cost_estimate: Optional[float] = None,
    ):
        """Log LLM interaction for cost tracking."""
        self.info(
            f"LLM interaction: {provider}/{model} used {tokens_used} tokens",
            event_type="llm_interaction",
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
        )


# Global logger instances
_loggers: Dict[str, EnhancedLogger] = {}


def get_logger(name: str, log_dir: Optional[str] = None) -> EnhancedLogger:
    """Get or create enhanced logger instance."""
    if name not in _loggers:
        _loggers[name] = EnhancedLogger(name, log_dir)
    return _loggers[name]


def setup_logging(log_level: str = "INFO", log_dir: Optional[str] = None):
    """Set up global logging configuration."""
    # Set root logger level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)

    # Create main application logger
    app_logger = get_logger("formal_circuits_gpt", log_dir)

    # Log startup
    app_logger.info(
        "Logging system initialized",
        event_type="system_start",
        log_level=log_level,
        log_dir=str(log_dir) if log_dir else "default",
    )
