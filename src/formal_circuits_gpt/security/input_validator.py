"""Input validation and sanitization for formal-circuits-gpt."""

import re
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from ..exceptions import VerificationError


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[Any] = None
    
    def __post_init__(self):
        self.errors = self.errors or []
        self.warnings = self.warnings or []


class SecurityError(Exception):
    """Exception raised for security-related validation failures."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # File path security patterns
    DANGEROUS_PATH_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'/etc/',   # System directories
        r'/proc/',
        r'/sys/',
        r'C:\\Windows\\',
        r'C:\\System32\\',
        r'~/',      # Home directory shortcuts
        r'\$\{.*\}',  # Environment variable injection
        r'`.*`',    # Command injection
        r'\|',      # Pipe injection
        r';',       # Command chaining
        r'&',       # Background execution
        r'<',       # Input redirection
        r'>',       # Output redirection
    ]
    
    # HDL code patterns that might be dangerous
    DANGEROUS_HDL_PATTERNS = [
        r'\$system\s*\(',  # Verilog system tasks
        r'\$display\s*\(',
        r'\$write\s*\(',
        r'\$readmem\s*\(',
        r'\$dumpfile\s*\(',
        r'`include\s+',    # File inclusion
        r'`define\s+',     # Macro definition (can be abused)
        r'\\\\',           # Excessive escaping
    ]
    
    # SQL injection patterns (for database queries)
    SQL_INJECTION_PATTERNS = [
        r';\s*(drop|delete|insert|update|create|alter)\s+',
        r'union\s+select\s+',
        r'or\s+1\s*=\s*1',
        r'and\s+1\s*=\s*1',
        r'--\s*',
        r'/\*.*\*/',
        r'xp_cmdshell',
        r'sp_executesql',
    ]
    
    def __init__(self, strict_mode: bool = True):
        """Initialize input validator.
        
        Args:
            strict_mode: Enable strict validation mode
        """
        self.strict_mode = strict_mode
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_hdl_length = 1_000_000  # 1M characters
        self.allowed_extensions = {'.v', '.vh', '.sv', '.vhd', '.vhdl'}
    
    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Validate file path for security issues."""
        errors = []
        warnings = []
        
        if not file_path:
            errors.append("File path cannot be empty")
            return ValidationResult(False, errors, warnings)
        
        # Convert to Path for normalization
        try:
            path = Path(file_path).resolve()
            normalized_path = str(path)
        except Exception as e:
            errors.append(f"Invalid file path: {str(e)}")
            return ValidationResult(False, errors, warnings)
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATH_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                errors.append(f"Potentially dangerous path pattern detected: {pattern}")
        
        # Check if path exists and is a file
        if not path.exists():
            errors.append(f"File does not exist: {file_path}")
        elif not path.is_file():
            errors.append(f"Path is not a file: {file_path}")
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            if self.strict_mode:
                errors.append(f"File extension not allowed: {path.suffix}")
            else:
                warnings.append(f"Unusual file extension: {path.suffix}")
        
        # Check file size
        try:
            if path.exists() and path.stat().st_size > self.max_file_size:
                errors.append(f"File too large: {path.stat().st_size} bytes > {self.max_file_size}")
        except OSError as e:
            warnings.append(f"Could not check file size: {str(e)}")
        
        # Check if file is readable
        try:
            if path.exists() and not os.access(path, os.R_OK):
                errors.append(f"File not readable: {file_path}")
        except OSError as e:
            warnings.append(f"Could not check file permissions: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, normalized_path)
    
    def validate_hdl_content(self, hdl_content: str) -> ValidationResult:
        """Validate HDL content for security and sanity."""
        errors = []
        warnings = []
        
        if not hdl_content:
            errors.append("HDL content cannot be empty")
            return ValidationResult(False, errors, warnings)
        
        # Check content length
        if len(hdl_content) > self.max_hdl_length:
            errors.append(f"HDL content too long: {len(hdl_content)} > {self.max_hdl_length}")
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_HDL_PATTERNS:
            matches = list(re.finditer(pattern, hdl_content, re.IGNORECASE))
            if matches:
                if self.strict_mode:
                    errors.append(f"Potentially dangerous HDL pattern: {pattern}")
                else:
                    warnings.append(f"Suspicious HDL pattern found: {pattern}")
        
        # Check for excessive nested structures (potential DoS)
        nesting_depth = self._check_nesting_depth(hdl_content)
        if nesting_depth > 50:
            if self.strict_mode:
                errors.append(f"Excessive nesting depth: {nesting_depth}")
            else:
                warnings.append(f"High nesting depth: {nesting_depth}")
        
        # Sanitize content
        sanitized_content = self._sanitize_hdl_content(hdl_content)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, sanitized_content)
    
    def validate_properties(self, properties: Union[List[str], str]) -> ValidationResult:
        """Validate property specifications."""
        errors = []
        warnings = []
        
        if not properties:
            # Empty properties is valid - will be auto-generated
            return ValidationResult(True, errors, warnings, [])
        
        # Convert to list if string
        if isinstance(properties, str):
            properties = [properties]
        
        if not isinstance(properties, list):
            errors.append("Properties must be a string or list of strings")
            return ValidationResult(False, errors, warnings)
        
        sanitized_properties = []
        
        for i, prop in enumerate(properties):
            if not isinstance(prop, str):
                errors.append(f"Property {i+1} must be a string")
                continue
            
            if len(prop) > 10000:  # Reasonable limit for property length
                errors.append(f"Property {i+1} too long: {len(prop)} characters")
                continue
            
            # Check for injection patterns
            for pattern in self.SQL_INJECTION_PATTERNS:
                if re.search(pattern, prop, re.IGNORECASE):
                    errors.append(f"Property {i+1} contains suspicious pattern: {pattern}")
                    break
            else:
                # Sanitize property
                sanitized_prop = self._sanitize_property(prop)
                sanitized_properties.append(sanitized_prop)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, sanitized_properties)
    
    def validate_model_name(self, model_name: str) -> ValidationResult:
        """Validate LLM model name."""
        errors = []
        warnings = []
        
        if not model_name:
            errors.append("Model name cannot be empty")
            return ValidationResult(False, errors, warnings)
        
        # Allow only alphanumeric, hyphens, underscores, dots
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', model_name):
            errors.append("Model name contains invalid characters")
        
        # Check reasonable length
        if len(model_name) > 100:
            errors.append("Model name too long")
        
        # Known valid model prefixes
        valid_prefixes = ['gpt-', 'claude-', 'text-', 'code-', 'llama-', 'mistral-']
        if not any(model_name.startswith(prefix) for prefix in valid_prefixes):
            warnings.append(f"Unknown model name format: {model_name}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, model_name.strip())
    
    def validate_prover_name(self, prover_name: str) -> ValidationResult:
        """Validate theorem prover name."""
        errors = []
        warnings = []
        
        if not prover_name:
            errors.append("Prover name cannot be empty")
            return ValidationResult(False, errors, warnings)
        
        valid_provers = {'isabelle', 'coq'}
        
        if prover_name.lower() not in valid_provers:
            errors.append(f"Unknown prover: {prover_name}. Valid options: {', '.join(valid_provers)}")
        
        is_valid = len(errors) == 0
        sanitized_name = prover_name.lower().strip() if is_valid else None
        return ValidationResult(is_valid, errors, warnings, sanitized_name)
    
    def validate_temperature(self, temperature: float) -> ValidationResult:
        """Validate LLM temperature parameter."""
        errors = []
        warnings = []
        
        if not isinstance(temperature, (int, float)):
            errors.append("Temperature must be a number")
            return ValidationResult(False, errors, warnings)
        
        if temperature < 0.0 or temperature > 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
        
        if temperature > 1.0:
            warnings.append("High temperature may produce inconsistent results")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, float(temperature))
    
    def validate_timeout(self, timeout: int) -> ValidationResult:
        """Validate timeout parameter."""
        errors = []
        warnings = []
        
        if not isinstance(timeout, int):
            errors.append("Timeout must be an integer")
            return ValidationResult(False, errors, warnings)
        
        if timeout <= 0:
            errors.append("Timeout must be positive")
        elif timeout > 3600:  # 1 hour max
            if self.strict_mode:
                errors.append("Timeout too large (max 3600 seconds)")
            else:
                warnings.append("Very large timeout specified")
        
        if timeout < 30:
            warnings.append("Short timeout may cause verification failures")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, timeout)
    
    def _check_nesting_depth(self, content: str) -> int:
        """Check maximum nesting depth in HDL content."""
        max_depth = 0
        current_depth = 0
        
        # Count braces and keywords that indicate nesting
        for line in content.split('\n'):
            line = line.strip()
            
            # Opening constructs
            opens = line.count('{') + line.count('begin') + line.count('(')
            
            # Closing constructs  
            closes = line.count('}') + line.count('end') + line.count(')')
            
            current_depth += opens - closes
            max_depth = max(max_depth, current_depth)
        
        return max_depth
    
    def _sanitize_hdl_content(self, content: str) -> str:
        """Sanitize HDL content while preserving structure."""
        # For HDL content, we need to be more careful to preserve structure
        sanitized = content
        
        # Only remove potentially dangerous shell commands in comments
        # Keep the HDL structure intact - be very specific about dangerous patterns
        dangerous_comment_patterns = [
            r'//.*(?:\$\([^)]*\)|`[^`]*`|\$\{[^}]*\})',  # Shell command substitution in comments
            r'/\*.*(?:\$\([^)]*\)|`[^`]*`|\$\{[^}]*\}).*?\*/',  # Shell commands in multiline comments
        ]
        
        for pattern in dangerous_comment_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.DOTALL | re.MULTILINE)
        
        # Remove null bytes and other control characters that could cause issues
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized
    
    def _sanitize_property(self, prop: str) -> str:
        """Sanitize property specification."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>|&;`]', '', prop)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        return sanitized.strip()
    
    def validate_api_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate API input data comprehensively."""
        errors = []
        warnings = []
        sanitized_data = {}
        
        # Validate each field
        for field, value in data.items():
            if field == 'hdl_file':
                result = self.validate_file_path(str(value))
                if not result.is_valid:
                    errors.extend([f"hdl_file: {error}" for error in result.errors])
                warnings.extend([f"hdl_file: {warning}" for warning in result.warnings])
                if result.sanitized_input:
                    sanitized_data[field] = result.sanitized_input
                    
            elif field == 'hdl_content':
                result = self.validate_hdl_content(str(value))
                if not result.is_valid:
                    errors.extend([f"hdl_content: {error}" for error in result.errors])
                warnings.extend([f"hdl_content: {warning}" for warning in result.warnings])
                if result.sanitized_input:
                    sanitized_data[field] = result.sanitized_input
                    
            elif field == 'properties':
                result = self.validate_properties(value)
                if not result.is_valid:
                    errors.extend([f"properties: {error}" for error in result.errors])
                warnings.extend([f"properties: {warning}" for warning in result.warnings])
                if result.sanitized_input is not None:
                    sanitized_data[field] = result.sanitized_input
                    
            elif field == 'model':
                result = self.validate_model_name(str(value))
                if not result.is_valid:
                    errors.extend([f"model: {error}" for error in result.errors])
                warnings.extend([f"model: {warning}" for warning in result.warnings])
                if result.sanitized_input:
                    sanitized_data[field] = result.sanitized_input
                    
            elif field == 'prover':
                result = self.validate_prover_name(str(value))
                if not result.is_valid:
                    errors.extend([f"prover: {error}" for error in result.errors])
                warnings.extend([f"prover: {warning}" for warning in result.warnings])
                if result.sanitized_input:
                    sanitized_data[field] = result.sanitized_input
                    
            elif field == 'temperature':
                result = self.validate_temperature(float(value))
                if not result.is_valid:
                    errors.extend([f"temperature: {error}" for error in result.errors])
                warnings.extend([f"temperature: {warning}" for warning in result.warnings])
                if result.sanitized_input is not None:
                    sanitized_data[field] = result.sanitized_input
                    
            elif field == 'timeout':
                result = self.validate_timeout(int(value))
                if not result.is_valid:
                    errors.extend([f"timeout: {error}" for error in result.errors])
                warnings.extend([f"timeout: {warning}" for warning in result.warnings])
                if result.sanitized_input is not None:
                    sanitized_data[field] = result.sanitized_input
            else:
                # Unknown field - pass through with basic sanitization
                if isinstance(value, str):
                    sanitized_data[field] = re.sub(r'[<>|&;`]', '', str(value))
                else:
                    sanitized_data[field] = value
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, sanitized_data)