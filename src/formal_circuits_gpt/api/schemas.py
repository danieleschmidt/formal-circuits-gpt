"""API request/response schemas and validation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from werkzeug.datastructures import FileStorage


@dataclass
class CircuitVerificationRequest:
    """Request schema for circuit verification."""
    hdl_code: str
    properties: Optional[List[str]] = None
    prover: str = "isabelle"
    temperature: float = 0.1
    timeout: int = 300
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class CircuitVerificationResponse:
    """Response schema for circuit verification."""
    status: str
    proof_code: str
    properties_verified: List[str]
    errors: List[str]
    prover_used: str
    execution_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CircuitListResponse:
    """Response schema for circuit listing."""
    circuits: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CacheStatsResponse:
    """Response schema for cache statistics."""
    database: Dict[str, Any]
    memory: Dict[str, Any]
    files: Dict[str, Any]
    cache_dir: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def validate_verification_request(data: Dict[str, Any]) -> List[str]:
    """Validate circuit verification request data.
    
    Args:
        data: Request data dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    if 'hdl_code' not in data:
        errors.append("'hdl_code' is required")
    elif not isinstance(data['hdl_code'], str):
        errors.append("'hdl_code' must be a string")
    elif not data['hdl_code'].strip():
        errors.append("'hdl_code' cannot be empty")
    
    # Optional fields validation
    if 'properties' in data:
        if not isinstance(data['properties'], (list, type(None))):
            errors.append("'properties' must be a list or null")
        elif data['properties'] is not None:
            for i, prop in enumerate(data['properties']):
                if not isinstance(prop, str):
                    errors.append(f"'properties[{i}]' must be a string")
    
    if 'prover' in data:
        if not isinstance(data['prover'], str):
            errors.append("'prover' must be a string")
        elif data['prover'] not in ['isabelle', 'coq']:
            errors.append("'prover' must be 'isabelle' or 'coq'")
    
    if 'temperature' in data:
        if not isinstance(data['temperature'], (int, float)):
            errors.append("'temperature' must be a number")
        elif not (0.0 <= data['temperature'] <= 2.0):
            errors.append("'temperature' must be between 0.0 and 2.0")
    
    if 'timeout' in data:
        if not isinstance(data['timeout'], int):
            errors.append("'timeout' must be an integer")
        elif data['timeout'] <= 0:
            errors.append("'timeout' must be positive")
        elif data['timeout'] > 3600:
            errors.append("'timeout' cannot exceed 3600 seconds (1 hour)")
    
    if 'name' in data:
        if not isinstance(data['name'], (str, type(None))):
            errors.append("'name' must be a string or null")
        elif data['name'] is not None and len(data['name']) > 100:
            errors.append("'name' cannot exceed 100 characters")
    
    return errors


def validate_file_upload(file: FileStorage) -> List[str]:
    """Validate uploaded HDL file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check filename
    if not file.filename:
        errors.append("Filename is required")
        return errors
    
    # Check file extension
    allowed_extensions = {'.v', '.vh', '.sv', '.vhd', '.vhdl'}
    file_ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        errors.append(f"File extension '{file_ext}' not allowed. Allowed: {', '.join(allowed_extensions)}")
    
    # Check file size (limit to 1MB for HDL files)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size = 1024 * 1024  # 1MB
    if file_size > max_size:
        errors.append(f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)")
    
    if file_size == 0:
        errors.append("File cannot be empty")
    
    return errors


def validate_cache_cleanup_request(data: Dict[str, Any]) -> List[str]:
    """Validate cache cleanup request.
    
    Args:
        data: Request data dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if 'max_age_days' in data:
        if not isinstance(data['max_age_days'], int):
            errors.append("'max_age_days' must be an integer")
        elif data['max_age_days'] <= 0:
            errors.append("'max_age_days' must be positive")
        elif data['max_age_days'] > 365:
            errors.append("'max_age_days' cannot exceed 365 days")
    
    return errors


def validate_pagination_params(limit: str, offset: str) -> tuple[int, int, List[str]]:
    """Validate pagination parameters.
    
    Args:
        limit: Limit parameter as string
        offset: Offset parameter as string
        
    Returns:
        Tuple of (validated_limit, validated_offset, errors)
    """
    errors = []
    validated_limit = 50  # Default
    validated_offset = 0  # Default
    
    # Validate limit
    if limit:
        try:
            validated_limit = int(limit)
            if validated_limit <= 0:
                errors.append("'limit' must be positive")
                validated_limit = 50
            elif validated_limit > 1000:
                errors.append("'limit' cannot exceed 1000")
                validated_limit = 1000
        except ValueError:
            errors.append("'limit' must be an integer")
    
    # Validate offset
    if offset:
        try:
            validated_offset = int(offset)
            if validated_offset < 0:
                errors.append("'offset' cannot be negative")
                validated_offset = 0
        except ValueError:
            errors.append("'offset' must be an integer")
    
    return validated_limit, validated_offset, errors


def sanitize_hdl_code(hdl_code: str) -> str:
    """Sanitize HDL code input.
    
    Args:
        hdl_code: Raw HDL code string
        
    Returns:
        Sanitized HDL code
    """
    # Remove null bytes and other dangerous characters
    sanitized = hdl_code.replace('\x00', '')
    
    # Normalize line endings
    sanitized = sanitized.replace('\r\n', '\n').replace('\r', '\n')
    
    # Limit size
    max_size = 100 * 1024  # 100KB
    if len(sanitized) > max_size:
        raise ValueError(f"HDL code exceeds maximum size ({max_size} characters)")
    
    return sanitized


def format_error_response(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format standardized error response.
    
    Args:
        error_type: Type of error
        message: Error message
        details: Additional error details
        
    Returns:
        Formatted error response dictionary
    """
    response = {
        'error': error_type,
        'message': message
    }
    
    if details:
        response['details'] = details
    
    return response