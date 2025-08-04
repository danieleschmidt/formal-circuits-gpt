"""API routes for formal verification endpoints."""

import os
import json
import tempfile
from typing import Dict, List, Any
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError

from ..core import CircuitVerifier
from ..exceptions import VerificationError
from ..database.repositories import CircuitRepository, VerificationRepository
from ..database.models import CircuitModel, VerificationResult
from .schemas import (
    CircuitVerificationRequest, CircuitVerificationResponse,
    validate_verification_request, validate_file_upload
)


# Blueprint definitions
status_bp = Blueprint('status', __name__)
verification_bp = Blueprint('verification', __name__)
circuits_bp = Blueprint('circuits', __name__)
cache_bp = Blueprint('cache', __name__)


@status_bp.route('/status')
def get_status():
    """Get system status and configuration."""
    try:
        status = {
            'service': 'formal-circuits-gpt',
            'version': '1.0.0',
            'status': 'running'
        }
        
        # Database status
        if hasattr(current_app, 'db_manager') and current_app.db_manager:
            try:
                db_stats = current_app.db_manager.get_stats()
                status['database'] = {
                    'status': 'connected',
                    'stats': db_stats
                }
            except Exception as e:
                status['database'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            status['database'] = {'status': 'not_configured'}
        
        # Cache status
        if hasattr(current_app, 'cache_manager') and current_app.cache_manager:
            try:
                cache_stats = current_app.cache_manager.get_cache_stats()
                status['cache'] = {
                    'status': 'available',
                    'stats': cache_stats
                }
            except Exception as e:
                status['cache'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            status['cache'] = {'status': 'not_configured'}
        
        # Environment info
        status['environment'] = {
            'python_version': os.environ.get('PYTHON_VERSION', 'unknown'),
            'debug_mode': current_app.debug,
            'max_content_length': current_app.config.get('MAX_CONTENT_LENGTH')
        }
        
        return jsonify(status)
        
    except Exception as e:
        current_app.logger.error(f"Status check failed: {e}")
        return jsonify({
            'service': 'formal-circuits-gpt',
            'status': 'error',
            'error': str(e)
        }), 500


@verification_bp.route('/verify', methods=['POST'])
def verify_circuit():
    """Verify a circuit from HDL code."""
    try:
        # Validate request
        data = request.get_json()
        if not data:
            raise BadRequest("JSON data required")
        
        validation_errors = validate_verification_request(data)
        if validation_errors:
            raise BadRequest(f"Validation errors: {', '.join(validation_errors)}")
        
        # Extract parameters
        hdl_code = data['hdl_code']
        properties = data.get('properties')
        prover = data.get('prover', 'isabelle')
        temperature = data.get('temperature', 0.1)
        timeout = data.get('timeout', 300)
        
        # Create verifier
        verifier = CircuitVerifier(
            prover=prover,
            temperature=temperature,
            debug_mode=current_app.debug
        )
        
        # Perform verification
        result = verifier.verify(
            hdl_code=hdl_code,
            properties=properties,
            timeout=timeout
        )
        
        # Save to database if available
        if hasattr(current_app, 'db_manager') and current_app.db_manager:
            try:
                circuit_repo = CircuitRepository(current_app.db_manager)
                verification_repo = VerificationRepository(current_app.db_manager)
                
                # Save circuit
                circuit_model = CircuitModel(
                    name=data.get('name', 'unnamed_circuit'),
                    hdl_type='verilog' if 'module' in hdl_code.lower() else 'vhdl',
                    source_code=hdl_code,
                    module_count=len(result.ast.modules) if result.ast else 0
                )
                circuit_id = circuit_repo.save_circuit(circuit_model)
                
                # Save verification result
                verification_result = VerificationResult(
                    circuit_id=circuit_id,
                    properties=result.properties_verified,
                    prover=prover,
                    status=result.status,
                    proof_code=result.proof_code,
                    errors=result.errors
                )
                verification_repo.save_result(verification_result)
                
            except Exception as e:
                current_app.logger.warning(f"Failed to save verification result: {e}")
        
        # Return response
        response = CircuitVerificationResponse(
            status=result.status,
            proof_code=result.proof_code,
            properties_verified=result.properties_verified,
            errors=result.errors,
            prover_used=prover,
            execution_metadata={
                'timeout': timeout,
                'temperature': temperature,
                'has_ast': result.ast is not None
            }
        )
        
        return jsonify(response.to_dict())
        
    except VerificationError as e:
        return jsonify({
            'error': 'VerificationError',
            'message': str(e),
            'status': 'FAILED'
        }), 400
    
    except BadRequest as e:
        return jsonify({
            'error': 'BadRequest',
            'message': str(e)
        }), 400
    
    except Exception as e:
        current_app.logger.error(f"Verification failed: {e}", exc_info=True)
        return jsonify({
            'error': 'InternalServerError',
            'message': 'Verification failed due to internal error'
        }), 500


@verification_bp.route('/verify/file', methods=['POST'])
def verify_circuit_file():
    """Verify a circuit from uploaded HDL file."""
    try:
        # Check if file is present
        if 'file' not in request.files:
            raise BadRequest("No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            raise BadRequest("No file selected")
        
        # Validate file
        validation_errors = validate_file_upload(file)
        if validation_errors:
            raise BadRequest(f"File validation errors: {', '.join(validation_errors)}")
        
        # Get additional parameters from form data
        prover = request.form.get('prover', 'isabelle')
        temperature = float(request.form.get('temperature', 0.1))
        timeout = int(request.form.get('timeout', 300))
        properties_str = request.form.get('properties')
        properties = json.loads(properties_str) if properties_str else None
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(mode='w+b', suffix=f"_{filename}", delete=False) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Create verifier and verify file
            verifier = CircuitVerifier(
                prover=prover,
                temperature=temperature,
                debug_mode=current_app.debug
            )
            
            result = verifier.verify_file(
                hdl_file=temp_path,
                properties=properties,
                timeout=timeout
            )
            
            # Return response
            response = CircuitVerificationResponse(
                status=result.status,
                proof_code=result.proof_code,
                properties_verified=result.properties_verified,
                errors=result.errors,
                prover_used=prover,
                execution_metadata={
                    'filename': filename,
                    'timeout': timeout,
                    'temperature': temperature
                }
            )
            
            return jsonify(response.to_dict())
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        
    except Exception as e:
        current_app.logger.error(f"File verification failed: {e}", exc_info=True)
        return jsonify({
            'error': 'InternalServerError',
            'message': 'File verification failed'
        }), 500


@circuits_bp.route('/circuits', methods=['GET'])
def list_circuits():
    """List all circuits in the database."""
    try:
        if not hasattr(current_app, 'db_manager') or not current_app.db_manager:
            raise InternalServerError("Database not available")
        
        circuit_repo = CircuitRepository(current_app.db_manager)
        
        # Get query parameters
        hdl_type = request.args.get('type')
        search = request.args.get('search')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # Get circuits based on filters
        if search:
            circuits = circuit_repo.search_circuits(search)
        elif hdl_type:
            circuits = circuit_repo.get_circuits_by_type(hdl_type)
        else:
            # Get all circuits (simplified - in real app would implement pagination)
            circuits = current_app.db_manager.execute_query(
                "SELECT * FROM circuit_models ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            circuits = [CircuitModel.from_row(row) for row in circuits]
        
        # Convert to response format
        circuit_list = []
        for circuit in circuits:
            circuit_list.append({
                'id': circuit.id,
                'name': circuit.name,
                'hdl_type': circuit.hdl_type,
                'module_count': circuit.module_count,
                'created_at': circuit.created_at.isoformat() if circuit.created_at else None,
                'updated_at': circuit.updated_at.isoformat() if circuit.updated_at else None
            })
        
        return jsonify({
            'circuits': circuit_list,
            'total': len(circuit_list),
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to list circuits: {e}")
        return jsonify({
            'error': 'InternalServerError',
            'message': 'Failed to retrieve circuits'
        }), 500


@circuits_bp.route('/circuits/<int:circuit_id>', methods=['GET'])
def get_circuit_detail(circuit_id):
    """Get detailed information about a specific circuit."""
    try:
        if not hasattr(current_app, 'db_manager') or not current_app.db_manager:
            raise InternalServerError("Database not available")
        
        circuit_repo = CircuitRepository(current_app.db_manager)
        verification_repo = VerificationRepository(current_app.db_manager)
        
        # Get circuit
        circuit = circuit_repo.get_circuit_by_id(circuit_id)
        if not circuit:
            raise NotFound(f"Circuit {circuit_id} not found")
        
        # Get verification results
        verification_results = verification_repo.get_results_by_circuit(circuit_id)
        
        # Format response
        response = {
            'id': circuit.id,
            'name': circuit.name,
            'hdl_type': circuit.hdl_type,
            'source_code': circuit.source_code,
            'module_count': circuit.module_count,
            'created_at': circuit.created_at.isoformat() if circuit.created_at else None,
            'updated_at': circuit.updated_at.isoformat() if circuit.updated_at else None,
            'metadata': circuit.metadata,
            'verification_results': []
        }
        
        # Add verification results
        for result in verification_results:
            response['verification_results'].append({
                'id': result.id,
                'status': result.status,
                'prover': result.prover,
                'properties': result.properties,
                'errors': result.errors,
                'execution_time': result.execution_time,
                'created_at': result.created_at.isoformat() if result.created_at else None
            })
        
        return jsonify(response)
        
    except NotFound as e:
        return jsonify({
            'error': 'NotFound',
            'message': str(e)
        }), 404
    
    except Exception as e:
        current_app.logger.error(f"Failed to get circuit detail: {e}")
        return jsonify({
            'error': 'InternalServerError',
            'message': 'Failed to retrieve circuit details'
        }), 500


@cache_bp.route('/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics."""
    try:
        if not hasattr(current_app, 'cache_manager') or not current_app.cache_manager:
            return jsonify({
                'error': 'CacheNotAvailable',
                'message': 'Cache manager not configured'
            }), 503
        
        stats = current_app.cache_manager.get_cache_stats()
        return jsonify(stats)
        
    except Exception as e:
        current_app.logger.error(f"Failed to get cache stats: {e}")
        return jsonify({
            'error': 'InternalServerError',
            'message': 'Failed to retrieve cache statistics'
        }), 500


@cache_bp.route('/cache', methods=['DELETE'])
def clear_cache():
    """Clear all caches."""
    try:
        if not hasattr(current_app, 'cache_manager') or not current_app.cache_manager:
            return jsonify({
                'error': 'CacheNotAvailable',
                'message': 'Cache manager not configured'
            }), 503
        
        # Clear cache (this is destructive!)
        current_app.cache_manager.clear_all_caches()
        
        return jsonify({
            'message': 'All caches cleared successfully'
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to clear cache: {e}")
        return jsonify({
            'error': 'InternalServerError',
            'message': 'Failed to clear cache'
        }), 500


@cache_bp.route('/cache/cleanup', methods=['POST'])
def cleanup_cache():
    """Clean up old cache entries."""
    try:
        if not hasattr(current_app, 'cache_manager') or not current_app.cache_manager:
            return jsonify({
                'error': 'CacheNotAvailable',
                'message': 'Cache manager not configured'
            }), 503
        
        # Get cleanup parameters
        max_age_days = int(request.json.get('max_age_days', 30)) if request.json else 30
        
        # Perform cleanup
        cleanup_result = current_app.cache_manager.cleanup_cache(max_age_days)
        
        return jsonify({
            'message': 'Cache cleanup completed',
            'cleanup_result': cleanup_result
        })
        
    except Exception as e:
        current_app.logger.error(f"Cache cleanup failed: {e}")
        return jsonify({
            'error': 'InternalServerError',
            'message': 'Cache cleanup failed'
        }), 500