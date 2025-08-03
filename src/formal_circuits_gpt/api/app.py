"""Flask application factory for formal verification API."""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from .routes import verification_bp, circuits_bp, cache_bp, status_bp
from ..database import DatabaseManager, run_migrations
from ..cache import CacheManager


def create_app(config=None):
    """Create and configure Flask application.
    
    Args:
        config: Configuration dictionary or object
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.update({
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key'),
        'DATABASE_URL': os.getenv('DATABASE_URL', 'sqlite:///formal_circuits.db'),
        'CACHE_DIR': os.getenv('CACHE_DIR', '.proof_cache'),
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True
    })
    
    if config:
        app.config.update(config)
    
    # Setup CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:8080"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # Initialize database
    try:
        db_manager = DatabaseManager(app.config['DATABASE_URL'])
        run_migrations(db_manager)
        app.db_manager = db_manager
        app.logger.info("Database initialized successfully")
    except Exception as e:
        app.logger.error(f"Database initialization failed: {e}")
        # Continue without database (degraded mode)
        app.db_manager = None
    
    # Initialize cache
    try:
        cache_manager = CacheManager(
            cache_dir=app.config['CACHE_DIR'],
            db_manager=app.db_manager
        )
        app.cache_manager = cache_manager
        app.logger.info("Cache manager initialized successfully")
    except Exception as e:
        app.logger.error(f"Cache initialization failed: {e}")
        app.cache_manager = None
    
    # Register blueprints
    app.register_blueprint(status_bp, url_prefix='/api')
    app.register_blueprint(verification_bp, url_prefix='/api')
    app.register_blueprint(circuits_bp, url_prefix='/api')
    app.register_blueprint(cache_bp, url_prefix='/api')
    
    # Error handlers
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Handle HTTP exceptions."""
        return jsonify({
            'error': e.name,
            'message': e.description,
            'status_code': e.code
        }), e.code
    
    @app.errorhandler(Exception)
    def handle_general_exception(e):
        """Handle general exceptions."""
        app.logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        status = {
            'status': 'healthy',
            'database': 'connected' if app.db_manager else 'disconnected',
            'cache': 'available' if app.cache_manager else 'unavailable'
        }
        
        # Check database connection
        if app.db_manager:
            try:
                app.db_manager.execute_query("SELECT 1")
                status['database'] = 'connected'
            except Exception:
                status['database'] = 'error'
                status['status'] = 'degraded'
        
        return jsonify(status)
    
    # API documentation endpoint
    @app.route('/api')
    def api_docs():
        """API documentation."""
        return jsonify({
            'name': 'Formal-Circuits-GPT API',
            'version': '1.0.0',
            'description': 'REST API for formal hardware verification',
            'endpoints': {
                'health': 'GET /health - Health check',
                'status': 'GET /api/status - System status',
                'verify': 'POST /api/verify - Verify circuit',
                'verify_file': 'POST /api/verify/file - Verify circuit from file',
                'circuits': 'GET /api/circuits - List circuits',
                'circuit_detail': 'GET /api/circuits/<id> - Get circuit details',
                'cache_stats': 'GET /api/cache/stats - Cache statistics',
                'cache_clear': 'DELETE /api/cache - Clear cache'
            },
            'documentation': '/api/docs'
        })
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )