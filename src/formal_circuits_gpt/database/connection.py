"""Database connection management."""

import os
import sqlite3
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager


class DatabaseError(Exception):
    """Database operation error."""
    pass


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL (uses SQLite by default)
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "sqlite:///formal_circuits.db"
        )
        self._local = threading.local()
        self._initialized = False
        
        # Parse database URL
        self._parse_database_url()
        
        # Initialize database
        self.initialize()
    
    def _parse_database_url(self):
        """Parse database URL to extract connection parameters."""
        if self.database_url.startswith("sqlite:///"):
            self.db_type = "sqlite"
            self.db_path = self.database_url[10:]  # Remove "sqlite:///"
            
            # Ensure directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
        else:
            raise DatabaseError(f"Unsupported database URL: {self.database_url}")
    
    def initialize(self):
        """Initialize database schema."""
        if self._initialized:
            return
            
        try:
            with self.get_connection() as conn:
                self._create_tables(conn)
            self._initialized = True
        except Exception as e:
            raise DatabaseError(f"Database initialization failed: {str(e)}") from e
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""
        cursor = conn.cursor()
        
        # Proof cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proof_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                circuit_hash TEXT UNIQUE NOT NULL,
                properties_hash TEXT NOT NULL,
                prover TEXT NOT NULL,
                proof_code TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Circuit models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                hdl_type TEXT NOT NULL,
                source_code TEXT NOT NULL,
                ast_json TEXT,
                module_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Verification results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                circuit_id INTEGER,
                properties TEXT NOT NULL,
                prover TEXT NOT NULL,
                status TEXT NOT NULL,
                proof_code TEXT,
                errors TEXT,
                execution_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (circuit_id) REFERENCES circuit_models(id)
            )
        """)
        
        # Lemma cache table for reusable proofs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lemma_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma_hash TEXT UNIQUE NOT NULL,
                lemma_name TEXT NOT NULL,
                statement TEXT NOT NULL,
                proof TEXT NOT NULL,
                prover TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_proof_cache_hash ON proof_cache(circuit_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_circuit_name ON circuit_models(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_verification_circuit ON verification_results(circuit_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lemma_hash ON lemma_cache(lemma_hash)")
        
        conn.commit()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection for current thread."""
        if not hasattr(self._local, 'connection'):
            if self.db_type == "sqlite":
                self._local.connection = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    check_same_thread=False
                )
                # Enable foreign keys and WAL mode for better performance
                self._local.connection.execute("PRAGMA foreign_keys = ON")
                self._local.connection.execute("PRAGMA journal_mode = WAL")
                self._local.connection.row_factory = sqlite3.Row
        
        return self._local.connection
    
    @contextmanager
    def transaction(self):
        """Database transaction context manager."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute a SELECT query and return results."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {str(e)}") from e
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT, UPDATE, or DELETE query."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.rowcount
        except Exception as e:
            raise DatabaseError(f"Update execution failed: {str(e)}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Table row counts
            tables = ['proof_cache', 'circuit_models', 'verification_results', 'lemma_cache']
            for table in tables:
                result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = result[0]['count'] if result else 0
            
            # Cache hit rate (simplified)
            cache_stats = self.execute_query("""
                SELECT 
                    AVG(access_count) as avg_access,
                    MAX(access_count) as max_access,
                    COUNT(*) as total_entries
                FROM proof_cache
            """)
            
            if cache_stats and cache_stats[0]['total_entries'] > 0:
                stats['cache_avg_access'] = cache_stats[0]['avg_access']
                stats['cache_max_access'] = cache_stats[0]['max_access']
            
            return stats
            
        except Exception as e:
            raise DatabaseError(f"Failed to get database stats: {str(e)}") from e
    
    def cleanup_old_entries(self, days: int = 30):
        """Clean up old cache entries."""
        try:
            query = """
                DELETE FROM proof_cache 
                WHERE last_accessed < datetime('now', '-{} days')
                AND access_count = 1
            """.format(days)
            
            deleted = self.execute_update(query)
            return deleted
            
        except Exception as e:
            raise DatabaseError(f"Cleanup failed: {str(e)}") from e