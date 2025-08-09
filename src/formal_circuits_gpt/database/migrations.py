"""Database migration management."""

import os
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
from .connection import DatabaseManager


class MigrationError(Exception):
    """Migration execution error."""
    pass


class Migration:
    """Individual database migration."""
    
    def __init__(self, version: int, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
    
    def __repr__(self):
        return f"Migration({self.version}, {self.name})"


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.migrations = self._load_migrations()
    
    def _load_migrations(self) -> List[Migration]:
        """Load all available migrations."""
        migrations = [
            Migration(
                version=1,
                name="initial_schema",
                up_sql="""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """,
                down_sql="DROP TABLE IF EXISTS schema_migrations;"
            ),
            
            Migration(
                version=2,
                name="add_performance_indexes",
                up_sql="""
                    CREATE INDEX IF NOT EXISTS idx_proof_cache_status 
                    ON proof_cache(verification_status);
                    
                    CREATE INDEX IF NOT EXISTS idx_verification_status 
                    ON verification_results(status);
                    
                    CREATE INDEX IF NOT EXISTS idx_circuit_updated 
                    ON circuit_models(updated_at);
                """,
                down_sql="""
                    DROP INDEX IF EXISTS idx_proof_cache_status;
                    DROP INDEX IF EXISTS idx_verification_status;
                    DROP INDEX IF EXISTS idx_circuit_updated;
                """
            ),
            
            Migration(
                version=3,
                name="add_proof_metrics",
                up_sql="""
                    ALTER TABLE proof_cache 
                    ADD COLUMN proof_size INTEGER DEFAULT 0;
                    
                    ALTER TABLE proof_cache 
                    ADD COLUMN complexity_score REAL DEFAULT 0.0;
                    
                    ALTER TABLE verification_results 
                    ADD COLUMN memory_usage INTEGER DEFAULT 0;
                """,
                down_sql=""  # Cannot easily rollback ALTER TABLE ADD COLUMN in SQLite
            ),
            
            Migration(
                version=4,
                name="add_circuit_tags",
                up_sql="""
                    CREATE TABLE IF NOT EXISTS circuit_tags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        circuit_id INTEGER NOT NULL,
                        tag TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (circuit_id) REFERENCES circuit_models(id) ON DELETE CASCADE,
                        UNIQUE(circuit_id, tag)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_circuit_tags_circuit 
                    ON circuit_tags(circuit_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_circuit_tags_tag 
                    ON circuit_tags(tag);
                """,
                down_sql="""
                    DROP TABLE IF EXISTS circuit_tags;
                """
            ),
            
            Migration(
                version=5,
                name="add_lemma_dependencies",
                up_sql="""
                    CREATE TABLE IF NOT EXISTS lemma_dependencies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lemma_id INTEGER NOT NULL,
                        depends_on_id INTEGER NOT NULL,
                        dependency_type TEXT DEFAULT 'uses',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (lemma_id) REFERENCES lemma_cache(id) ON DELETE CASCADE,
                        FOREIGN KEY (depends_on_id) REFERENCES lemma_cache(id) ON DELETE CASCADE,
                        UNIQUE(lemma_id, depends_on_id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_lemma_deps_lemma 
                    ON lemma_dependencies(lemma_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_lemma_deps_depends 
                    ON lemma_dependencies(depends_on_id);
                """,
                down_sql="""
                    DROP TABLE IF EXISTS lemma_dependencies;
                """
            )
        ]
        
        return sorted(migrations, key=lambda m: m.version)
    
    def get_current_version(self) -> int:
        """Get current database schema version."""
        try:
            # Check if migrations table exists
            result = self.db.execute_query("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_migrations'
            """)
            
            if not result:
                return 0
            
            # Get latest migration version
            result = self.db.execute_query("""
                SELECT MAX(version) as version FROM schema_migrations
            """)
            
            return result[0]['version'] if result and result[0]['version'] else 0
            
        except Exception:
            return 0
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get migrations that haven't been applied yet."""
        current_version = self.get_current_version()
        return [m for m in self.migrations if m.version > current_version]
    
    def run_migrations(self) -> List[Migration]:
        """Run all pending migrations."""
        pending = self.get_pending_migrations()
        applied = []
        
        for migration in pending:
            try:
                self._apply_migration(migration)
                applied.append(migration)
                print(f"✅ Applied migration {migration.version}: {migration.name}")
            except Exception as e:
                print(f"❌ Failed to apply migration {migration.version}: {e}")
                break
        
        return applied
    
    def _apply_migration(self, migration: Migration):
        """Apply a single migration."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Execute migration SQL
            for statement in migration.up_sql.split(';'):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            
            # Record migration in schema_migrations table
            cursor.execute("""
                INSERT INTO schema_migrations (version, name) 
                VALUES (?, ?)
            """, (migration.version, migration.name))
    
    def rollback_migration(self, target_version: int) -> List[Migration]:
        """Rollback to a specific migration version."""
        current_version = self.get_current_version()
        
        if target_version >= current_version:
            return []
        
        # Get migrations to rollback (in reverse order)
        to_rollback = [
            m for m in reversed(self.migrations) 
            if target_version < m.version <= current_version
        ]
        
        rolled_back = []
        
        for migration in to_rollback:
            try:
                if migration.down_sql:
                    self._rollback_migration(migration)
                    rolled_back.append(migration)
                    print(f"⬇️ Rolled back migration {migration.version}: {migration.name}")
                else:
                    print(f"⚠️ Cannot rollback migration {migration.version}: no down_sql provided")
                    break
            except Exception as e:
                print(f"❌ Failed to rollback migration {migration.version}: {e}")
                break
        
        return rolled_back
    
    def _rollback_migration(self, migration: Migration):
        """Rollback a single migration."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Execute rollback SQL
            for statement in migration.down_sql.split(';'):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            
            # Remove migration record
            cursor.execute("""
                DELETE FROM schema_migrations WHERE version = ?
            """, (migration.version,))
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        current_version = self.get_current_version()
        pending = self.get_pending_migrations()
        
        applied_migrations = []
        try:
            results = self.db.execute_query("""
                SELECT version, name, applied_at 
                FROM schema_migrations 
                ORDER BY version
            """)
            applied_migrations = [dict(row) for row in results]
        except Exception:
            pass
        
        return {
            'current_version': current_version,
            'latest_available': max(m.version for m in self.migrations) if self.migrations else 0,
            'pending_count': len(pending),
            'pending_migrations': [{'version': m.version, 'name': m.name} for m in pending],
            'applied_migrations': applied_migrations
        }


def run_migrations(db_manager: Optional[DatabaseManager] = None) -> List[Migration]:
    """Run all pending migrations."""
    if db_manager is None:
        db_manager = DatabaseManager()
    
    migration_manager = MigrationManager(db_manager)
    return migration_manager.run_migrations()


def get_migration_status(db_manager: Optional[DatabaseManager] = None) -> Dict[str, Any]:
    """Get migration status."""
    if db_manager is None:
        db_manager = DatabaseManager()
    
    migration_manager = MigrationManager(db_manager)
    return migration_manager.get_migration_status()