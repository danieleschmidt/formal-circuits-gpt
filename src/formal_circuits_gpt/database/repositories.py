"""Repository classes for data access."""

from typing import List, Optional, Dict, Any
from .connection import DatabaseManager
from .models import ProofCache, CircuitModel, VerificationResult, LemmaCache


class BaseRepository:
    """Base repository class."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize repository with database manager."""
        self.db = db_manager


class ProofRepository(BaseRepository):
    """Repository for proof cache operations."""

    def get_cached_proof(
        self, circuit_hash: str, properties_hash: str, prover: str
    ) -> Optional[ProofCache]:
        """Get cached proof if available."""
        query = """
            SELECT * FROM proof_cache 
            WHERE circuit_hash = ? AND properties_hash = ? AND prover = ?
        """

        results = self.db.execute_query(query, (circuit_hash, properties_hash, prover))

        if results:
            # Update access statistics
            self._update_access_stats(results[0]["id"])
            return ProofCache.from_row(results[0])

        return None

    def cache_proof(self, proof: ProofCache) -> int:
        """Cache a proof result."""
        query = """
            INSERT OR REPLACE INTO proof_cache 
            (circuit_hash, properties_hash, prover, proof_code, verification_status, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        params = (
            proof.circuit_hash,
            proof.properties_hash,
            proof.prover,
            proof.proof_code,
            proof.verification_status,
            proof.to_dict()["metadata"],
        )

        return self.db.execute_update(query, params)

    def _update_access_stats(self, proof_id: int):
        """Update access statistics for cached proof."""
        query = """
            UPDATE proof_cache 
            SET last_accessed = CURRENT_TIMESTAMP, 
                access_count = access_count + 1
            WHERE id = ?
        """

        self.db.execute_update(query, (proof_id,))

    def get_all_cached_proofs(self, limit: int = 100) -> List[ProofCache]:
        """Get all cached proofs with optional limit."""
        query = """
            SELECT * FROM proof_cache 
            ORDER BY last_accessed DESC
            LIMIT ?
        """

        results = self.db.execute_query(query, (limit,))
        return [ProofCache.from_row(row) for row in results]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats_query = """
            SELECT 
                COUNT(*) as total_entries,
                SUM(access_count) as total_accesses,
                AVG(access_count) as avg_accesses,
                COUNT(DISTINCT prover) as unique_provers,
                COUNT(CASE WHEN verification_status = 'VERIFIED' THEN 1 END) as verified_count,
                COUNT(CASE WHEN verification_status = 'FAILED' THEN 1 END) as failed_count
            FROM proof_cache
        """

        result = self.db.execute_query(stats_query)

        if result:
            return dict(result[0])

        return {
            "total_entries": 0,
            "total_accesses": 0,
            "avg_accesses": 0,
            "unique_provers": 0,
            "verified_count": 0,
            "failed_count": 0,
        }

    def cleanup_old_entries(self, days: int = 30) -> int:
        """Clean up old, rarely accessed cache entries."""
        query = """
            DELETE FROM proof_cache 
            WHERE last_accessed < datetime('now', '-{} days')
            AND access_count <= 2
        """.format(
            days
        )

        return self.db.execute_update(query)


class CircuitRepository(BaseRepository):
    """Repository for circuit model operations."""

    def save_circuit(self, circuit: CircuitModel) -> int:
        """Save circuit model to database."""
        if circuit.id:
            # Update existing
            query = """
                UPDATE circuit_models 
                SET name = ?, hdl_type = ?, source_code = ?, ast_json = ?, 
                    module_count = ?, updated_at = CURRENT_TIMESTAMP, metadata = ?
                WHERE id = ?
            """
            params = (
                circuit.name,
                circuit.hdl_type,
                circuit.source_code,
                circuit.ast_json,
                circuit.module_count,
                circuit.to_dict()["metadata"],
                circuit.id,
            )
        else:
            # Insert new
            query = """
                INSERT INTO circuit_models 
                (name, hdl_type, source_code, ast_json, module_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            params = (
                circuit.name,
                circuit.hdl_type,
                circuit.source_code,
                circuit.ast_json,
                circuit.module_count,
                circuit.to_dict()["metadata"],
            )

        return self.db.execute_update(query, params)

    def get_circuit_by_id(self, circuit_id: int) -> Optional[CircuitModel]:
        """Get circuit by ID."""
        query = "SELECT * FROM circuit_models WHERE id = ?"
        results = self.db.execute_query(query, (circuit_id,))

        if results:
            return CircuitModel.from_row(results[0])

        return None

    def get_circuit_by_name(self, name: str) -> Optional[CircuitModel]:
        """Get circuit by name."""
        query = "SELECT * FROM circuit_models WHERE name = ?"
        results = self.db.execute_query(query, (name,))

        if results:
            return CircuitModel.from_row(results[0])

        return None

    def get_circuits_by_type(self, hdl_type: str) -> List[CircuitModel]:
        """Get circuits by HDL type."""
        query = (
            "SELECT * FROM circuit_models WHERE hdl_type = ? ORDER BY created_at DESC"
        )
        results = self.db.execute_query(query, (hdl_type,))

        return [CircuitModel.from_row(row) for row in results]

    def search_circuits(self, search_term: str) -> List[CircuitModel]:
        """Search circuits by name or content."""
        query = """
            SELECT * FROM circuit_models 
            WHERE name LIKE ? OR source_code LIKE ?
            ORDER BY created_at DESC
        """

        search_pattern = f"%{search_term}%"
        results = self.db.execute_query(query, (search_pattern, search_pattern))

        return [CircuitModel.from_row(row) for row in results]

    def delete_circuit(self, circuit_id: int) -> int:
        """Delete circuit and related verification results."""
        # Delete verification results first (foreign key constraint)
        self.db.execute_update(
            "DELETE FROM verification_results WHERE circuit_id = ?", (circuit_id,)
        )

        # Delete circuit
        return self.db.execute_update(
            "DELETE FROM circuit_models WHERE id = ?", (circuit_id,)
        )


class VerificationRepository(BaseRepository):
    """Repository for verification result operations."""

    def save_result(self, result: VerificationResult) -> int:
        """Save verification result."""
        query = """
            INSERT INTO verification_results 
            (circuit_id, properties, prover, status, proof_code, errors, execution_time, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        result_dict = result.to_dict()
        params = (
            result.circuit_id,
            result_dict["properties"],
            result.prover,
            result.status,
            result.proof_code,
            result_dict["errors"],
            result.execution_time,
            result_dict["metadata"],
        )

        return self.db.execute_update(query, params)

    def get_results_by_circuit(self, circuit_id: int) -> List[VerificationResult]:
        """Get all verification results for a circuit."""
        query = """
            SELECT * FROM verification_results 
            WHERE circuit_id = ? 
            ORDER BY created_at DESC
        """

        results = self.db.execute_query(query, (circuit_id,))
        return [VerificationResult.from_row(row) for row in results]

    def get_recent_results(self, limit: int = 50) -> List[VerificationResult]:
        """Get recent verification results."""
        query = """
            SELECT vr.*, cm.name as circuit_name
            FROM verification_results vr
            JOIN circuit_models cm ON vr.circuit_id = cm.id
            ORDER BY vr.created_at DESC
            LIMIT ?
        """

        results = self.db.execute_query(query, (limit,))
        return [VerificationResult.from_row(row) for row in results]

    def get_success_rate(self, prover: Optional[str] = None) -> Dict[str, float]:
        """Get verification success rate statistics."""
        base_query = """
            SELECT 
                status,
                COUNT(*) as count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
            FROM verification_results
        """

        if prover:
            base_query += " WHERE prover = ?"
            params = (prover,)
        else:
            params = ()

        base_query += " GROUP BY status"

        results = self.db.execute_query(base_query, params)

        stats = {}
        for row in results:
            stats[row["status"]] = {
                "count": row["count"],
                "percentage": row["percentage"],
            }

        return stats


class LemmaRepository(BaseRepository):
    """Repository for lemma cache operations."""

    def get_lemma(self, lemma_hash: str) -> Optional[LemmaCache]:
        """Get cached lemma by hash."""
        query = "SELECT * FROM lemma_cache WHERE lemma_hash = ?"
        results = self.db.execute_query(query, (lemma_hash,))

        if results:
            # Update usage count
            self._increment_usage(results[0]["id"])
            return LemmaCache.from_row(results[0])

        return None

    def cache_lemma(self, lemma: LemmaCache) -> int:
        """Cache a lemma for reuse."""
        query = """
            INSERT OR REPLACE INTO lemma_cache 
            (lemma_hash, lemma_name, statement, proof, prover, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        params = (
            lemma.lemma_hash,
            lemma.lemma_name,
            lemma.statement,
            lemma.proof,
            lemma.prover,
            lemma.to_dict()["metadata"],
        )

        return self.db.execute_update(query, params)

    def _increment_usage(self, lemma_id: int):
        """Increment usage count for lemma."""
        query = "UPDATE lemma_cache SET usage_count = usage_count + 1 WHERE id = ?"
        self.db.execute_update(query, (lemma_id,))

    def get_popular_lemmas(self, prover: str, limit: int = 20) -> List[LemmaCache]:
        """Get most frequently used lemmas for a prover."""
        query = """
            SELECT * FROM lemma_cache 
            WHERE prover = ? 
            ORDER BY usage_count DESC 
            LIMIT ?
        """

        results = self.db.execute_query(query, (prover, limit))
        return [LemmaCache.from_row(row) for row in results]
