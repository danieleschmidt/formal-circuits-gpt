"""Cache management system for formal verification results."""

import os
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..database import DatabaseManager, ProofRepository, LemmaRepository
from ..database.models import ProofCache, LemmaCache


class CacheManager:
    """Manages caching for proof results and lemmas."""
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 db_manager: Optional[DatabaseManager] = None):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for file-based cache
            db_manager: Database manager for persistent storage
        """
        self.cache_dir = Path(cache_dir or os.getenv("CACHE_DIR", ".proof_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_manager = db_manager or DatabaseManager()
        self.proof_repo = ProofRepository(self.db_manager)
        self.lemma_repo = LemmaRepository(self.db_manager)
        
        # In-memory caches for fast access
        self._memory_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self._max_memory_entries = 1000
        self._cleanup_interval = 3600  # 1 hour
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def get_cached_proof(self, circuit_code: str, properties: List[str], 
                        prover: str) -> Optional[str]:
        """Get cached proof if available.
        
        Args:
            circuit_code: HDL circuit code
            properties: List of properties to verify
            prover: Theorem prover name
            
        Returns:
            Cached proof code or None
        """
        # Create hashes for lookup
        circuit_hash = ProofCache.create_hash(circuit_code)
        properties_hash = ProofCache.create_properties_hash(properties)
        
        # Check memory cache first
        memory_key = f"{circuit_hash}:{properties_hash}:{prover}"
        with self._cache_lock:
            if memory_key in self._memory_cache:
                return self._memory_cache[memory_key]['proof_code']
        
        # Check database cache
        cached_proof = self.proof_repo.get_cached_proof(
            circuit_hash, properties_hash, prover
        )
        
        if cached_proof and cached_proof.verification_status == "VERIFIED":
            # Store in memory cache for faster future access
            with self._cache_lock:
                self._memory_cache[memory_key] = {
                    'proof_code': cached_proof.proof_code,
                    'timestamp': time.time()
                }
                self._trim_memory_cache()
            
            return cached_proof.proof_code
        
        return None
    
    def cache_proof(self, circuit_code: str, properties: List[str], 
                   prover: str, proof_code: str, verification_status: str,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Cache a proof result.
        
        Args:
            circuit_code: HDL circuit code
            properties: List of properties verified
            prover: Theorem prover used
            proof_code: Generated proof code
            verification_status: "VERIFIED" or "FAILED"
            metadata: Additional metadata
        """
        circuit_hash = ProofCache.create_hash(circuit_code)
        properties_hash = ProofCache.create_properties_hash(properties)
        
        # Create proof cache entry
        proof_cache = ProofCache(
            circuit_hash=circuit_hash,
            properties_hash=properties_hash,
            prover=prover,
            proof_code=proof_code,
            verification_status=verification_status,
            metadata=metadata or {}
        )
        
        # Store in database
        self.proof_repo.cache_proof(proof_cache)
        
        # Store in memory cache if successful
        if verification_status == "VERIFIED":
            memory_key = f"{circuit_hash}:{properties_hash}:{prover}"
            with self._cache_lock:
                self._memory_cache[memory_key] = {
                    'proof_code': proof_code,
                    'timestamp': time.time()
                }
                self._trim_memory_cache()
    
    def get_cached_lemma(self, statement: str, prover: str) -> Optional[str]:
        """Get cached lemma proof if available."""
        lemma_hash = LemmaCache.create_hash(statement, prover)
        
        # Check memory cache
        memory_key = f"lemma:{lemma_hash}"
        with self._cache_lock:
            if memory_key in self._memory_cache:
                return self._memory_cache[memory_key]['proof']
        
        # Check database
        cached_lemma = self.lemma_repo.get_lemma(lemma_hash)
        
        if cached_lemma:
            # Store in memory cache
            with self._cache_lock:
                self._memory_cache[memory_key] = {
                    'proof': cached_lemma.proof,
                    'timestamp': time.time()
                }
                self._trim_memory_cache()
            
            return cached_lemma.proof
        
        return None
    
    def cache_lemma(self, name: str, statement: str, proof: str, 
                   prover: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Cache a lemma for reuse."""
        lemma_hash = LemmaCache.create_hash(statement, prover)
        
        # Create lemma cache entry
        lemma_cache = LemmaCache(
            lemma_hash=lemma_hash,
            lemma_name=name,
            statement=statement,
            proof=proof,
            prover=prover,
            metadata=metadata or {}
        )
        
        # Store in database
        self.lemma_repo.cache_lemma(lemma_cache)
        
        # Store in memory cache
        memory_key = f"lemma:{lemma_hash}"
        with self._cache_lock:
            self._memory_cache[memory_key] = {
                'proof': proof,
                'timestamp': time.time()
            }
            self._trim_memory_cache()
    
    def get_popular_lemmas(self, prover: str, limit: int = 20) -> List[LemmaCache]:
        """Get most frequently used lemmas."""
        return self.lemma_repo.get_popular_lemmas(prover, limit)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # Database stats
        db_stats = self.proof_repo.get_cache_stats()
        
        # Memory cache stats
        with self._cache_lock:
            memory_stats = {
                'memory_entries': len(self._memory_cache),
                'memory_max_entries': self._max_memory_entries,
                'memory_usage_percent': (len(self._memory_cache) / self._max_memory_entries) * 100
            }
        
        # File cache stats
        file_stats = self._get_file_cache_stats()
        
        return {
            'database': db_stats,
            'memory': memory_stats,
            'files': file_stats,
            'cache_dir': str(self.cache_dir)
        }
    
    def cleanup_cache(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old cache entries."""
        # Clean database cache
        deleted_proofs = self.proof_repo.cleanup_old_entries(max_age_days)
        
        # Clean memory cache (entries older than 1 hour)
        deleted_memory = 0
        current_time = time.time()
        with self._cache_lock:
            keys_to_delete = [
                key for key, value in self._memory_cache.items()
                if current_time - value['timestamp'] > 3600
            ]
            for key in keys_to_delete:
                del self._memory_cache[key]
            deleted_memory = len(keys_to_delete)
        
        # Clean file cache
        deleted_files = self._cleanup_file_cache(max_age_days)
        
        return {
            'deleted_proofs': deleted_proofs,
            'deleted_memory': deleted_memory,
            'deleted_files': deleted_files
        }
    
    def _trim_memory_cache(self):
        """Trim memory cache to max size."""
        if len(self._memory_cache) <= self._max_memory_entries:
            return
        
        # Remove oldest entries
        sorted_items = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        items_to_remove = len(self._memory_cache) - self._max_memory_entries
        for i in range(items_to_remove):
            key, _ = sorted_items[i]
            del self._memory_cache[key]
    
    def _get_file_cache_stats(self) -> Dict[str, Any]:
        """Get file cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'file_count': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024)
            }
        except Exception:
            return {
                'file_count': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0
            }
    
    def _cleanup_file_cache(self, max_age_days: int) -> int:
        """Clean up old cache files."""
        try:
            max_age_seconds = max_age_days * 24 * 3600
            current_time = time.time()
            deleted_count = 0
            
            for cache_file in self.cache_dir.glob("*.cache"):
                if current_time - cache_file.stat().st_mtime > max_age_seconds:
                    cache_file.unlink()
                    deleted_count += 1
            
            return deleted_count
        except Exception:
            return 0
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self._cleanup_interval)
                    self.cleanup_cache()
                except Exception:
                    pass  # Ignore cleanup errors
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def clear_all_caches(self) -> None:
        """Clear all caches (memory, database, and files)."""
        # Clear memory cache
        with self._cache_lock:
            self._memory_cache.clear()
        
        # Clear database cache (use with caution!)
        self.db_manager.execute_update("DELETE FROM proof_cache")
        self.db_manager.execute_update("DELETE FROM lemma_cache")
        
        # Clear file cache
        try:
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()
        except Exception:
            pass