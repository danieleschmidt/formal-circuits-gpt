"""Advanced optimized caching system for formal-circuits-gpt."""

import os
import json
import pickle  # Only for backward compatibility, prefer JSON when possible
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading

from ..monitoring.logger import get_logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    tags: List[str] = None
    
    def __post_init__(self):
        self.tags = self.tags or []
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self):
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_key(key)
                self.stats['misses'] += 1
                return None
            
            # Update access and move to end
            entry.update_access()
            self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats['hits'] += 1
            return entry.data
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None, tags: List[str] = None):
        """Put item in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(data))
            except:
                size_bytes = 0
            
            # Create entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                tags=tags or []
            )
            
            # Remove existing if present
            if key in self.cache:
                self._remove_key(key)
            
            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)
            
            # Evict if necessary
            while len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _remove_key(self, key: str):
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order[0]
            self._remove_key(lru_key)
            self.stats['evictions'] += 1
    
    def clear_by_tags(self, tags: List[str]):
        """Clear entries with any of the specified tags."""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_key(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'total_size_bytes': total_size,
                'usage_percent': len(self.cache) / self.max_size * 100
            }


class PersistentCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        self.lock = threading.Lock()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from persistent cache."""
        with self.lock:
            if key not in self.index:
                return None
            
            metadata = self.index[key]
            
            # Check expiration
            if metadata.get('ttl_seconds') and time.time() - metadata['created_at'] > metadata['ttl_seconds']:
                self.remove(key)
                return None
            
            # Load data from disk
            file_path = self._get_file_path(key)
            if not file_path.exists():
                # Clean up stale index entry
                del self.index[key]
                self._save_index()
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access metadata
                self.index[key]['accessed_at'] = time.time()
                self.index[key]['access_count'] = self.index[key].get('access_count', 0) + 1
                self._save_index()
                
                return data
            except:
                # Remove corrupted entry
                self.remove(key)
                return None
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None, tags: List[str] = None):
        """Put item in persistent cache."""
        with self.lock:
            file_path = self._get_file_path(key)
            
            try:
                # Save data to disk
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Update index
                self.index[key] = {
                    'created_at': time.time(),
                    'accessed_at': time.time(),
                    'access_count': 1,
                    'size_bytes': file_path.stat().st_size,
                    'ttl_seconds': ttl_seconds,
                    'tags': tags or [],
                    'file_path': str(file_path)
                }
                
                self._save_index()
                
            except Exception as e:
                # Clean up on failure
                if file_path.exists():
                    file_path.unlink()
                raise
    
    def remove(self, key: str):
        """Remove item from cache."""
        with self.lock:
            if key in self.index:
                file_path = Path(self.index[key]['file_path'])
                if file_path.exists():
                    file_path.unlink()
                del self.index[key]
                self._save_index()
    
    def clear_by_tags(self, tags: List[str]):
        """Clear entries with any of the specified tags."""
        with self.lock:
            keys_to_remove = []
            for key, metadata in self.index.items():
                if any(tag in metadata.get('tags', []) for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.remove(key)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self.lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, metadata in self.index.items():
                if metadata.get('ttl_seconds') and current_time - metadata['created_at'] > metadata['ttl_seconds']:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.remove(key)
            
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(metadata.get('size_bytes', 0) for metadata in self.index.values())
            total_files = len(self.index)
            
            return {
                'file_count': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }


class OptimizedCacheManager:
    """High-performance cache manager with multiple cache layers."""
    
    def __init__(self, cache_dir: Optional[str] = None, memory_cache_size: int = 1000):
        self.logger = get_logger("cache_manager")
        
        # Initialize cache layers
        self.memory_cache = LRUCache(max_size=memory_cache_size)
        
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), '.cache', 'formal_circuits_gpt')
        
        self.persistent_cache = PersistentCache(cache_dir)
        
        # Cache configurations
        self.default_ttl = 3600 * 24 * 7  # 1 week
        self.proof_ttl = 3600 * 24 * 30   # 1 month
        
        self.logger.info(f"Optimized cache manager initialized with directory: {cache_dir}")
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        # Sort kwargs for consistent key generation
        sorted_items = sorted(kwargs.items())
        key_data = json.dumps(sorted_items, sort_keys=True)
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get_proof_cache(self, hdl_code: str, prover: str, model: str, properties: List[str]) -> Optional[Any]:
        """Get cached proof result."""
        key = self._generate_key(
            "proof",
            hdl_code=hashlib.sha256(hdl_code.encode()).hexdigest(),
            prover=prover,
            model=model,
            properties=sorted(properties)
        )
        
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            self.logger.debug(f"Proof cache hit (memory): {key[:16]}...")
            return result
        
        # Try persistent cache
        result = self.persistent_cache.get(key)
        if result is not None:
            self.logger.debug(f"Proof cache hit (disk): {key[:16]}...")
            # Promote to memory cache
            self.memory_cache.put(key, result, ttl_seconds=self.proof_ttl, tags=["proof"])
            return result
        
        self.logger.debug(f"Proof cache miss: {key[:16]}...")
        return None
    
    def put_proof_cache(self, hdl_code: str, prover: str, model: str, properties: List[str], result: Any):
        """Cache proof result."""
        key = self._generate_key(
            "proof",
            hdl_code=hashlib.sha256(hdl_code.encode()).hexdigest(),
            prover=prover,
            model=model,
            properties=sorted(properties)
        )
        
        tags = ["proof", f"prover:{prover}", f"model:{model}"]
        
        # Store in both caches
        self.memory_cache.put(key, result, ttl_seconds=self.proof_ttl, tags=tags)
        self.persistent_cache.put(key, result, ttl_seconds=self.proof_ttl, tags=tags)
        
        self.logger.debug(f"Cached proof result: {key[:16]}...")
    
    def get_parsed_ast_cache(self, hdl_code: str) -> Optional[Any]:
        """Get cached parsed AST."""
        key = self._generate_key(
            "ast",
            hdl_code=hashlib.sha256(hdl_code.encode()).hexdigest()
        )
        
        result = self.memory_cache.get(key)
        if result is not None:
            self.logger.debug(f"AST cache hit: {key[:16]}...")
            return result
        
        return None
    
    def put_parsed_ast_cache(self, hdl_code: str, ast: Any):
        """Cache parsed AST."""
        key = self._generate_key(
            "ast",
            hdl_code=hashlib.sha256(hdl_code.encode()).hexdigest()
        )
        
        self.memory_cache.put(key, ast, ttl_seconds=self.default_ttl, tags=["ast"])
        self.logger.debug(f"Cached AST: {key[:16]}...")
    
    def get_translation_cache(self, ast_hash: str, prover: str) -> Optional[Any]:
        """Get cached translation."""
        key = self._generate_key("translation", ast_hash=ast_hash, prover=prover)
        
        result = self.memory_cache.get(key)
        if result is not None:
            self.logger.debug(f"Translation cache hit: {key[:16]}...")
            return result
        
        return None
    
    def put_translation_cache(self, ast_hash: str, prover: str, translation: Any):
        """Cache translation."""
        key = self._generate_key("translation", ast_hash=ast_hash, prover=prover)
        
        self.memory_cache.put(key, translation, ttl_seconds=self.default_ttl, 
                            tags=["translation", f"prover:{prover}"])
        self.logger.debug(f"Cached translation: {key[:16]}...")
    
    def invalidate_by_model(self, model: str):
        """Invalidate cache entries for specific model."""
        tags = [f"model:{model}"]
        self.memory_cache.clear_by_tags(tags)
        self.persistent_cache.clear_by_tags(tags)
        self.logger.info(f"Invalidated cache for model: {model}")
    
    def invalidate_by_prover(self, prover: str):
        """Invalidate cache entries for specific prover."""
        tags = [f"prover:{prover}"]
        self.memory_cache.clear_by_tags(tags)
        self.persistent_cache.clear_by_tags(tags)
        self.logger.info(f"Invalidated cache for prover: {prover}")
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.memory_cache = LRUCache(max_size=self.memory_cache.max_size)
        
        # Clear persistent cache
        for key in list(self.persistent_cache.index.keys()):
            self.persistent_cache.remove(key)
        
        self.logger.info("All caches cleared")
    
    def cleanup_cache(self, max_age_days: int) -> Dict[str, int]:
        """Cleanup old cache entries."""
        # Cleanup persistent cache
        expired_count = self.persistent_cache.cleanup_expired()
        
        # TODO: Add cleanup by age for non-expired entries
        
        self.logger.info(f"Cache cleanup completed: {expired_count} expired entries removed")
        
        return {
            "expired_entries_removed": expired_count
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        persistent_stats = self.persistent_cache.get_stats()
        
        return {
            "memory": memory_stats,
            "files": persistent_stats,
            "cache_dir": str(self.persistent_cache.cache_dir)
        }