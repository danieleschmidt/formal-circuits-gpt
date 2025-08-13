"""Adaptive caching system with ML-driven optimization."""

import time
import threading
import hashlib
import pickle
import json
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
import statistics


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # ML-driven adaptive policy
    TTL = "ttl"  # Time To Live
    COST_AWARE = "cost_aware"  # Based on computation cost


@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    access_count: int = 0
    last_access: float = 0.0
    creation_time: float = 0.0
    computation_cost_ms: float = 0.0
    size_bytes: int = 0
    hit_rate: float = 0.0
    eviction_score: float = 0.0
    
    def __post_init__(self):
        if self.creation_time == 0.0:
            self.creation_time = time.time()
        if self.last_access == 0.0:
            self.last_access = self.creation_time


class AdaptiveCacheManager:
    """Adaptive cache manager with ML-driven optimization."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 500.0,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.ADAPTIVE,
        ttl_seconds: float = 3600.0,
        learning_enabled: bool = True,
        predictive_prefetch: bool = True,
        compression_enabled: bool = True
    ):
        """Initialize adaptive cache manager.
        
        Args:
            max_size: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
            eviction_policy: Cache eviction policy
            ttl_seconds: Default TTL for items
            learning_enabled: Enable ML-based optimization
            predictive_prefetch: Enable predictive prefetching
            compression_enabled: Enable value compression
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.eviction_policy = eviction_policy
        self.ttl_seconds = ttl_seconds
        self.learning_enabled = learning_enabled
        self.predictive_prefetch = predictive_prefetch
        self.compression_enabled = compression_enabled
        
        self._cache: Dict[str, CacheItem] = {}
        self._lock = threading.RLock()
        
        # Statistics and learning
        self._access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._hit_rates: deque = deque(maxlen=1000)
        self._total_requests = 0
        self._total_hits = 0
        self._memory_usage_mb = 0.0
        
        # Adaptive learning parameters
        self._feature_weights = {
            "access_frequency": 0.3,
            "recency": 0.25,
            "computation_cost": 0.2,
            "size": -0.1,  # Negative because smaller is better
            "hit_rate": 0.15
        }
        
        # Performance tracking
        self._eviction_history: deque = deque(maxlen=100)
        self._optimization_intervals = deque(maxlen=50)
        self._last_optimization = time.time()
        
        # Predictive prefetching
        self._access_sequences: deque = deque(maxlen=1000)  # Track access patterns
        self._pattern_predictor: Dict[str, List[str]] = {}  # key -> likely next keys
        
        # Compression support
        if compression_enabled:
            try:
                import zlib
                self._compressor = zlib
            except ImportError:
                self._compressor = None
                self.compression_enabled = False
        else:
            self._compressor = None
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            self._total_requests += 1
            
            if key not in self._cache:
                self._hit_rates.append(0)
                return None
            
            item = self._cache[key]
            
            # Check TTL
            if time.time() - item.creation_time > self.ttl_seconds:
                self._remove_item(key)
                self._hit_rates.append(0)
                return None
            
            # Update access statistics
            item.access_count += 1
            item.last_access = time.time()
            self._access_patterns[key].append(time.time())
            
            self._total_hits += 1
            self._hit_rates.append(1)
            
            # Update hit rate for this item
            recent_accesses = len(self._access_patterns[key])
            item.hit_rate = recent_accesses / max(1, self._total_requests)
            
            # Track access sequence for predictive prefetching
            self._access_sequences.append(key)
            if self.predictive_prefetch:
                self._update_access_patterns(key)
                self._trigger_predictive_prefetch(key)
            
            # Decompress value if needed
            value = item.value
            if self.compression_enabled and hasattr(item, '_compressed') and item._compressed:
                value = self._decompress_value(value)
            
            return value
    
    def put(
        self,
        key: str,
        value: Any,
        computation_cost_ms: float = 0.0,
        ttl_override: Optional[float] = None
    ) -> bool:
        """Put item in cache."""
        with self._lock:
            # Compress value if enabled
            stored_value = value
            compressed = False
            if self.compression_enabled and self._compressor:
                try:
                    compressed_data = self._compress_value(value)
                    if len(compressed_data) < len(pickle.dumps(value)) * 0.8:  # Only compress if >20% savings
                        stored_value = compressed_data
                        compressed = True
                except:
                    pass  # Fall back to uncompressed
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(stored_value))
            except:
                size_bytes = 1024  # Default estimate
            
            # Check if we need to evict items
            if self._should_evict(size_bytes):
                self._evict_items(size_bytes)
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=stored_value,
                computation_cost_ms=computation_cost_ms,
                size_bytes=size_bytes
            )
            
            # Mark compression status
            if compressed:
                item._compressed = True
            
            # Remove existing item if present
            if key in self._cache:
                self._remove_item(key)
            
            # Add new item
            self._cache[key] = item
            self._memory_usage_mb += size_bytes / (1024 * 1024)
            
            # Trigger learning if enabled
            if self.learning_enabled and len(self._cache) % 50 == 0:
                self._update_adaptive_weights()
            
            return True
    
    def _should_evict(self, new_item_size: int) -> bool:
        """Check if eviction is needed."""
        current_size = len(self._cache)
        current_memory = self._memory_usage_mb + (new_item_size / (1024 * 1024))
        
        return (current_size >= self.max_size or 
                current_memory >= self.max_memory_mb)
    
    def _evict_items(self, required_space_bytes: int):
        """Evict items based on current policy."""
        if not self._cache:
            return
        
        start_time = time.time()
        items_evicted = 0
        space_freed = 0
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            candidates = self._get_lru_candidates()
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            candidates = self._get_lfu_candidates()
        elif self.eviction_policy == CacheEvictionPolicy.COST_AWARE:
            candidates = self._get_cost_aware_candidates()
        elif self.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            candidates = self._get_adaptive_candidates()
        else:
            candidates = list(self._cache.keys())
        
        # Evict candidates until we have enough space
        required_memory = required_space_bytes / (1024 * 1024)
        
        for key in candidates:
            if key not in self._cache:
                continue
            
            item = self._cache[key]
            self._remove_item(key)
            
            items_evicted += 1
            space_freed += item.size_bytes
            
            # Check if we've freed enough space
            if (len(self._cache) < self.max_size * 0.9 and 
                self._memory_usage_mb + required_memory < self.max_memory_mb * 0.9):
                break
        
        # Record eviction statistics
        eviction_time = (time.time() - start_time) * 1000
        self._eviction_history.append({
            "timestamp": time.time(),
            "items_evicted": items_evicted,
            "space_freed_bytes": space_freed,
            "eviction_time_ms": eviction_time,
            "policy": self.eviction_policy.value
        })
    
    def _get_lru_candidates(self) -> List[str]:
        """Get LRU eviction candidates."""
        return sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_access
        )
    
    def _get_lfu_candidates(self) -> List[str]:
        """Get LFU eviction candidates."""
        return sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].access_count
        )
    
    def _get_cost_aware_candidates(self) -> List[str]:
        """Get cost-aware eviction candidates."""
        def cost_benefit_ratio(key: str) -> float:
            item = self._cache[key]
            benefit = item.computation_cost_ms * item.access_count
            cost = item.size_bytes / 1024  # KB
            return benefit / max(cost, 1)
        
        return sorted(
            self._cache.keys(),
            key=cost_benefit_ratio
        )
    
    def _get_adaptive_candidates(self) -> List[str]:
        """Get adaptive eviction candidates using ML-based scoring."""
        def adaptive_score(key: str) -> float:
            item = self._cache[key]
            current_time = time.time()
            
            # Feature extraction
            features = {
                "access_frequency": item.access_count / max(current_time - item.creation_time, 1),
                "recency": 1 / max(current_time - item.last_access + 1, 1),
                "computation_cost": item.computation_cost_ms / 1000,  # Normalize to seconds
                "size": item.size_bytes / (1024 * 1024),  # MB
                "hit_rate": item.hit_rate
            }
            
            # Calculate weighted score
            score = sum(
                features[feature] * weight 
                for feature, weight in self._feature_weights.items()
                if feature in features
            )
            
            return score
        
        # Sort by score (lower scores are evicted first)
        return sorted(self._cache.keys(), key=adaptive_score)
    
    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self._cache:
            item = self._cache[key]
            self._memory_usage_mb -= item.size_bytes / (1024 * 1024)
            del self._cache[key]
    
    def _update_adaptive_weights(self):
        """Update adaptive weights based on performance."""
        if not self.learning_enabled or len(self._eviction_history) < 10:
            return
        
        current_time = time.time()
        if current_time - self._last_optimization < 300:  # 5 minutes
            return
        
        # Analyze recent performance
        recent_evictions = [e for e in self._eviction_history if current_time - e["timestamp"] < 1800]
        
        if len(recent_evictions) < 5:
            return
        
        # Calculate performance metrics
        avg_hit_rate = statistics.mean(self._hit_rates) if self._hit_rates else 0
        eviction_frequency = len(recent_evictions) / (1800 / 60)  # per minute
        avg_eviction_time = statistics.mean([e["eviction_time_ms"] for e in recent_evictions])
        
        # Adaptive weight adjustment based on performance
        if avg_hit_rate < 0.5:  # Low hit rate
            self._feature_weights["recency"] += 0.05
            self._feature_weights["access_frequency"] -= 0.02
        elif avg_hit_rate > 0.8:  # High hit rate
            self._feature_weights["computation_cost"] += 0.03
            self._feature_weights["size"] -= 0.02
        
        if eviction_frequency > 10:  # Too frequent evictions
            self._feature_weights["size"] -= 0.05
            self._feature_weights["access_frequency"] += 0.03
        
        # Normalize weights
        total_weight = sum(abs(w) for w in self._feature_weights.values())
        for feature in self._feature_weights:
            self._feature_weights[feature] /= total_weight
        
        self._last_optimization = current_time
    
    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching pattern."""
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                self._remove_item(key)
            return len(keys_to_remove)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._memory_usage_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            hit_rate = self._total_hits / max(self._total_requests, 1)
            
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._memory_usage_mb,
                "max_memory_mb": self.max_memory_mb,
                "memory_utilization": self._memory_usage_mb / self.max_memory_mb,
                "hit_rate": hit_rate,
                "total_requests": self._total_requests,
                "total_hits": self._total_hits,
                "eviction_policy": self.eviction_policy.value,
                "learning_enabled": self.learning_enabled,
                "feature_weights": dict(self._feature_weights),
                "recent_evictions": len(self._eviction_history),
                "items_by_access_count": {
                    "1": len([item for item in self._cache.values() if item.access_count == 1]),
                    "2-5": len([item for item in self._cache.values() if 2 <= item.access_count <= 5]),
                    "6-20": len([item for item in self._cache.values() if 6 <= item.access_count <= 20]),
                    "20+": len([item for item in self._cache.values() if item.access_count > 20])
                }
            }
    
    def optimize(self) -> Dict[str, Any]:
        """Manually trigger cache optimization."""
        with self._lock:
            start_time = time.time()
            initial_size = len(self._cache)
            initial_memory = self._memory_usage_mb
            
            # Force adaptive weight update
            if self.learning_enabled:
                self._update_adaptive_weights()
            
            # Preemptive eviction if near limits
            if len(self._cache) > self.max_size * 0.8 or self._memory_usage_mb > self.max_memory_mb * 0.8:
                self._evict_items(0)  # Evict to free up space
            
            optimization_time = (time.time() - start_time) * 1000
            
            return {
                "optimization_time_ms": optimization_time,
                "items_before": initial_size,
                "items_after": len(self._cache),
                "memory_before_mb": initial_memory,
                "memory_after_mb": self._memory_usage_mb,
                "items_evicted": initial_size - len(self._cache),
                "memory_freed_mb": initial_memory - self._memory_usage_mb
            }
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value using zlib."""
        if not self._compressor:
            return pickle.dumps(value)
        
        pickled = pickle.dumps(value)
        return self._compressor.compress(pickled)
    
    def _decompress_value(self, compressed_data: bytes) -> Any:
        """Decompress value using zlib."""
        if not self._compressor:
            return pickle.loads(compressed_data)
        
        decompressed = self._compressor.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _update_access_patterns(self, accessed_key: str):
        """Update access patterns for predictive prefetching."""
        if len(self._access_sequences) < 2:
            return
        
        # Look at recent access patterns
        recent_sequences = list(self._access_sequences)[-10:]  # Last 10 accesses
        
        # Find patterns where this key was accessed
        for i in range(len(recent_sequences) - 1):
            if recent_sequences[i] == accessed_key:
                next_key = recent_sequences[i + 1]
                
                if accessed_key not in self._pattern_predictor:
                    self._pattern_predictor[accessed_key] = []
                
                # Add to predictions with frequency tracking
                predictions = self._pattern_predictor[accessed_key]
                if next_key not in predictions:
                    predictions.append(next_key)
                
                # Keep only top 5 predictions
                if len(predictions) > 5:
                    self._pattern_predictor[accessed_key] = predictions[-5:]
    
    def _trigger_predictive_prefetch(self, accessed_key: str):
        """Trigger predictive prefetching based on access patterns."""
        if accessed_key not in self._pattern_predictor:
            return
        
        # Get predicted next accesses
        predicted_keys = self._pattern_predictor[accessed_key]
        
        # Check if predicted keys are not in cache and could be prefetched
        for predicted_key in predicted_keys:
            if predicted_key not in self._cache:
                # In a real implementation, this would trigger background prefetch
                # For now, we just track the prediction
                pass