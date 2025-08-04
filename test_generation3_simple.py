#!/usr/bin/env python3
"""Simple Generation 3 optimization tests."""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.cache.optimized_cache import LRUCache, OptimizedCacheManager


def test_basic_lru_cache():
    """Test basic LRU cache functionality."""
    print("Testing basic LRU cache...")
    
    cache = LRUCache(max_size=3)
    
    # Basic put/get
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Test eviction
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key1
    
    assert cache.get("key1") is None
    assert cache.get("key4") == "value4"
    
    print("✓ Basic LRU cache working")
    return True


def test_cache_manager():
    """Test optimized cache manager."""
    print("Testing cache manager...")
    
    cache = OptimizedCacheManager(memory_cache_size=10)
    
    # Test AST caching
    test_hdl = "module test(); endmodule"
    test_ast = {"modules": [{"name": "test"}]}
    
    cache.put_parsed_ast_cache(test_hdl, test_ast)
    retrieved = cache.get_parsed_ast_cache(test_hdl)
    
    assert retrieved == test_ast
    
    print("✓ Cache manager working")
    return True


def test_cache_stats():
    """Test cache statistics."""
    print("Testing cache statistics...")
    
    cache = OptimizedCacheManager()
    
    # Add some data
    for i in range(5):
        cache.put_parsed_ast_cache(f"hdl_{i}", f"ast_{i}")
    
    stats = cache.get_cache_stats()
    
    assert "memory" in stats
    assert "files" in stats
    assert stats["memory"]["size"] == 5
    
    print("✓ Cache statistics working")
    return True


def main():
    """Run simple Generation 3 tests."""
    print("=== Simple Generation 3 Optimization Tests ===\n")
    
    tests = [
        test_basic_lru_cache,
        test_cache_manager,
        test_cache_stats
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 Generation 3 core optimization features working!")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)