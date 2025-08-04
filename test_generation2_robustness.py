#!/usr/bin/env python3
"""Generation 2 robustness and error handling tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.core import CircuitVerifier
from formal_circuits_gpt.security import SecurityError
from formal_circuits_gpt.exceptions import VerificationError


def test_input_validation():
    """Test comprehensive input validation."""
    print("Testing input validation...")
    
    verifier = CircuitVerifier(strict_mode=True)
    
    # Test invalid HDL content
    try:
        result = verifier.verify("$system(\"rm -rf /\")")  # Dangerous system call
        print("✗ Should have blocked dangerous HDL content")
        return False
    except SecurityError:
        print("✓ Dangerous HDL content blocked")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    # Test excessively long content
    try:
        long_content = "a" * 2_000_000  # Exceeds max length
        result = verifier.verify(long_content)
        print("✗ Should have blocked excessively long content")
        return False
    except SecurityError:
        print("✓ Excessively long content blocked")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    # Test invalid properties
    try:
        result = verifier.verify("module test(); endmodule", ["'; DROP TABLE users; --"])
        print("✗ Should have blocked SQL injection in properties")
        return False
    except SecurityError:
        print("✓ SQL injection attempt blocked")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    return True


def test_error_handling():
    """Test comprehensive error handling."""
    print("Testing error handling...")
    
    # Test invalid initialization parameters
    try:
        verifier = CircuitVerifier(prover="invalid_prover")
        print("✗ Should have rejected invalid prover")
        return False
    except SecurityError:
        print("✓ Invalid prover rejected")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    try:
        verifier = CircuitVerifier(temperature=-1.0)
        print("✗ Should have rejected invalid temperature")
        return False
    except SecurityError:
        print("✓ Invalid temperature rejected")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    try:
        verifier = CircuitVerifier(refinement_rounds=100)
        print("✗ Should have rejected excessive refinement rounds")
        return False
    except SecurityError:
        print("✓ Excessive refinement rounds rejected")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    return True


def test_logging_and_monitoring():
    """Test logging and monitoring features."""
    print("Testing logging and monitoring...")
    
    try:
        verifier = CircuitVerifier()
        
        # Check that logger is initialized
        if not hasattr(verifier, 'logger'):
            print("✗ Logger not initialized")
            return False
        
        # Check that health checker is initialized
        if not hasattr(verifier, 'health_checker'):
            print("✗ Health checker not initialized")
            return False
        
        # Check that session ID is generated
        if not hasattr(verifier, 'session_id') or not verifier.session_id:
            print("✗ Session ID not generated")
            return False
        
        print("✓ Logging and monitoring components initialized")
        
        # Test health check
        health_results = verifier.health_checker.check_all()
        if not health_results:
            print("✗ Health check returned no results")
            return False
        
        print(f"✓ Health check completed - {len(health_results)} components checked")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging/monitoring test failed: {e}")
        return False


def test_sanitization():
    """Test input sanitization."""
    print("Testing input sanitization...")
    
    try:
        verifier = CircuitVerifier(strict_mode=False)  # Allow warnings but sanitize
        
        # Test HDL with comments that could contain injection
        hdl_with_comments = """
        module test();
            // This is a comment with <script>alert('xss')</script>
            wire a;  // Another comment with | dangerous | pipes
        endmodule
        """
        
        result = verifier.verify(hdl_with_comments)
        
        # Should succeed after sanitization
        if result.status in ["VERIFIED", "FAILED"]:  # Either is OK, point is it didn't crash
            print("✓ HDL sanitization working")
        else:
            print(f"✗ Unexpected status: {result.status}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Sanitization test failed: {e}")
        return False


def test_performance_logging():
    """Test performance logging."""
    print("Testing performance logging...")
    
    try:
        verifier = CircuitVerifier()
        
        simple_verilog = """
        module simple_gate(
            input a,
            input b,
            output out
        );
            assign out = a & b;
        endmodule
        """
        
        result = verifier.verify(simple_verilog)
        
        # Check that duration is tracked
        if not hasattr(result, 'duration_ms') or result.duration_ms <= 0:
            print("✗ Performance metrics not tracked")
            return False
        
        # Check that verification ID is assigned
        if not hasattr(result, 'verification_id') or not result.verification_id:
            print("✗ Verification ID not assigned")
            return False
        
        print(f"✓ Performance logged - Duration: {result.duration_ms:.2f}ms")
        print(f"✓ Verification ID: {result.verification_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance logging test failed: {e}")
        return False


def test_graceful_degradation():
    """Test graceful degradation when components fail."""
    print("Testing graceful degradation...")
    
    try:
        # Test with debug mode to see full errors
        verifier = CircuitVerifier(debug_mode=False)  # Should handle errors gracefully
        
        # Test with malformed HDL that should be parsed but fail later
        malformed_hdl = """
        module incomplete_module(
            input a
            // Missing closing parenthesis and other syntax errors
        """
        
        try:
            result = verifier.verify(malformed_hdl)
            print("✗ Should have failed with malformed HDL")
            return False
        except VerificationError as e:
            print("✓ Gracefully handled malformed HDL with VerificationError")
        except Exception as e:
            print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Graceful degradation test failed: {e}")
        return False


def main():
    """Run all Generation 2 robustness tests."""
    print("=== Formal-Circuits-GPT Generation 2 Robustness Tests ===\n")
    
    tests = [
        test_input_validation,
        test_error_handling,
        test_logging_and_monitoring,
        test_sanitization,
        test_performance_logging,
        test_graceful_degradation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print(f"=== Results: {passed}/{total} robustness tests passed ===")
    
    if passed == total:
        print("🎉 All Generation 2 robustness tests passed!")
        print("✓ Comprehensive error handling implemented")
        print("✓ Input validation and sanitization working")  
        print("✓ Logging and monitoring operational")
        print("✓ Security measures active")
        return True
    else:
        print("❌ Some robustness tests failed. Generation 2 needs fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)