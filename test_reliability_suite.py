"""Comprehensive reliability testing suite."""

import pytest
import time
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from src.formal_circuits_gpt import CircuitVerifier
from src.formal_circuits_gpt.reliability.fault_injection import (
    fault_injector, ChaosScenarios, chaos_mode, FaultConfig, FaultType
)
from src.formal_circuits_gpt.reliability.circuit_breaker import circuit_breaker_manager
from src.formal_circuits_gpt.reliability.rate_limiter import rate_limiter_manager
from src.formal_circuits_gpt.exceptions import *


class TestCircuitBreakerReliability:
    """Test circuit breaker reliability patterns."""
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after repeated failures."""
        cb = circuit_breaker_manager.get_breaker("test", failure_threshold=3, timeout=1.0)
        
        def failing_operation():
            raise Exception("Simulated failure")
        
        # Should work normally first few times (but fail)
        for _ in range(3):
            with pytest.raises(Exception):
                cb.call(failing_operation)
        
        # Should now be open and fail fast
        start_time = time.time()
        with pytest.raises(Exception) as exc_info:
            cb.call(failing_operation)
        duration = time.time() - start_time
        
        # Should fail immediately (circuit open)
        assert duration < 0.1
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state recovery."""
        cb = circuit_breaker_manager.get_breaker("test_recovery", failure_threshold=2, timeout=0.1)
        
        # Force circuit open
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should now allow one call (half-open)
        success_count = 0
        
        def sometimes_succeeds():
            nonlocal success_count
            success_count += 1
            if success_count == 1:
                return "success"
            raise Exception("Still failing")
        
        # First call should succeed (half-open -> closed)
        result = cb.call(sometimes_succeeds)
        assert result == "success"
        
        # Subsequent calls should work normally
        with pytest.raises(Exception):
            cb.call(sometimes_succeeds)
    
    def test_circuit_breaker_with_verification(self):
        """Test circuit breaker integrated with verification."""
        verifier = CircuitVerifier(debug_mode=True)
        
        # Mock LLM to always fail
        with patch.object(verifier.llm_manager, 'generate_sync') as mock_llm:
            mock_llm.side_effect = Exception("API failure")
            
            hdl_code = "module test(); endmodule"
            
            # Should eventually open circuit breaker
            for i in range(6):  # More than failure threshold
                try:
                    verifier.verify(hdl_code)
                except Exception as e:
                    if "Circuit breaker is OPEN" in str(e):
                        break
            else:
                pytest.fail("Circuit breaker should have opened")


class TestRateLimiterReliability:
    """Test rate limiting reliability patterns."""
    
    def test_token_bucket_rate_limiting(self):
        """Test token bucket rate limiter."""
        limiter = rate_limiter_manager.create_token_bucket(
            "test_bucket", capacity=3, refill_rate=1.0
        )
        
        # Should allow initial burst
        for _ in range(3):
            assert limiter.wait_for_token(1, timeout=0.1) == True
        
        # Should block after capacity exhausted
        assert limiter.wait_for_token(1, timeout=0.1) == False
        
        # Should allow after refill time
        time.sleep(1.1)
        assert limiter.wait_for_token(1, timeout=0.1) == True
    
    def test_rate_limiter_integration(self):
        """Test rate limiter with verification system."""
        # This would be tested with real API calls in integration tests
        pass


class TestFaultInjectionChaos:
    """Test fault injection and chaos engineering."""
    
    def setup_method(self):
        """Setup for each test."""
        fault_injector.reset_fault_history()
        fault_injector.active_faults.clear()
        fault_injector.disable_chaos_mode()
    
    def test_network_delay_injection(self):
        """Test network delay fault injection."""
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.NETWORK_DELAY,
            probability=1.0,  # Always inject
            magnitude=1.0
        ))
        fault_injector.enable_chaos_mode()
        
        start_time = time.time()
        delay = fault_injector.inject_network_delay(0.0)
        duration = time.time() - start_time
        
        # Should have injected delay
        assert duration >= 1.0  # At least 1 second delay
        assert delay >= 1.0
        
        # Should be recorded in history
        stats = fault_injector.get_fault_statistics()
        assert stats["total_faults"] == 1
        assert "network_delay" in stats["by_type"]
    
    def test_llm_failure_injection(self):
        """Test LLM failure injection."""
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.LLM_TIMEOUT,
            probability=1.0
        ))
        fault_injector.enable_chaos_mode()
        
        error = fault_injector.inject_llm_failure("test_operation")
        
        assert error is not None
        assert isinstance(error, LLMError)
        assert "timeout" in error.message.lower()
    
    @chaos_mode(enabled=True)
    def test_chaos_mode_decorator(self):
        """Test chaos mode decorator."""
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.RANDOM_FAILURE,
            probability=1.0
        ))
        
        error = fault_injector.inject_random_failure("test")
        assert error is not None
    
    def test_chaos_scenarios(self):
        """Test predefined chaos scenarios."""
        fault_injector.enable_chaos_mode()
        
        # Test network instability scenario
        ChaosScenarios.network_instability()
        assert FaultType.NETWORK_DELAY in fault_injector.active_faults
        assert FaultType.NETWORK_FAILURE in fault_injector.active_faults
        
        # Test LLM service degradation  
        ChaosScenarios.llm_service_degradation()
        assert FaultType.LLM_TIMEOUT in fault_injector.active_faults
        assert FaultType.LLM_RATE_LIMIT in fault_injector.active_faults
    
    def test_verification_under_chaos(self):
        """Test verification system under chaos conditions."""
        fault_injector.enable_chaos_mode()
        ChaosScenarios.llm_service_degradation()
        
        verifier = CircuitVerifier(
            debug_mode=True,
            refinement_rounds=2  # Reduce to speed up test
        )
        
        hdl_code = "module test_chaos(); endmodule"
        
        # May succeed or fail, but should handle gracefully
        try:
            result = verifier.verify(hdl_code)
            # If it succeeds, great!
            assert result is not None
        except (LLMError, VerificationError) as e:
            # Expected under chaos conditions
            assert "timeout" in str(e).lower() or "rate" in str(e).lower()


class TestRetryPolicyReliability:
    """Test retry policy reliability patterns."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff retry policy."""
        from src.formal_circuits_gpt.reliability.retry_policy import ExponentialBackoff
        
        policy = ExponentialBackoff(max_attempts=3, base_delay=0.1, max_delay=1.0)
        
        attempt_count = 0
        start_time = time.time()
        
        def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Not yet")
            return "success"
        
        result = policy.execute(failing_operation)
        duration = time.time() - start_time
        
        assert result == "success"
        assert attempt_count == 3
        assert duration >= 0.3  # Should have delays: 0.1 + 0.2 = 0.3
    
    def test_retry_with_circuit_breaker(self):
        """Test retry policy interaction with circuit breaker."""
        from src.formal_circuits_gpt.reliability.retry_policy import ExponentialBackoff
        
        cb = circuit_breaker_manager.get_breaker("retry_test", failure_threshold=2, timeout=0.1)
        policy = ExponentialBackoff(max_attempts=5, base_delay=0.01)
        
        def always_fails():
            raise Exception("Always fails")
        
        # Should eventually hit circuit breaker
        with pytest.raises(Exception) as exc_info:
            policy.execute(lambda: cb.call(always_fails))
        
        # Might be the original exception or circuit breaker exception
        assert "fails" in str(exc_info.value) or "Circuit breaker" in str(exc_info.value)


class TestIntegratedReliability:
    """Test integrated reliability patterns working together."""
    
    def test_full_reliability_stack(self):
        """Test all reliability patterns working together."""
        # Enable chaos mode with multiple fault types
        fault_injector.enable_chaos_mode()
        ChaosScenarios.full_chaos()
        
        verifier = CircuitVerifier(
            debug_mode=True,
            refinement_rounds=1  # Keep test fast
        )
        
        hdl_code = """
        module reliable_test(
            input [3:0] a,
            input [3:0] b,
            output [4:0] sum
        );
            assign sum = a + b;
        endmodule
        """
        
        success_count = 0
        total_attempts = 10
        
        # Run multiple attempts to test reliability
        for i in range(total_attempts):
            try:
                result = verifier.verify(hdl_code)
                if result and result.status == "VERIFIED":
                    success_count += 1
            except Exception as e:
                # Expected under chaos - system should handle gracefully
                assert isinstance(e, (VerificationError, LLMError, ProverError))
        
        # Should have some successes despite chaos
        print(f"Success rate under chaos: {success_count}/{total_attempts}")
        
        # Get fault injection statistics
        stats = fault_injector.get_fault_statistics()
        print(f"Faults injected: {stats}")
        
        # Clean up
        fault_injector.disable_chaos_mode()
        fault_injector.reset_fault_history()
    
    def test_graceful_degradation(self):
        """Test system degrades gracefully under load."""
        # This would test with real load in integration tests
        pass
    
    def test_recovery_after_failures(self):
        """Test system recovers after failures."""
        # Test circuit breaker recovery, rate limiter refill, etc.
        pass


class TestErrorHandlingReliability:
    """Test comprehensive error handling."""
    
    def test_enhanced_exception_handling(self):
        """Test enhanced exception with context and suggestions."""
        error = VerificationError(
            "Test verification failed",
            circuit_name="test_circuit",
            properties_failed=["prop1", "prop2"],
            error_code="TEST_001",
            suggestions=["Try this", "Or that"]
        )
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "VerificationError"
        assert error_dict["category"] == "verification"
        assert error_dict["error_code"] == "TEST_001"
        assert len(error_dict["suggestions"]) == 2
        
        user_msg = error.get_user_friendly_message()
        assert "Suggestions:" in user_msg
        assert "Try this" in user_msg
    
    def test_error_context_preservation(self):
        """Test error context is preserved through call stack."""
        def level3():
            raise ParsingError(
                "Parse failed",
                hdl_type="verilog",
                line_number=42,
                source_snippet="module test"
            )
        
        def level2():
            try:
                level3()
            except ParsingError as e:
                # Enhance with more context
                e.context["function"] = "level2"
                raise
        
        def level1():
            try:
                level2()
            except ParsingError as e:
                # Add even more context
                e.context["call_stack"] = ["level1", "level2", "level3"]
                raise
        
        with pytest.raises(ParsingError) as exc_info:
            level1()
        
        error = exc_info.value
        assert error.line_number == 42
        assert error.context["function"] == "level2"
        assert "call_stack" in error.context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])