"""Fault injection and chaos engineering for reliability testing."""

import random
import time
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass


class FaultType(Enum):
    """Types of faults that can be injected."""
    NETWORK_DELAY = "network_delay"
    NETWORK_FAILURE = "network_failure"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    PROVER_CRASH = "prover_crash"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_FULL = "disk_full"
    CORRUPT_RESPONSE = "corrupt_response"
    RANDOM_FAILURE = "random_failure"


@dataclass
class FaultConfig:
    """Configuration for fault injection."""
    fault_type: FaultType
    probability: float  # 0.0 to 1.0
    duration_seconds: Optional[float] = None
    magnitude: float = 1.0  # Intensity multiplier
    enabled: bool = True


class FaultInjector:
    """Chaos engineering fault injection system."""
    
    def __init__(self):
        """Initialize fault injector."""
        self.active_faults: Dict[FaultType, FaultConfig] = {}
        self.fault_history: List[Dict[str, Any]] = []
        self.enabled = False
    
    def enable_chaos_mode(self):
        """Enable chaos engineering mode."""
        self.enabled = True
    
    def disable_chaos_mode(self):
        """Disable chaos engineering mode."""
        self.enabled = False
    
    def register_fault(self, config: FaultConfig):
        """Register a fault configuration."""
        self.active_faults[config.fault_type] = config
    
    def should_inject_fault(self, fault_type: FaultType) -> bool:
        """Determine if a fault should be injected."""
        if not self.enabled:
            return False
        
        config = self.active_faults.get(fault_type)
        if not config or not config.enabled:
            return False
        
        return random.random() < config.probability
    
    def inject_network_delay(self, base_delay: float = 0.0) -> float:
        """Inject network delay."""
        if self.should_inject_fault(FaultType.NETWORK_DELAY):
            config = self.active_faults[FaultType.NETWORK_DELAY]
            additional_delay = random.uniform(1.0, 5.0) * config.magnitude
            
            self._record_fault(FaultType.NETWORK_DELAY, {
                "base_delay": base_delay,
                "injected_delay": additional_delay
            })
            
            time.sleep(additional_delay)
            return base_delay + additional_delay
        
        return base_delay
    
    def inject_llm_failure(self, operation: str) -> Optional[Exception]:
        """Inject LLM-related failures."""
        if self.should_inject_fault(FaultType.LLM_TIMEOUT):
            self._record_fault(FaultType.LLM_TIMEOUT, {"operation": operation})
            from ..exceptions import LLMError
            return LLMError(
                "Injected timeout failure",
                provider="test",
                api_error_code="CHAOS_TIMEOUT"
            )
        
        if self.should_inject_fault(FaultType.LLM_RATE_LIMIT):
            self._record_fault(FaultType.LLM_RATE_LIMIT, {"operation": operation})
            from ..exceptions import LLMError
            return LLMError(
                "Injected rate limit error", 
                provider="test",
                api_error_code="CHAOS_RATE_LIMIT"
            )
        
        if self.should_inject_fault(FaultType.CORRUPT_RESPONSE):
            self._record_fault(FaultType.CORRUPT_RESPONSE, {"operation": operation})
            from ..exceptions import LLMError
            return LLMError(
                "Injected response corruption",
                provider="test", 
                api_error_code="CHAOS_CORRUPT"
            )
        
        return None
    
    def inject_prover_failure(self, prover_name: str) -> Optional[Exception]:
        """Inject theorem prover failures."""
        if self.should_inject_fault(FaultType.PROVER_CRASH):
            self._record_fault(FaultType.PROVER_CRASH, {"prover": prover_name})
            from ..exceptions import ProverError
            return ProverError(
                "Injected prover crash",
                prover_name=prover_name,
                command="chaos_injection"
            )
        
        return None
    
    def inject_random_failure(self, operation: str) -> Optional[Exception]:
        """Inject random failure."""
        if self.should_inject_fault(FaultType.RANDOM_FAILURE):
            self._record_fault(FaultType.RANDOM_FAILURE, {"operation": operation})
            from ..exceptions import VerificationError
            return VerificationError(
                f"Injected random failure in {operation}",
                error_code="CHAOS_RANDOM"
            )
        
        return None
    
    def _record_fault(self, fault_type: FaultType, context: Dict[str, Any]):
        """Record fault injection for analysis."""
        self.fault_history.append({
            "fault_type": fault_type.value,
            "timestamp": time.time(),
            "context": context
        })
        
        # Keep history limited
        if len(self.fault_history) > 1000:
            self.fault_history = self.fault_history[-500:]
    
    def get_fault_statistics(self) -> Dict[str, Any]:
        """Get statistics on injected faults."""
        if not self.fault_history:
            return {"total_faults": 0, "by_type": {}}
        
        total = len(self.fault_history)
        by_type = {}
        
        for fault in self.fault_history:
            fault_type = fault["fault_type"]
            by_type[fault_type] = by_type.get(fault_type, 0) + 1
        
        return {
            "total_faults": total,
            "by_type": by_type,
            "fault_rate": total / max(1, time.time() - self.fault_history[0]["timestamp"])
        }
    
    def reset_fault_history(self):
        """Reset fault injection history."""
        self.fault_history.clear()


# Global fault injector instance
fault_injector = FaultInjector()


def chaos_mode(enabled: bool = True):
    """Context manager or decorator for chaos mode."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_state = fault_injector.enabled
            fault_injector.enabled = enabled
            try:
                return func(*args, **kwargs)
            finally:
                fault_injector.enabled = original_state
        return wrapper
    
    if callable(enabled):
        # Used as decorator without arguments
        func = enabled
        enabled = True
        return decorator(func)
    else:
        # Used as decorator with arguments
        return decorator


# Predefined fault scenarios
class ChaosScenarios:
    """Predefined chaos engineering scenarios."""
    
    @staticmethod
    def network_instability():
        """Simulate network instability."""
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.NETWORK_DELAY,
            probability=0.3,
            magnitude=2.0
        ))
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.NETWORK_FAILURE, 
            probability=0.1
        ))
    
    @staticmethod
    def llm_service_degradation():
        """Simulate LLM service issues."""
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.LLM_TIMEOUT,
            probability=0.2
        ))
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.LLM_RATE_LIMIT,
            probability=0.15
        ))
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.CORRUPT_RESPONSE,
            probability=0.05
        ))
    
    @staticmethod
    def prover_instability():
        """Simulate theorem prover instability."""
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.PROVER_CRASH,
            probability=0.1
        ))
    
    @staticmethod
    def full_chaos():
        """Enable all chaos scenarios."""
        ChaosScenarios.network_instability()
        ChaosScenarios.llm_service_degradation() 
        ChaosScenarios.prover_instability()
        fault_injector.register_fault(FaultConfig(
            fault_type=FaultType.RANDOM_FAILURE,
            probability=0.05
        ))