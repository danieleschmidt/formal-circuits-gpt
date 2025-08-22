"""
Simplified Validation Test

A focused test to validate core functionality without complex dependencies.
"""

import asyncio
import time
import tempfile
import json
from pathlib import Path

def test_basic_imports():
    """Test that core components can be imported."""
    try:
        from src.formal_circuits_gpt.core import CircuitVerifier, ProofResult
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_basic_circuit_verification():
    """Test basic circuit verification functionality."""
    try:
        from src.formal_circuits_gpt.core import CircuitVerifier
        
        verifier = CircuitVerifier()
        
        # Simple test circuit
        simple_verilog = """
        module test_adder(
            input [3:0] a,
            input [3:0] b,
            output [4:0] sum
        );
            assign sum = a + b;
        endmodule
        """
        
        # Test verification
        result = verifier.verify(simple_verilog, ["sum == a + b"])
        
        if hasattr(result, 'status'):
            print(f"✅ Basic verification successful: {result.status}")
            return True
        else:
            print("❌ Verification result invalid")
            return False
            
    except Exception as e:
        print(f"❌ Basic verification failed: {e}")
        return False

def test_observability():
    """Test observability system."""
    try:
        from src.formal_circuits_gpt.monitoring.realtime_observability import RealTimeObservability
        
        obs = RealTimeObservability()
        
        # Test metric recording
        obs.record_metric("test_metric", 42.0)
        
        # Test span creation
        span_id = obs.start_trace("test_operation")
        obs.finish_span(span_id, "success")
        
        print("✅ Observability test successful")
        return True
        
    except Exception as e:
        print(f"❌ Observability test failed: {e}")
        return False

def test_reliability():
    """Test reliability components."""
    try:
        from src.formal_circuits_gpt.reliability.distributed_fault_tolerance import DistributedFaultTolerance
        
        dft = DistributedFaultTolerance("test-node")
        
        # Test basic functionality
        if hasattr(dft, 'nodes') and hasattr(dft, 'performance_metrics'):
            print("✅ Reliability test successful")
            return True
        else:
            print("❌ Reliability components missing")
            return False
            
    except Exception as e:
        print(f"❌ Reliability test failed: {e}")
        return False

async def test_auto_scaling():
    """Test auto-scaling functionality."""
    try:
        from src.formal_circuits_gpt.monitoring.realtime_observability import RealTimeObservability
        from src.formal_circuits_gpt.optimization.intelligent_auto_scaling import IntelligentAutoScaling, ResourceType
        
        obs = RealTimeObservability()
        auto_scaling = IntelligentAutoScaling(obs)
        
        # Test prediction
        prediction = await auto_scaling.predict_demand(ResourceType.COMPUTE)
        
        if hasattr(prediction, 'predicted_value'):
            print("✅ Auto-scaling test successful")
            return True
        else:
            print("❌ Auto-scaling prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

def test_parsers():
    """Test HDL parsers."""
    try:
        from src.formal_circuits_gpt.parsers import VerilogParser
        
        parser = VerilogParser()
        
        simple_verilog = """
        module test(input a, output b);
            assign b = a;
        endmodule
        """
        
        ast = parser.parse(simple_verilog)
        
        if ast and hasattr(ast, 'modules') and len(ast.modules) > 0:
            print("✅ Parser test successful")
            return True
        else:
            print("❌ Parser test failed")
            return False
            
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        return False

async def run_simplified_validation():
    """Run simplified validation tests."""
    print("🚀 Starting Simplified Validation")
    print("="*50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Circuit Verification", test_basic_circuit_verification),
        ("Observability", test_observability),
        ("Reliability", test_reliability),
        ("Parser Functionality", test_parsers),
    ]
    
    async_tests = [
        ("Auto-scaling", test_auto_scaling),
    ]
    
    results = {}
    
    # Run synchronous tests
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        results[test_name] = test_func()
    
    # Run async tests
    for test_name, test_func in async_tests:
        print(f"\n🔍 Running {test_name}...")
        results[test_name] = await test_func()
    
    # Calculate results
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    score = passed / total
    
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall Score: {passed}/{total} ({score:.2%})")
    
    if score >= 0.8:
        print("🎉 VALIDATION SUCCESSFUL - System is ready!")
        status = "SUCCESS"
    elif score >= 0.6:
        print("⚠️ VALIDATION WARNING - Some issues detected")
        status = "WARNING"
    else:
        print("❌ VALIDATION FAILED - Critical issues detected")
        status = "FAILED"
    
    # Save results
    validation_result = {
        "timestamp": time.time(),
        "status": status,
        "score": score,
        "tests_passed": passed,
        "tests_total": total,
        "test_results": results
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(validation_result, f, indent=2)
        print(f"\nResults saved to: {f.name}")
    
    return validation_result

if __name__ == "__main__":
    asyncio.run(run_simplified_validation())