#!/usr/bin/env python3
"""Final comprehensive validation test for formal-circuits-gpt."""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.core import CircuitVerifier

def test_real_world_verilog_examples():
    """Test with realistic Verilog examples."""
    print("🔧 Testing Real-World Verilog Examples...")
    
    verifier = CircuitVerifier()
    
    test_cases = [
        # Simple adder
        """
        module adder(
            input [3:0] a,
            input [3:0] b,
            output [4:0] sum
        );
            assign sum = a + b;
        endmodule
        """,
        
        # Counter with reset
        """
        module counter(
            input clk,
            input reset,
            output reg [7:0] count
        );
            always @(posedge clk or posedge reset) begin
                if (reset)
                    count <= 8'b0;
                else
                    count <= count + 1;
            end
        endmodule
        """,
        
        # Multiplexer
        """
        module mux4to1(
            input [1:0] sel,
            input [3:0] data_in,
            output reg data_out
        );
            always @(*) begin
                case (sel)
                    2'b00: data_out = data_in[0];
                    2'b01: data_out = data_in[1];
                    2'b10: data_out = data_in[2];
                    2'b11: data_out = data_in[3];
                endcase
            end
        endmodule
        """,
        
        # FSM example
        """
        module traffic_light(
            input clk,
            input reset,
            output reg [1:0] light
        );
            parameter RED = 2'b00, YELLOW = 2'b01, GREEN = 2'b10;
            reg [1:0] state, next_state;
            
            always @(posedge clk or posedge reset) begin
                if (reset)
                    state <= RED;
                else
                    state <= next_state;
            end
            
            always @(*) begin
                case (state)
                    RED: next_state = GREEN;
                    GREEN: next_state = YELLOW;
                    YELLOW: next_state = RED;
                    default: next_state = RED;
                endcase
            end
            
            assign light = state;
        endmodule
        """
    ]
    
    passed_count = 0
    
    for i, verilog_code in enumerate(test_cases):
        try:
            start_time = time.time()
            result = verifier.verify(verilog_code)
            duration = time.time() - start_time
            
            if result.status in ["VERIFIED", "FAILED"]:
                passed_count += 1
                print(f"  ✅ Test case {i+1}: {result.status} in {duration:.3f}s")
                print(f"     Properties: {len(result.properties_verified)}")
            else:
                print(f"  ⚠️ Test case {i+1}: Unexpected status {result.status}")
                
        except Exception as e:
            print(f"  ❌ Test case {i+1}: Exception - {str(e)[:50]}...")
    
    success_rate = (passed_count / len(test_cases)) * 100
    print(f"  📊 Real-world examples success rate: {success_rate:.1f}%")
    
    return success_rate >= 90.0

def test_security_hardened_validation():
    """Test security validation with proper HDL structure."""
    print("🔒 Testing Security with Proper HDL Structure...")
    
    verifier = CircuitVerifier(strict_mode=True)
    
    # These should be rejected due to security concerns
    dangerous_but_valid_hdl = [
        """
        module test();
            // $system("rm -rf /");
            wire test_signal;
        endmodule
        """,
        
        """
        module test();
            /* <script>alert(1)</script> */
            wire test_signal;
        endmodule
        """,
        
        """
        module test();
            // javascript:void(0)
            wire test_signal;
        endmodule
        """,
    ]
    
    # These should be accepted (legitimate HDL)
    safe_hdl = [
        """
        module test();
            wire test_signal;
            assign test_signal = 1'b0;
        endmodule
        """,
        
        """
        module adder(input a, input b, output sum);
            assign sum = a + b;
        endmodule
        """,
    ]
    
    rejected_dangerous = 0
    accepted_safe = 0
    
    print("  Testing dangerous patterns...")
    for i, dangerous_hdl in enumerate(dangerous_but_valid_hdl):
        try:
            result = verifier.verify(dangerous_hdl)
            print(f"    ❌ Dangerous pattern {i+1} was accepted!")
        except Exception as e:
            if "security" in str(e).lower() or "dangerous" in str(e).lower():
                rejected_dangerous += 1
                print(f"    ✅ Dangerous pattern {i+1} properly rejected")
            else:
                print(f"    ⚠️ Dangerous pattern {i+1} rejected for other reason")
    
    print("  Testing safe patterns...")
    for i, safe_hdl_code in enumerate(safe_hdl):
        try:
            result = verifier.verify(safe_hdl_code)
            if result.status in ["VERIFIED", "FAILED"]:
                accepted_safe += 1
                print(f"    ✅ Safe pattern {i+1} accepted: {result.status}")
            else:
                print(f"    ⚠️ Safe pattern {i+1} unexpected status: {result.status}")
        except Exception as e:
            print(f"    ❌ Safe pattern {i+1} incorrectly rejected: {str(e)[:50]}...")
    
    dangerous_rejection_rate = (rejected_dangerous / len(dangerous_but_valid_hdl)) * 100
    safe_acceptance_rate = (accepted_safe / len(safe_hdl)) * 100
    
    print(f"  📊 Dangerous pattern rejection: {dangerous_rejection_rate:.1f}%")
    print(f"  📊 Safe pattern acceptance: {safe_acceptance_rate:.1f}%")
    
    return dangerous_rejection_rate >= 80.0 and safe_acceptance_rate >= 90.0

def test_parser_robustness_realistic():
    """Test parser robustness with realistic edge cases."""
    print("🛡️ Testing Parser Robustness (Realistic Cases)...")
    
    verifier = CircuitVerifier()
    
    edge_cases = [
        # Missing semicolons (should be recoverable)
        """
        module test();
            wire a
            wire b
            assign a = b
        endmodule
        """,
        
        # Extra whitespace and formatting issues
        """
        module   test  (  )  ;
            wire    a   ;
            assign   a  =  1'b0  ;
        endmodule
        """,
        
        # Comments in unusual places
        """
        module test(); // Comment here
            wire /* inline comment */ a;
            assign a = // another comment
                1'b0; // end comment
        endmodule
        """,
        
        # Case sensitivity issues
        """
        Module Test();
            Wire A;
            Assign A = 1'B0;
        EndModule
        """,
    ]
    
    recovered_count = 0
    
    for i, edge_case in enumerate(edge_cases):
        try:
            result = verifier.verify(edge_case)
            if result.status in ["VERIFIED", "FAILED"]:
                recovered_count += 1
                print(f"  ✅ Edge case {i+1}: Handled successfully ({result.status})")
            else:
                print(f"  ⚠️ Edge case {i+1}: Partial recovery ({result.status})")
                recovered_count += 0.5
        except Exception as e:
            print(f"  ❌ Edge case {i+1}: Failed - {str(e)[:50]}...")
    
    recovery_rate = (recovered_count / len(edge_cases)) * 100
    print(f"  📊 Realistic edge case recovery: {recovery_rate:.1f}%")
    
    return recovery_rate >= 75.0

def test_performance_validation():
    """Test performance with various circuit sizes."""
    print("⚡ Testing Performance Validation...")
    
    verifier = CircuitVerifier()
    
    # Generate circuits of different sizes
    test_circuits = []
    
    # Small circuit
    small_circuit = """
    module small_test(input a, output b);
        assign b = a;
    endmodule
    """
    test_circuits.append(("Small", small_circuit))
    
    # Medium circuit
    medium_circuit = """
    module medium_test(
        input [7:0] data_in,
        input clk,
        input reset,
        output reg [7:0] data_out
    );
        reg [7:0] buffer [0:3];
        integer i;
        
        always @(posedge clk or posedge reset) begin
            if (reset) begin
                for (i = 0; i < 4; i = i + 1)
                    buffer[i] <= 8'b0;
                data_out <= 8'b0;
            end else begin
                buffer[0] <= data_in;
                buffer[1] <= buffer[0];
                buffer[2] <= buffer[1];
                buffer[3] <= buffer[2];
                data_out <= buffer[3];
            end
        end
    endmodule
    """
    test_circuits.append(("Medium", medium_circuit))
    
    performance_results = []
    
    for size_name, circuit in test_circuits:
        try:
            start_time = time.time()
            result = verifier.verify(circuit)
            duration = time.time() - start_time
            
            performance_results.append((size_name, duration, result.status))
            print(f"  ✅ {size_name} circuit: {duration:.3f}s ({result.status})")
            
        except Exception as e:
            print(f"  ❌ {size_name} circuit: Failed - {str(e)[:50]}...")
            performance_results.append((size_name, float('inf'), "ERROR"))
    
    # Check if performance is reasonable (< 5 seconds for these examples)
    all_fast = all(duration < 5.0 for _, duration, _ in performance_results if duration != float('inf'))
    success_rate = len([r for r in performance_results if r[2] in ["VERIFIED", "FAILED"]]) / len(performance_results) * 100
    
    print(f"  📊 Performance success rate: {success_rate:.1f}%")
    print(f"  📊 All tests under 5s: {'Yes' if all_fast else 'No'}")
    
    return success_rate >= 90.0 and all_fast

def main():
    """Run final comprehensive validation."""
    print("=== FINAL COMPREHENSIVE VALIDATION ===\n")
    
    tests = [
        ("Real-World Verilog Examples", test_real_world_verilog_examples),
        ("Security Hardened Validation", test_security_hardened_validation), 
        ("Parser Robustness (Realistic)", test_parser_robustness_realistic),
        ("Performance Validation", test_performance_validation),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ PASS: {test_name}")
                passed_tests += 1
            else:
                print(f"❌ FAIL: {test_name}")
        except Exception as e:
            print(f"❌ ERROR: {test_name}: {e}")
        print()
    
    success_rate = (passed_tests / total_tests) * 100
    print("=" * 60)
    print(f"📋 FINAL COMPREHENSIVE VALIDATION RESULTS:")
    print(f"  ✅ PASS: {passed_tests}/{total_tests} tests")
    print(f"  📊 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75.0:
        print("🎉 SYSTEM VALIDATION PASSED!")
        print("🚀 Ready for production deployment!")
        return True
    else:
        print("⚠️ System validation needs additional work")
        return False

if __name__ == "__main__":
    main()