#!/usr/bin/env python3
"""Basic functionality test for formal-circuits-gpt Generation 1."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.core import CircuitVerifier
from formal_circuits_gpt.parsers.ast_nodes import CircuitAST, Module, Port, SignalType

def test_basic_initialization():
    """Test basic CircuitVerifier initialization."""
    print("Testing CircuitVerifier initialization...")
    verifier = CircuitVerifier()
    assert verifier.prover == "isabelle"
    assert verifier.model == "gpt-4-turbo"
    assert verifier.temperature == 0.1
    print("✓ CircuitVerifier initialization passed")
    return True

def test_verilog_parsing():
    """Test basic Verilog parsing."""
    print("Testing Verilog parsing...")
    verifier = CircuitVerifier()
    
    simple_verilog = """
    module simple_adder(
        input [3:0] a,
        input [3:0] b,
        output [4:0] sum
    );
        assign sum = a + b;
    endmodule
    """
    
    try:
        ast = verifier._parse_hdl(simple_verilog)
        assert isinstance(ast, CircuitAST)
        assert len(ast.modules) == 1
        
        module = ast.modules[0]
        assert module.name == "simple_adder"
        assert len(module.ports) >= 2  # Should have inputs and outputs
        assert len(module.assignments) >= 1  # Should have the assign statement
        
        print("✓ Verilog parsing passed")
    except Exception as e:
        print(f"✗ Verilog parsing failed: {e}")
        return False
    
    return True

def test_property_generation():
    """Test property generation."""
    print("Testing property generation...")  
    verifier = CircuitVerifier()
    
    # Create a simple module AST
    ports = [
        Port("a", SignalType.INPUT, width=4),
        Port("b", SignalType.INPUT, width=4),
        Port("sum", SignalType.OUTPUT, width=5)
    ]
    
    module = Module(
        name="test_adder",
        ports=ports,
        signals=[],
        assignments=[],
        always_blocks=[],
        submodules=[],
        parameters={}
    )
    
    ast = CircuitAST(modules=[module])
    
    try:
        properties = verifier.property_generator.generate_properties(ast)
        assert len(properties) > 0
        print(f"✓ Property generation passed - generated {len(properties)} properties")
        
        # Print first few properties
        for i, prop in enumerate(properties[:3]):
            print(f"  Property {i+1}: {prop.name}")
            
    except Exception as e:
        print(f"✗ Property generation failed: {e}")
        return False
    
    return True

def test_isabelle_translation():
    """Test Isabelle translation."""
    print("Testing Isabelle translation...")
    
    # Create simple AST
    ports = [
        Port("a", SignalType.INPUT, width=1),
        Port("b", SignalType.INPUT, width=1),
        Port("out", SignalType.OUTPUT, width=1)
    ]
    
    module = Module(
        name="simple_gate",
        ports=ports,
        signals=[],
        assignments=[],
        always_blocks=[],
        submodules=[],
        parameters={}
    )
    
    ast = CircuitAST(modules=[module])
    
    try:
        from formal_circuits_gpt.translators import IsabelleTranslator
        translator = IsabelleTranslator()
        
        theory_content = translator.translate(ast, "TestCircuit")
        assert "theory TestCircuit" in theory_content
        assert "simple_gate" in theory_content
        assert "end" in theory_content
        
        print("✓ Isabelle translation passed")
        print("  Sample theory content:")
        print("  " + theory_content.split('\n')[0])
        
    except Exception as e:
        print(f"✗ Isabelle translation failed: {e}")
        return False
    
    return True

def main():
    """Run all basic functionality tests."""
    print("=== Formal-Circuits-GPT Generation 1 Basic Tests ===\n")
    
    tests = [
        test_basic_initialization,
        test_verilog_parsing, 
        test_property_generation,
        test_isabelle_translation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result is None or result:  # Treat None as success
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All Generation 1 basic functionality tests passed!")
        return True
    else:
        print("❌ Some tests failed. Generation 1 needs fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)