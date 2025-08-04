#!/usr/bin/env python3
"""Debug Verilog parser."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.parsers.verilog_parser import VerilogParser

def debug_parsing():
    simple_verilog = """
    module simple_adder(
        input [3:0] a,
        input [3:0] b,
        output [4:0] sum
    );
        assign sum = a + b;
    endmodule
    """
    
    parser = VerilogParser()
    
    # Remove comments
    cleaned = parser._remove_comments(simple_verilog)
    print("Cleaned code:")
    print(repr(cleaned))
    print()
    
    # Parse modules
    modules = parser._parse_modules(cleaned)
    print(f"Found {len(modules)} modules")
    
    if modules:
        module = modules[0]
        print(f"Module name: {module.name}")
        print(f"Ports: {[p.name + ':' + p.signal_type.value for p in module.ports]}")
        print(f"Signals: {[s.name for s in module.signals]}")
        print(f"Assignments: {[(a.target, a.expression) for a in module.assignments]}")
        
        # Debug validation
        all_signals = {p.name for p in module.ports} | {s.name for s in module.signals}
        print(f"All signals: {all_signals}")
        
        for assignment in module.assignments:
            print(f"Assignment target '{assignment.target}' in all_signals: {assignment.target in all_signals}")
            is_output = any(p.name == assignment.target and p.signal_type.value == "output" 
                          for p in module.ports)
            print(f"Is output port: {is_output}")

if __name__ == "__main__":
    debug_parsing()