#!/usr/bin/env python3
"""Debug full parser flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.parsers.verilog_parser import VerilogParser
from formal_circuits_gpt.parsers.ast_nodes import SignalType

def debug_full_parsing():
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
    
    # Step by step debugging
    cleaned = parser._remove_comments(simple_verilog)
    print("1. Cleaned code - OK")
    
    # Find module match
    import re
    module_pattern = re.compile(
        r'module\s+(\w+)\s*(?:\#\([^)]*\))?\s*\(([^)]*)\)\s*;', 
        re.MULTILINE | re.DOTALL
    )
    
    match = module_pattern.search(cleaned)
    if match:
        module_name = match.group(1)
        port_list = match.group(2) if match.group(2) else ""
        print(f"2. Module name: {module_name}")
        print(f"3. Port list: {repr(port_list)}")
        
        # Get module body
        module_start = match.end()
        module_body = parser._extract_module_body(cleaned, module_start)
        print(f"4. Module body: {repr(module_body)}")
        
        # Parse ports
        ports = parser._parse_ports(port_list, module_body)
        print(f"5. Parsed ports:")
        for port in ports:
            print(f"   {port.name} ({port.signal_type.value}, width={port.width})")
        
        # Parse assignments  
        assignments = parser._parse_assignments(module_body)
        print(f"6. Parsed assignments:")
        for assign in assignments:
            print(f"   {assign.target} = {assign.expression}")
        
        # Check validation logic
        all_signals = {p.name for p in ports}
        print(f"7. All signal names: {all_signals}")
        
        for assignment in assignments:
            print(f"8. Checking assignment '{assignment.target}':")
            print(f"   In all_signals: {assignment.target in all_signals}")
            
            # Check if output
            is_output = any(p.name == assignment.target and p.signal_type == SignalType.OUTPUT 
                          for p in ports)
            print(f"   Is output port: {is_output}")
            
            # List all ports for debugging
            for p in ports:
                print(f"   Port: {p.name}, type: {p.signal_type}, equals target: {p.name == assignment.target}")

if __name__ == "__main__":
    debug_full_parsing()