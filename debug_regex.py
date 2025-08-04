#!/usr/bin/env python3
"""Debug regex patterns."""

import re

def debug_port_pattern():
    port_pattern = re.compile(
        r'(input|output|inout)\s+(?:\[(\d+):(\d+)\])?\s*(\w+)', 
        re.MULTILINE
    )
    
    module_body = """
        input [3:0] a,
        input [3:0] b,
        output [4:0] sum
    );
        assign sum = a + b;
    """
    
    print("Port pattern matches:")
    for match in port_pattern.finditer(module_body):
        print(f"Direction: '{match.group(1)}'")
        print(f"MSB: '{match.group(2)}'") 
        print(f"LSB: '{match.group(3)}'")
        print(f"Name: '{match.group(4)}'")
        print("---")

if __name__ == "__main__":
    debug_port_pattern()