"""Tests for PropertyGenerator."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from formal_circuits.verilog_parser import VerilogParser
from formal_circuits.property_gen import PropertyGenerator, Property

ADDER_V = """
module simple_adder (
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum
);
    assign sum = a + b;
endmodule
"""

COUNTER_V = """
module counter (
    input clk,
    input rst,
    input enable,
    output reg [7:0] count
);
    always @(posedge clk) begin
        if (rst)
            count <= 8'b0;
        else if (enable)
            count <= count + 1;
    end
endmodule
"""


def test_property_gen_combinational():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    assert len(props) > 0
    types = {p.prop_type for p in props}
    # Should have combinational and/or safety properties
    assert "combinational" in types or "safety" in types

    # Each property should have required fields
    for prop in props:
        assert prop.name
        assert prop.description
        assert prop.formal_expr
        assert 0.0 <= prop.confidence <= 1.0


def test_property_gen_sequential():
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    assert len(props) > 0
    types = {p.prop_type for p in props}
    assert "sequential" in types

    # Should have reset property
    reset_props = [p for p in props if "reset" in p.name.lower()]
    assert len(reset_props) > 0


def test_property_describes_reset():
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    reset_prop = next((p for p in props if "reset" in p.name.lower()), None)
    assert reset_prop is not None
    assert "rst" in reset_prop.formal_expr or "reset" in reset_prop.formal_expr.lower()


def test_property_describe():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    for prop in props:
        desc = gen.describe_property(prop)
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_safety_props_generated():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    safety = [p for p in props if p.prop_type == "safety"]
    assert len(safety) > 0
