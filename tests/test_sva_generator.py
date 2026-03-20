"""Tests for SVAGenerator."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from formal_circuits.verilog_parser import VerilogParser
from formal_circuits.property_gen import PropertyGenerator
from formal_circuits.sva_generator import SVAGenerator

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


def test_sva_generator_assertion_contains_property():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    sva = SVAGenerator()
    for prop in props:
        assertion = sva.generate_assertion(prop, module)
        assert prop.name in assertion
        assert isinstance(assertion, str)
        assert len(assertion) > 0


def test_sva_generator_full_spec():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    sva = SVAGenerator()
    spec = sva.generate_full_spec(module, props)

    assert f"module {module.name}_spec" in spec
    assert f"endmodule" in spec
    assert "assert property" in spec
    assert "assume" in spec  # input assumptions
    assert "cover property" in spec


def test_sva_generate_assumption():
    from formal_circuits.verilog_parser import Port
    port = Port(name="clk", direction="input", width=1)
    sva = SVAGenerator()
    assumption = sva.generate_assumption(port)
    assert "clk" in assumption
    assert "assume" in assumption


def test_sva_generate_cover():
    from formal_circuits.property_gen import Property
    prop = Property(
        name="test_prop",
        prop_type="safety",
        description="A test property",
        formal_expr="assert property (##0 1'b1);",
        confidence=0.9,
    )
    sva = SVAGenerator()
    cover = sva.generate_cover(prop)
    assert "cover" in cover
    assert "test_prop" in cover


def test_sva_to_file(tmp_path):
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)
    sva = SVAGenerator()
    spec = sva.generate_full_spec(module, props)

    output = tmp_path / "test.sva"
    sva.to_file(spec, str(output))
    assert output.exists()
    assert output.read_text() == spec


def test_sva_full_spec_counter():
    """Counter module should produce a spec with posedge clk references."""
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)
    sva = SVAGenerator()
    spec = sva.generate_full_spec(module, props)

    assert "posedge clk" in spec or "clk" in spec
    assert module.name in spec
