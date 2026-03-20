"""Tests for VerilogParser."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from formal_circuits.verilog_parser import VerilogParser, Port, AlwaysBlock

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

MUX_V = """
module mux4 (
    input [1:0] sel,
    input [7:0] a, b, c, d,
    output reg [7:0] out
);
    always @(*) begin
        case (sel)
            2'b00: out = a;
            2'b01: out = b;
            2'b10: out = c;
            2'b11: out = d;
        endcase
    end
endmodule
"""


def test_parse_module_name_adder():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    assert module.name == "simple_adder"


def test_parse_module_name_counter():
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)
    assert module.name == "counter"


def test_parse_ports_adder():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    port_names = {p.name for p in module.ports}
    assert "a" in port_names
    assert "b" in port_names
    assert "sum" in port_names

    a_port = next(p for p in module.ports if p.name == "a")
    assert a_port.direction == "input"
    assert a_port.width == 8

    sum_port = next(p for p in module.ports if p.name == "sum")
    assert sum_port.direction == "output"
    assert sum_port.width == 9


def test_parse_always_blocks_counter():
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)
    assert len(module.always_blocks) == 1
    block = module.always_blocks[0]
    assert "posedge clk" in block.sensitivity
    assert block.block_type == "sequential"


def test_parse_sequential_detection():
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)
    blocks = module.always_blocks
    assert len(blocks) == 1
    assert parser.is_sequential(blocks[0]) is True


def test_parse_combinational_detection():
    parser = VerilogParser()
    module = parser.parse(MUX_V)
    blocks = module.always_blocks
    assert len(blocks) == 1
    assert parser.is_sequential(blocks[0]) is False
    assert blocks[0].block_type == "combinational"


def test_parse_file(tmp_path):
    """Test parsing from a file."""
    f = tmp_path / "test.v"
    f.write_text(ADDER_V)
    parser = VerilogParser()
    module = parser.parse_file(str(f))
    assert module.name == "simple_adder"
    assert len(module.ports) >= 3


def test_parse_parameters():
    """Test parameter extraction."""
    v = """
    module paramtest #(parameter WIDTH = 8, parameter DEPTH = 16) (
        input clk,
        output [7:0] out
    );
    endmodule
    """
    parser = VerilogParser()
    module = parser.parse(v)
    assert "WIDTH" in module.parameters
    assert module.parameters["WIDTH"] == "8"


def test_raw_text_preserved():
    parser = VerilogParser()
    module = parser.parse(ADDER_V)
    assert "assign sum = a + b" in module.raw_text
