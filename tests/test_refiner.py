"""Tests for SelfRefiner."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from formal_circuits.verilog_parser import VerilogParser
from formal_circuits.property_gen import PropertyGenerator
from formal_circuits.sva_generator import SVAGenerator
from formal_circuits.refiner import SelfRefiner, RefinementStep

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

ADDER_V = """
module simple_adder (
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum
);
    assign sum = a + b;
endmodule
"""


def test_refiner_check_spec():
    """check_spec should detect issues in a minimal spec."""
    refiner = SelfRefiner()
    minimal_spec = """
    module foo_spec(clk, rst, out);
    input clk;
    input rst;
    output out;
    assert property (@(posedge clk) 1'b1);
    endmodule
    """
    issues = refiner.check_spec(minimal_spec)
    # Should find at least some issues
    assert isinstance(issues, list)
    # No cover properties
    assert "no_cover_properties" in issues


def test_refiner_check_clean_spec():
    """A spec with all required elements should return fewer issues."""
    refiner = SelfRefiner()
    good_spec = """
    module foo_spec(clk, rst, out);
    input clk; input rst; output out;
    assume_clk: assume property (@(posedge clk) 1'b1);
    assume_rst_sync: assume property (@(posedge clk) 1'b1);
    my_prop: assert property (@(posedge clk) (out == 1'b0));
    cover_basic: cover property (##[1:10] 1'b1);
    endmodule
    """
    issues = refiner.check_spec(good_spec)
    # Should not flag clock/reset if assume present
    assert "missing_clock_assumption" not in issues
    assert "no_cover_properties" not in issues


def test_refiner_iterates():
    """Refiner should produce at least one refinement step."""
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)

    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    sva = SVAGenerator()
    spec = sva.generate_full_spec(module, props)

    refiner = SelfRefiner(max_iterations=3)
    steps = refiner.refine(spec, module)

    assert isinstance(steps, list)
    assert len(steps) >= 1
    for step in steps:
        assert isinstance(step, RefinementStep)
        assert step.iteration >= 1
        assert isinstance(step.issues_found, list)
        assert isinstance(step.refined_spec, str)
        assert 0.0 <= step.improvement_score <= 1.0


def test_refiner_max_iterations():
    """Refiner should respect max_iterations."""
    parser = VerilogParser()
    module = parser.parse(COUNTER_V)

    gen = PropertyGenerator()
    props = gen.generate_for_module(module)

    sva = SVAGenerator()
    spec = sva.generate_full_spec(module, props)

    refiner = SelfRefiner(max_iterations=2)
    steps = refiner.refine(spec, module)
    assert len(steps) <= 2


def test_refiner_apply_fix_cover():
    """apply_fix should add cover properties."""
    refiner = SelfRefiner()
    spec = "module foo_spec(a);\n  input a;\nendmodule"
    fixed = refiner.apply_fix(spec, "no_cover_properties")
    assert "cover property" in fixed


def test_refiner_apply_fix_clock():
    """apply_fix should add clock assumption."""
    refiner = SelfRefiner()
    spec = "module foo_spec(clk);\n  input clk;\nassert property (@(posedge clk) 1'b1);\nendmodule"
    fixed = refiner.apply_fix(spec, "missing_clock_assumption")
    assert "assume" in fixed
