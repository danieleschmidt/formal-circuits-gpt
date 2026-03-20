"""Tests for FormalReport."""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from formal_circuits.verilog_parser import VerilogParser
from formal_circuits.property_gen import PropertyGenerator
from formal_circuits.sva_generator import SVAGenerator
from formal_circuits.refiner import SelfRefiner
from formal_circuits.report import FormalReport

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


def make_report(verilog_text):
    parser = VerilogParser()
    module = parser.parse(verilog_text)
    gen = PropertyGenerator()
    props = gen.generate_for_module(module)
    sva = SVAGenerator()
    spec = sva.generate_full_spec(module, props)
    refiner = SelfRefiner()
    steps = refiner.refine(spec, module)
    return FormalReport(module, props, spec, steps)


def test_report_to_dict():
    report = make_report(ADDER_V)
    d = report.to_dict()

    assert isinstance(d, dict)
    assert "module" in d
    assert d["module"]["name"] == "simple_adder"
    assert "ports" in d["module"]
    assert "properties" in d
    assert "sva_spec" in d
    assert "refinement_steps" in d

    # Check property structure
    for prop in d["properties"]:
        assert "name" in prop
        assert "type" in prop
        assert "description" in prop
        assert "confidence" in prop


def test_report_to_json():
    report = make_report(ADDER_V)
    j = report.to_json()
    assert isinstance(j, str)
    parsed = json.loads(j)
    assert parsed["module"]["name"] == "simple_adder"


def test_report_to_markdown():
    report = make_report(ADDER_V)
    md = report.to_markdown()

    assert isinstance(md, str)
    assert "# Formal Verification Report" in md
    assert "simple_adder" in md
    assert "## Ports" in md
    assert "## Formal Properties" in md
    assert "## Generated SVA Specification" in md


def test_report_to_markdown_counter():
    report = make_report(COUNTER_V)
    md = report.to_markdown()
    assert "counter" in md
    assert "Sequential" in md


def test_report_save_json(tmp_path):
    report = make_report(ADDER_V)
    out = tmp_path / "report.json"
    report.save(str(out), format="json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["module"]["name"] == "simple_adder"


def test_report_save_markdown(tmp_path):
    report = make_report(ADDER_V)
    out = tmp_path / "report.md"
    report.save(str(out), format="markdown")
    assert out.exists()
    content = out.read_text()
    assert "# Formal Verification Report" in content


def test_report_invalid_format():
    report = make_report(ADDER_V)
    with pytest.raises(ValueError):
        report.save("/tmp/test.xyz", format="pdf")


def test_report_refinement_steps():
    report = make_report(COUNTER_V)
    d = report.to_dict()
    assert isinstance(d["refinement_steps"], list)
    for step in d["refinement_steps"]:
        assert "iteration" in step
        assert "issues_found" in step
        assert "improvement_score" in step
