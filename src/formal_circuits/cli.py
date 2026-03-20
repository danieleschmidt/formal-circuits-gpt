"""CLI entry point for formal-circuits-gpt."""

import argparse
import sys
from pathlib import Path

from .verilog_parser import VerilogParser
from .property_gen import PropertyGenerator
from .sva_generator import SVAGenerator
from .refiner import SelfRefiner
from .report import FormalReport


def cmd_convert(args):
    """Convert a Verilog file to an SVA spec."""
    parser = VerilogParser()
    module = parser.parse_file(args.input)

    prop_gen = PropertyGenerator()
    properties = prop_gen.generate_for_module(module)

    sva_gen = SVAGenerator()
    spec = sva_gen.generate_full_spec(module, properties)

    output = args.output or (Path(args.input).stem + ".sva")
    sva_gen.to_file(spec, output)
    print(f"SVA spec written to: {output}")
    print(f"  Module: {module.name}")
    print(f"  Ports:  {len(module.ports)}")
    print(f"  Props:  {len(properties)}")


def cmd_analyze(args):
    """Analyze a Verilog file and print property summary."""
    parser = VerilogParser()
    module = parser.parse_file(args.input)

    prop_gen = PropertyGenerator()
    properties = prop_gen.generate_for_module(module)

    has_seq = any(ab.block_type == "sequential" for ab in module.always_blocks)

    print(f"Module: {module.name}")
    print(f"  Logic type: {'Sequential' if has_seq else 'Combinational'}")
    print(f"  Ports ({len(module.ports)}):")
    for p in module.ports:
        w = f"[{p.width-1}:0] " if p.width > 1 else ""
        reg = " reg" if p.is_reg else ""
        print(f"    {p.direction}{reg} {w}{p.name}")
    if module.parameters:
        print(f"  Parameters: {module.parameters}")
    print(f"\n  Properties ({len(properties)}):")
    for prop in properties:
        conf = f"{int(prop.confidence*100)}%"
        print(f"    [{conf}] {prop.name} ({prop.prop_type})")
        print(f"         {prop.description[:70]}")


def cmd_report(args):
    """Generate a full formal report for a Verilog file."""
    parser = VerilogParser()
    module = parser.parse_file(args.input)

    prop_gen = PropertyGenerator()
    properties = prop_gen.generate_for_module(module)

    sva_gen = SVAGenerator()
    spec = sva_gen.generate_full_spec(module, properties)

    refiner = SelfRefiner()
    steps = refiner.refine(spec, module)
    refined_spec = steps[-1].refined_spec if steps else spec

    report = FormalReport(module, properties, refined_spec, steps)

    fmt = args.format or "json"
    output = args.output or f"report.{fmt}"
    report.save(output, format=fmt)
    print(f"Report written to: {output}")


def main():
    top = argparse.ArgumentParser(
        prog="formal-circuits",
        description="Verilog/VHDL to formal proof converter",
    )
    sub = top.add_subparsers(dest="command", required=True)

    # convert
    p_conv = sub.add_parser("convert", help="Convert Verilog to SVA spec")
    p_conv.add_argument("--input", "-i", required=True, help="Input .v file")
    p_conv.add_argument("--output", "-o", help="Output .sva file (default: <input>.sva)")
    p_conv.set_defaults(func=cmd_convert)

    # analyze
    p_ana = sub.add_parser("analyze", help="Analyze Verilog and show properties")
    p_ana.add_argument("--input", "-i", required=True, help="Input .v file")
    p_ana.set_defaults(func=cmd_analyze)

    # report
    p_rep = sub.add_parser("report", help="Generate full formal report")
    p_rep.add_argument("--input", "-i", required=True, help="Input .v file")
    p_rep.add_argument("--output", "-o", help="Output file (default: report.<format>)")
    p_rep.add_argument("--format", "-f", choices=["json", "markdown", "md"], default="json",
                       help="Output format (default: json)")
    p_rep.set_defaults(func=cmd_report)

    args = top.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
