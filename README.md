# formal-circuits-gpt

A self-refining agent for converting Verilog/VHDL to formal proofs (SystemVerilog Assertions).

## Features

- **Verilog Parser**: Regex-based extraction of module name, ports, always blocks, and parameters
- **Property Generator**: Automatic generation of combinational, sequential, safety, and liveness properties
- **SVA Generator**: Output as SystemVerilog Assertions with full spec files
- **Self-Refiner**: Iterative rule-based refinement of formal specs
- **Report Generator**: JSON and Markdown reports with plain English property descriptions
- **CLI**: `formal-circuits convert/analyze/report`

## Usage

```bash
# Install
pip install -e .

# Convert Verilog to SVA spec
formal-circuits convert --input examples/counter.v --output counter.sva

# Analyze a module
formal-circuits analyze --input examples/simple_adder.v

# Generate full report
formal-circuits report --input examples/counter.v --output report.md --format markdown
```

## Structure

```
src/formal_circuits/
    verilog_parser.py   # Regex-based Verilog parser
    property_gen.py     # Formal property generator
    sva_generator.py    # SystemVerilog Assertion generator
    refiner.py          # Self-refining spec improver
    report.py           # Report generation (JSON/Markdown)
    cli.py              # CLI entry point
examples/
    simple_adder.v      # 8-bit adder
    counter.v           # Synchronous counter with enable/reset
    mux.v               # 4-to-1 multiplexer
tests/                  # 10+ passing tests
```

## Running Tests

```bash
pytest tests/ -v
```
