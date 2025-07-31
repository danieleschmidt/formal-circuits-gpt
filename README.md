# Formal-Circuits-GPT

Self-refining agent that converts hardware descriptions (Verilog/VHDL) into formal proofs using Large Language Models and deductive search. Based on MIT's "LLM-Assisted Circuit Proofs" research (May 2025), this tool automates the verification of digital circuits using Isabelle and Coq theorem provers.

## Overview

Formal-Circuits-GPT bridges the gap between hardware design and formal verification by automatically generating and refining mathematical proofs of circuit correctness. The system uses LLMs to understand circuit behavior, generate proof obligations, and iteratively refine proofs through interaction with theorem provers.

## Features

- **Multi-HDL Support**: Parses both Verilog and VHDL designs
- **Dual Prover Backend**: Generates proofs for both Isabelle/HOL and Coq
- **Self-Refinement Loop**: Automatically fixes proof errors through LLM-guided search
- **Property Synthesis**: Infers likely correctness properties from circuit structure
- **Incremental Verification**: Supports modular proof composition
- **Counter-example Analysis**: Learns from failed proof attempts

## Installation

```bash
# Install the Python package
pip install formal-circuits-gpt

# Install theorem provers (required)
# For Ubuntu/Debian:
sudo apt-get install isabelle
sudo apt-get install coq

# For macOS:
brew install isabelle
brew install coq

# Verify installation
formal-circuits-gpt --check-setup
```

## Quick Start

### Basic Usage

```python
from formal_circuits_gpt import CircuitVerifier

# Initialize verifier
verifier = CircuitVerifier(
    prover="isabelle",  # or "coq"
    model="gpt-4-turbo",
    temperature=0.1
)

# Verify a simple Verilog module
verilog_code = """
module adder(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
"""

# Generate and verify proof
proof = verifier.verify(
    hdl_code=verilog_code,
    properties=["sum == a + b", "sum < 32"]
)

print(proof.status)  # "VERIFIED" or "FAILED"
print(proof.isabelle_code)  # Generated Isabelle proof
```

### Advanced Example with Custom Properties

```python
from formal_circuits_gpt import CircuitVerifier, PropertySpec

# Define custom properties
properties = [
    PropertySpec(
        name="associativity",
        formula="∀a b c. add(add(a,b),c) = add(a,add(b,c))",
        proof_strategy="induction"
    ),
    PropertySpec(
        name="overflow_detection",
        formula="∀a b. (a + b ≥ 2^n) ↔ overflow_flag"
    )
]

# Verify complex arithmetic unit
verifier = CircuitVerifier(prover="coq", refinement_rounds=10)

proof = verifier.verify_file(
    "designs/arithmetic_unit.v",
    properties=properties,
    timeout=3600  # 1 hour timeout
)

# Export proof for paper/documentation
proof.export_latex("proofs/arithmetic_unit.tex")
```

## Architecture

```
formal-circuits-gpt/
├── formal_circuits_gpt/
│   ├── parsers/           # HDL parsing
│   │   ├── verilog/       # Verilog/SystemVerilog parser
│   │   └── vhdl/          # VHDL parser
│   ├── translators/       # HDL to formal language
│   │   ├── to_isabelle/   # Isabelle/HOL generation
│   │   └── to_coq/        # Coq generation
│   ├── llm/              # LLM integration
│   │   ├── prompts/      # Proof generation prompts
│   │   └── refinement/   # Error correction strategies
│   ├── provers/          # Theorem prover interfaces
│   ├── properties/       # Property inference/checking
│   └── search/           # Deductive search algorithms
├── benchmarks/           # Standard verification benchmarks
├── examples/            # Example circuits and proofs
└── tests/              # Test suite
```

## Workflow

1. **Parse**: Extract circuit structure from Verilog/VHDL
2. **Translate**: Convert to formal representation
3. **Infer**: Generate likely correctness properties
4. **Prove**: Attempt proof with theorem prover
5. **Refine**: Use LLM to fix errors and retry
6. **Verify**: Check final proof validity

## Supported Circuit Types

### Combinational Logic
- Arithmetic units (adders, multipliers, dividers)
- Encoders/decoders
- Multiplexers
- Parity generators
- Boolean function implementations

### Sequential Logic
- Finite state machines
- Counters
- Shift registers
- Memory controllers
- Clock domain crossings

### Parameterized Designs
- Generic/parameterized modules
- Generate blocks
- Recursive structures

## Property Specification

### Built-in Property Templates

```python
from formal_circuits_gpt.properties import CommonProperties

# Arithmetic properties
properties = CommonProperties.arithmetic(
    overflow_check=True,
    associativity=True,
    commutativity=False  # For non-commutative operations
)

# FSM properties
properties = CommonProperties.fsm(
    deadlock_free=True,
    reachability=["IDLE", "DONE"],
    mutual_exclusion=[("STATE_A", "STATE_B")]
)
```

### Custom Property Language

```python
# Define properties using our DSL
property_spec = """
property no_deadlock:
    always_eventually (state == IDLE);

property data_integrity:
    forall addr data.
        write_enable && write_addr == addr && write_data == data
        implies
        next(read_enable && read_addr == addr -> read_data == data);

property timing_constraint:
    rise(clock) -> next[3](data_valid);
"""

verifier.verify(hdl_code, properties=property_spec)
```

## LLM-Guided Refinement

The system uses advanced prompting strategies to refine failed proofs:

```python
# Configure refinement behavior
verifier = CircuitVerifier(
    refinement_strategy="chain_of_thought",
    max_refinement_rounds=20,
    use_counterexamples=True,
    proof_search_depth=5
)

# The system will automatically:
# 1. Analyze proof failures
# 2. Generate hypotheses about fixes
# 3. Test modifications
# 4. Learn from successful patterns
```

## Integration with Existing Workflows

### CI/CD Integration

```yaml
# .github/workflows/formal-verification.yml
name: Formal Verification
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Formal-Circuits-GPT
        run: pip install formal-circuits-gpt
      - name: Verify Circuits
        run: |
          formal-circuits-gpt verify \
            --input src/hdl/*.v \
            --properties specs/*.prop \
            --prover isabelle \
            --output reports/
```

### EDA Tool Integration

```python
# Export to standard formats
proof.export_systemverilog_assertions("assertions.sv")
proof.export_psl("properties.psl")
proof.export_smtlib("constraints.smt2")
```

## Performance Optimization

### Parallel Verification

```python
from formal_circuits_gpt import ParallelVerifier

# Verify multiple modules in parallel
verifier = ParallelVerifier(
    num_workers=8,
    prover="coq",
    shared_lemma_cache=True
)

results = verifier.verify_directory(
    "designs/",
    property_file="specs/global_properties.yaml"
)
```

### Proof Caching

```python
# Enable proof caching for incremental verification
verifier = CircuitVerifier(
    cache_dir=".proof_cache/",
    reuse_lemmas=True,
    incremental_mode=True
)
```

## Benchmarks

Run standard benchmarks:

```bash
# Run AIGER benchmark suite
formal-circuits-gpt benchmark --suite aiger

# Run custom benchmark
formal-circuits-gpt benchmark --circuits my_designs/ --properties my_specs/

# Compare with baseline tools
formal-circuits-gpt benchmark --compare-with abc,nusmv
```

## Debugging Failed Proofs

```python
# Enable detailed debugging
verifier = CircuitVerifier(debug_mode=True)

try:
    proof = verifier.verify(hdl_code, properties)
except ProofFailure as e:
    # Analyze failure
    print(f"Proof failed at: {e.failed_goal}")
    print(f"Counterexample: {e.counterexample}")
    
    # Get LLM's analysis
    analysis = verifier.analyze_failure(e)
    print(f"Suggested fix: {analysis.suggestion}")
    
    # Try alternative strategies
    proof = verifier.verify(
        hdl_code,
        properties,
        alternative_strategy="bounded_model_checking"
    )
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Prover Backends

```python
from formal_circuits_gpt.provers import BaseProver

class MyProver(BaseProver):
    def translate(self, circuit_ast):
        # Convert to prover's language
        pass
    
    def prove(self, goal):
        # Attempt proof
        pass
    
    def parse_result(self, output):
        # Parse prover output
        pass
```

## Citation

```bibtex
@software{formal_circuits_gpt,
  title={Formal-Circuits-GPT: LLM-Assisted Hardware Verification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/formal-circuits-gpt}
}

@article{mit_llm_circuits_2025,
  title={LLM-Assisted Circuit Proofs},
  author={MIT Research Group},
  journal={Formal Methods in System Design},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- MIT CSAIL for the foundational LLM-assisted proof research
- The Isabelle and Coq communities
- Contributors to hardware verification tools
