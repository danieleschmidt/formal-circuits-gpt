# User Guide: Formal-Circuits-GPT

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Getting Started

### Prerequisites
- Python 3.9 or higher
- One or more theorem provers (Isabelle, Coq)
- LLM API access (OpenAI, Anthropic, or local model)

### Installation

#### Quick Install
```bash
pip install formal-circuits-gpt
```

#### Development Install
```bash
git clone https://github.com/terragonlabs/formal-circuits-gpt
cd formal-circuits-gpt
pip install -e .[dev]
```

#### Theorem Prover Setup

**Isabelle/HOL**
```bash
# Ubuntu/Debian
sudo apt-get install isabelle

# macOS
brew install isabelle

# Manual installation
wget https://isabelle.in.tum.de/dist/Isabelle2024.tar.gz
tar -xzf Isabelle2024.tar.gz
export PATH=$PATH:$(pwd)/Isabelle2024/bin
```

**Coq**
```bash
# Ubuntu/Debian
sudo apt-get install coq

# macOS
brew install coq

# Using opam
opam install coq
```

### Initial Setup
```bash
# Verify installation
formal-circuits-gpt --check-setup

# Configure API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Test basic functionality
formal-circuits-gpt verify examples/simple_adder.v
```

## Basic Usage

### Command Line Interface

#### Basic Verification
```bash
# Verify a single Verilog file
formal-circuits-gpt verify design.v

# Verify with specific prover
formal-circuits-gpt verify design.v --prover isabelle

# Verify with custom properties
formal-circuits-gpt verify design.v --properties properties.yaml
```

#### Common Options
```bash
# Set timeout (in seconds)
formal-circuits-gpt verify design.v --timeout 1800

# Verbose output
formal-circuits-gpt verify design.v --verbose

# Save results to file
formal-circuits-gpt verify design.v --output results.json

# Use specific LLM model
formal-circuits-gpt verify design.v --model gpt-4-turbo
```

### Python API

#### Basic Example
```python
from formal_circuits_gpt import CircuitVerifier

# Initialize verifier
verifier = CircuitVerifier(
    prover="isabelle",
    model="gpt-4-turbo"
)

# Verify HDL code
verilog_code = """
module adder(
    input [3:0] a,
    input [3:0] b,  
    output [4:0] sum
);
    assign sum = a + b;
endmodule
"""

result = verifier.verify(verilog_code)
print(f"Status: {result.status}")
print(f"Proof: {result.proof_text}")
```

#### Working with Files
```python
# Verify from file
result = verifier.verify_file("designs/counter.v")

# Verify multiple files
results = verifier.verify_directory("designs/")

# Save results
result.save("verification_results.json")
result.export_proof("proof.thy")  # Isabelle format
```

### Property Specification

#### Using Built-in Properties
```python
from formal_circuits_gpt.properties import CommonProperties

# Arithmetic properties
properties = CommonProperties.arithmetic(
    overflow_check=True,
    associativity=True,
    commutativity=True
)

result = verifier.verify(code, properties=properties)
```

#### Custom Properties
```yaml
# properties.yaml
properties:
  - name: "no_overflow"
    formula: "∀a b. a + b < 2^n → no_overflow_flag"
    description: "Addition should not overflow for valid inputs"
    
  - name: "deterministic"
    formula: "∀inputs. same_inputs → same_outputs"
    description: "Circuit should be deterministic"
```

```python
# Load from file
result = verifier.verify_file("design.v", properties="properties.yaml")
```

## Advanced Features

### Self-Refinement Configuration
```python
verifier = CircuitVerifier(
    prover="coq",
    refinement_strategy="chain_of_thought",
    max_refinement_rounds=10,
    use_counterexamples=True
)
```

### Parallel Verification
```python
from formal_circuits_gpt import ParallelVerifier

verifier = ParallelVerifier(
    num_workers=4,
    shared_cache=True
)

results = verifier.verify_directory("large_project/")
```

### Performance Optimization

#### Caching
```python
verifier = CircuitVerifier(
    cache_dir=".verification_cache",
    reuse_lemmas=True,
    incremental_mode=True
)
```

#### Timeouts and Limits
```python
verifier = CircuitVerifier(
    timeout=3600,  # 1 hour total timeout
    proof_timeout=600,  # 10 min per proof attempt
    max_proof_size=10000  # Max lines in proof
)
```

## Configuration

### Configuration File
Create `~/.formal-circuits-gpt/config.yaml`:

```yaml
# Default settings
default:
  prover: "isabelle"
  model: "gpt-4-turbo"
  timeout: 1800
  
# LLM providers
llm:
  openai:
    model: "gpt-4-turbo"
    temperature: 0.1
    max_tokens: 4000
  anthropic:
    model: "claude-3-sonnet"
    temperature: 0.1
    max_tokens: 4000

# Theorem provers
provers:
  isabelle:
    path: "/usr/local/bin/isabelle"
    timeout: 600
  coq:
    path: "/usr/bin/coq"
    timeout: 600

# Refinement settings
refinement:
  max_rounds: 15
  strategy: "chain_of_thought"
  use_counterexamples: true

# Caching
cache:
  enabled: true
  directory: "~/.formal-circuits-gpt/cache"
  max_size: "1GB"
```

### Environment Variables
```bash
# Required
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional
export FORMAL_CIRCUITS_CONFIG="custom-config.yaml"
export FORMAL_CIRCUITS_CACHE_DIR="/tmp/verification-cache"
export FORMAL_CIRCUITS_LOG_LEVEL="INFO"
```

## Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Missing theorem prover
Error: Isabelle not found in PATH
Solution: Install Isabelle and add to PATH

# Python version issues
Error: formal-circuits-gpt requires Python 3.9+
Solution: Upgrade Python or use virtual environment
```

#### API Issues
```bash
# Invalid API key
Error: OpenAI API authentication failed
Solution: Check OPENAI_API_KEY environment variable

# Rate limiting
Error: API rate limit exceeded
Solution: Add delays or use different provider
```

#### Verification Failures
```python
# Enable debugging
verifier = CircuitVerifier(debug_mode=True)

# Analyze failures
try:
    result = verifier.verify(code)
except ProofFailure as e:
    print(f"Failed at: {e.failed_goal}")
    print(f"Error: {e.error_message}")
    print(f"Suggestion: {e.suggested_fix}")
```

### Debug Mode
```bash
# Enable verbose logging
formal-circuits-gpt verify design.v --debug

# Save intermediate files
formal-circuits-gpt verify design.v --save-intermediate
```

### Getting Help
```bash
# Command help
formal-circuits-gpt --help
formal-circuits-gpt verify --help

# Check system status
formal-circuits-gpt --check-setup

# Version information
formal-circuits-gpt --version
```

## Best Practices

### Circuit Design
1. **Use clear naming**: Descriptive signal and module names
2. **Add comments**: Document complex logic
3. **Modular design**: Break large circuits into smaller modules
4. **Avoid complex expressions**: Simplify for better verification

### Property Specification
1. **Start simple**: Begin with basic properties
2. **Be specific**: Precise property definitions work better
3. **Use templates**: Leverage built-in property templates
4. **Test incrementally**: Verify properties one at a time

### Performance Optimization
1. **Enable caching**: Reuse verification results
2. **Use timeouts**: Prevent infinite verification attempts
3. **Parallel processing**: Verify multiple modules simultaneously
4. **Monitor resources**: Track LLM API usage and costs

### Debugging
1. **Enable debug mode**: Get detailed verification logs
2. **Save intermediate files**: Examine generated proofs
3. **Start with known-good circuits**: Verify your setup works
4. **Use smaller examples**: Debug with simple test cases

### Production Usage
1. **Pin versions**: Use specific version numbers in production
2. **Set timeouts**: Prevent runaway verification jobs
3. **Monitor costs**: Track LLM API usage
4. **Backup results**: Save verification artifacts
5. **Test thoroughly**: Validate on known circuits first

---

For more detailed information, see:
- [Developer Guide](developer-guide.md)
- [API Reference](../api/README.md)
- [Examples](../../examples/README.md)
- [FAQ](faq.md)