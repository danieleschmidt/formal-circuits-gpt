# Getting Started with Formal-Circuits-GPT

Welcome to Formal-Circuits-GPT! This guide will help you get up and running with automated formal verification of digital circuits.

## Prerequisites

### Required Knowledge
- Basic understanding of digital circuits (combinational and sequential logic)
- Familiarity with Verilog or VHDL
- Basic command-line experience
- Optional: Knowledge of formal verification concepts

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows (with WSL2)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for LLM API access

## Installation

### Method 1: Using pip (Recommended)

```bash
# Install the package
pip install formal-circuits-gpt

# Verify installation
formal-circuits-gpt --version
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/formal-circuits-gpt.git
cd formal-circuits-gpt

# Install in development mode
pip install -e .

# Verify installation
python -m formal_circuits_gpt --version
```

### Method 3: Using Docker

```bash
# Pull the Docker image
docker pull formal-circuits-gpt:latest

# Run with Docker
docker run -it formal-circuits-gpt:latest --help
```

## Initial Setup

### 1. Install Theorem Provers

#### Isabelle/HOL (Recommended for beginners)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install isabelle

# macOS
brew install isabelle

# Verify installation
isabelle version
```

#### Coq (Advanced users)
```bash
# Ubuntu/Debian
sudo apt-get install coq

# macOS
brew install coq

# Verify installation
coq --version
```

### 2. Configure LLM Access

Create a configuration file at `~/.formal-circuits-gpt/config.yaml`:

```yaml
llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4-turbo"
  api_key_env: "OPENAI_API_KEY"  # Environment variable name
  
theorem_provers:
  isabelle:
    path: "/usr/bin/isabelle"
  coq:
    path: "/usr/bin/coq"
```

Set your API key as an environment variable:
```bash
# Add to your ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="your-api-key-here"

# Reload your shell or run:
source ~/.bashrc
```

### 3. Verify Setup

```bash
# Check that everything is working
formal-circuits-gpt --check-setup
```

Expected output:
```
✅ Python environment: OK
✅ Isabelle/HOL: Found at /usr/bin/isabelle
✅ LLM API: OpenAI GPT-4 accessible
✅ Configuration: Valid
🎉 Setup complete! Ready to verify circuits.
```

## Your First Verification

Let's verify a simple 4-bit adder circuit to make sure everything works.

### 1. Create a Simple Circuit

Create a file named `simple_adder.v`:

```verilog
module simple_adder(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
```

### 2. Run Basic Verification

```bash
# Verify the circuit with automatic property inference
formal-circuits-gpt verify simple_adder.v

# Or with explicit properties
formal-circuits-gpt verify simple_adder.v \
  --property "sum == a + b" \
  --property "sum <= 30"
```

### 3. Understanding the Output

```
🔍 Parsing Verilog...
✅ Successfully parsed simple_adder.v

🧠 Generating properties...
✅ Inferred 3 properties: arithmetic correctness, overflow bounds, bit width

🤖 Generating formal proof...
✅ Initial proof generated (142 lines)

🔬 Checking with Isabelle...
✅ Proof verified successfully!

📊 Results:
  - Circuit: simple_adder (4-bit adder)
  - Properties verified: 3/3
  - Proof size: 142 lines
  - Verification time: 23.4 seconds
  - Prover: Isabelle/HOL

💾 Proof saved to: proofs/simple_adder.thy
```

Congratulations! You've successfully verified your first circuit.

## Common Usage Patterns

### Verifying Combinational Logic

```bash
# Verify all Verilog files in a directory
formal-circuits-gpt verify designs/*.v --output-dir proofs/

# Verify with specific properties
formal-circuits-gpt verify multiplier.v \
  --property "product == a * b" \
  --property "product < 256"

# Use Coq instead of Isabelle
formal-circuits-gpt verify encoder.v --prover coq
```

### Batch Processing

```bash
# Process multiple circuits with custom configuration
formal-circuits-gpt batch \
  --input designs/ \
  --properties specs/properties.yaml \
  --output reports/ \
  --parallel 4
```

### Working with VHDL

```bash
# VHDL support (experimental)
formal-circuits-gpt verify counter.vhdl --prover coq
```

## Understanding Verification Results

### Success Cases

When verification succeeds, you'll get:
- ✅ **Proof file**: Mathematical proof in Isabelle/Coq format
- 📊 **Verification report**: Summary of properties and results
- 🎯 **Property analysis**: Which properties were verified and how

### Failure Cases

When verification fails, the tool will:
- 🔍 **Analyze the failure**: Identify why the proof failed
- 🤖 **Attempt refinement**: Try to fix the proof automatically
- 📝 **Provide feedback**: Suggest possible issues or fixes

Example failure output:
```
❌ Verification failed for property: "sum < 16"

🔍 Analysis:
  - Property violated for inputs: a=15, b=15
  - Expected: sum < 16, Actual: sum = 30
  - Suggestion: Consider overflow behavior

🤖 Attempting refinement...
✅ Refined property: "sum <= a + b"
✅ Verification successful with refined property!
```

## Best Practices

### Writing Verifiable Circuits

1. **Keep modules focused**: Smaller modules are easier to verify
2. **Use meaningful names**: Clear signal names help property inference
3. **Avoid complex expressions**: Break down complex logic into steps
4. **Document assumptions**: Use comments to specify expected behavior

### Property Specification

1. **Start simple**: Begin with basic properties like "output equals expected function"
2. **Consider edge cases**: Think about corner cases and boundary conditions
3. **Use templates**: Leverage built-in property templates for common patterns
4. **Be specific**: Precise properties are easier to verify than vague ones

### Performance Optimization

1. **Use appropriate provers**: Isabelle for beginners, Coq for advanced users
2. **Enable caching**: Reuse proofs for similar circuits
3. **Parallel processing**: Use `--parallel` for batch verification
4. **Incremental verification**: Verify modules individually before combining

## Troubleshooting

### Common Issues

#### "LLM API Error"
- **Cause**: Invalid API key or network issues
- **Solution**: Check your API key and internet connection
```bash
# Test API access
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### "Theorem Prover Not Found"
- **Cause**: Prover not installed or not in PATH
- **Solution**: Install the prover and update your configuration
```bash
# Check if Isabelle is in PATH
which isabelle

# If not found, install it
sudo apt-get install isabelle
```

#### "Parsing Error"
- **Cause**: Unsupported Verilog/VHDL features
- **Solution**: Check the supported features list or simplify the circuit

#### "Verification Timeout"
- **Cause**: Circuit too complex or properties too difficult
- **Solution**: Increase timeout or break down the verification
```bash
# Increase timeout to 10 minutes
formal-circuits-gpt verify complex.v --timeout 600
```

### Getting Help

1. **Documentation**: Check the full documentation at [docs/](../README.md)
2. **Examples**: Look at [examples/](../../examples/) for sample circuits
3. **Community**: Join our [Discord server](https://discord.gg/formal-circuits-gpt)
4. **Issues**: Report bugs on [GitHub Issues](https://github.com/danieleschmidt/formal-circuits-gpt/issues)

## Next Steps

Now that you have the basics working, you can:

1. **Explore examples**: Check out the [examples directory](../../examples/) for more complex circuits
2. **Learn about properties**: Read the [Property Specification Guide](property-specification.md)
3. **Advanced features**: Explore [Advanced Usage](advanced-usage.md) for power user features
4. **Integration**: Set up [CI/CD Integration](../workflows/ci-cd-requirements.md) for your projects
5. **Contribute**: Help improve the tool by contributing to our [open source project](../../CONTRIBUTING.md)

Happy verifying! 🎉