# Property Specification Guide

This guide covers how to specify and work with properties in Formal-Circuits-GPT for effective circuit verification.

## Overview

Properties are mathematical assertions about circuit behavior that we want to verify. They form the foundation of formal verification by precisely defining what "correctness" means for your circuit.

## Property Types

### Safety Properties
**Definition**: "Something bad never happens"

Examples:
- Array bounds are never exceeded
- Signals never take invalid values
- State machines never enter invalid states

```python
# Array access safety
property array_bounds_safe:
    always (array_index < ARRAY_SIZE)

# Signal value constraints
property valid_state:
    always (state in {IDLE, ACTIVE, DONE})
```

### Liveness Properties
**Definition**: "Something good eventually happens"

Examples:
- Requests are eventually processed
- State machines eventually return to idle
- Outputs eventually become stable

```python
# Request processing
property request_processed:
    always (request_valid -> eventually response_valid)

# Return to idle
property eventually_idle:
    always (eventually (state == IDLE))
```

### Invariant Properties
**Definition**: Properties that must always hold

Examples:
- Conservation laws (what goes in equals what comes out)
- Mutual exclusion (two things can't be true simultaneously)
- Resource constraints

```python
# Conservation
property conservation:
    always (input_count == output_count + internal_count)

# Mutual exclusion
property mutex:
    always (!(enable_a && enable_b))
```

## Property Specification Methods

### 1. Command Line Properties

Simple properties can be specified directly on the command line:

```bash
# Single property
formal-circuits-gpt verify adder.v --property "sum == a + b"

# Multiple properties
formal-circuits-gpt verify counter.v \
  --property "count <= MAX_COUNT" \
  --property "count >= 0"

# Using property expressions
formal-circuits-gpt verify fsm.v \
  --property "state == IDLE -> next_state in {IDLE, START}"
```

### 2. Property Files

For complex properties, use YAML property files:

```yaml
# properties.yaml
circuit: arithmetic_unit
properties:
  - name: addition_correctness
    type: safety
    description: "Addition operation produces correct result"
    formula: "mode == ADD -> result == a + b"
    
  - name: overflow_detection
    type: safety
    description: "Overflow flag is set correctly"
    formula: "(a + b) > MAX_VALUE <-> overflow_flag"
    
  - name: eventually_ready
    type: liveness
    description: "Unit eventually becomes ready"
    formula: "always (eventually ready)"

templates:
  - arithmetic
  - finite_state_machine
```

Usage:
```bash
formal-circuits-gpt verify unit.v --properties properties.yaml
```

### 3. Built-in Property Templates

Use pre-defined templates for common circuit patterns:

```python
from formal_circuits_gpt.properties import CommonProperties

# Arithmetic unit properties
props = CommonProperties.arithmetic(
    overflow_check=True,
    associativity=True,
    commutativity=True,
    identity_element=0
)

# Finite state machine properties
props = CommonProperties.fsm(
    deadlock_free=True,
    reachability=["IDLE", "DONE"],
    mutual_exclusion=[("STATE_A", "STATE_B")]
)

# Memory properties
props = CommonProperties.memory(
    data_integrity=True,
    address_bounds=True,
    write_read_consistency=True
)
```

### 4. Custom Property DSL

For advanced users, we provide a domain-specific language:

```python
property_spec = """
// Arithmetic properties
property addition_commutative:
    forall a, b: int.
        add(a, b) == add(b, a);

property addition_associative:
    forall a, b, c: int.
        add(add(a, b), c) == add(a, add(b, c));

// Temporal properties
property request_response:
    always (req_valid && ready -> 
        next[1..10](resp_valid));

property no_deadlock:
    always (eventually (state == IDLE));

// Data integrity
property write_read_consistency:
    forall addr, data: bitvec[32].
        write_enable && write_addr == addr && write_data == data
        implies
        next(read_enable && read_addr == addr -> read_data == data);
"""
```

## Property Categories by Circuit Type

### Combinational Logic

#### Arithmetic Circuits
```python
# Basic arithmetic correctness
properties = [
    "result == a + b",  # Adder
    "product == a * b", # Multiplier
    "quotient * divisor + remainder == dividend", # Divider
    "remainder < divisor" # Division remainder constraint
]

# Overflow and underflow
properties.extend([
    "(a + b) > MAX_VALUE -> overflow",
    "(a - b) < 0 -> underflow"
])

# Bit width constraints
properties.extend([
    "result[WIDTH-1:0] == result", # No extra bits
    "carry_out == (a + b)[WIDTH]"  # Carry generation
])
```

#### Encoders/Decoders
```python
# Encoder properties
properties = [
    "input[i] -> output == i",  # Encoding correctness
    "count_ones(input) <= 1",   # At most one input active
    "input == 0 -> output == 0" # Zero input handling
]

# Decoder properties  
properties = [
    "enable && (input == i) -> output[i]", # Decoding correctness
    "!enable -> output == 0",              # Enable control
    "count_ones(output) <= 1"              # At most one output
]
```

#### Multiplexers
```python
properties = [
    "sel == 0 -> out == in0",
    "sel == 1 -> out == in1",
    "forall i. sel == i -> out == in[i]" # General case
]
```

### Sequential Logic

#### Counters
```python
properties = [
    # Basic counting
    "count_up && count < MAX -> next(count) == count + 1",
    "count_down && count > 0 -> next(count) == count - 1",
    
    # Overflow behavior
    "count_up && count == MAX -> next(count) == 0", # Wrap around
    "count_down && count == 0 -> next(count) == MAX",
    
    # Control signals
    "reset -> next(count) == 0",
    "!enable -> next(count) == count" # Hold value
]
```

#### Finite State Machines
```python
properties = [
    # State validity
    "state in {IDLE, START, PROCESS, DONE}",
    
    # Transition constraints
    "state == IDLE && start -> next(state) == START",
    "state == PROCESS && done -> next(state) == DONE",
    
    # Liveness
    "always (eventually (state == IDLE))", # Return to idle
    "start -> eventually (state == DONE)", # Progress guarantee
    
    # Safety
    "state == PROCESS -> !start" # No restart during processing
]
```

#### Shift Registers
```python
properties = [
    # Shift operation
    "shift_left -> next(reg) == {reg[WIDTH-2:0], shift_in}",
    "shift_right -> next(reg) == {shift_in, reg[WIDTH-1:1]}",
    
    # Parallel load
    "load -> next(reg) == data_in",
    
    # Hold
    "!(shift_left || shift_right || load) -> next(reg) == reg"
]
```

## Advanced Property Patterns

### Temporal Logic Properties

#### Eventually Patterns
```python
# Something eventually happens
"eventually (ready)"
"eventually (state == DONE)"
"eventually (buffer_empty)"

# Eventually with conditions
"req_valid -> eventually (resp_valid)"
"start -> eventually (finished)"
```

#### Always Eventually (Liveness)
```python
# Periodic behavior
"always (eventually (clock_tick))"
"always (eventually (refresh_cycle))"

# Progress guarantees
"always (req_pending -> eventually (req_processed))"
```

#### Until Patterns
```python
# State transitions
"(state == WAIT) until (ready)"
"(collecting_data) until (buffer_full)"

# Resource allocation
"(resource_requested) until (resource_granted)"
```

### Quantified Properties

#### Universal Quantification
```python
# For all inputs
"forall x: int[8]. f(x) >= 0"
"forall addr: addr_t. valid_addr(addr) -> accessible(addr)"

# For all time
"forall t: time. P(t) -> eventually Q(t)"
```

#### Existential Quantification  
```python
# There exists a solution
"exists x: int. solve(x)"
"exists path: path_t. reachable(start, end, path)"
```

### Parametric Properties

For parameterized circuits:

```python
# Width-parameterized adder
template_properties = """
parameter WIDTH: int > 0;

property addition_correct:
    forall a, b: bitvec[WIDTH].
        let sum = add(a, b) in
        sum[WIDTH-1:0] == (a + b)[WIDTH-1:0];

property overflow_flag:
    forall a, b: bitvec[WIDTH].
        let sum = add(a, b) in
        overflow <-> (a + b) >= 2^WIDTH;
"""
```

## Property Verification Strategies

### Compositional Verification

Break complex properties into simpler ones:

```python
# Instead of one complex property
"complex_circuit_correct(inputs) == expected_output(inputs)"

# Use multiple simpler properties
properties = [
    "stage1_correct(inputs) == stage1_expected(inputs)",
    "stage2_correct(stage1_out) == stage2_expected(stage1_out)", 
    "stage3_correct(stage2_out) == expected_output(inputs)"
]
```

### Assume-Guarantee Reasoning

Specify assumptions and guarantees:

```python
# Module interface specification
module_spec = """
assumptions:
    - input_valid -> input in valid_range
    - clock frequency >= 100MHz
    - reset properly synchronized

guarantees:
    - output_valid -> output correct
    - response within 10 clock cycles
    - no internal state corruption
"""
```

### Bounded Verification

For performance, use bounded properties:

```python
properties = [
    # Bounded liveness (within N cycles)
    "req_valid -> next[1..10](resp_valid)",
    
    # Bounded safety (for first N cycles)  
    "bounded[100] (always (count <= MAX_COUNT))",
    
    # Inductive properties
    "inductive (P(0) && (P(n) -> P(n+1)))"
]
```

## Property Debugging

### Understanding Failures

When properties fail, the tool provides debugging information:

```
❌ Property failed: "sum == a + b"

🔍 Counterexample:
  Inputs: a=15, b=3, carry_in=1
  Expected: sum=18
  Actual: sum=19
  
🤖 Analysis:
  - Carry input not accounted for in property
  - Suggested fix: "sum == a + b + carry_in"
```

### Property Refinement

Iteratively improve properties:

```python
# Initial (too strict)
"output_valid -> output == expected"

# Refined (account for delays)  
"output_valid -> next[0..2](output == expected)"

# Further refined (account for reset)
"!reset && output_valid -> next[0..2](output == expected)"
```

### Coverage Analysis

Check property coverage:

```bash
# Generate coverage report
formal-circuits-gpt coverage circuit.v properties.yaml

# Output shows which circuit behaviors are covered
✅ All input combinations covered
⚠️  Reset behavior not fully covered  
❌ Error handling paths not covered
```

## Best Practices

### 1. Start Simple
- Begin with basic functional correctness
- Add edge cases and corner conditions
- Gradually increase complexity

### 2. Use Appropriate Abstraction
- Don't over-specify implementation details
- Focus on interface behavior
- Use appropriate levels of abstraction

### 3. Make Properties Readable
- Use descriptive names
- Add documentation
- Group related properties

### 4. Test Your Properties
- Verify properties make sense
- Check with known good/bad examples
- Use property simulation when available

### 5. Maintain Property Suites
- Version control your properties
- Keep properties up-to-date with circuit changes
- Reuse properties across similar circuits

## Common Pitfalls

### 1. Over-Specification
```python
# BAD: Too implementation-specific
"internal_counter == 42 && state == PROCESS_STAGE_3"

# GOOD: Interface-level specification  
"processing -> eventually (done)"
```

### 2. Under-Specification
```python
# BAD: Too vague
"output reasonable"

# GOOD: Precise specification
"0 <= output && output <= MAX_VALUE"
```

### 3. Timing Issues
```python
# BAD: Immediate expectation
"request -> response"

# GOOD: Proper timing
"request -> next[1..5](response)"
```

### 4. Reset Handling
```python
# BAD: Ignores reset
"always (counter_increasing)"

# GOOD: Accounts for reset
"!reset -> (counter_increasing || counter_max)"
```

## Integration with Development Workflow

### CI/CD Integration

```yaml
# .github/workflows/formal-verification.yml
- name: Verify Properties
  run: |
    formal-circuits-gpt verify src/**/*.v \
      --properties specs/properties.yaml \
      --output verification-report.json
      
- name: Check Coverage
  run: |
    formal-circuits-gpt coverage src/**/*.v \
      --properties specs/properties.yaml \
      --min-coverage 90%
```

### Property Maintenance

1. **Review Process**: Include property reviews in code reviews
2. **Regression Testing**: Test properties against known circuit versions
3. **Documentation**: Keep properties documented and up-to-date
4. **Reuse**: Build libraries of reusable property templates

This guide should help you effectively specify and work with properties in Formal-Circuits-GPT. For more examples, see the [examples directory](../../examples/) and the [API documentation](../api/README.md).