"""Property generator for inferring circuit correctness properties."""

from typing import List, Dict, Any, Optional, Set
from enum import Enum
from ..parsers.ast_nodes import CircuitAST, Module, Port, Signal, Assignment, SignalType


class PropertyType(Enum):
    """Types of properties that can be generated."""

    FUNCTIONAL = "functional"
    SAFETY = "safety"
    LIVENESS = "liveness"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"


class PropertySpec:
    """Specification for a circuit property."""

    def __init__(
        self,
        name: str,
        formula: str,
        property_type: PropertyType,
        description: str = "",
        proof_strategy: str = "auto",
    ):
        self.name = name
        self.formula = formula
        self.property_type = property_type
        self.description = description
        self.proof_strategy = proof_strategy

    def __repr__(self) -> str:
        return f"PropertySpec({self.name}: {self.formula})"


class PropertyGenerator:
    """Generates correctness properties from circuit structure."""

    def __init__(self):
        """Initialize the property generator with enhanced pattern recognition."""
        # Component type patterns for intelligent property inference
        self.component_patterns = {
            "adder": {
                "keywords": ["add", "sum", "plus", "carry"],
                "signals": ["sum", "cout", "carry_out"],
                "properties": ["commutativity", "associativity", "overflow_detection"],
            },
            "multiplier": {
                "keywords": ["mul", "mult", "product", "multiply"],
                "signals": ["product", "result"],
                "properties": ["commutativity", "distributivity", "zero_property"],
            },
            "counter": {
                "keywords": ["count", "counter", "increment", "decrement"],
                "signals": ["count", "overflow", "underflow"],
                "properties": ["increment_behavior", "reset_behavior", "overflow_wrap"],
            },
            "fsm": {
                "keywords": ["state", "next_state", "current_state"],
                "signals": ["state", "next_state"],
                "properties": ["reachability", "deadlock_freedom", "determinism"],
            },
            "mux": {
                "keywords": ["mux", "select", "multiplex", "switch"],
                "signals": ["sel", "select", "out"],
                "properties": ["selection_correctness", "all_inputs_reachable"],
            },
            "memory": {
                "keywords": ["mem", "memory", "ram", "rom", "storage"],
                "signals": ["addr", "data", "we", "oe"],
                "properties": ["read_after_write", "data_integrity", "address_decode"],
            },
            "fifo": {
                "keywords": ["fifo", "queue", "buffer"],
                "signals": ["empty", "full", "push", "pop"],
                "properties": [
                    "fifo_order",
                    "empty_full_mutual_exclusion",
                    "capacity_limits",
                ],
            },
        }

        # Advanced property templates
        self.property_templates = {
            "temporal": {
                "always": "G({condition})",
                "eventually": "F({condition})",
                "until": "{condition1} U {condition2}",
                "next": "X({condition})",
            },
            "functional": {
                "equality": "{output} = {expression}",
                "implication": "{condition} → {result}",
                "equivalence": "{expr1} ↔ {expr2}",
            },
            "bounds": {
                "range": "{min} ≤ {signal} ≤ {max}",
                "positive": "{signal} ≥ 0",
                "overflow": "{signal} < 2^{width}",
            },
        }

    def generate_properties(
        self, ast: CircuitAST, include_types: Optional[List[PropertyType]] = None
    ) -> List[PropertySpec]:
        """Generate properties for the entire circuit.

        Args:
            ast: Circuit AST to analyze
            include_types: Property types to generate (all if None)

        Returns:
            List of generated properties
        """
        if include_types is None:
            include_types = list(PropertyType)

        properties = []

        for module in ast.modules:
            module_props = self.generate_module_properties(module, include_types)
            properties.extend(module_props)

        # Add global properties
        if PropertyType.STRUCTURAL in include_types:
            global_props = self._generate_global_properties(ast)
            properties.extend(global_props)

        return properties

    def infer_component_type(self, module: Module) -> Optional[str]:
        """Intelligently infer component type from module structure and naming."""
        module_name = module.name.lower()

        # Direct name matching
        for comp_type, patterns in self.component_patterns.items():
            if any(keyword in module_name for keyword in patterns["keywords"]):
                return comp_type

        # Signal pattern matching
        signal_names = [sig.name.lower() for sig in module.signals]
        port_names = [port.name.lower() for port in module.ports]
        all_names = signal_names + port_names

        component_scores = {}
        for comp_type, patterns in self.component_patterns.items():
            score = 0
            for signal_pattern in patterns["signals"]:
                if any(signal_pattern in name for name in all_names):
                    score += 2
            for keyword in patterns["keywords"]:
                if any(keyword in name for name in all_names):
                    score += 1
            if score > 0:
                component_scores[comp_type] = score

        # Return highest scoring component type
        if component_scores:
            return max(component_scores.items(), key=lambda x: x[1])[0]

        return None

    def generate_component_specific_properties(
        self, module: Module, component_type: str
    ) -> List[PropertySpec]:
        """Generate properties specific to the identified component type."""
        properties = []

        if component_type == "adder":
            properties.extend(self._generate_adder_properties(module))
        elif component_type == "counter":
            properties.extend(self._generate_counter_properties(module))
        elif component_type == "fsm":
            properties.extend(self._generate_fsm_properties(module))
        elif component_type == "mux":
            properties.extend(self._generate_mux_properties(module))
        elif component_type == "memory":
            properties.extend(self._generate_memory_properties(module))

        return properties

    def _generate_adder_properties(self, module: Module) -> List[PropertySpec]:
        """Generate adder-specific properties."""
        properties = []

        # Find adder inputs and outputs
        from ..parsers.ast_nodes import SignalType

        inputs = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        outputs = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]

        if len(inputs) >= 2 and len(outputs) >= 1:
            input_a = inputs[0].name
            input_b = inputs[1].name if len(inputs) > 1 else inputs[0].name
            output_sum = outputs[0].name

            # Commutativity
            properties.append(
                PropertySpec(
                    name="adder_commutativity",
                    formula=f"∀ {input_a} {input_b}. {module.name}({input_a}, {input_b}) = {module.name}({input_b}, {input_a})",
                    property_type=PropertyType.FUNCTIONAL,
                    description="Addition is commutative",
                    proof_strategy="simp",
                )
            )

            # Zero identity
            properties.append(
                PropertySpec(
                    name="adder_zero_identity",
                    formula=f"∀ {input_a}. {module.name}({input_a}, 0) = {input_a}",
                    property_type=PropertyType.FUNCTIONAL,
                    description="Adding zero preserves the value",
                )
            )

            # Overflow bounds
            if hasattr(outputs[0], "width") and outputs[0].width:
                width = outputs[0].width
                properties.append(
                    PropertySpec(
                        name="adder_overflow_bounds",
                        formula=f"∀ inputs. {output_sum} < 2^{width}",
                        property_type=PropertyType.SAFETY,
                        description="Sum stays within output bit width bounds",
                    )
                )

        return properties

    def _generate_counter_properties(self, module: Module) -> List[PropertySpec]:
        """Generate counter-specific properties."""
        properties = []

        # Find counter signals
        count_signal = next(
            (p.name for p in module.ports if "count" in p.name.lower()), None
        )
        reset_signal = next(
            (p.name for p in module.ports if "reset" in p.name.lower()), None
        )
        enable_signal = next(
            (p.name for p in module.ports if "enable" in p.name.lower()), None
        )

        if count_signal:
            # Reset property
            if reset_signal:
                properties.append(
                    PropertySpec(
                        name="counter_reset",
                        formula=f"{reset_signal} → ◯({count_signal} = 0)",
                        property_type=PropertyType.SAFETY,
                        description="Counter resets to zero",
                    )
                )

            # Increment property
            if enable_signal:
                properties.append(
                    PropertySpec(
                        name="counter_increment",
                        formula=f"{enable_signal} ∧ ¬{reset_signal or 'reset'} → ◯({count_signal} = {count_signal} + 1)",
                        property_type=PropertyType.FUNCTIONAL,
                        description="Counter increments when enabled",
                    )
                )

            # Overflow wrap-around
            count_port = next((p for p in module.ports if p.name == count_signal), None)
            if count_port and hasattr(count_port, "width") and count_port.width:
                max_val = 2**count_port.width - 1
                properties.append(
                    PropertySpec(
                        name="counter_overflow_wrap",
                        formula=f"({count_signal} = {max_val}) ∧ {enable_signal or 'enable'} → ◯({count_signal} = 0)",
                        property_type=PropertyType.FUNCTIONAL,
                        description="Counter wraps to zero at maximum value",
                    )
                )

        return properties

    def _generate_fsm_properties(self, module: Module) -> List[PropertySpec]:
        """Generate FSM-specific properties."""
        properties = []

        # Find state signals
        state_signals = [
            p.name for p in module.ports + module.signals if "state" in p.name.lower()
        ]

        if state_signals:
            state_sig = state_signals[0]

            # Determinism
            properties.append(
                PropertySpec(
                    name="fsm_determinism",
                    formula=f"∀ inputs state. ∃! next_state. transition(state, inputs) = next_state",
                    property_type=PropertyType.SAFETY,
                    description="FSM transitions are deterministic",
                )
            )

            # Reachability (all states reachable)
            properties.append(
                PropertySpec(
                    name="fsm_reachability",
                    formula=f"∀ state. ◇({state_sig} = state)",
                    property_type=PropertyType.LIVENESS,
                    description="All states are reachable",
                )
            )

        return properties

    def _generate_mux_properties(self, module: Module) -> List[PropertySpec]:
        """Generate multiplexer-specific properties."""
        properties = []

        # Find selection and data signals
        from ..parsers.ast_nodes import SignalType

        sel_signal = next(
            (p.name for p in module.ports if "sel" in p.name.lower()), None
        )
        data_inputs = [
            p.name
            for p in module.ports
            if p.name.startswith("data") or p.name.startswith("in")
        ]
        output_signal = next(
            (p.name for p in module.ports if p.signal_type == SignalType.OUTPUT), None
        )

        if sel_signal and data_inputs and output_signal:
            # Selection correctness
            for i, data_input in enumerate(data_inputs):
                properties.append(
                    PropertySpec(
                        name=f"mux_select_{i}",
                        formula=f"{sel_signal} = {i} → {output_signal} = {data_input}",
                        property_type=PropertyType.FUNCTIONAL,
                        description=f"Correct selection of input {i}",
                    )
                )

        return properties

    def _generate_memory_properties(self, module: Module) -> List[PropertySpec]:
        """Generate memory-specific properties."""
        properties = []

        # Find memory control signals
        addr_sig = next(
            (p.name for p in module.ports if "addr" in p.name.lower()), None
        )
        data_sig = next(
            (p.name for p in module.ports if "data" in p.name.lower()), None
        )
        we_sig = next(
            (
                p.name
                for p in module.ports
                if "we" in p.name.lower() or "write" in p.name.lower()
            ),
            None,
        )

        if addr_sig and data_sig and we_sig:
            # Read-after-write consistency
            properties.append(
                PropertySpec(
                    name="memory_read_after_write",
                    formula=f"{we_sig} ∧ ({addr_sig} = addr) → ◯(read({addr_sig}) = {data_sig})",
                    property_type=PropertyType.FUNCTIONAL,
                    description="Read after write returns written value",
                )
            )

        return properties

    def generate_module_properties(
        self, module: Module, include_types: List[PropertyType]
    ) -> List[PropertySpec]:
        """Generate properties for a single module with intelligent inference."""
        properties = []

        # First, try to infer component type for specialized properties
        component_type = self.infer_component_type(module)
        if component_type:
            properties.extend(
                self.generate_component_specific_properties(module, component_type)
            )

        # General functional properties
        if PropertyType.FUNCTIONAL in include_types:
            properties.extend(self._generate_functional_properties(module))

        # Safety properties
        if PropertyType.SAFETY in include_types:
            properties.extend(self._generate_safety_properties(module))

        # Structural properties
        if PropertyType.STRUCTURAL in include_types:
            properties.extend(self._generate_structural_properties(module))

        # Liveness properties
        if PropertyType.LIVENESS in include_types:
            properties.extend(self._generate_liveness_properties(module))

        # Temporal properties
        if PropertyType.TEMPORAL in include_types:
            properties.extend(self._generate_temporal_properties(module))

        return properties

    def _generate_functional_properties(self, module: Module) -> List[PropertySpec]:
        """Generate functional correctness properties."""
        properties = []

        module_type = self._classify_module(module)
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]

        # Arithmetic modules
        if module_type == "adder" and len(input_ports) >= 2 and len(output_ports) == 1:
            in1, in2 = input_ports[0].name, input_ports[1].name
            out = output_ports[0].name

            properties.append(
                PropertySpec(
                    name=f"{module.name}_addition_correctness",
                    formula=f"∀ {in1} {in2}. {module.name}({in1}, {in2}) = {in1} + {in2}",
                    property_type=PropertyType.FUNCTIONAL,
                    description=f"Addition correctness for {module.name}",
                    proof_strategy="arithmetic",
                )
            )

            # Commutativity
            properties.append(
                PropertySpec(
                    name=f"{module.name}_commutativity",
                    formula=f"∀ {in1} {in2}. {module.name}({in1}, {in2}) = {module.name}({in2}, {in1})",
                    property_type=PropertyType.FUNCTIONAL,
                    description="Addition is commutative",
                )
            )

            # Associativity (if 3+ inputs)
            if len(input_ports) >= 3:
                in3 = input_ports[2].name
                properties.append(
                    PropertySpec(
                        name=f"{module.name}_associativity",
                        formula=f"∀ {in1} {in2} {in3}. {module.name}({module.name}({in1}, {in2}), {in3}) = {module.name}({in1}, {module.name}({in2}, {in3}))",
                        property_type=PropertyType.FUNCTIONAL,
                        description="Addition is associative",
                    )
                )

        elif (
            module_type == "multiplier"
            and len(input_ports) >= 2
            and len(output_ports) == 1
        ):
            in1, in2 = input_ports[0].name, input_ports[1].name
            out = output_ports[0].name

            properties.append(
                PropertySpec(
                    name=f"{module.name}_multiplication_correctness",
                    formula=f"∀ {in1} {in2}. {module.name}({in1}, {in2}) = {in1} * {in2}",
                    property_type=PropertyType.FUNCTIONAL,
                    description=f"Multiplication correctness for {module.name}",
                )
            )

        elif module_type == "comparator" and len(input_ports) >= 2:
            in1, in2 = input_ports[0].name, input_ports[1].name

            # Find equality output
            eq_output = None
            for port in output_ports:
                if "eq" in port.name.lower() or "equal" in port.name.lower():
                    eq_output = port.name
                    break

            if eq_output:
                properties.append(
                    PropertySpec(
                        name=f"{module.name}_equality_correctness",
                        formula=f"∀ {in1} {in2}. {eq_output} ↔ ({in1} = {in2})",
                        property_type=PropertyType.FUNCTIONAL,
                        description="Equality comparison correctness",
                    )
                )

        # Generic input-output relationship
        if not properties and input_ports and output_ports:
            # Generate a generic property
            input_vars = " ".join(p.name for p in input_ports)
            properties.append(
                PropertySpec(
                    name=f"{module.name}_well_defined",
                    formula=f"∀ {input_vars}. ∃ result. {module.name}({input_vars}) = result",
                    property_type=PropertyType.FUNCTIONAL,
                    description="Module is well-defined for all inputs",
                )
            )

        return properties

    def _generate_safety_properties(self, module: Module) -> List[PropertySpec]:
        """Generate safety properties (nothing bad happens)."""
        properties = []

        # Overflow detection for arithmetic modules
        module_type = self._classify_module(module)
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]

        if module_type in ["adder", "multiplier"] and input_ports and output_ports:
            output_port = output_ports[0]
            max_value = 2**output_port.width - 1

            properties.append(
                PropertySpec(
                    name=f"{module.name}_no_overflow",
                    formula=f"∀ inputs. {module.name}(inputs) ≤ {max_value}",
                    property_type=PropertyType.SAFETY,
                    description="No arithmetic overflow occurs",
                )
            )

        # Signal bounds
        for port in output_ports:
            if port.width > 1:
                max_val = 2**port.width - 1
                properties.append(
                    PropertySpec(
                        name=f"{module.name}_{port.name}_bounds",
                        formula=f"∀ inputs. 0 ≤ {port.name} ≤ {max_val}",
                        property_type=PropertyType.SAFETY,
                        description=f"Output {port.name} within valid range",
                    )
                )

        # No undefined outputs
        if output_ports:
            output_names = [p.name for p in output_ports]
            properties.append(
                PropertySpec(
                    name=f"{module.name}_outputs_defined",
                    formula=f"∀ inputs. defined({', '.join(output_names)})",
                    property_type=PropertyType.SAFETY,
                    description="All outputs are always defined",
                )
            )

        return properties

    def _generate_structural_properties(self, module: Module) -> List[PropertySpec]:
        """Generate structural properties about module organization."""
        properties = []

        # Port consistency
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]

        if input_ports:
            properties.append(
                PropertySpec(
                    name=f"{module.name}_has_inputs",
                    formula=f"inputs_exist({', '.join(p.name for p in input_ports)})",
                    property_type=PropertyType.STRUCTURAL,
                    description="Module has required input ports",
                )
            )

        if output_ports:
            properties.append(
                PropertySpec(
                    name=f"{module.name}_has_outputs",
                    formula=f"outputs_exist({', '.join(p.name for p in output_ports)})",
                    property_type=PropertyType.STRUCTURAL,
                    description="Module has required output ports",
                )
            )

        # Assignment coverage
        assigned_signals = {a.target for a in module.assignments}
        for assignment in module.always_blocks:
            assigned_signals.update(a.target for a in assignment.statements)

        output_signal_names = {p.name for p in output_ports}
        unassigned_outputs = output_signal_names - assigned_signals

        if not unassigned_outputs:
            properties.append(
                PropertySpec(
                    name=f"{module.name}_complete_assignment",
                    formula="∀ output ∈ outputs. assigned(output)",
                    property_type=PropertyType.STRUCTURAL,
                    description="All outputs are assigned",
                )
            )

        return properties

    def _generate_liveness_properties(self, module: Module) -> List[PropertySpec]:
        """Generate liveness properties (something good eventually happens)."""
        properties = []

        # For sequential modules, generate eventual response properties
        has_clock = any(
            "clk" in p.name.lower() or "clock" in p.name.lower()
            for p in module.ports
            if p.signal_type == SignalType.INPUT
        )

        if has_clock:
            properties.append(
                PropertySpec(
                    name=f"{module.name}_eventually_responds",
                    formula="◇(response_ready)",
                    property_type=PropertyType.LIVENESS,
                    description="Module eventually produces output",
                )
            )

        # Counter-specific liveness
        if "count" in module.name.lower():
            properties.append(
                PropertySpec(
                    name=f"{module.name}_eventually_reaches_max",
                    formula="◇(counter = max_value)",
                    property_type=PropertyType.LIVENESS,
                    description="Counter eventually reaches maximum",
                )
            )

        return properties

    def _generate_temporal_properties(self, module: Module) -> List[PropertySpec]:
        """Generate temporal logic properties."""
        properties = []

        # Clock-based temporal properties
        has_clock = any(
            "clk" in p.name.lower() or "clock" in p.name.lower()
            for p in module.ports
            if p.signal_type == SignalType.INPUT
        )

        if has_clock:
            # Reset behavior
            reset_ports = [
                p
                for p in module.ports
                if "rst" in p.name.lower() or "reset" in p.name.lower()
            ]
            if reset_ports:
                properties.append(
                    PropertySpec(
                        name=f"{module.name}_reset_behavior",
                        formula="reset → ◯(all_outputs = 0)",
                        property_type=PropertyType.TEMPORAL,
                        description="Reset properly initializes all outputs",
                    )
                )

            # Stability
            properties.append(
                PropertySpec(
                    name=f"{module.name}_stable_outputs",
                    formula="stable_inputs → stable_outputs",
                    property_type=PropertyType.TEMPORAL,
                    description="Stable inputs produce stable outputs",
                )
            )

        return properties

    def _generate_global_properties(self, ast: CircuitAST) -> List[PropertySpec]:
        """Generate global properties across all modules."""
        properties = []

        # Module connectivity
        all_instances = []
        for module in ast.modules:
            all_instances.extend(module.submodules)

        if all_instances:
            properties.append(
                PropertySpec(
                    name="global_connectivity",
                    formula="∀ instance. properly_connected(instance)",
                    property_type=PropertyType.STRUCTURAL,
                    description="All module instances are properly connected",
                )
            )

        # No combinational loops
        properties.append(
            PropertySpec(
                name="no_combinational_loops",
                formula="acyclic(combinational_dependencies)",
                property_type=PropertyType.STRUCTURAL,
                description="No combinational feedback loops exist",
            )
        )

        return properties

    def _classify_module(self, module: Module) -> str:
        """Classify module type based on name and structure."""
        name_lower = module.name.lower()

        for module_type, pattern_dict in self.component_patterns.items():
            if any(pattern in name_lower for pattern in pattern_dict["keywords"]):
                return module_type

        # Classify by port structure
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]

        if len(input_ports) == 2 and len(output_ports) == 1:
            # Could be binary operation
            if any(word in name_lower for word in ["add", "sum"]):
                return "adder"
            elif any(word in name_lower for word in ["mul", "mult"]):
                return "multiplier"

        # Default classification
        return "generic"

    def generate_custom_property(
        self,
        name: str,
        formula: str,
        property_type: PropertyType,
        description: str = "",
    ) -> PropertySpec:
        """Generate a custom property specification."""
        return PropertySpec(
            name=name,
            formula=formula,
            property_type=property_type,
            description=description,
        )
