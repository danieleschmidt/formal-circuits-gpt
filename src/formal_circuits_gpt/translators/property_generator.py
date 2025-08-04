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
    
    def __init__(self, name: str, formula: str, property_type: PropertyType, 
                 description: str = "", proof_strategy: str = "auto"):
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
        """Initialize the property generator."""
        self.arithmetic_patterns = {
            'adder': ['add', 'sum', 'plus'],
            'multiplier': ['mul', 'mult', 'product'],
            'subtractor': ['sub', 'minus', 'diff'],
            'comparator': ['cmp', 'compare', 'equal', 'less', 'greater'],
            'counter': ['count', 'counter', 'increment'],
            'decoder': ['decode', 'decoder'],
            'encoder': ['encode', 'encoder'],
            'mux': ['mux', 'select', 'multiplex'],
            'demux': ['demux', 'demultiplex']
        }
    
    def generate_properties(self, ast: CircuitAST, 
                          include_types: Optional[List[PropertyType]] = None) -> List[PropertySpec]:
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
    
    def generate_module_properties(self, module: Module, 
                                 include_types: List[PropertyType]) -> List[PropertySpec]:
        """Generate properties for a single module."""
        properties = []
        
        # Functional properties
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
        if module_type == 'adder' and len(input_ports) >= 2 and len(output_ports) == 1:
            in1, in2 = input_ports[0].name, input_ports[1].name
            out = output_ports[0].name
            
            properties.append(PropertySpec(
                name=f"{module.name}_addition_correctness",
                formula=f"∀ {in1} {in2}. {module.name}({in1}, {in2}) = {in1} + {in2}",
                property_type=PropertyType.FUNCTIONAL,
                description=f"Addition correctness for {module.name}",
                proof_strategy="arithmetic"
            ))
            
            # Commutativity
            properties.append(PropertySpec(
                name=f"{module.name}_commutativity", 
                formula=f"∀ {in1} {in2}. {module.name}({in1}, {in2}) = {module.name}({in2}, {in1})",
                property_type=PropertyType.FUNCTIONAL,
                description="Addition is commutative"
            ))
            
            # Associativity (if 3+ inputs)
            if len(input_ports) >= 3:
                in3 = input_ports[2].name
                properties.append(PropertySpec(
                    name=f"{module.name}_associativity",
                    formula=f"∀ {in1} {in2} {in3}. {module.name}({module.name}({in1}, {in2}), {in3}) = {module.name}({in1}, {module.name}({in2}, {in3}))",
                    property_type=PropertyType.FUNCTIONAL,
                    description="Addition is associative"
                ))
        
        elif module_type == 'multiplier' and len(input_ports) >= 2 and len(output_ports) == 1:
            in1, in2 = input_ports[0].name, input_ports[1].name
            out = output_ports[0].name
            
            properties.append(PropertySpec(
                name=f"{module.name}_multiplication_correctness",
                formula=f"∀ {in1} {in2}. {module.name}({in1}, {in2}) = {in1} * {in2}",
                property_type=PropertyType.FUNCTIONAL,
                description=f"Multiplication correctness for {module.name}"
            ))
        
        elif module_type == 'comparator' and len(input_ports) >= 2:
            in1, in2 = input_ports[0].name, input_ports[1].name
            
            # Find equality output
            eq_output = None
            for port in output_ports:
                if 'eq' in port.name.lower() or 'equal' in port.name.lower():
                    eq_output = port.name
                    break
            
            if eq_output:
                properties.append(PropertySpec(
                    name=f"{module.name}_equality_correctness",
                    formula=f"∀ {in1} {in2}. {eq_output} ↔ ({in1} = {in2})",
                    property_type=PropertyType.FUNCTIONAL,
                    description="Equality comparison correctness"
                ))
        
        # Generic input-output relationship
        if not properties and input_ports and output_ports:
            # Generate a generic property
            input_vars = " ".join(p.name for p in input_ports)
            properties.append(PropertySpec(
                name=f"{module.name}_well_defined",
                formula=f"∀ {input_vars}. ∃ result. {module.name}({input_vars}) = result",
                property_type=PropertyType.FUNCTIONAL,
                description="Module is well-defined for all inputs"
            ))
        
        return properties
    
    def _generate_safety_properties(self, module: Module) -> List[PropertySpec]:
        """Generate safety properties (nothing bad happens)."""
        properties = []
        
        # Overflow detection for arithmetic modules
        module_type = self._classify_module(module)
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]
        
        if module_type in ['adder', 'multiplier'] and input_ports and output_ports:
            output_port = output_ports[0]
            max_value = 2 ** output_port.width - 1
            
            properties.append(PropertySpec(
                name=f"{module.name}_no_overflow",
                formula=f"∀ inputs. {module.name}(inputs) ≤ {max_value}",
                property_type=PropertyType.SAFETY,
                description="No arithmetic overflow occurs"
            ))
        
        # Signal bounds
        for port in output_ports:
            if port.width > 1:
                max_val = 2 ** port.width - 1
                properties.append(PropertySpec(
                    name=f"{module.name}_{port.name}_bounds",
                    formula=f"∀ inputs. 0 ≤ {port.name} ≤ {max_val}",
                    property_type=PropertyType.SAFETY,
                    description=f"Output {port.name} within valid range"
                ))
        
        # No undefined outputs
        if output_ports:
            output_names = [p.name for p in output_ports]
            properties.append(PropertySpec(
                name=f"{module.name}_outputs_defined",
                formula=f"∀ inputs. defined({', '.join(output_names)})",
                property_type=PropertyType.SAFETY,
                description="All outputs are always defined"
            ))
        
        return properties
    
    def _generate_structural_properties(self, module: Module) -> List[PropertySpec]:
        """Generate structural properties about module organization."""
        properties = []
        
        # Port consistency
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]
        
        if input_ports:
            properties.append(PropertySpec(
                name=f"{module.name}_has_inputs",
                formula=f"inputs_exist({', '.join(p.name for p in input_ports)})",
                property_type=PropertyType.STRUCTURAL,
                description="Module has required input ports"
            ))
        
        if output_ports:
            properties.append(PropertySpec(
                name=f"{module.name}_has_outputs",
                formula=f"outputs_exist({', '.join(p.name for p in output_ports)})",
                property_type=PropertyType.STRUCTURAL,
                description="Module has required output ports"
            ))
        
        # Assignment coverage
        assigned_signals = {a.target for a in module.assignments}
        for assignment in module.always_blocks:
            assigned_signals.update(a.target for a in assignment.statements)
        
        output_signal_names = {p.name for p in output_ports}
        unassigned_outputs = output_signal_names - assigned_signals
        
        if not unassigned_outputs:
            properties.append(PropertySpec(
                name=f"{module.name}_complete_assignment",
                formula="∀ output ∈ outputs. assigned(output)",
                property_type=PropertyType.STRUCTURAL,
                description="All outputs are assigned"
            ))
        
        return properties
    
    def _generate_liveness_properties(self, module: Module) -> List[PropertySpec]:
        """Generate liveness properties (something good eventually happens)."""
        properties = []
        
        # For sequential modules, generate eventual response properties
        has_clock = any('clk' in p.name.lower() or 'clock' in p.name.lower() 
                       for p in module.ports if p.signal_type == SignalType.INPUT)
        
        if has_clock:
            properties.append(PropertySpec(
                name=f"{module.name}_eventually_responds",
                formula="◇(response_ready)",
                property_type=PropertyType.LIVENESS,
                description="Module eventually produces output"
            ))
        
        # Counter-specific liveness
        if 'count' in module.name.lower():
            properties.append(PropertySpec(
                name=f"{module.name}_eventually_reaches_max",
                formula="◇(counter = max_value)",
                property_type=PropertyType.LIVENESS,
                description="Counter eventually reaches maximum"
            ))
        
        return properties
    
    def _generate_temporal_properties(self, module: Module) -> List[PropertySpec]:
        """Generate temporal logic properties."""
        properties = []
        
        # Clock-based temporal properties
        has_clock = any('clk' in p.name.lower() or 'clock' in p.name.lower()
                       for p in module.ports if p.signal_type == SignalType.INPUT)
        
        if has_clock:
            # Reset behavior
            reset_ports = [p for p in module.ports 
                          if 'rst' in p.name.lower() or 'reset' in p.name.lower()]
            if reset_ports:
                properties.append(PropertySpec(
                    name=f"{module.name}_reset_behavior",
                    formula="reset → ◯(all_outputs = 0)",
                    property_type=PropertyType.TEMPORAL,
                    description="Reset properly initializes all outputs"
                ))
            
            # Stability
            properties.append(PropertySpec(
                name=f"{module.name}_stable_outputs",
                formula="stable_inputs → stable_outputs",
                property_type=PropertyType.TEMPORAL,
                description="Stable inputs produce stable outputs"
            ))
        
        return properties
    
    def _generate_global_properties(self, ast: CircuitAST) -> List[PropertySpec]:
        """Generate global properties across all modules."""
        properties = []
        
        # Module connectivity
        all_instances = []
        for module in ast.modules:
            all_instances.extend(module.submodules)
        
        if all_instances:
            properties.append(PropertySpec(
                name="global_connectivity",
                formula="∀ instance. properly_connected(instance)",
                property_type=PropertyType.STRUCTURAL,
                description="All module instances are properly connected"
            ))
        
        # No combinational loops
        properties.append(PropertySpec(
            name="no_combinational_loops",
            formula="acyclic(combinational_dependencies)",
            property_type=PropertyType.STRUCTURAL,
            description="No combinational feedback loops exist"
        ))
        
        return properties
    
    def _classify_module(self, module: Module) -> str:
        """Classify module type based on name and structure."""
        name_lower = module.name.lower()
        
        for module_type, patterns in self.arithmetic_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return module_type
        
        # Classify by port structure
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]
        
        if len(input_ports) == 2 and len(output_ports) == 1:
            # Could be binary operation
            if any(word in name_lower for word in ['add', 'sum']):
                return 'adder'
            elif any(word in name_lower for word in ['mul', 'mult']):
                return 'multiplier'
        
        # Default classification
        return 'generic'
    
    def generate_custom_property(self, name: str, formula: str, 
                                property_type: PropertyType, 
                                description: str = "") -> PropertySpec:
        """Generate a custom property specification."""
        return PropertySpec(
            name=name,
            formula=formula,
            property_type=property_type,
            description=description
        )