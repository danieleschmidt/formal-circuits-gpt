"""Abstract Syntax Tree nodes for HDL representations."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class SignalType(Enum):
    """Signal type enumeration."""
    INPUT = "input"
    OUTPUT = "output"
    WIRE = "wire"
    REG = "reg"
    INOUT = "inout"


class OperatorType(Enum):
    """Operator type enumeration."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    AND = "&"
    OR = "|"
    XOR = "^"
    NOT = "~"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


@dataclass
class Port:
    """Represents a module port."""
    name: str
    signal_type: SignalType
    width: int = 1
    msb: Optional[int] = None
    lsb: Optional[int] = None
    
    @property
    def bit_width(self) -> int:
        """Calculate bit width from MSB/LSB."""
        if self.msb is not None and self.lsb is not None:
            return abs(self.msb - self.lsb) + 1
        return self.width


@dataclass
class Signal:
    """Represents an internal signal."""
    name: str
    signal_type: SignalType
    width: int = 1
    initial_value: Optional[str] = None
    
    def is_vector(self) -> bool:
        """Check if signal is a vector (width > 1)."""
        return self.width > 1


@dataclass
class Assignment:
    """Represents a signal assignment."""
    target: str
    expression: str
    is_blocking: bool = False
    delay: Optional[str] = None
    
    def __post_init__(self):
        """Validate assignment after initialization."""
        if not self.target or not self.expression:
            raise ValueError("Assignment must have target and expression")


@dataclass
class AlwaysBlock:
    """Represents an always block."""
    sensitivity_list: List[str]
    statements: List[Assignment]
    block_type: str = "combinational"  # or "sequential"
    
    def is_sequential(self) -> bool:
        """Check if this is a sequential block."""
        return any("posedge" in sig or "negedge" in sig for sig in self.sensitivity_list)


@dataclass
class Module:
    """Represents a hardware module."""
    name: str
    ports: List[Port]
    signals: List[Signal]
    assignments: List[Assignment]
    always_blocks: List[AlwaysBlock]
    submodules: List["ModuleInstance"]
    parameters: Dict[str, Any]
    
    def __post_init__(self):
        """Validate module after initialization."""
        if not self.name:
            raise ValueError("Module must have a name")
        self.submodules = self.submodules or []
        self.parameters = self.parameters or {}
    
    def get_port_by_name(self, name: str) -> Optional[Port]:
        """Find port by name."""
        for port in self.ports:
            if port.name == name:
                return port
        return None
    
    def get_signal_by_name(self, name: str) -> Optional[Signal]:
        """Find signal by name."""
        for signal in self.signals:
            if signal.name == name:
                return signal
        return None
    
    def get_input_ports(self) -> List[Port]:
        """Get all input ports."""
        return [p for p in self.ports if p.signal_type == SignalType.INPUT]
    
    def get_output_ports(self) -> List[Port]:
        """Get all output ports."""
        return [p for p in self.ports if p.signal_type == SignalType.OUTPUT]


@dataclass
class ModuleInstance:
    """Represents an instance of a module."""
    instance_name: str
    module_name: str
    port_connections: Dict[str, str]
    parameter_overrides: Dict[str, Any]
    
    def __post_init__(self):
        """Validate instance after initialization."""
        if not self.instance_name or not self.module_name:
            raise ValueError("Instance must have instance_name and module_name")
        self.parameter_overrides = self.parameter_overrides or {}


@dataclass
class CircuitAST:
    """Top-level Abstract Syntax Tree for a circuit."""
    modules: List[Module]
    top_module: Optional[str] = None
    
    def __post_init__(self):
        """Validate AST after initialization."""
        if not self.modules:
            raise ValueError("CircuitAST must contain at least one module")
        
        # Set top module if not specified
        if not self.top_module and len(self.modules) == 1:
            self.top_module = self.modules[0].name
    
    def get_module_by_name(self, name: str) -> Optional[Module]:
        """Find module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None
    
    def get_top_module(self) -> Optional[Module]:
        """Get the top-level module."""
        if self.top_module:
            return self.get_module_by_name(self.top_module)
        return None
    
    def validate(self) -> List[str]:
        """Validate the AST and return list of errors."""
        errors = []
        
        # Check for duplicate module names
        module_names = [m.name for m in self.modules]
        if len(module_names) != len(set(module_names)):
            errors.append("Duplicate module names found")
        
        # Validate each module
        for module in self.modules:
            # Check for duplicate port names
            port_names = [p.name for p in module.ports]
            if len(port_names) != len(set(port_names)):
                errors.append(f"Module {module.name}: Duplicate port names")
            
            # Check for undefined signals in assignments
            all_signals = {p.name for p in module.ports} | {s.name for s in module.signals}
            for assignment in module.assignments:
                if assignment.target not in all_signals:
                    # Check if it's an output port (outputs can be assigned to)
                    is_output = any(p.name == assignment.target and p.signal_type == SignalType.OUTPUT 
                                  for p in module.ports)
                    if not is_output:
                        errors.append(f"Module {module.name}: Undefined signal '{assignment.target}'")
        
        return errors