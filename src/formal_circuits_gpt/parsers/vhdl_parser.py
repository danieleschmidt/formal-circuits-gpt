"""VHDL parser implementation."""

import re
from typing import List, Dict, Any, Optional, Tuple
from .ast_nodes import (
    CircuitAST, Module, Port, Signal, Assignment, AlwaysBlock,
    SignalType, ModuleInstance
)


class VHDLParseError(Exception):
    """Exception raised for VHDL parsing errors."""
    pass


class VHDLParser:
    """Parser for VHDL code."""
    
    def __init__(self):
        """Initialize the parser with regex patterns."""
        self.entity_pattern = re.compile(
            r'entity\s+(\w+)\s+is\s+(.*?)end\s+(?:entity\s+)?(?:\w+\s*)?;',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        self.architecture_pattern = re.compile(
            r'architecture\s+(\w+)\s+of\s+(\w+)\s+is\s+(.*?)begin\s+(.*?)end\s+(?:architecture\s+)?(?:\w+\s*)?;',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        self.port_pattern = re.compile(
            r'(\w+)\s*:\s*(in|out|inout)\s+([\w_]+)(?:\s*\(\s*(\d+)\s+downto\s+(\d+)\s*\))?\s*(?:;|$)',
            re.MULTILINE | re.IGNORECASE
        )
        self.signal_pattern = re.compile(
            r'signal\s+(\w+)\s*:\s*(\w+)(?:\((\d+)\s+downto\s+(\d+)\))?(?:\s*:=\s*([^;]+))?;',
            re.MULTILINE | re.IGNORECASE
        )
        self.assignment_pattern = re.compile(
            r'(\w+)\s*<=\s*([^;]+);',
            re.MULTILINE | re.IGNORECASE
        )
        self.process_pattern = re.compile(
            r'(?:(\w+)\s*:\s*)?process\s*(?:\(([^)]*)\))?\s+(.*?)end\s+process(?:\s+\w+)?;',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        self.component_pattern = re.compile(
            r'(\w+)\s*:\s*(\w+)(?:\s+generic\s+map\s*\([^)]*\))?\s*port\s+map\s*\(([^)]*)\);',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
    
    def parse(self, vhdl_code: str) -> CircuitAST:
        """Parse VHDL code into AST.
        
        Args:
            vhdl_code: VHDL source code string
            
        Returns:
            CircuitAST representation of the code
            
        Raises:
            VHDLParseError: If parsing fails
        """
        try:
            # Remove comments
            cleaned_code = self._remove_comments(vhdl_code)
            
            # Extract entities and architectures
            modules = self._parse_entities_and_architectures(cleaned_code)
            
            if not modules:
                raise VHDLParseError("No entities found in VHDL code")
            
            # Create AST
            ast = CircuitAST(modules=modules)
            
            # Validate AST
            errors = ast.validate()
            if errors:
                raise VHDLParseError(f"AST validation failed: {'; '.join(errors)}")
            
            return ast
            
        except Exception as e:
            raise VHDLParseError(f"Failed to parse VHDL: {str(e)}") from e
    
    def _remove_comments(self, code: str) -> str:
        """Remove VHDL comments (-- to end of line)."""
        return re.sub(r'--.*$', '', code, flags=re.MULTILINE)
    
    def _parse_entities_and_architectures(self, code: str) -> List[Module]:
        """Parse entities and their corresponding architectures."""
        modules = []
        
        # Find all entities
        entities = {}
        for match in self.entity_pattern.finditer(code):
            entity_name = match.group(1)
            entity_body = match.group(2)
            entities[entity_name] = self._parse_entity_ports(entity_body)
        
        # Find all architectures and match with entities
        for match in self.architecture_pattern.finditer(code):
            arch_name = match.group(1)
            entity_name = match.group(2)
            arch_declarations = match.group(3)
            arch_body = match.group(4)
            
            if entity_name not in entities:
                continue  # Skip orphaned architectures
            
            # Get ports from entity
            ports = entities[entity_name]
            
            # Parse architecture components
            signals = self._parse_signals(arch_declarations)
            assignments = self._parse_assignments(arch_body)
            processes = self._parse_processes(arch_body)
            instances = self._parse_component_instances(arch_body)
            
            # Convert processes to always blocks
            always_blocks = [self._process_to_always_block(p) for p in processes]
            
            # Add implicit signals for ports that are assigned to (VHDL allows direct port assignment)
            port_names = {p.name for p in ports}
            assigned_targets = {a.target for a in assignments}
            
            # For any assignment target that is a port but not in signals, 
            # the port itself acts as the signal
            for target in assigned_targets:
                if target in port_names and target not in {s.name for s in signals}:
                    # Find the port to get its type info
                    port = next(p for p in ports if p.name == target)
                    # Convert port to signal for consistency
                    signal = Signal(
                        name=target,
                        signal_type=SignalType.WIRE if port.signal_type == SignalType.OUTPUT else SignalType.WIRE,
                        width=port.width,
                        initial_value=None
                    )
                    signals.append(signal)
            
            module = Module(
                name=entity_name,
                ports=ports,
                signals=signals,
                assignments=assignments,
                always_blocks=always_blocks,
                submodules=instances,
                parameters={}
            )
            
            modules.append(module)
        
        return modules
    
    def _parse_entity_ports(self, entity_body: str) -> List[Port]:
        """Parse entity port declarations."""
        ports = []
        
        # Find port clause - handle nested parentheses properly
        port_start = entity_body.find('port')
        if port_start == -1:
            return ports
        
        # Find the opening parenthesis after 'port'
        paren_start = entity_body.find('(', port_start)
        if paren_start == -1:
            return ports
        
        # Find matching closing parenthesis (handle nested parens)
        paren_count = 0
        paren_end = -1
        for i, char in enumerate(entity_body[paren_start:], paren_start):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    paren_end = i
                    break
        
        if paren_end == -1:
            return ports
        
        port_declarations = entity_body[paren_start + 1:paren_end]
        
        # Parse individual port declarations
        for match in self.port_pattern.finditer(port_declarations):
            name = match.group(1)
            direction = match.group(2).lower()
            type_name = match.group(3)
            msb = int(match.group(4)) if match.group(4) else None
            lsb = int(match.group(5)) if match.group(5) else None
            
            # Map VHDL directions to SignalType
            signal_type_map = {
                'in': SignalType.INPUT,
                'out': SignalType.OUTPUT,
                'inout': SignalType.INOUT
            }
            signal_type = signal_type_map.get(direction, SignalType.INPUT)
            
            # Calculate width
            width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
            
            port = Port(
                name=name,
                signal_type=signal_type,
                width=width,
                msb=msb,
                lsb=lsb
            )
            ports.append(port)
        
        return ports
    
    def _parse_signals(self, declarations: str) -> List[Signal]:
        """Parse signal declarations."""
        signals = []
        
        for match in self.signal_pattern.finditer(declarations):
            name = match.group(1)
            type_name = match.group(2)
            msb = int(match.group(3)) if match.group(3) else None
            lsb = int(match.group(4)) if match.group(4) else None
            initial_value = match.group(5) if match.group(5) else None
            
            # Calculate width
            width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
            
            # VHDL signals are similar to Verilog wires
            signal = Signal(
                name=name,
                signal_type=SignalType.WIRE,
                width=width,
                initial_value=initial_value.strip() if initial_value else None
            )
            signals.append(signal)
        
        return signals
    
    def _parse_assignments(self, body: str) -> List[Assignment]:
        """Parse concurrent signal assignments."""
        assignments = []
        
        for match in self.assignment_pattern.finditer(body):
            target = match.group(1)
            expression = match.group(2).strip()
            
            assignment = Assignment(
                target=target,
                expression=expression,
                is_blocking=False  # VHDL concurrent assignments are non-blocking
            )
            assignments.append(assignment)
        
        return assignments
    
    def _parse_processes(self, body: str) -> List[Dict[str, Any]]:
        """Parse process statements."""
        processes = []
        
        for match in self.process_pattern.finditer(body):
            process_label = match.group(1) if match.group(1) else "unnamed"
            sensitivity_list = match.group(2) if match.group(2) else ""
            process_body = match.group(3)
            
            # Parse sensitivity list
            sensitivity_signals = []
            if sensitivity_list:
                sensitivity_signals = [s.strip() for s in sensitivity_list.split(',') if s.strip()]
            
            # Parse process statements
            statements = self._parse_process_statements(process_body)
            
            process_info = {
                'label': process_label,
                'sensitivity_list': sensitivity_signals,
                'statements': statements
            }
            processes.append(process_info)
        
        return processes
    
    def _parse_process_statements(self, process_body: str) -> List[Assignment]:
        """Parse statements within a process."""
        statements = []
        
        # Look for variable assignments and signal assignments
        assignment_pattern = re.compile(r'(\w+)\s*(?::=|<=)\s*([^;]+);', re.MULTILINE | re.IGNORECASE)
        
        for match in assignment_pattern.finditer(process_body):
            target = match.group(1)
            expression = match.group(2).strip()
            is_blocking = ":=" in match.group(0)  # := is variable assignment (blocking)
            
            assignment = Assignment(
                target=target,
                expression=expression,
                is_blocking=is_blocking
            )
            statements.append(assignment)
        
        return statements
    
    def _process_to_always_block(self, process_info: Dict[str, Any]) -> AlwaysBlock:
        """Convert VHDL process to Verilog-style always block."""
        sensitivity_list = process_info['sensitivity_list']
        statements = process_info['statements']
        
        # Determine block type based on sensitivity list
        block_type = "sequential"
        for signal in sensitivity_list:
            if "clock" in signal.lower() or "clk" in signal.lower():
                block_type = "sequential"
                break
        else:
            block_type = "combinational"
        
        return AlwaysBlock(
            sensitivity_list=sensitivity_list,
            statements=statements,
            block_type=block_type
        )
    
    def _parse_component_instances(self, body: str) -> List[ModuleInstance]:
        """Parse component instantiations."""
        instances = []
        
        for match in self.component_pattern.finditer(body):
            instance_name = match.group(1)
            component_name = match.group(2)
            port_map = match.group(3)
            
            # Parse port map
            port_connections = self._parse_port_map(port_map)
            
            instance = ModuleInstance(
                instance_name=instance_name,
                module_name=component_name,
                port_connections=port_connections,
                parameter_overrides={}
            )
            instances.append(instance)
        
        return instances
    
    def _parse_port_map(self, port_map: str) -> Dict[str, str]:
        """Parse VHDL port map connections."""
        connections = {}
        
        if not port_map.strip():
            return connections
        
        # Parse named associations (port => signal)
        association_pattern = re.compile(r'(\w+)\s*=>\s*(\w+)', re.IGNORECASE)
        
        for match in association_pattern.finditer(port_map):
            port_name = match.group(1)
            signal_name = match.group(2)
            connections[port_name] = signal_name
        
        return connections