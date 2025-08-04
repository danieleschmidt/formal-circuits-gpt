"""Verilog parser implementation."""

import re
from typing import List, Dict, Any, Optional, Tuple
from .ast_nodes import (
    CircuitAST, Module, Port, Signal, Assignment, AlwaysBlock,
    SignalType, ModuleInstance
)


class VerilogParseError(Exception):
    """Exception raised for Verilog parsing errors."""
    pass


class VerilogParser:
    """Parser for Verilog HDL code."""
    
    def __init__(self):
        """Initialize the parser with regex patterns."""
        self.module_pattern = re.compile(
            r'module\s+(\w+)\s*(?:\#\([^)]*\))?\s*\(([^)]*)\)\s*;', 
            re.MULTILINE | re.DOTALL
        )
        self.port_pattern = re.compile(
            r'(input|output|inout)\s+(?:\[(\d+):(\d+)\])?\s*(\w+)', 
            re.MULTILINE
        )
        self.wire_pattern = re.compile(
            r'wire\s+(?:\[(\d+):(\d+)\])?\s*(\w+)\s*;', 
            re.MULTILINE
        )
        self.reg_pattern = re.compile(
            r'reg\s+(?:\[(\d+):(\d+)\])?\s*(\w+)\s*;', 
            re.MULTILINE
        )
        self.assign_pattern = re.compile(
            r'assign\s+(\w+)\s*=\s*([^;]+)\s*;', 
            re.MULTILINE
        )
        self.always_pattern = re.compile(
            r'always\s*@\s*\(([^)]+)\)\s*(begin)?(.*?)(?:end)?', 
            re.MULTILINE | re.DOTALL
        )
        self.instance_pattern = re.compile(
            r'(\w+)\s+(?:#\([^)]*\))?\s*(\w+)\s*\(([^)]*)\)\s*;',
            re.MULTILINE | re.DOTALL
        )
    
    def parse(self, verilog_code: str) -> CircuitAST:
        """Parse Verilog code into AST.
        
        Args:
            verilog_code: Verilog source code string
            
        Returns:
            CircuitAST representation of the code
            
        Raises:
            VerilogParseError: If parsing fails
        """
        try:
            # Remove comments
            cleaned_code = self._remove_comments(verilog_code)
            
            # Extract modules
            modules = self._parse_modules(cleaned_code)
            
            if not modules:
                raise VerilogParseError("No modules found in Verilog code")
            
            # Create AST
            ast = CircuitAST(modules=modules)
            
            # Validate AST
            errors = ast.validate()
            if errors:
                raise VerilogParseError(f"AST validation failed: {'; '.join(errors)}")
            
            return ast
            
        except Exception as e:
            raise VerilogParseError(f"Failed to parse Verilog: {str(e)}") from e
    
    def _remove_comments(self, code: str) -> str:
        """Remove single-line and multi-line comments."""
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code
    
    def _parse_modules(self, code: str) -> List[Module]:
        """Parse all modules from code."""
        modules = []
        
        for match in self.module_pattern.finditer(code):
            module_name = match.group(1)
            port_list = match.group(2) if match.group(2) else ""
            
            # Find module body
            module_start = match.end()
            module_body = self._extract_module_body(code, module_start)
            
            # Parse module components
            ports = self._parse_ports(port_list, module_body)
            signals = self._parse_signals(module_body)
            assignments = self._parse_assignments(module_body)
            always_blocks = self._parse_always_blocks(module_body)
            instances = self._parse_instances(module_body)
            
            module = Module(
                name=module_name,
                ports=ports,
                signals=signals,
                assignments=assignments,
                always_blocks=always_blocks,
                submodules=instances,
                parameters={}
            )
            
            modules.append(module)
        
        return modules
    
    def _extract_module_body(self, code: str, start_pos: int) -> str:
        """Extract module body from start position to endmodule."""
        lines = code[start_pos:].split('\n')
        body_lines = []
        
        for line in lines:
            if line.strip().startswith('endmodule'):
                break
            body_lines.append(line)
        
        return '\n'.join(body_lines)
    
    def _parse_ports(self, port_list: str, module_body: str) -> List[Port]:
        """Parse module ports."""
        ports = []
        
        # Parse port directions from module body
        port_directions = {}
        for match in self.port_pattern.finditer(module_body):
            direction = match.group(1)
            msb = int(match.group(2)) if match.group(2) else None
            lsb = int(match.group(3)) if match.group(3) else None
            name = match.group(4)
            
            signal_type = SignalType(direction)
            width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
            
            port = Port(
                name=name,
                signal_type=signal_type,
                width=width,
                msb=msb,
                lsb=lsb
            )
            ports.append(port)
            port_directions[name] = port
        
        # If ports are declared in port list only, infer from usage
        if not ports and port_list.strip():
            port_names = [name.strip() for name in port_list.split(',') if name.strip()]
            for name in port_names:
                # Try to find direction in module body
                if name in port_directions:
                    ports.append(port_directions[name])
                else:
                    # Default to input if not found
                    ports.append(Port(name=name, signal_type=SignalType.INPUT))
        
        return ports
    
    def _parse_signals(self, module_body: str) -> List[Signal]:
        """Parse wire and reg declarations."""
        signals = []
        
        # Parse wire declarations
        for match in self.wire_pattern.finditer(module_body):
            msb = int(match.group(1)) if match.group(1) else None
            lsb = int(match.group(2)) if match.group(2) else None
            name = match.group(3)
            
            width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
            signal = Signal(name=name, signal_type=SignalType.WIRE, width=width)
            signals.append(signal)
        
        # Parse reg declarations
        for match in self.reg_pattern.finditer(module_body):
            msb = int(match.group(1)) if match.group(1) else None
            lsb = int(match.group(2)) if match.group(2) else None
            name = match.group(3)
            
            width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
            signal = Signal(name=name, signal_type=SignalType.REG, width=width)
            signals.append(signal)
        
        return signals
    
    def _parse_assignments(self, module_body: str) -> List[Assignment]:
        """Parse assign statements."""
        assignments = []
        
        for match in self.assign_pattern.finditer(module_body):
            target = match.group(1)
            expression = match.group(2).strip()
            
            assignment = Assignment(
                target=target,
                expression=expression,
                is_blocking=False  # assign statements are non-blocking
            )
            assignments.append(assignment)
        
        return assignments
    
    def _parse_always_blocks(self, module_body: str) -> List[AlwaysBlock]:
        """Parse always blocks."""
        always_blocks = []
        
        for match in self.always_pattern.finditer(module_body):
            sensitivity = match.group(1).strip()
            block_body = match.group(3) if match.group(3) else ""
            
            # Parse sensitivity list
            sensitivity_list = [s.strip() for s in sensitivity.split(',') if s.strip()]
            
            # Parse statements within always block
            statements = self._parse_always_statements(block_body)
            
            # Determine block type
            block_type = "sequential" if any("posedge" in s or "negedge" in s 
                                           for s in sensitivity_list) else "combinational"
            
            always_block = AlwaysBlock(
                sensitivity_list=sensitivity_list,
                statements=statements,
                block_type=block_type
            )
            always_blocks.append(always_block)
        
        return always_blocks
    
    def _parse_always_statements(self, block_body: str) -> List[Assignment]:
        """Parse statements within always block."""
        statements = []
        
        # Simple assignment parsing within always blocks
        # This is a simplified version - real parser would handle more complex statements
        assignment_pattern = re.compile(r'(\w+)\s*[<]?=\s*([^;]+)\s*;')
        
        for match in assignment_pattern.finditer(block_body):
            target = match.group(1)
            expression = match.group(2).strip()
            is_blocking = "<=" not in match.group(0)  # <= is non-blocking
            
            assignment = Assignment(
                target=target,
                expression=expression,
                is_blocking=is_blocking
            )
            statements.append(assignment)
        
        return statements
    
    def _parse_instances(self, module_body: str) -> List[ModuleInstance]:
        """Parse module instantiations."""
        instances = []
        
        for match in self.instance_pattern.finditer(module_body):
            module_name = match.group(1)
            instance_name = match.group(2)
            connections = match.group(3)
            
            # Skip if this looks like a built-in gate or keyword
            if module_name.lower() in ['and', 'or', 'not', 'buf', 'wire', 'reg', 'input', 'output']:
                continue
            
            # Parse port connections
            port_connections = self._parse_port_connections(connections)
            
            instance = ModuleInstance(
                instance_name=instance_name,
                module_name=module_name,
                port_connections=port_connections,
                parameter_overrides={}
            )
            instances.append(instance)
        
        return instances
    
    def _parse_port_connections(self, connections: str) -> Dict[str, str]:
        """Parse port connections in module instantiation."""
        port_map = {}
        
        if not connections.strip():
            return port_map
        
        # Handle named port connections (.port(signal))
        named_pattern = re.compile(r'\.(\w+)\s*\(\s*(\w+)\s*\)')
        for match in named_pattern.finditer(connections):
            port_name = match.group(1)
            signal_name = match.group(2)
            port_map[port_name] = signal_name
        
        # Handle positional connections (simplified)
        if not port_map:
            signals = [s.strip() for s in connections.split(',') if s.strip()]
            for i, signal in enumerate(signals):
                port_map[f"port_{i}"] = signal
        
        return port_map