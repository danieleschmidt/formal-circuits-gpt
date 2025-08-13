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
            r'always\s*@\s*\(([^)]+)\)\s*(begin)?\s*(.*?)(?:end\s*)?', 
            re.MULTILINE | re.DOTALL
        )
        # Enhanced patterns for better parsing
        self.reg_declaration_pattern = re.compile(
            r'reg\s+(?:\[(\d+):(\d+)\]\s*)?([\w,\s]+)\s*;',
            re.MULTILINE
        )
        self.wire_declaration_pattern = re.compile(
            r'wire\s+(?:\[(\d+):(\d+)\]\s*)?([\w,\s]+)\s*;',
            re.MULTILINE
        )
        self.instance_pattern = re.compile(
            r'(\w+)\s+(?:#\([^)]*\))?\s*(\w+)\s*\(([^)]*)\)\s*;',
            re.MULTILINE | re.DOTALL
        )
    
    def parse(self, verilog_code: str) -> CircuitAST:
        """Parse Verilog code into AST with enhanced error handling.
        
        Args:
            verilog_code: Verilog source code string
            
        Returns:
            CircuitAST representation of the code
            
        Raises:
            VerilogParseError: If parsing fails
        """
        try:
            # Input validation
            if not verilog_code or not verilog_code.strip():
                raise VerilogParseError("Verilog code cannot be empty")
            
            # Remove comments
            cleaned_code = self._remove_comments(verilog_code)
            
            # Try to recover from malformed input
            cleaned_code = self._attempt_basic_recovery(cleaned_code)
            
            # Extract modules
            modules = self._parse_modules(cleaned_code)
            
            if not modules:
                # Try alternative parsing strategies
                modules = self._try_alternative_parsing(cleaned_code)
                
            if not modules:
                raise VerilogParseError("No modules found in Verilog code")
            
            # Create AST
            ast = CircuitAST(modules=modules)
            
            # Validate AST
            errors = ast.validate()
            if errors:
                raise VerilogParseError(f"AST validation failed: {'; '.join(errors)}")
            
            return ast
            
        except VerilogParseError:
            # Re-raise known parsing errors
            raise
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
        """Parse module ports with enhanced robustness."""
        ports = []
        seen_ports = set()  # Track port names to avoid duplicates
        
        # Parse port declarations from port list (Verilog-2001 style)
        combined_text = port_list + "\n" + module_body
        
        # Enhanced regex for better port matching
        enhanced_port_pattern = re.compile(
            r'(input|output|inout)\s+(?:(wire|reg)\s+)?(?:\[(\d+):(\d+)\]\s*)?([\w,\s]+)',
            re.MULTILINE | re.IGNORECASE
        )
        
        for match in enhanced_port_pattern.finditer(combined_text):
            direction = match.group(1).lower()
            signal_kind = match.group(2)  # wire/reg qualifier
            msb = int(match.group(3)) if match.group(3) else None
            lsb = int(match.group(4)) if match.group(4) else None
            names_str = match.group(5)
            
            # Handle multiple port names in one declaration
            port_names = [name.strip() for name in names_str.split(',') if name.strip()]
            
            for name in port_names:
                if name in seen_ports:
                    continue  # Skip duplicate ports
                seen_ports.add(name)
                
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
        
        # Fallback: Parse from simple port list if no declarations found
        if not ports and port_list.strip():
            simple_names = [name.strip() for name in port_list.split(',') if name.strip()]
            for name in simple_names:
                if name not in seen_ports:
                    # Default to input if not specified
                    port = Port(name=name, signal_type=SignalType.INPUT, width=1)
                    ports.append(port)
                    seen_ports.add(name)
        
        return ports
    
    def _parse_signals(self, module_body: str) -> List[Signal]:
        """Parse wire and reg declarations with enhanced support."""
        signals = []
        seen_signals = set()
        
        # Parse wire declarations with multiple names
        for match in self.wire_declaration_pattern.finditer(module_body):
            msb = int(match.group(1)) if match.group(1) else None
            lsb = int(match.group(2)) if match.group(2) else None
            names_str = match.group(3)
            
            signal_names = [name.strip() for name in names_str.split(',') if name.strip()]
            
            for name in signal_names:
                if name not in seen_signals:
                    width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
                    signal = Signal(name=name, signal_type=SignalType.WIRE, width=width)
                    signals.append(signal)
                    seen_signals.add(name)
        
        # Parse reg declarations with multiple names
        for match in self.reg_declaration_pattern.finditer(module_body):
            msb = int(match.group(1)) if match.group(1) else None
            lsb = int(match.group(2)) if match.group(2) else None
            names_str = match.group(3)
            
            signal_names = [name.strip() for name in names_str.split(',') if name.strip()]
            
            for name in signal_names:
                if name not in seen_signals:
                    width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
                    signal = Signal(name=name, signal_type=SignalType.REG, width=width)
                    signals.append(signal)
                    seen_signals.add(name)
        
        # Fallback to old patterns for compatibility
        for match in self.wire_pattern.finditer(module_body):
            name = match.group(3)
            if name not in seen_signals:
                msb = int(match.group(1)) if match.group(1) else None
                lsb = int(match.group(2)) if match.group(2) else None
                width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
                signal = Signal(name=name, signal_type=SignalType.WIRE, width=width)
                signals.append(signal)
                seen_signals.add(name)
        
        for match in self.reg_pattern.finditer(module_body):
            name = match.group(3)
            if name not in seen_signals:
                msb = int(match.group(1)) if match.group(1) else None
                lsb = int(match.group(2)) if match.group(2) else None
                width = abs(msb - lsb) + 1 if msb is not None and lsb is not None else 1
                signal = Signal(name=name, signal_type=SignalType.REG, width=width)
                signals.append(signal)
                seen_signals.add(name)
        
        return signals
    
    def _attempt_basic_recovery(self, code: str) -> str:
        """Attempt to recover from common malformed Verilog patterns."""
        if not code or not code.strip():
            return code
            
        # Remove null bytes and control characters
        code = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', code)
        
        # Fix common missing semicolons
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.endswith((';', '{', '}', ')', ',')):
                # Check if it looks like a statement that needs a semicolon
                if any(keyword in stripped for keyword in ['assign', 'wire', 'reg', 'input', 'output']):
                    if not stripped.endswith(':'):  # Don't add semicolon to labels
                        line = line + ';'
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _try_alternative_parsing(self, code: str) -> List[Module]:
        """Try alternative parsing strategies for difficult cases."""
        modules = []
        
        # Strategy 1: Look for module-like structures without strict syntax
        module_candidates = re.findall(r'(\w+)\s*\([^)]*\)', code)
        if module_candidates:
            # Create a minimal module from the first candidate
            name = module_candidates[0]
            module = Module(
                name=name,
                ports=[],
                signals=[],
                assignments=[],
                always_blocks=[],
                submodules=[],
                parameters={}
            )
            modules.append(module)
        
        # Strategy 2: Create a default module if we find any HDL-like content
        if not modules and any(keyword in code.lower() for keyword in ['assign', 'wire', 'reg', 'input', 'output']):
            module = Module(
                name="recovered_module",
                ports=[],
                signals=[],
                assignments=self._parse_assignments(code),
                always_blocks=[],
                submodules=[],
                parameters={}
            )
            modules.append(module)
        
        return modules
    
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