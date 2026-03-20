"""Verilog parser using regex-based extraction."""

import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Port:
    name: str
    direction: str  # "input", "output", "inout"
    width: int = 1  # bit width
    is_reg: bool = False

    def __repr__(self):
        w = f"[{self.width-1}:0] " if self.width > 1 else ""
        reg = " reg" if self.is_reg else ""
        return f"Port({self.direction}{reg} {w}{self.name})"


@dataclass
class AlwaysBlock:
    sensitivity: str  # "posedge clk", "*", etc.
    body: str
    block_type: str  # "combinational" or "sequential"

    def __repr__(self):
        return f"AlwaysBlock({self.block_type}, @({self.sensitivity}))"


@dataclass
class VerilogModule:
    name: str
    ports: List[Port]
    always_blocks: List[AlwaysBlock]
    parameters: Dict[str, str]
    raw_text: str

    def __repr__(self):
        return f"VerilogModule({self.name}, ports={len(self.ports)}, always={len(self.always_blocks)})"


class VerilogParser:
    """Regex-based Verilog module parser."""

    def parse(self, verilog_text: str) -> VerilogModule:
        """Parse Verilog text into a VerilogModule."""
        name = self.extract_module_name(verilog_text)
        ports = self.extract_ports(verilog_text)
        always_blocks = self.extract_always_blocks(verilog_text)
        parameters = self.extract_parameters(verilog_text)
        return VerilogModule(
            name=name,
            ports=ports,
            always_blocks=always_blocks,
            parameters=parameters,
            raw_text=verilog_text,
        )

    def parse_file(self, path: str) -> VerilogModule:
        """Parse a Verilog file."""
        with open(path, "r") as f:
            text = f.read()
        return self.parse(text)

    def extract_module_name(self, text: str) -> str:
        """Extract module name from Verilog text."""
        # Match: module <name> (  or  module <name>\n
        m = re.search(r'\bmodule\s+(\w+)\s*[(\s#]', text)
        if m:
            return m.group(1)
        # Fallback: module <name> at end of line
        m = re.search(r'\bmodule\s+(\w+)', text)
        if m:
            return m.group(1)
        return "unknown"

    def extract_ports(self, text: str) -> List[Port]:
        """Extract port declarations from Verilog text."""
        ports = []
        seen = set()

        # Remove comments
        text_clean = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text_clean = re.sub(r'/\*.*?\*/', '', text_clean, flags=re.DOTALL)

        # Pattern for port declarations: direction [reg] [width] name[, name]*
        # e.g.: input [7:0] a, b, c
        #       output reg [7:0] count
        port_pattern = re.compile(
            r'\b(input|output|inout)\s+'
            r'(reg\s+)?'
            r'(?:\[(\d+)\s*:\s*(\d+)\]\s*)?'
            r'([\w\s,]+?)(?=;|\)|\binput\b|\boutput\b|\binout\b)',
            re.MULTILINE
        )

        for m in port_pattern.finditer(text_clean):
            direction = m.group(1)
            is_reg = bool(m.group(2))
            msb = int(m.group(3)) if m.group(3) is not None else 0
            lsb = int(m.group(4)) if m.group(4) is not None else 0
            width = abs(msb - lsb) + 1 if m.group(3) is not None else 1
            names_str = m.group(5)

            # Extract individual port names
            names = [n.strip() for n in names_str.split(',') if n.strip() and re.match(r'^\w+$', n.strip())]
            for name in names:
                if name not in seen and name not in ('input', 'output', 'inout', 'reg', 'wire', 'integer'):
                    seen.add(name)
                    ports.append(Port(
                        name=name,
                        direction=direction,
                        width=width,
                        is_reg=is_reg,
                    ))

        return ports

    def extract_always_blocks(self, text: str) -> List[AlwaysBlock]:
        """Extract always blocks from Verilog text."""
        blocks = []

        # Remove comments
        text_clean = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text_clean = re.sub(r'/\*.*?\*/', '', text_clean, flags=re.DOTALL)

        # Find all always @(...) blocks
        always_pattern = re.compile(r'\balways\s*@\s*\(([^)]+)\)\s*(begin\b.*?end\b|[^;]+;)', re.DOTALL)

        for m in always_pattern.finditer(text_clean):
            sensitivity = m.group(1).strip()
            body = m.group(2).strip()
            block_type = "sequential" if self.is_sequential_sensitivity(sensitivity) else "combinational"
            blocks.append(AlwaysBlock(
                sensitivity=sensitivity,
                body=body,
                block_type=block_type,
            ))

        return blocks

    def extract_parameters(self, text: str) -> Dict[str, str]:
        """Extract parameter declarations."""
        params = {}
        # Match: parameter NAME = value
        param_pattern = re.compile(r'\bparameter\s+(\w+)\s*=\s*([^,;)]+)', re.MULTILINE)
        for m in param_pattern.finditer(text):
            params[m.group(1)] = m.group(2).strip()
        return params

    def is_sequential(self, always_block: AlwaysBlock) -> bool:
        """Check if an always block is sequential (has posedge/negedge)."""
        return self.is_sequential_sensitivity(always_block.sensitivity)

    def is_sequential_sensitivity(self, sensitivity: str) -> bool:
        """Check if a sensitivity string indicates sequential logic."""
        return bool(re.search(r'\b(posedge|negedge)\b', sensitivity, re.IGNORECASE))
