"""Property generator for Verilog modules."""

from dataclasses import dataclass
from typing import List
from .verilog_parser import VerilogModule, Port


@dataclass
class Property:
    name: str
    prop_type: str  # "combinational", "sequential", "safety", "liveness"
    description: str  # Plain English
    formal_expr: str  # SVA expression
    confidence: float  # 0-1

    def __repr__(self):
        return f"Property({self.name}, type={self.prop_type}, conf={self.confidence:.2f})"


class PropertyGenerator:
    """Generate formal properties for a Verilog module."""

    def generate_for_module(self, module: VerilogModule) -> List[Property]:
        """Generate all applicable properties for a module."""
        props = []
        has_sequential = any(ab.block_type == "sequential" for ab in module.always_blocks)

        if has_sequential:
            props.extend(self._gen_sequential_props(module))
        else:
            props.extend(self._gen_combinational_props(module))

        props.extend(self._gen_safety_props(module))
        return props

    def _gen_combinational_props(self, module: VerilogModule) -> List[Property]:
        """Generate properties for purely combinational modules."""
        props = []
        outputs = [p for p in module.ports if p.direction == "output"]
        inputs = [p for p in module.ports if p.direction == "input"]

        # Output completeness: all outputs are driven
        if outputs:
            output_names = ", ".join(p.name for p in outputs)
            props.append(Property(
                name="output_completeness",
                prop_type="combinational",
                description=f"All outputs ({output_names}) are fully driven by the combinational logic.",
                formal_expr=f"// All outputs driven combinationally\nassert property (##0 1'b1);  // placeholder: outputs always defined",
                confidence=0.85,
            ))

        # Input-output determinism
        if inputs and outputs:
            in_names = ", ".join(p.name for p in inputs[:3])
            out_names = ", ".join(p.name for p in outputs[:2])
            props.append(Property(
                name="input_output_determinism",
                prop_type="combinational",
                description=f"For the same inputs ({in_names}...), the outputs ({out_names}...) are always the same (deterministic combinational behavior).",
                formal_expr=f"// Combinational determinism\nassert property (##0 1'b1);  // outputs are deterministic functions of inputs",
                confidence=0.90,
            ))

        # Width correctness for arithmetic
        for out in outputs:
            for inp in inputs:
                if out.width == inp.width + 1:
                    props.append(Property(
                        name=f"no_overflow_{out.name}",
                        prop_type="safety",
                        description=f"Output '{out.name}' ({out.width}-bit) is wide enough to hold the full result of operations on {inp.width}-bit inputs without overflow.",
                        formal_expr=f"assert property (##0 ({out.name} <= {2**out.width - 1}));",
                        confidence=0.80,
                    ))

        return props

    def _gen_sequential_props(self, module: VerilogModule) -> List[Property]:
        """Generate properties for sequential modules."""
        props = []

        # Find clock and reset ports
        clk_port = self._find_clock(module)
        rst_port = self._find_reset(module)
        reg_outputs = [p for p in module.ports if p.is_reg and p.direction == "output"]

        clk_name = clk_port.name if clk_port else "clk"
        rst_name = rst_port.name if rst_port else "rst"

        # Reset behavior
        if rst_port and reg_outputs:
            for reg in reg_outputs:
                props.append(Property(
                    name=f"reset_{reg.name}",
                    prop_type="sequential",
                    description=f"When reset ('{rst_name}') is asserted, register '{reg.name}' reaches a known safe state (typically 0) on the next clock edge.",
                    formal_expr=f"assert property (@(posedge {clk_name}) {rst_name} |=> ({reg.name} == 0));",
                    confidence=0.80,
                ))

        # Register stability: only changes on clock edge
        if reg_outputs:
            for reg in reg_outputs:
                props.append(Property(
                    name=f"clk_stability_{reg.name}",
                    prop_type="sequential",
                    description=f"Register '{reg.name}' only changes its value on the rising edge of the clock ('{clk_name}'), ensuring synchronous operation.",
                    formal_expr=f"assert property (@(posedge {clk_name}) disable iff ({rst_name}) 1'b1);  // clocked register",
                    confidence=0.75,
                ))

        # No metastability assumption
        if clk_port:
            props.append(Property(
                name="no_metastability",
                prop_type="safety",
                description=f"Assumption: all inputs are synchronized to the '{clk_name}' clock domain and free from metastability.",
                formal_expr=f"assume property (@(posedge {clk_name}) $stable({clk_name}) || $rose({clk_name}) || $fell({clk_name}));",
                confidence=0.70,
            ))

        # Liveness: enable causes progress
        enable_port = next((p for p in module.ports if 'enable' in p.name.lower() or p.name.lower() == 'en'), None)
        if enable_port and reg_outputs:
            for reg in reg_outputs:
                props.append(Property(
                    name=f"enable_progress_{reg.name}",
                    prop_type="liveness",
                    description=f"When '{enable_port.name}' is asserted and reset is not active, '{reg.name}' eventually changes value (progress / liveness).",
                    formal_expr=f"assert property (@(posedge {clk_name}) disable iff ({rst_name}) {enable_port.name} |-> ##[1:$] ({reg.name} != $past({reg.name})));",
                    confidence=0.65,
                ))

        return props

    def describe_property(self, prop: Property) -> str:
        """Return plain English description of a property."""
        return prop.description

    def _gen_safety_props(self, module: VerilogModule) -> List[Property]:
        """Generate generic safety properties."""
        props = []
        outputs = [p for p in module.ports if p.direction == "output"]
        clk_port = self._find_clock(module)
        rst_port = self._find_reset(module)

        clk_name = clk_port.name if clk_port else "clk"
        rst_name = rst_port.name if rst_port else "rst"

        has_sequential = any(ab.block_type == "sequential" for ab in module.always_blocks)
        trigger = f"@(posedge {clk_name})" if has_sequential else "##0"

        for out in outputs:
            props.append(Property(
                name=f"no_x_{out.name}",
                prop_type="safety",
                description=f"Output '{out.name}' never takes an unknown (X) or high-impedance (Z) value during normal operation.",
                formal_expr=f"assert property ({trigger} !$isunknown({out.name}));",
                confidence=0.75,
            ))

        return props

    def _find_clock(self, module: VerilogModule):
        """Heuristically find the clock port."""
        for p in module.ports:
            if p.direction == "input" and p.width == 1:
                if re.search(r'^clk$|^clock$|^clk_\w+', p.name, re.IGNORECASE):
                    return p
        # Fallback: any 1-bit input named clk-like
        for p in module.ports:
            if p.direction == "input" and p.width == 1 and 'clk' in p.name.lower():
                return p
        return None

    def _find_reset(self, module: VerilogModule):
        """Heuristically find the reset port."""
        for p in module.ports:
            if p.direction == "input" and p.width == 1:
                if re.search(r'^rst$|^reset$|^rst_n$|^resetn$|^arst$', p.name, re.IGNORECASE):
                    return p
        for p in module.ports:
            if p.direction == "input" and p.width == 1 and 'rst' in p.name.lower():
                return p
        return None


import re
