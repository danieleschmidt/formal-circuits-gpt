"""Formal report generation."""

import json
from typing import List, Optional
from .verilog_parser import VerilogModule
from .property_gen import Property
from .refiner import RefinementStep


class FormalReport:
    """Generate human-readable and machine-readable formal verification reports."""

    def __init__(
        self,
        module: VerilogModule,
        properties: List[Property],
        sva_spec: str,
        refinement_steps: Optional[List[RefinementStep]] = None,
    ):
        self.module = module
        self.properties = properties
        self.sva_spec = sva_spec
        self.refinement_steps = refinement_steps or []

    def to_dict(self) -> dict:
        """Serialize the report to a dict."""
        return {
            "module": {
                "name": self.module.name,
                "ports": [
                    {
                        "name": p.name,
                        "direction": p.direction,
                        "width": p.width,
                        "is_reg": p.is_reg,
                    }
                    for p in self.module.ports
                ],
                "always_blocks": [
                    {
                        "sensitivity": ab.sensitivity,
                        "block_type": ab.block_type,
                    }
                    for ab in self.module.always_blocks
                ],
                "parameters": self.module.parameters,
            },
            "properties": [
                {
                    "name": p.name,
                    "type": p.prop_type,
                    "description": p.description,
                    "formal_expr": p.formal_expr,
                    "confidence": p.confidence,
                }
                for p in self.properties
            ],
            "sva_spec": self.sva_spec,
            "refinement_steps": [
                {
                    "iteration": s.iteration,
                    "issues_found": s.issues_found,
                    "improvement_score": s.improvement_score,
                }
                for s in self.refinement_steps
            ],
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Generate a human-readable Markdown report."""
        lines = []

        lines.append(f"# Formal Verification Report: `{self.module.name}`")
        lines.append("")

        # Module summary
        lines.append("## Module Summary")
        lines.append(f"- **Name:** `{self.module.name}`")
        lines.append(f"- **Ports:** {len(self.module.ports)}")
        lines.append(f"- **Always blocks:** {len(self.module.always_blocks)}")
        lines.append(f"- **Parameters:** {len(self.module.parameters)}")
        has_seq = any(ab.block_type == "sequential" for ab in self.module.always_blocks)
        lines.append(f"- **Logic type:** {'Sequential (clocked)' if has_seq else 'Combinational'}")
        lines.append("")

        # Port list
        lines.append("## Ports")
        lines.append("| Name | Direction | Width | Reg |")
        lines.append("|------|-----------|-------|-----|")
        for p in self.module.ports:
            lines.append(f"| `{p.name}` | {p.direction} | {p.width} | {'yes' if p.is_reg else 'no'} |")
        lines.append("")

        # Parameters
        if self.module.parameters:
            lines.append("## Parameters")
            for name, val in self.module.parameters.items():
                lines.append(f"- `{name}` = `{val}`")
            lines.append("")

        # Properties table
        lines.append("## Formal Properties")
        lines.append(f"Total: {len(self.properties)}")
        lines.append("")
        lines.append("| # | Name | Type | Confidence | Description |")
        lines.append("|---|------|------|-----------|-------------|")
        for i, prop in enumerate(self.properties, 1):
            conf_pct = f"{int(prop.confidence * 100)}%"
            lines.append(f"| {i} | `{prop.name}` | {prop.prop_type} | {conf_pct} | {prop.description[:80]} |")
        lines.append("")

        # Plain English descriptions
        lines.append("## Property Descriptions (Plain English)")
        for prop in self.properties:
            lines.append(f"### `{prop.name}`")
            lines.append(f"- **Type:** {prop.prop_type}")
            lines.append(f"- **Confidence:** {int(prop.confidence * 100)}%")
            lines.append(f"- **Description:** {prop.description}")
            lines.append(f"```systemverilog")
            lines.append(prop.formal_expr)
            lines.append("```")
            lines.append("")

        # SVA spec
        lines.append("## Generated SVA Specification")
        lines.append("```systemverilog")
        lines.append(self.sva_spec)
        lines.append("```")
        lines.append("")

        # Refinement steps
        if self.refinement_steps:
            lines.append("## Refinement History")
            for step in self.refinement_steps:
                lines.append(f"### Iteration {step.iteration}")
                lines.append(f"- **Issues found:** {len(step.issues_found)}")
                for issue in step.issues_found:
                    lines.append(f"  - {issue}")
                lines.append(f"- **Improvement score:** {step.improvement_score:.0%}")
                lines.append("")

        return "\n".join(lines)

    def save(self, path: str, format: str = "json") -> None:
        """Save report to file in the given format."""
        if format == "json":
            content = self.to_json()
        elif format == "markdown" or format == "md":
            content = self.to_markdown()
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'markdown'.")

        with open(path, "w") as f:
            f.write(content)
