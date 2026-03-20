"""Self-refining agent for iterative SVA spec improvement."""

import re
from dataclasses import dataclass, field
from typing import List
from .verilog_parser import VerilogModule


@dataclass
class RefinementStep:
    iteration: int
    input_spec: str
    issues_found: List[str]
    refined_spec: str
    improvement_score: float  # 0-1, higher = better


class SelfRefiner:
    """
    Iterative refinement of formal specs.
    In absence of LLM, does rule-based refinement:
    - Checks for missing assumptions
    - Checks for vacuous assertions
    - Adds missing clock/reset infrastructure
    """

    KNOWN_ISSUES = {
        "missing_clock_assumption": "Missing clock assumption: no `assume` for clock signal",
        "missing_reset_assumption": "Missing reset assumption: no constraint on reset signal",
        "vacuous_assertion": "Potentially vacuous assertion: `1'b1` used as assertion body",
        "missing_port_declarations": "Missing port declarations in spec module",
        "no_cover_properties": "No cover properties for reachability checking",
    }

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations

    def refine(self, spec: str, module: VerilogModule) -> List[RefinementStep]:
        """Iteratively refine a spec, returning the full refinement history."""
        history = []
        current_spec = spec

        for i in range(1, self.max_iterations + 1):
            issues = self.check_spec(current_spec)
            if not issues:
                # Nothing left to fix
                history.append(RefinementStep(
                    iteration=i,
                    input_spec=current_spec,
                    issues_found=[],
                    refined_spec=current_spec,
                    improvement_score=1.0,
                ))
                break

            # Apply fixes
            refined = current_spec
            for issue in issues:
                refined = self.apply_fix(refined, issue)

            # Score: fraction of issues resolved
            remaining = self.check_spec(refined)
            resolved = len(issues) - len(remaining)
            score = resolved / len(issues) if issues else 1.0

            history.append(RefinementStep(
                iteration=i,
                input_spec=current_spec,
                issues_found=list(issues),
                refined_spec=refined,
                improvement_score=round(score, 2),
            ))

            if refined == current_spec:
                # No changes made; stop
                break

            current_spec = refined

        return history

    def check_spec(self, spec: str) -> List[str]:
        """Identify issues in the SVA spec string."""
        issues = []

        # Check for clock assumption
        if "posedge" in spec and "assume" not in spec:
            issues.append("missing_clock_assumption")

        # Check for reset assumption
        if re.search(r'\b(rst|reset)\b', spec) and "assume_rst" not in spec and "assume_reset" not in spec:
            issues.append("missing_reset_assumption")

        # Check for vacuous assertions (body is just 1'b1)
        if re.search(r'assert\s+property\s*\([^)]*1\'b1\s*\)', spec):
            issues.append("vacuous_assertion")

        # Check for port declarations
        if "endmodule" in spec and not re.search(r'\b(input|output|inout)\b', spec):
            issues.append("missing_port_declarations")

        # Check for cover properties
        if "cover property" not in spec:
            issues.append("no_cover_properties")

        return issues

    def apply_fix(self, spec: str, issue: str) -> str:
        """Apply a rule-based fix for a known issue type."""
        if issue == "missing_clock_assumption":
            # Add clock assumption before endmodule
            insertion = (
                "\n  // [Refiner] Added clock assumption\n"
                "  assume_clk_period: assume property (@(posedge clk) ##1 1'b1);  // clock is toggling\n"
            )
            spec = spec.replace("endmodule", insertion + "endmodule", 1)

        elif issue == "missing_reset_assumption":
            insertion = (
                "\n  // [Refiner] Added reset assumption\n"
                "  assume_rst_sync: assume property (@(posedge clk) $stable(rst) || $rose(rst) || $fell(rst));\n"
            )
            spec = spec.replace("endmodule", insertion + "endmodule", 1)

        elif issue == "vacuous_assertion":
            # Add a comment flagging vacuous assertions
            spec = re.sub(
                r'(assert\s+property\s*\([^)]*1\'b1\s*\))',
                r'// [Refiner] WARNING: possibly vacuous assertion below\n  \1',
                spec,
            )

        elif issue == "missing_port_declarations":
            insertion = (
                "\n  // [Refiner] Port declarations missing — add them manually\n"
                "  // Example: input clk; input rst; ...\n"
            )
            # Insert after first line of module
            spec = re.sub(r'(module \w+_spec\([^)]*\);)', r'\1' + insertion, spec)

        elif issue == "no_cover_properties":
            insertion = (
                "\n  // [Refiner] Added basic cover property\n"
                "  cover_basic: cover property (##[1:10] 1'b1);  // at least one cycle of activity\n"
            )
            spec = spec.replace("endmodule", insertion + "endmodule", 1)

        return spec
