"""Translator from HDL AST to Isabelle/HOL specifications."""

from typing import List, Dict, Any, Optional
from ..parsers.ast_nodes import CircuitAST, Module, Port, Signal, Assignment, SignalType


class IsabelleTranslationError(Exception):
    """Exception raised for Isabelle translation errors."""

    pass


class IsabelleTranslator:
    """Translates HDL AST to Isabelle/HOL theory files."""

    def __init__(self):
        """Initialize the Isabelle translator."""
        self.theory_name = "Circuit_Verification"
        self.imports = ["Main", "HOL.Nat", "HOL.Bool"]

    def translate(self, ast: CircuitAST, theory_name: Optional[str] = None) -> str:
        """Translate AST to Isabelle theory.

        Args:
            ast: Circuit AST to translate
            theory_name: Name for the theory (optional)

        Returns:
            Isabelle theory file content

        Raises:
            IsabelleTranslationError: If translation fails
        """
        try:
            if theory_name:
                self.theory_name = theory_name

            theory_content = []

            # Theory header
            theory_content.append(self._generate_theory_header())

            # Type definitions
            theory_content.append(self._generate_type_definitions(ast))

            # Function definitions for each module
            for module in ast.modules:
                theory_content.append(self._translate_module(module))

            # Property definitions
            theory_content.append(self._generate_properties(ast))

            # Theory footer
            theory_content.append("end")

            return "\n\n".join(theory_content)

        except Exception as e:
            raise IsabelleTranslationError(
                f"Failed to translate to Isabelle: {str(e)}"
            ) from e

    def _generate_theory_header(self) -> str:
        """Generate Isabelle theory header."""
        imports_str = " ".join(self.imports)
        return f"""theory {self.theory_name}
  imports {imports_str}
begin"""

    def _generate_type_definitions(self, ast: CircuitAST) -> str:
        """Generate type definitions for signals and values."""
        content = []

        # Basic bit vector type
        content.append("(* Bit vector type for hardware signals *)")
        content.append("type_synonym bit = bool")
        content.append("type_synonym 'a bitvec = \"'a list\"")

        # Signal state type for each module
        for module in ast.modules:
            if module.ports or module.signals:
                content.append(f"")
                content.append(f"(* State type for module {module.name} *)")

                fields = []
                for port in module.ports:
                    if port.width > 1:
                        fields.append(f'  {port.name} :: "nat bitvec"')
                    else:
                        fields.append(f"  {port.name} :: bit")

                for signal in module.signals:
                    if signal.width > 1:
                        fields.append(f'  {signal.name} :: "nat bitvec"')
                    else:
                        fields.append(f"  {signal.name} :: bit")

                if fields:
                    content.append("record " + f"{module.name}_state =")
                    content.extend(fields)

        return "\n".join(content)

    def _translate_module(self, module: Module) -> str:
        """Translate a single module to Isabelle functions."""
        content = []

        content.append(f"(* Module: {module.name} *)")

        # Create module function signature
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]

        if input_ports and output_ports:
            # Function signature
            sig_parts = []
            for port in input_ports:
                if port.width > 1:
                    sig_parts.append(f'{port.name} :: "nat bitvec"')
                else:
                    sig_parts.append(f"{port.name} :: bit")

            return_type = self._generate_return_type(output_ports)
            signature = " ⇒ ".join(sig_parts + [return_type])

            content.append(f'fun {module.name} :: "{signature}" where')

            # Function definition
            input_vars = " ".join(p.name for p in input_ports)
            function_body = self._generate_function_body(
                module, input_ports, output_ports
            )

            content.append(f'"{module.name} {input_vars} = {function_body}"')

        # Translate assignments as lemmas
        if module.assignments:
            content.append("")
            content.append(f"(* Assignments for module {module.name} *)")
            for i, assignment in enumerate(module.assignments):
                lemma_name = f"{module.name}_assign_{i+1}"
                lemma = self._translate_assignment_to_lemma(
                    assignment, module, lemma_name
                )
                content.append(lemma)

        return "\n".join(content)

    def _generate_return_type(self, output_ports: List[Port]) -> str:
        """Generate return type for module function."""
        if len(output_ports) == 1:
            port = output_ports[0]
            if port.width > 1:
                return '"nat bitvec"'
            else:
                return "bit"
        else:
            # Multiple outputs - use tuple
            types = []
            for port in output_ports:
                if port.width > 1:
                    types.append('"nat bitvec"')
                else:
                    types.append("bit")
            return f"({' × '.join(types)})"

    def _generate_function_body(
        self, module: Module, input_ports: List[Port], output_ports: List[Port]
    ) -> str:
        """Generate function body for module."""
        # Simple implementation - translate assignments
        if len(output_ports) == 1 and module.assignments:
            # Find assignment for output
            output_name = output_ports[0].name
            for assignment in module.assignments:
                if assignment.target == output_name:
                    return self._translate_expression(assignment.expression)

        # Default case - return input (identity function)
        if len(input_ports) == 1 and len(output_ports) == 1:
            return input_ports[0].name

        # Multiple outputs - return tuple of expressions
        output_exprs = []
        for port in output_ports:
            # Try to find assignment
            expr_found = False
            for assignment in module.assignments:
                if assignment.target == port.name:
                    output_exprs.append(
                        self._translate_expression(assignment.expression)
                    )
                    expr_found = True
                    break
            if not expr_found:
                output_exprs.append("False")  # Default value

        if len(output_exprs) > 1:
            return f"({', '.join(output_exprs)})"
        else:
            return output_exprs[0] if output_exprs else "False"

    def _translate_expression(self, expression: str) -> str:
        """Translate HDL expression to Isabelle."""
        # Basic expression translation
        expr = expression.strip()

        # Boolean operators
        expr = expr.replace("&&", "∧")
        expr = expr.replace("||", "∨")
        expr = expr.replace("!", "¬")
        expr = expr.replace("==", "=")
        expr = expr.replace("!=", "≠")

        # Arithmetic operators (keep as is for now)
        # expr = expr.replace("+", "+")  # Addition is the same

        return expr

    def _translate_assignment_to_lemma(
        self, assignment: Assignment, module: Module, lemma_name: str
    ) -> str:
        """Translate assignment to Isabelle lemma."""
        target = assignment.target
        expression = self._translate_expression(assignment.expression)

        # Create lemma statement
        lemma = f"lemma {lemma_name}:\n"
        lemma += f'  "{target} = {expression}"\n'
        lemma += "  sorry (* Proof to be completed *)"

        return lemma

    def _generate_properties(self, ast: CircuitAST) -> str:
        """Generate correctness properties for the circuit."""
        content = []

        content.append("(* Correctness Properties *)")

        for module in ast.modules:
            # Generate basic properties
            content.append(f"")
            content.append(f"(* Properties for module {module.name} *)")

            # Well-formedness property
            if module.ports:
                content.append(f"lemma {module.name}_well_formed:")
                content.append('  "True" (* Well-formedness condition *)')
                content.append("  sorry")

            # Functional correctness (example)
            input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
            output_ports = [
                p for p in module.ports if p.signal_type == SignalType.OUTPUT
            ]

            if (
                input_ports
                and output_ports
                and len(input_ports) == 2
                and len(output_ports) == 1
            ):
                # Assume it's an adder-like circuit
                if "add" in module.name.lower() or "sum" in module.name.lower():
                    in1, in2 = input_ports[0].name, input_ports[1].name
                    out = output_ports[0].name

                    content.append(f"")
                    content.append(f"lemma {module.name}_correctness:")
                    content.append(f'  "{module.name} {in1} {in2} = {in1} + {in2}"')
                    content.append("  sorry (* Proof to be completed *)")

        return "\n".join(content)

    def generate_verification_goals(
        self, ast: CircuitAST, properties: List[str]
    ) -> str:
        """Generate verification goals for given properties."""
        content = []

        content.append("(* Verification Goals *)")

        for i, prop in enumerate(properties):
            goal_name = f"verification_goal_{i+1}"
            content.append(f"")
            content.append(f"lemma {goal_name}:")
            content.append(f'  "{prop}"')
            content.append("  sorry (* Proof to be generated by LLM *)")

        return "\n".join(content)
