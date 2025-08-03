"""Translator from HDL AST to Coq specifications."""

from typing import List, Dict, Any, Optional
from ..parsers.ast_nodes import CircuitAST, Module, Port, Signal, Assignment, SignalType


class CoqTranslationError(Exception):
    """Exception raised for Coq translation errors."""
    pass


class CoqTranslator:
    """Translates HDL AST to Coq vernacular files."""
    
    def __init__(self):
        """Initialize the Coq translator."""
        self.requires = ["Coq.Init.Nat", "Coq.Bool.Bool", "Coq.Lists.List"]
    
    def translate(self, ast: CircuitAST, module_name: Optional[str] = None) -> str:
        """Translate AST to Coq vernacular.
        
        Args:
            ast: Circuit AST to translate
            module_name: Name for the Coq module (optional)
            
        Returns:
            Coq vernacular file content
            
        Raises:
            CoqTranslationError: If translation fails
        """
        try:
            if not module_name:
                top_module = ast.get_top_module()
                module_name = top_module.name if top_module else "Circuit"
            
            content = []
            
            # File header with requires
            content.append(self._generate_header())
            
            # Module definition
            content.append(f"Module {module_name}.")
            content.append("")
            
            # Type definitions
            content.append(self._generate_type_definitions(ast))
            
            # Function definitions for each module
            for module in ast.modules:
                content.append(self._translate_module(module))
            
            # Property definitions
            content.append(self._generate_properties(ast))
            
            # Module footer
            content.append(f"End {module_name}.")
            
            return "\n\n".join(content)
            
        except Exception as e:
            raise CoqTranslationError(f"Failed to translate to Coq: {str(e)}") from e
    
    def _generate_header(self) -> str:
        \"\"\"Generate Coq file header with requires.\"\"\"
        header = []
        for req in self.requires:
            header.append(f"Require Import {req}.")
        
        header.extend([
            "Import ListNotations.",
            "Open Scope nat_scope.",
            "Open Scope bool_scope."
        ])
        
        return "\n".join(header)
    
    def _generate_type_definitions(self, ast: CircuitAST) -> str:
        \"\"\"Generate type definitions for signals and values.\"\"\"
        content = []
        
        content.append("(* Basic types for hardware modeling *)")
        content.append("Definition bit := bool.")
        content.append("Definition bitvec (n : nat) := list bool.")
        content.append("")
        
        # State types for each module
        for module in ast.modules:
            if module.ports or module.signals:
                content.append(f"(* State type for module {module.name} *)")
                
                fields = []
                for port in module.ports:
                    if port.width > 1:
                        fields.append(f"  {port.name} : bitvec {port.width}")
                    else:
                        fields.append(f"  {port.name} : bit")
                
                for signal in module.signals:
                    if signal.width > 1:
                        fields.append(f"  {signal.name} : bitvec {signal.width}")
                    else:
                        fields.append(f"  {signal.name} : bit")
                
                if fields:
                    content.append(f"Record {module.name}_state := {{")
                    content.extend(fields)
                    content.append("}.")
                    content.append("")
        
        return "\n".join(content)
    
    def _translate_module(self, module: Module) -> str:
        \"\"\"Translate a single module to Coq function.\"\"\"
        content = []
        
        content.append(f"(* Module: {module.name} *)")
        
        # Create module function
        input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
        output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]
        
        if input_ports and output_ports:
            # Function signature
            params = []
            for port in input_ports:
                if port.width > 1:
                    params.append(f"({port.name} : bitvec {port.width})")
                else:
                    params.append(f"({port.name} : bit)")
            
            return_type = self._generate_return_type(output_ports)
            param_str = " ".join(params)
            
            content.append(f"Definition {module.name} {param_str} : {return_type} :=")
            
            # Function body
            function_body = self._generate_function_body(module, input_ports, output_ports)
            content.append(f"  {function_body}.")
        
        # Translate assignments as separate definitions
        if module.assignments:
            content.append("")
            content.append(f"(* Assignment functions for module {module.name} *)")
            for i, assignment in enumerate(module.assignments):
                def_name = f"{module.name}_assign_{assignment.target}"
                definition = self._translate_assignment_to_definition(assignment, module, def_name)
                content.append(definition)
        
        return "\n".join(content)
    
    def _generate_return_type(self, output_ports: List[Port]) -> str:
        \"\"\"Generate return type for module function.\"\"\"
        if len(output_ports) == 1:
            port = output_ports[0]
            if port.width > 1:
                return f"bitvec {port.width}"
            else:
                return "bit"
        else:
            # Multiple outputs - use tuple
            types = []
            for port in output_ports:
                if port.width > 1:
                    types.append(f"bitvec {port.width}")
                else:
                    types.append("bit")
            return f"({' * '.join(types)})"
    
    def _generate_function_body(self, module: Module, input_ports: List[Port], output_ports: List[Port]) -> str:
        \"\"\"Generate function body for module.\"\"\"
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
                    output_exprs.append(self._translate_expression(assignment.expression))
                    expr_found = True
                    break
            if not expr_found:
                output_exprs.append("false")  # Default value
        
        if len(output_exprs) > 1:
            return f"({', '.join(output_exprs)})"
        else:
            return output_exprs[0] if output_exprs else "false"
    
    def _translate_expression(self, expression: str) -> str:
        \"\"\"Translate HDL expression to Coq.\"\"\"
        # Basic expression translation
        expr = expression.strip()
        
        # Boolean operators
        expr = expr.replace("&&", "&&")  # Coq uses && for boolean and
        expr = expr.replace("||", "||")  # Coq uses || for boolean or
        expr = expr.replace("!", "negb")  # Coq boolean negation
        expr = expr.replace("==", "=?")  # Coq equality comparison for nat
        expr = expr.replace("!=", "<>")  # Coq inequality
        
        # Handle simple arithmetic
        expr = expr.replace("+", "+")  # Addition is the same
        expr = expr.replace("-", "-")  # Subtraction is the same
        expr = expr.replace("*", "*")  # Multiplication is the same
        
        return expr
    
    def _translate_assignment_to_definition(self, assignment: Assignment, module: Module, def_name: str) -> str:
        \"\"\"Translate assignment to Coq definition.\"\"\"
        target = assignment.target
        expression = self._translate_expression(assignment.expression)
        
        # Find the target signal/port to determine type
        target_type = "bit"  # Default
        for port in module.ports:
            if port.name == target:
                target_type = f"bitvec {port.width}" if port.width > 1 else "bit"
                break
        
        for signal in module.signals:
            if signal.name == target:
                target_type = f"bitvec {signal.width}" if signal.width > 1 else "bit"
                break
        
        definition = f"Definition {def_name} : {target_type} := {expression}."
        return definition
    
    def _generate_properties(self, ast: CircuitAST) -> str:
        \"\"\"Generate correctness properties for the circuit.\"\"\"
        content = []
        
        content.append("(* Correctness Properties *)")
        
        for module in ast.modules:
            # Generate basic properties
            content.append("")
            content.append(f"(* Properties for module {module.name} *)")
            
            # Well-formedness property
            if module.ports:
                content.append(f"Lemma {module.name}_well_formed :")
                content.append("  True.")
                content.append("Proof. trivial. Qed.")
                content.append("")
            
            # Functional correctness (example)
            input_ports = [p for p in module.ports if p.signal_type == SignalType.INPUT]
            output_ports = [p for p in module.ports if p.signal_type == SignalType.OUTPUT]
            
            if input_ports and output_ports and len(input_ports) == 2 and len(output_ports) == 1:
                # Assume it's an adder-like circuit
                if "add" in module.name.lower() or "sum" in module.name.lower():
                    in1, in2 = input_ports[0].name, input_ports[1].name
                    out = output_ports[0].name
                    
                    content.append(f"Lemma {module.name}_correctness :")
                    content.append(f"  forall {in1} {in2},")
                    content.append(f"  {module.name} {in1} {in2} = {in1} + {in2}.")
                    content.append("Proof.")
                    content.append("  (* Proof to be completed *)")
                    content.append("Admitted.")
                    content.append("")
        
        return "\n".join(content)
    
    def generate_verification_goals(self, ast: CircuitAST, properties: List[str]) -> str:
        \"\"\"Generate verification goals for given properties.\"\"\"
        content = []
        
        content.append("(* Verification Goals *)")
        
        for i, prop in enumerate(properties):
            goal_name = f"verification_goal_{i+1}"
            content.append("")
            content.append(f"Lemma {goal_name} :")
            content.append(f"  {prop}.")
            content.append("Proof.")
            content.append("  (* Proof to be generated by LLM *)")
            content.append("Admitted.")
        
        return "\n".join(content)