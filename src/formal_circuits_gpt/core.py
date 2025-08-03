"""Core CircuitVerifier class for formal verification."""

import os
import asyncio
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .exceptions import VerificationError
from .parsers import VerilogParser, VHDLParser, CircuitAST
from .translators import IsabelleTranslator, CoqTranslator, PropertyGenerator
from .llm import LLMManager, LLMResponse
from .provers import IsabelleInterface, CoqInterface


class CircuitVerifier:
    """Main interface for circuit verification using LLMs and theorem provers."""
    
    def __init__(
        self,
        prover: str = "isabelle",
        model: str = "gpt-4-turbo", 
        temperature: float = 0.1,
        refinement_rounds: int = 5,
        debug_mode: bool = False
    ):
        """Initialize the circuit verifier.
        
        Args:
            prover: Theorem prover to use ("isabelle" or "coq")
            model: LLM model to use for proof generation
            temperature: LLM temperature setting
            refinement_rounds: Maximum refinement attempts
            debug_mode: Enable detailed debugging output
        """
        self.prover = prover
        self.model = model
        self.temperature = temperature
        self.refinement_rounds = refinement_rounds
        self.debug_mode = debug_mode
        
        # Initialize components
        self.llm_manager = LLMManager.create_default()
        self.property_generator = PropertyGenerator()
        
        # Initialize parsers
        self.verilog_parser = VerilogParser()
        self.vhdl_parser = VHDLParser()
        
        # Initialize translators
        self.isabelle_translator = IsabelleTranslator()
        self.coq_translator = CoqTranslator()
        
        # Initialize prover interfaces (will be created when needed)
        self._prover_interface = None
        
    def verify(
        self,
        hdl_code: str,
        properties: Union[List[str], str, None] = None,
        timeout: int = 300
    ) -> "ProofResult":
        """Verify circuit properties.
        
        Args:
            hdl_code: Verilog or VHDL source code
            properties: List of properties to verify (auto-generated if None)
            timeout: Verification timeout in seconds
            
        Returns:
            ProofResult containing verification status and proof
            
        Raises:
            VerificationError: If verification fails
        """
        try:
            # Step 1: Parse HDL code
            ast = self._parse_hdl(hdl_code)
            
            # Step 2: Generate or use provided properties
            if properties is None:
                property_specs = self.property_generator.generate_properties(ast)
                property_list = [prop.formula for prop in property_specs]
            elif isinstance(properties, str):
                property_list = [properties]
            else:
                property_list = properties
            
            # Step 3: Translate to formal specification
            if self.prover == "isabelle":
                formal_spec = self.isabelle_translator.translate(ast)
                verification_goals = self.isabelle_translator.generate_verification_goals(ast, property_list)
            else:  # coq
                formal_spec = self.coq_translator.translate(ast)
                verification_goals = self.coq_translator.generate_verification_goals(ast, property_list)
            
            # Step 4: Generate initial proof using LLM
            proof_content = self._generate_proof_with_llm(formal_spec, verification_goals, property_list)
            
            # Step 5: Verify proof with theorem prover
            verification_result = self._verify_with_prover(proof_content)
            
            # Step 6: Refine proof if needed
            if not verification_result.success and self.refinement_rounds > 0:
                proof_content = self._refine_proof(proof_content, verification_result.errors)
                verification_result = self._verify_with_prover(proof_content)
            
            # Create result
            status = "VERIFIED" if verification_result.success else "FAILED"
            return ProofResult(
                status=status,
                proof_code=proof_content,
                errors=verification_result.errors if not verification_result.success else [],
                properties_verified=property_list,
                ast=ast
            )
            
        except Exception as e:
            if self.debug_mode:
                raise
            raise VerificationError(f"Verification failed: {str(e)}") from e
    
    def verify_file(
        self,
        hdl_file: str,
        properties: Optional[Union[List[str], str]] = None,
        timeout: int = 3600
    ) -> "ProofResult":
        """Verify circuit from file.
        
        Args:
            hdl_file: Path to HDL source file
            properties: Properties to verify (auto-inferred if None)
            timeout: Verification timeout in seconds
            
        Returns:
            ProofResult containing verification status and proof
        """
        try:
            # Read file content
            file_path = Path(hdl_file)
            if not file_path.exists():
                raise VerificationError(f"File not found: {hdl_file}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                hdl_code = f.read()
            
            # Verify using the main verify method
            return self.verify(hdl_code, properties, timeout)
            
        except Exception as e:
            raise VerificationError(f"File verification failed: {str(e)}") from e
    
    def _parse_hdl(self, hdl_code: str) -> CircuitAST:
        """Parse HDL code into AST."""
        # Try to detect HDL type from code content
        code_lower = hdl_code.lower()
        
        if any(keyword in code_lower for keyword in ['module', 'endmodule', 'assign', 'always']):
            # Looks like Verilog
            return self.verilog_parser.parse(hdl_code)
        elif any(keyword in code_lower for keyword in ['entity', 'architecture', 'signal', 'process']):
            # Looks like VHDL
            return self.vhdl_parser.parse(hdl_code)
        else:
            # Default to Verilog
            return self.verilog_parser.parse(hdl_code)
    
    def _generate_proof_with_llm(self, formal_spec: str, verification_goals: str, properties: List[str]) -> str:
        """Generate proof using LLM."""
        prompt = self._create_proof_generation_prompt(formal_spec, verification_goals, properties)
        
        try:
            response = self.llm_manager.generate_sync(
                prompt,
                temperature=self.temperature,
                max_tokens=3000
            )
            return response.content
        except Exception as e:
            raise VerificationError(f"LLM proof generation failed: {str(e)}") from e
    
    def _create_proof_generation_prompt(self, formal_spec: str, verification_goals: str, properties: List[str]) -> str:
        """Create prompt for LLM proof generation."""
        prover_name = "Isabelle/HOL" if self.prover == "isabelle" else "Coq"
        
        prompt = f"""You are an expert in formal verification using {prover_name}. 

Given the following formal specification and verification goals, please generate complete proofs for all the properties.

FORMAL SPECIFICATION:
{formal_spec}

VERIFICATION GOALS:
{verification_goals}

PROPERTIES TO VERIFY:
{chr(10).join(f'- {prop}' for prop in properties)}

Requirements:
1. Generate complete, valid {prover_name} proofs
2. Use appropriate proof tactics and strategies
3. Handle all cases and edge conditions
4. Provide clear proof structure and comments
5. Ensure all lemmas and theorems are properly proven

Please provide the complete proof code:"""
        
        return prompt
    
    def _verify_with_prover(self, proof_content: str) -> "ProverResult":
        """Verify proof with theorem prover."""
        if not self._prover_interface:
            if self.prover == "isabelle":
                self._prover_interface = IsabelleInterface()
            else:
                self._prover_interface = CoqInterface()
        
        return self._prover_interface.verify_proof(proof_content)
    
    def _refine_proof(self, proof_content: str, errors: List[str]) -> str:
        """Refine proof based on errors."""
        refinement_prompt = f"""The following proof has errors. Please fix them and provide a corrected version.

ORIGINAL PROOF:
{proof_content}

ERRORS:
{chr(10).join(f'- {error}' for error in errors)}

Please provide the corrected proof addressing all the errors above:"""
        
        try:
            response = self.llm_manager.generate_sync(
                refinement_prompt,
                temperature=self.temperature,
                max_tokens=3000
            )
            return response.content
        except Exception as e:
            if self.debug_mode:
                print(f"Proof refinement failed: {e}")
            return proof_content  # Return original if refinement fails


class ProofResult:
    """Result of formal verification attempt."""
    
    def __init__(self, status: str, proof_code: str = "", errors: List[str] = None, 
                 properties_verified: List[str] = None, ast: CircuitAST = None):
        self.status = status
        self.proof_code = proof_code
        self.errors = errors or []
        self.properties_verified = properties_verified or []
        self.ast = ast
        
    @property 
    def isabelle_code(self) -> str:
        """Get Isabelle proof code."""
        return self.proof_code if "theory" in self.proof_code.lower() else ""
    
    @property
    def coq_code(self) -> str:
        """Get Coq proof code.""" 
        return self.proof_code if "Require" in self.proof_code or "Definition" in self.proof_code else ""
        
    def export_latex(self, filename: str) -> None:
        """Export proof to LaTeX format."""
        latex_content = f"""\\documentclass{{article}}
\\usepackage{{amsmath,amsthm,amssymb}}
\\usepackage{{listings}}

\\title{{Formal Verification Report}}
\\author{{Formal-Circuits-GPT}}

\\begin{{document}}
\\maketitle

\\section{{Verification Status}}
Status: {self.status}

\\section{{Properties Verified}}
\\begin{{itemize}}
{chr(10).join(f'\\item {prop}' for prop in self.properties_verified)}
\\end{{itemize}}

\\section{{Proof Code}}
\\begin{{lstlisting}}
{self.proof_code}
\\end{{lstlisting}}

{"\\section{Errors}" if self.errors else ""}
{"\\begin{itemize}" if self.errors else ""}
{chr(10).join(f'\\item {error}' for error in self.errors) if self.errors else ""}
{"\\end{itemize}" if self.errors else ""}

\\end{{document}}"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
    
    def export_systemverilog_assertions(self, filename: str) -> None:
        """Export properties as SystemVerilog assertions."""
        if not self.properties_verified:
            return
            
        sva_content = "// Generated SystemVerilog Assertions\n\n"
        for i, prop in enumerate(self.properties_verified):
            sva_content += f"property prop_{i+1};\n"
            sva_content += f"  // {prop}\n"
            sva_content += f"  @(posedge clk) {self._property_to_sva(prop)};\n"
            sva_content += "endproperty\n\n"
            sva_content += f"assert property (prop_{i+1});\n\n"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(sva_content)
    
    def _property_to_sva(self, property_formula: str) -> str:
        """Convert property formula to SystemVerilog assertion."""
        # Basic conversion - would need more sophisticated parsing
        sva = property_formula.replace("∀", "").replace("∃", "")
        sva = sva.replace("→", "->").replace("↔", "<->")
        sva = sva.replace("∧", "&&").replace("∨", "||")
        return sva.strip()


class ProverResult:
    """Result from theorem prover verification."""
    
    def __init__(self, success: bool, errors: List[str] = None, output: str = ""):
        self.success = success
        self.errors = errors or []
        self.output = output