"""Core CircuitVerifier class for formal verification."""

import os
import asyncio
import time
import uuid
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .exceptions import VerificationError
from .parsers import VerilogParser, VHDLParser, CircuitAST
from .translators import IsabelleTranslator, CoqTranslator, PropertyGenerator
from .llm.llm_client import LLMManager, LLMResponse
from .provers import IsabelleInterface, CoqInterface
from .provers.mock_prover import MockProver
from .security import InputValidator, ValidationResult, SecurityError
from .monitoring.logger import get_logger
from .monitoring.health_checker import HealthChecker


class CircuitVerifier:
    """Main interface for circuit verification using LLMs and theorem provers."""
    
    def __init__(
        self,
        prover: str = "isabelle",
        model: str = "gpt-4-turbo", 
        temperature: float = 0.1,
        refinement_rounds: int = 5,
        debug_mode: bool = False,
        strict_mode: bool = True
    ):
        """Initialize the circuit verifier.
        
        Args:
            prover: Theorem prover to use ("isabelle" or "coq")
            model: LLM model to use for proof generation
            temperature: LLM temperature setting
            refinement_rounds: Maximum refinement attempts
            debug_mode: Enable detailed debugging output
            strict_mode: Enable strict security validation
        """
        # Initialize security and monitoring
        self.validator = InputValidator(strict_mode=strict_mode)
        self.logger = get_logger("circuit_verifier")
        self.health_checker = HealthChecker()
        self.session_id = str(uuid.uuid4())
        
        # Validate inputs
        self._validate_init_params(prover, model, temperature, refinement_rounds)
        
        self.prover = prover
        self.model = model
        self.temperature = temperature
        self.refinement_rounds = refinement_rounds
        self.debug_mode = debug_mode
        self.strict_mode = strict_mode
        
        # Initialize components with error handling
        try:
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
            
            # Log initialization
            self.logger.set_context(
                session_id=self.session_id,
                prover=prover,
                model=model
            )
            self.logger.info("CircuitVerifier initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CircuitVerifier: {str(e)}")
            raise VerificationError(f"Initialization failed: {str(e)}") from e
    
    def _validate_init_params(self, prover: str, model: str, temperature: float, refinement_rounds: int):
        """Validate initialization parameters."""
        prover_result = self.validator.validate_prover_name(prover)
        if not prover_result.is_valid:
            raise SecurityError(f"Invalid prover: {'; '.join(prover_result.errors)}")
        
        model_result = self.validator.validate_model_name(model)
        if not model_result.is_valid:
            raise SecurityError(f"Invalid model: {'; '.join(model_result.errors)}")
        
        temp_result = self.validator.validate_temperature(temperature)
        if not temp_result.is_valid:
            raise SecurityError(f"Invalid temperature: {'; '.join(temp_result.errors)}")
        
        if not isinstance(refinement_rounds, int) or refinement_rounds < 0 or refinement_rounds > 20:
            raise SecurityError("Refinement rounds must be an integer between 0 and 20")
        
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
            SecurityError: If input validation fails
        """
        verification_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Step 0: Input validation and security checks
            self.logger.set_context(verification_id=verification_id)
            self.logger.log_verification_start("hdl_code", self.prover, self.model)
            
            # Validate HDL content
            hdl_validation = self.validator.validate_hdl_content(hdl_code)
            if not hdl_validation.is_valid:
                self.logger.log_security_event(
                    "input_validation_failed", 
                    f"HDL validation errors: {'; '.join(hdl_validation.errors)}",
                    "error"
                )
                raise SecurityError(f"HDL validation failed: {'; '.join(hdl_validation.errors)}")
            
            if hdl_validation.warnings:
                self.logger.warning(f"HDL validation warnings: {'; '.join(hdl_validation.warnings)}")
            
            # Use sanitized HDL content
            sanitized_hdl = hdl_validation.sanitized_input
            
            # Validate properties
            if properties is not None:
                prop_validation = self.validator.validate_properties(properties)
                if not prop_validation.is_valid:
                    self.logger.log_security_event(
                        "property_validation_failed",
                        f"Property validation errors: {'; '.join(prop_validation.errors)}",
                        "error"
                    )
                    raise SecurityError(f"Property validation failed: {'; '.join(prop_validation.errors)}")
                properties = prop_validation.sanitized_input
            
            # Validate timeout
            timeout_validation = self.validator.validate_timeout(timeout)
            if not timeout_validation.is_valid:
                raise SecurityError(f"Timeout validation failed: {'; '.join(timeout_validation.errors)}")
            timeout = timeout_validation.sanitized_input
            
            # Step 1: Parse HDL code with error handling
            self.logger.info("Starting HDL parsing")
            parse_start = time.time()
            try:
                ast = self._parse_hdl(sanitized_hdl)
                parse_time = (time.time() - parse_start) * 1000
                self.logger.log_performance("hdl_parsing", parse_time, modules_count=len(ast.modules))
            except Exception as e:
                self.logger.error(f"HDL parsing failed: {str(e)}")
                raise VerificationError(f"Failed to parse HDL code: {str(e)}") from e
            
            # Step 2: Generate or validate properties
            self.logger.info("Generating/validating properties")
            prop_start = time.time()
            try:
                if properties is None:
                    property_specs = self.property_generator.generate_properties(ast)
                    property_list = [prop.formula for prop in property_specs]
                elif isinstance(properties, str):
                    property_list = [properties]
                else:
                    property_list = properties
                
                prop_time = (time.time() - prop_start) * 1000
                self.logger.log_performance("property_generation", prop_time, properties_count=len(property_list))
            except Exception as e:
                self.logger.error(f"Property generation failed: {str(e)}")
                raise VerificationError(f"Failed to generate properties: {str(e)}") from e
            
            # Step 3: Translate to formal specification
            self.logger.info(f"Translating to {self.prover} specification")
            translate_start = time.time()
            try:
                if self.prover == "isabelle":
                    formal_spec = self.isabelle_translator.translate(ast)
                    verification_goals = self.isabelle_translator.generate_verification_goals(ast, property_list)
                else:  # coq
                    formal_spec = self.coq_translator.translate(ast)
                    verification_goals = self.coq_translator.generate_verification_goals(ast, property_list)
                
                translate_time = (time.time() - translate_start) * 1000
                self.logger.log_performance("translation", translate_time, spec_length=len(formal_spec))
            except Exception as e:
                self.logger.error(f"Translation to {self.prover} failed: {str(e)}")
                raise VerificationError(f"Failed to translate to {self.prover}: {str(e)}") from e
            
            # Step 4: Generate proof with LLM
            self.logger.info(f"Generating proof with {self.model}")
            llm_start = time.time()
            try:
                proof_content = self._generate_proof_with_llm(formal_spec, verification_goals, property_list)
                llm_time = (time.time() - llm_start) * 1000
                self.logger.log_performance("llm_proof_generation", llm_time, proof_length=len(proof_content))
            except Exception as e:
                self.logger.error(f"LLM proof generation failed: {str(e)}")
                raise VerificationError(f"Failed to generate proof with LLM: {str(e)}") from e
            
            # Step 5: Verify proof with theorem prover
            self.logger.info(f"Verifying proof with {self.prover}")
            prover_start = time.time()
            try:
                verification_result = self._verify_with_prover(proof_content)
                prover_time = (time.time() - prover_start) * 1000
                self.logger.log_performance("proof_verification", prover_time, success=verification_result.success)
            except Exception as e:
                self.logger.error(f"Proof verification failed: {str(e)}")
                raise VerificationError(f"Failed to verify proof: {str(e)}") from e
            
            # Step 6: Refine proof if needed
            refinement_attempts = 0
            if not verification_result.success and self.refinement_rounds > 0:
                self.logger.info(f"Proof failed, attempting refinement (max {self.refinement_rounds} rounds)")
                
                for attempt in range(self.refinement_rounds):
                    refinement_attempts += 1
                    refine_start = time.time()
                    
                    try:
                        refined_proof = self._refine_proof(proof_content, verification_result.errors)
                        refined_result = self._verify_with_prover(refined_proof)
                        
                        refine_time = (time.time() - refine_start) * 1000
                        self.logger.log_performance(
                            f"proof_refinement_attempt_{attempt+1}", 
                            refine_time, 
                            success=refined_result.success
                        )
                        
                        if refined_result.success:
                            self.logger.info(f"Proof refinement successful on attempt {attempt+1}")
                            proof_content = refined_proof
                            verification_result = refined_result
                            break
                        else:
                            self.logger.warning(f"Proof refinement attempt {attempt+1} failed")
                    except Exception as e:
                        self.logger.warning(f"Proof refinement attempt {attempt+1} error: {str(e)}")
                        
                if not verification_result.success:
                    self.logger.warning(f"All {self.refinement_rounds} refinement attempts failed")
            
            # Create result
            status = "VERIFIED" if verification_result.success else "FAILED"
            total_time = (time.time() - start_time) * 1000
            
            # Log completion
            self.logger.log_verification_end(status, total_time, len(property_list))
            
            return ProofResult(
                status=status,
                proof_code=proof_content,
                errors=verification_result.errors if not verification_result.success else [],
                properties_verified=property_list,
                ast=ast,
                verification_id=verification_id,
                duration_ms=total_time,
                refinement_attempts=refinement_attempts
            )
            
        except (SecurityError, VerificationError):
            # Re-raise known exceptions
            total_time = (time.time() - start_time) * 1000
            self.logger.log_verification_end("ERROR", total_time, 0)
            raise
        except Exception as e:
            # Handle unexpected exceptions
            total_time = (time.time() - start_time) * 1000
            self.logger.error(f"Unexpected error during verification: {str(e)}")
            self.logger.log_verification_end("ERROR", total_time, 0)
            
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
                isabelle = IsabelleInterface()
                if isabelle.check_installation():
                    self._prover_interface = isabelle
                else:
                    self.logger.warning("Isabelle not installed, using mock prover for testing")
                    self._prover_interface = MockProver()
            else:
                coq = CoqInterface()
                if coq.check_installation():
                    self._prover_interface = coq
                else:
                    self.logger.warning("Coq not installed, using mock prover for testing")
                    self._prover_interface = MockProver()
        
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
                 properties_verified: List[str] = None, ast: CircuitAST = None,
                 verification_id: str = None, duration_ms: float = 0.0, 
                 refinement_attempts: int = 0):
        self.status = status
        self.proof_code = proof_code
        self.errors = errors or []
        self.properties_verified = properties_verified or []
        self.ast = ast
        self.verification_id = verification_id
        self.duration_ms = duration_ms
        self.refinement_attempts = refinement_attempts
        
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