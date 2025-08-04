"""Proof refinement using LLMs to fix errors."""

from typing import List, Dict, Any, Optional, Tuple
from .llm_client import LLMManager, LLMResponse
from .prompt_manager import PromptManager
from .response_parser import ResponseParser


class ProofRefinementError(Exception):
    """Exception raised for proof refinement errors."""
    pass


class ProofRefiner:
    """Refines formal proofs by fixing errors using LLMs."""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """Initialize proof refiner.
        
        Args:
            llm_manager: LLM manager instance (creates default if None)
        """
        self.llm_manager = llm_manager or LLMManager.create_default()
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()
    
    def refine_proof(self, original_proof: str, errors: List[str],
                    prover: str = "isabelle", max_attempts: int = 3,
                    temperature: float = 0.1) -> Tuple[str, bool]:
        """Refine proof by fixing errors.
        
        Args:
            original_proof: Original proof with errors
            errors: List of error messages
            prover: Target theorem prover
            max_attempts: Maximum refinement attempts
            temperature: LLM temperature
            
        Returns:
            Tuple of (refined_proof, success)
            
        Raises:
            ProofRefinementError: If refinement fails
        """
        current_proof = original_proof
        current_errors = errors
        
        for attempt in range(max_attempts):
            try:
                # Create refinement prompt
                prompt = self.prompt_manager.create_refinement_prompt(
                    original_proof=current_proof,
                    errors=current_errors,
                    prover=prover
                )
                
                # Generate refined proof
                response = self.llm_manager.generate_sync(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=4000
                )
                
                # Extract refined proof code
                refined_proof = self.response_parser.extract_proof_code(
                    response.content,
                    prover=prover
                )
                
                # Basic validation
                is_valid, syntax_issues = self.response_parser.validate_syntax(
                    refined_proof, 
                    prover
                )
                
                if is_valid:
                    return refined_proof, True
                else:
                    # Use syntax issues as errors for next iteration
                    current_proof = refined_proof
                    current_errors = syntax_issues
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise ProofRefinementError(f"Refinement failed after {max_attempts} attempts: {str(e)}") from e
                continue
        
        # If we get here, all attempts failed
        return current_proof, False
    
    def fix_specific_error(self, proof: str, error_type: str, error_message: str,
                          prover: str = "isabelle") -> str:
        """Fix a specific type of error in the proof.
        
        Args:
            proof: Original proof
            error_type: Type of error (e.g., "syntax", "type", "unfinished")
            error_message: Specific error message
            prover: Target theorem prover
            
        Returns:
            Fixed proof code
        """
        try:
            # Create specialized prompt based on error type
            if error_type == "syntax":
                prompt = self._create_syntax_fix_prompt(proof, error_message, prover)
            elif error_type == "type":
                prompt = self._create_type_fix_prompt(proof, error_message, prover)
            elif error_type == "unfinished":
                prompt = self._create_completion_prompt(proof, error_message, prover)
            else:
                # Generic error fix
                prompt = self.prompt_manager.create_refinement_prompt(
                    original_proof=proof,
                    errors=[error_message],
                    prover=prover
                )
            
            response = self.llm_manager.generate_sync(
                prompt=prompt,
                temperature=0.1,
                max_tokens=3000
            )
            
            return self.response_parser.extract_proof_code(
                response.content,
                prover=prover
            )
            
        except Exception as e:
            raise ProofRefinementError(f"Failed to fix {error_type} error: {str(e)}") from e
    
    def optimize_proof(self, proof: str, prover: str = "isabelle") -> str:
        """Optimize proof for efficiency and readability.
        
        Args:
            proof: Original proof
            prover: Target theorem prover
            
        Returns:
            Optimized proof code
        """
        try:
            prompt = self.prompt_manager.create_optimization_prompt(proof, prover)
            
            response = self.llm_manager.generate_sync(
                prompt=prompt,
                temperature=0.1,
                max_tokens=3000
            )
            
            return self.response_parser.extract_proof_code(
                response.content,
                prover=prover
            )
            
        except Exception as e:
            raise ProofRefinementError(f"Proof optimization failed: {str(e)}") from e
    
    def complete_partial_proof(self, partial_proof: str, missing_parts: List[str],
                             prover: str = "isabelle") -> str:
        """Complete a partial proof by filling in missing parts.
        
        Args:
            partial_proof: Incomplete proof
            missing_parts: List of missing components
            prover: Target theorem prover
            
        Returns:
            Completed proof
        """
        try:
            missing_str = "\n".join(f"- {part}" for part in missing_parts)
            
            prompt = f"""Complete the following partial {prover.title()} proof by adding the missing parts.

PARTIAL PROOF:
{partial_proof}

MISSING PARTS:
{missing_str}

Requirements:
1. Add all missing components
2. Ensure syntactic correctness
3. Maintain logical consistency
4. Use appropriate {prover.title()} constructs

COMPLETED PROOF:"""
            
            response = self.llm_manager.generate_sync(
                prompt=prompt,
                temperature=0.1,
                max_tokens=3000
            )
            
            return self.response_parser.extract_proof_code(
                response.content,
                prover=prover
            )
            
        except Exception as e:
            raise ProofRefinementError(f"Proof completion failed: {str(e)}") from e
    
    def _create_syntax_fix_prompt(self, proof: str, error_message: str, prover: str) -> str:
        """Create prompt for fixing syntax errors."""
        return f"""Fix the syntax error in the following {prover.title()} proof.

PROOF WITH SYNTAX ERROR:
{proof}

SYNTAX ERROR:
{error_message}

Please provide the corrected proof with proper {prover.title()} syntax:"""
    
    def _create_type_fix_prompt(self, proof: str, error_message: str, prover: str) -> str:
        """Create prompt for fixing type errors."""
        return f"""Fix the type error in the following {prover.title()} proof.

PROOF WITH TYPE ERROR:
{proof}

TYPE ERROR:
{error_message}

Please provide the corrected proof with proper types and type annotations:"""
    
    def _create_completion_prompt(self, proof: str, error_message: str, prover: str) -> str:
        """Create prompt for completing unfinished proofs."""
        return f"""Complete the unfinished {prover.title()} proof.

UNFINISHED PROOF:
{proof}

ERROR MESSAGE:
{error_message}

Please provide the completed proof with all necessary steps:"""
    
    def iterative_refinement(self, proof: str, validator_func: callable,
                           prover: str = "isabelle", max_iterations: int = 5) -> Tuple[str, bool]:
        """Perform iterative refinement using external validator.
        
        Args:
            proof: Initial proof
            validator_func: Function that returns (success, errors) for proof
            prover: Target theorem prover
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (final_proof, success)
        """
        current_proof = proof
        
        for iteration in range(max_iterations):
            # Validate current proof
            success, errors = validator_func(current_proof)
            
            if success:
                return current_proof, True
            
            # Refine proof based on errors
            try:
                current_proof, _ = self.refine_proof(
                    original_proof=current_proof,
                    errors=errors,
                    prover=prover,
                    max_attempts=1
                )
            except ProofRefinementError:
                # If refinement fails, try optimization
                try:
                    current_proof = self.optimize_proof(current_proof, prover)
                except ProofRefinementError:
                    break
        
        # Final validation
        success, _ = validator_func(current_proof)
        return current_proof, success