"""Proof generation using LLMs."""

from typing import List, Dict, Any, Optional
from .llm_client import LLMManager, LLMResponse
from .prompt_manager import PromptManager
from .response_parser import ResponseParser


class ProofGenerationError(Exception):
    """Exception raised for proof generation errors."""
    pass


class ProofGenerator:
    """Generates formal proofs using LLMs."""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None, provider=None):
        """Initialize proof generator.
        
        Args:
            llm_manager: LLM manager instance (creates default if None)
            provider: LLM provider (for backward compatibility)
        """
        if provider is not None:
            # Backward compatibility: create manager from provider
            from .llm_client import LLMManager
            manager = LLMManager()
            manager.add_client("default", provider, set_as_default=True)
            self.llm_manager = manager
            self.provider = provider  # Store for backward compatibility
        else:
            self.llm_manager = llm_manager or LLMManager.create_default()
            self.provider = None
        
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()
    
    def generate_proof(self, formal_spec: str, verification_goals: str, 
                      properties: List[str], prover: str = "isabelle",
                      temperature: float = 0.1) -> str:
        """Generate proof for given specification and properties.
        
        Args:
            formal_spec: Formal specification of the circuit
            verification_goals: Verification goals to prove
            properties: List of properties to verify
            prover: Target theorem prover ("isabelle" or "coq")
            temperature: LLM temperature for generation
            
        Returns:
            Generated proof code
            
        Raises:
            ProofGenerationError: If proof generation fails
        """
        try:
            # Create prompt for proof generation
            prompt = self.prompt_manager.create_proof_prompt(
                formal_spec=formal_spec,
                verification_goals=verification_goals,
                properties=properties,
                prover=prover
            )
            
            # Generate proof using LLM
            response = self.llm_manager.generate_sync(
                prompt=prompt,
                temperature=temperature,
                max_tokens=4000
            )
            
            # Parse and validate response
            proof_code = self.response_parser.extract_proof_code(
                response.content, 
                prover=prover
            )
            
            return proof_code
            
        except Exception as e:
            raise ProofGenerationError(f"Proof generation failed: {str(e)}") from e
    
    def generate_lemma(self, lemma_statement: str, context: str = "",
                      prover: str = "isabelle", temperature: float = 0.1) -> str:
        """Generate proof for a specific lemma.
        
        Args:
            lemma_statement: The lemma to prove
            context: Additional context and definitions
            prover: Target theorem prover
            temperature: LLM temperature
            
        Returns:
            Generated lemma proof
        """
        try:
            prompt = self.prompt_manager.create_lemma_prompt(
                lemma_statement=lemma_statement,
                context=context,
                prover=prover
            )
            
            response = self.llm_manager.generate_sync(
                prompt=prompt,
                temperature=temperature,
                max_tokens=2000
            )
            
            return self.response_parser.extract_proof_code(
                response.content,
                prover=prover
            )
            
        except Exception as e:
            raise ProofGenerationError(f"Lemma generation failed: {str(e)}") from e
    
    def generate_inductive_proof(self, base_case: str, inductive_step: str,
                               context: str = "", prover: str = "isabelle") -> str:
        """Generate inductive proof structure.
        
        Args:
            base_case: Base case statement
            inductive_step: Inductive step statement  
            context: Additional context
            prover: Target theorem prover
            
        Returns:
            Generated inductive proof
        """
        try:
            prompt = self.prompt_manager.create_inductive_prompt(
                base_case=base_case,
                inductive_step=inductive_step,
                context=context,
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
            raise ProofGenerationError(f"Inductive proof generation failed: {str(e)}") from e
    
    def generate_proof_sketch(self, goal: str, context: str = "",
                            prover: str = "isabelle") -> str:
        """Generate high-level proof sketch.
        
        Args:
            goal: Goal to prove
            context: Additional context
            prover: Target theorem prover
            
        Returns:
            Generated proof sketch
        """
        try:
            prompt = self.prompt_manager.create_sketch_prompt(
                goal=goal,
                context=context,
                prover=prover
            )
            
            response = self.llm_manager.generate_sync(
                prompt=prompt,
                temperature=0.2,  # Slightly higher for creativity
                max_tokens=1500
            )
            
            return response.content.strip()
            
        except Exception as e:
            raise ProofGenerationError(f"Proof sketch generation failed: {str(e)}") from e
    
    # Backward compatibility methods for tests
    def generate_initial_proof(self, circuit_ast, properties: List[str], **kwargs) -> str:
        """Generate initial proof for circuit AST and properties."""
        if self.provider and hasattr(self.provider, 'generate_proof'):
            # Use provider directly for backward compatibility
            prompt = self.construct_prompt("circuit", str(circuit_ast), properties)
            return self.provider.generate_proof(prompt, **kwargs)
        else:
            # Convert AST to formal spec (simplified for testing)
            formal_spec = f"Circuit: {circuit_ast}"
            verification_goals = "; ".join(properties)
            
            return self.generate_proof(
                formal_spec=formal_spec,
                verification_goals=verification_goals,
                properties=properties,
                **kwargs
            )
    
    def construct_prompt(self, circuit_type: str, circuit_code: str, properties: List[str]) -> str:
        """Construct prompt for proof generation."""
        return self.prompt_manager.create_proof_prompt(
            formal_spec=circuit_code,
            verification_goals=f"Verify {circuit_type} circuit",
            properties=properties,
            prover="isabelle"
        )