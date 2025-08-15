"""Prompt management for LLM interactions."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProofTemplate:
    """Template for proof generation."""

    name: str
    prover: str
    template: str
    variables: List[str]


class PromptManager:
    """Manages prompts for different proof generation tasks."""

    def __init__(self):
        """Initialize prompt manager with templates."""
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, ProofTemplate]:
        """Load proof templates for different provers."""
        templates = {}

        # Isabelle proof template
        templates["isabelle_main"] = ProofTemplate(
            name="isabelle_main",
            prover="isabelle",
            template="""You are an expert in formal verification using Isabelle/HOL.

Given the following formal specification and verification goals, generate complete, valid Isabelle proofs.

CONTEXT:
{context}

FORMAL SPECIFICATION:
{formal_spec}

VERIFICATION GOALS:
{verification_goals}

PROPERTIES TO VERIFY:
{properties}

Requirements:
1. Generate complete, syntactically correct Isabelle/HOL proofs
2. Use appropriate proof methods: auto, simp, blast, metis, etc.
3. Break down complex goals into manageable lemmas
4. Provide clear proof structure with proper indentation
5. Include helpful comments explaining proof steps
6. Handle all edge cases and corner conditions

Generate the complete Isabelle theory with all proofs:""",
            variables=["context", "formal_spec", "verification_goals", "properties"],
        )

        # Coq proof template
        templates["coq_main"] = ProofTemplate(
            name="coq_main",
            prover="coq",
            template="""You are an expert in formal verification using Coq.

Given the following formal specification and verification goals, generate complete, valid Coq proofs.

CONTEXT:
{context}

FORMAL SPECIFICATION:
{formal_spec}

VERIFICATION GOALS:
{verification_goals}

PROPERTIES TO VERIFY:
{properties}

Requirements:
1. Generate complete, syntactically correct Coq vernacular
2. Use appropriate tactics: auto, simpl, rewrite, induction, etc.
3. Define necessary lemmas and helper functions
4. Provide clear proof structure using bullets and braces
5. Include informative comments
6. Handle all cases systematically

Generate the complete Coq file with all proofs:""",
            variables=["context", "formal_spec", "verification_goals", "properties"],
        )

        # Lemma-specific templates
        templates["isabelle_lemma"] = ProofTemplate(
            name="isabelle_lemma",
            prover="isabelle",
            template="""Generate an Isabelle proof for the following lemma:

CONTEXT:
{context}

LEMMA:
{lemma_statement}

Provide a complete proof using appropriate Isabelle tactics. The proof should be:
1. Syntactically correct
2. Logically sound
3. Well-structured with clear steps
4. Use efficient proof methods

Proof:""",
            variables=["context", "lemma_statement"],
        )

        templates["coq_lemma"] = ProofTemplate(
            name="coq_lemma",
            prover="coq",
            template="""Generate a Coq proof for the following lemma:

CONTEXT:
{context}

LEMMA:
{lemma_statement}

Provide a complete proof using appropriate Coq tactics. The proof should be:
1. Syntactically correct
2. Logically sound  
3. Well-structured with clear steps
4. Use efficient tactics

Proof:""",
            variables=["context", "lemma_statement"],
        )

        return templates

    def create_proof_prompt(
        self,
        formal_spec: str,
        verification_goals: str,
        properties: List[str],
        prover: str = "isabelle",
        context: str = "",
    ) -> str:
        """Create prompt for main proof generation.

        Args:
            formal_spec: Formal specification
            verification_goals: Goals to verify
            properties: List of properties
            prover: Target prover
            context: Additional context

        Returns:
            Generated prompt string
        """
        template_name = f"{prover}_main"
        template = self.templates.get(template_name)

        if not template:
            raise ValueError(f"No template found for {template_name}")

        # Format properties as string
        properties_str = "\n".join(f"- {prop}" for prop in properties)

        return template.template.format(
            context=context,
            formal_spec=formal_spec,
            verification_goals=verification_goals,
            properties=properties_str,
        )

    def create_lemma_prompt(
        self, lemma_statement: str, context: str = "", prover: str = "isabelle"
    ) -> str:
        """Create prompt for lemma proof generation."""
        template_name = f"{prover}_lemma"
        template = self.templates.get(template_name)

        if not template:
            raise ValueError(f"No template found for {template_name}")

        return template.template.format(
            context=context, lemma_statement=lemma_statement
        )

    def create_inductive_prompt(
        self,
        base_case: str,
        inductive_step: str,
        context: str = "",
        prover: str = "isabelle",
    ) -> str:
        """Create prompt for inductive proof generation."""
        inductive_template = f"""Generate an inductive proof using {prover.title()}.

CONTEXT:
{context}

BASE CASE:
{base_case}

INDUCTIVE STEP:
{inductive_step}

Generate a complete inductive proof that:
1. Clearly establishes the base case
2. Proves the inductive step assuming the inductive hypothesis
3. Uses appropriate induction principles
4. Is syntactically correct for {prover.title()}

Proof:"""

        return inductive_template

    def create_sketch_prompt(
        self, goal: str, context: str = "", prover: str = "isabelle"
    ) -> str:
        """Create prompt for proof sketch generation."""
        sketch_template = f"""Generate a high-level proof sketch for the following goal using {prover.title()}.

CONTEXT:
{context}

GOAL:
{goal}

Provide a structured proof sketch that outlines:
1. Main proof strategy and approach
2. Key lemmas that need to be established
3. Major proof steps and their justification
4. Potential challenges and how to address them
5. Alternative approaches if the main strategy fails

Focus on the logical structure rather than detailed tactics.

Proof Sketch:"""

        return sketch_template

    def create_refinement_prompt(
        self, original_proof: str, errors: List[str], prover: str = "isabelle"
    ) -> str:
        """Create prompt for proof refinement."""
        error_str = "\n".join(f"- {error}" for error in errors)

        refinement_template = f"""The following {prover.title()} proof has errors. Please fix them and provide a corrected version.

ORIGINAL PROOF:
{original_proof}

ERRORS ENCOUNTERED:
{error_str}

Please provide a corrected proof that:
1. Addresses all the errors listed above
2. Maintains the original proof strategy where possible
3. Is syntactically correct for {prover.title()}
4. Is logically sound and complete

CORRECTED PROOF:"""

        return refinement_template

    def create_optimization_prompt(self, proof: str, prover: str = "isabelle") -> str:
        """Create prompt for proof optimization."""
        optimization_template = f"""Optimize the following {prover.title()} proof to make it more concise and efficient.

ORIGINAL PROOF:
{proof}

Please provide an optimized version that:
1. Uses more efficient tactics and methods
2. Eliminates redundant steps
3. Combines related proof steps where possible
4. Maintains correctness and readability
5. Is syntactically correct for {prover.title()}

OPTIMIZED PROOF:"""

        return optimization_template

    def add_custom_template(
        self, name: str, prover: str, template: str, variables: List[str]
    ) -> None:
        """Add custom proof template."""
        self.templates[name] = ProofTemplate(
            name=name, prover=prover, template=template, variables=variables
        )

    def get_template(self, name: str) -> Optional[ProofTemplate]:
        """Get template by name."""
        return self.templates.get(name)
