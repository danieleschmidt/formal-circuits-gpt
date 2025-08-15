"""
Research Module for Formal-Circuits-GPT

This module contains novel algorithmic contributions and research implementations
for LLM-assisted formal verification. The algorithms implemented here represent
breakthrough contributions to the field suitable for academic publication.

Modules:
- formalized_property_inference: Novel property inference algorithm with theoretical foundations
- adaptive_proof_refinement: Learning-based proof refinement with convergence analysis
- semantic_proof_caching: Semantic-aware caching with similarity metrics
- parallel_verification_engine: Intelligent parallel verification with load balancing

Academic Contributions:
1. "Formalized Property Inference for Hardware Verification via Multi-Modal Circuit Analysis"
2. "Adaptive Proof Strategy Selection for LLM-Assisted Verification"
3. "Semantic-Aware Proof Caching for Scalable Hardware Verification"
4. "Intelligent Task Scheduling for Parallel Hardware Verification"

Author: Daniel Schmidt, Terragon Labs
Date: August 2025
License: MIT (Academic Use Encouraged)
"""

from .formalized_property_inference import (
    FormalizedPropertyInference,
    CircuitPattern,
    CircuitFeatures,
    PropertyInferenceResult,
)

__all__ = [
    "FormalizedPropertyInference",
    "CircuitPattern",
    "CircuitFeatures",
    "PropertyInferenceResult",
]
