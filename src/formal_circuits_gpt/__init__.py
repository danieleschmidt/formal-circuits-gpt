"""
Formal-Circuits-GPT: LLM-Assisted Hardware Verification

A Python package for automated formal verification of hardware circuits
using Large Language Models and theorem provers.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import CircuitVerifier
from .exceptions import ProofFailure, VerificationError

__all__ = ["CircuitVerifier", "ProofFailure", "VerificationError"]