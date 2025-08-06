"""LLM integration for proof generation and refinement."""

from .proof_generator import ProofGenerator
from .proof_refiner import ProofRefiner
from .prompt_manager import PromptManager
from .response_parser import ResponseParser
from .llm_client import (
    LLMClient,
    LLMProvider,
    LLMError,
    LLMResponse,
    OpenAIClient,
    AnthropicClient,
    LLMManager
)

# Aliases for backward compatibility with test expectations
OpenAIProvider = OpenAIClient
AnthropicProvider = AnthropicClient

__all__ = [
    "ProofGenerator",
    "ProofRefiner",
    "PromptManager", 
    "ResponseParser",
    "LLMClient",
    "LLMProvider",
    "LLMError", 
    "LLMResponse",
    "OpenAIClient",
    "AnthropicClient",
    "OpenAIProvider",  # Alias
    "AnthropicProvider",  # Alias
    "LLMManager"
]