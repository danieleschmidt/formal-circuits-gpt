"""LLM client for interfacing with different language model providers."""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    tokens_used: int
    model: str
    finish_reason: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}


class LLMError(Exception):
    """Base exception for LLM errors."""
    def __init__(self, message: str, provider: str = None, model: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs) -> LLMResponse:
        """Synchronous generation method."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("OpenAI API key not found")
        
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.sync_client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMError("OpenAI package not installed. Run: pip install openai")
    
    def generate_proof(self, prompt: str, **kwargs) -> str:
        """Generate proof using OpenAI API (synchronous)."""
        response = self.generate_sync(prompt, **kwargs)
        return response.content
    
    def refine_proof(self, proof: str, error_message: str, **kwargs) -> str:
        """Refine proof based on error feedback."""
        refinement_prompt = f"""The following proof has errors. Please fix them and provide a corrected version.

ORIGINAL PROOF:
{proof}

ERRORS:
{error_message}

Please provide the corrected proof addressing all the errors above:"""
        
        response = self.generate_sync(refinement_prompt, **kwargs)
        return response.content
    
    def estimate_cost(self, prompt: str) -> float:
        """Estimate cost for the prompt in USD."""
        # Rough token estimation (4 chars per token)
        input_tokens = len(prompt) // 4
        # Assume similar output length
        output_tokens = input_tokens
        
        # GPT-4 pricing (approximate as of 2024)
        input_cost_per_token = 0.00003  # $0.03 per 1K tokens
        output_cost_per_token = 0.00006  # $0.06 per 1K tokens
        
        total_cost = (input_tokens * input_cost_per_token + 
                     output_tokens * output_cost_per_token)
        
        return total_cost
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            temperature = kwargs.get("temperature", 0.1)
            max_tokens = kwargs.get("max_tokens", 2000)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                model=self.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={"provider": "openai"}
            )
            
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}") from e
    
    def generate_sync(self, prompt: str, **kwargs) -> LLMResponse:
        """Synchronous generation using OpenAI API."""
        try:
            temperature = kwargs.get("temperature", 0.1)
            max_tokens = kwargs.get("max_tokens", 2000)
            
            response = self.sync_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                model=self.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={"provider": "openai"}
            )
            
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}") from e


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic client."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("Anthropic API key not found")
        
        self.model = model
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.sync_client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise LLMError("Anthropic package not installed. Run: pip install anthropic")
    
    def generate_proof(self, prompt: str, **kwargs) -> str:
        """Generate proof using Anthropic API (synchronous)."""
        response = self.generate_sync(prompt, **kwargs)
        return response.content
    
    def refine_proof(self, proof: str, error_message: str, **kwargs) -> str:
        """Refine proof based on error feedback."""
        refinement_prompt = f"""The following proof has errors. Please fix them and provide a corrected version.

ORIGINAL PROOF:
{proof}

ERRORS:
{error_message}

Please provide the corrected proof addressing all the errors above:"""
        
        response = self.generate_sync(refinement_prompt, **kwargs)
        return response.content
    
    def estimate_cost(self, prompt: str) -> float:
        """Estimate cost for the prompt in USD."""
        # Rough token estimation (4 chars per token)
        input_tokens = len(prompt) // 4
        # Assume similar output length  
        output_tokens = input_tokens
        
        # Claude pricing (approximate as of 2024)
        input_cost_per_token = 0.000015  # $0.015 per 1K tokens
        output_cost_per_token = 0.000075  # $0.075 per 1K tokens
        
        total_cost = (input_tokens * input_cost_per_token + 
                     output_tokens * output_cost_per_token)
        
        return total_cost
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            temperature = kwargs.get("temperature", 0.1)
            max_tokens = kwargs.get("max_tokens", 2000)
            
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.content[0].text if response.content else ""
            
            return LLMResponse(
                content=content,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                model=self.model,
                finish_reason=response.stop_reason,
                metadata={"provider": "anthropic"}
            )
            
        except Exception as e:
            raise LLMError(f"Anthropic API error: {str(e)}") from e
    
    def generate_sync(self, prompt: str, **kwargs) -> LLMResponse:
        """Synchronous generation using Anthropic API."""
        try:
            temperature = kwargs.get("temperature", 0.1)
            max_tokens = kwargs.get("max_tokens", 2000)
            
            response = self.sync_client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.content[0].text if response.content else ""
            
            return LLMResponse(
                content=content,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                model=self.model,
                finish_reason=response.stop_reason,
                metadata={"provider": "anthropic"}
            )
            
        except Exception as e:
            raise LLMError(f"Anthropic API error: {str(e)}") from e


class LocalClient(LLMClient):
    """Local model client (placeholder implementation)."""
    
    def __init__(self, model_path: str, api_endpoint: str = "http://localhost:8000"):
        """Initialize local client."""
        self.model_path = model_path
        self.api_endpoint = api_endpoint
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using local model."""
        # Placeholder implementation
        return LLMResponse(
            content="Local model response placeholder",
            tokens_used=100,
            model=self.model_path,
            finish_reason="completed",
            metadata={"provider": "local"}
        )
    
    def generate_sync(self, prompt: str, **kwargs) -> LLMResponse:
        """Synchronous generation using local model."""
        # Placeholder implementation
        return LLMResponse(
            content="Local model response placeholder",
            tokens_used=100,
            model=self.model_path,
            finish_reason="completed",
            metadata={"provider": "local"}
        )


class LLMManager:
    """Manager for multiple LLM clients."""
    
    def __init__(self):
        """Initialize LLM manager."""
        self.clients: Dict[str, LLMClient] = {}
        self.default_client: Optional[str] = None
    
    def add_client(self, name: str, client: LLMClient, set_as_default: bool = False) -> None:
        """Add LLM client."""
        self.clients[name] = client
        if set_as_default or not self.default_client:
            self.default_client = name
    
    def get_client(self, name: Optional[str] = None) -> Optional[LLMClient]:
        """Get LLM client by name."""
        if name:
            return self.clients.get(name)
        elif self.default_client:
            return self.clients.get(self.default_client)
        return None
    
    async def generate(self, prompt: str, client_name: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response using specified or default client."""
        client = self.get_client(client_name)
        if not client:
            raise LLMError(f"Client '{client_name or self.default_client}' not found")
        
        return await client.generate(prompt, **kwargs)
    
    def generate_sync(self, prompt: str, client_name: Optional[str] = None, **kwargs) -> LLMResponse:
        """Synchronous generation using specified or default client."""
        client = self.get_client(client_name)
        if not client:
            raise LLMError(f"Client '{client_name or self.default_client}' not found")
        
        return client.generate_sync(prompt, **kwargs)
    
    @classmethod
    def create_default(cls) -> "LLMManager":
        """Create manager with default clients."""
        manager = cls()
        
        # Try to add OpenAI client
        try:
            openai_client = OpenAIClient()
            manager.add_client("openai", openai_client, set_as_default=True)
        except LLMError:
            pass
        
        # Try to add Anthropic client
        try:
            anthropic_client = AnthropicClient()
            manager.add_client("anthropic", anthropic_client, 
                             set_as_default=(not manager.default_client))
        except LLMError:
            pass
        
        # If no API clients available, add mock client for testing
        if not manager.default_client:
            mock_client = MockLLMClient()
            manager.add_client("mock", mock_client, set_as_default=True)
        
        return manager


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and offline use."""
    
    def __init__(self):
        """Initialize mock client."""
        self.model = "mock-model"
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response."""
        # Generate plausible Isabelle proof content
        proof_content = """
theory MockProof
imports Main
begin

definition circuit_correct :: "bool" where
  "circuit_correct = True"

theorem circuit_verification:
  "circuit_correct"
  by (simp add: circuit_correct_def)

end
"""
        return LLMResponse(
            content=proof_content,
            tokens_used=150,
            model=self.model,
            finish_reason="completed",
            metadata={"provider": "mock", "test_mode": True}
        )
    
    def generate_sync(self, prompt: str, **kwargs) -> LLMResponse:
        """Synchronous mock generation."""
        # Generate plausible Isabelle proof content
        proof_content = """
theory MockProof
imports Main
begin

definition circuit_correct :: "bool" where
  "circuit_correct = True"

theorem circuit_verification:
  "circuit_correct"
  by (simp add: circuit_correct_def)

end
"""
        return LLMResponse(
            content=proof_content,
            tokens_used=150,
            model=self.model,
            finish_reason="completed",
            metadata={"provider": "mock", "test_mode": True}
        )