# ADR-0002: LLM Integration Architecture

## Status
Accepted

## Context
The formal-circuits-gpt system requires integration with Large Language Models (LLMs) for proof generation and refinement. We need to decide on the architecture for LLM integration that supports multiple providers, manages API keys securely, handles rate limiting, and provides extensibility for future models.

Key requirements:
- Support for multiple LLM providers (OpenAI, Anthropic, local models)
- Secure API key management
- Rate limiting and error handling
- Prompt template management
- Response parsing and validation
- Cost tracking and optimization

## Decision
We will implement a plugin-based LLM integration architecture with the following components:

1. **Provider Interface**: Abstract base class defining the contract for LLM providers
2. **Provider Implementations**: Concrete implementations for each LLM service
3. **Prompt Manager**: Centralized prompt template management with versioning
4. **Response Parser**: Structured parsing and validation of LLM responses
5. **Rate Limiter**: Token bucket algorithm for API rate limiting
6. **Cost Tracker**: Monitor and log API usage and costs
7. **Configuration Manager**: Secure credential management and provider selection

### Architecture Components:

```python
# Abstract provider interface
class LLMProvider:
    def generate_proof(self, prompt: str, **kwargs) -> ProofResponse
    def refine_proof(self, failed_proof: str, error: str) -> ProofResponse
    def estimate_cost(self, prompt: str) -> float
```

### Provider Selection Strategy:
- Primary: GPT-4 for complex proof generation
- Fallback: GPT-3.5-turbo for simple proofs and refinement
- Local: Support for open-source models via API
- Cost optimization: Automatic provider selection based on prompt complexity

## Consequences

### Positive
- **Flexibility**: Easy to add new LLM providers or switch between them
- **Reliability**: Built-in fallback mechanisms and error handling
- **Security**: Centralized credential management with environment variable support
- **Cost Control**: Monitoring and cost optimization features
- **Maintainability**: Clear separation of concerns and modular design
- **Testing**: Easy to mock providers for testing
- **Performance**: Rate limiting prevents API quota exhaustion

### Negative
- **Complexity**: More complex than a single-provider implementation
- **Dependencies**: Multiple API dependencies increase potential failure points
- **Configuration**: Requires careful configuration management
- **Testing**: Need to test multiple provider implementations

## Implementation Details

### Configuration Example:
```yaml
llm:
  primary_provider: "openai"
  fallback_provider: "anthropic"
  providers:
    openai:
      model: "gpt-4-turbo"
      api_key_env: "OPENAI_API_KEY"
      rate_limit: 60  # requests per minute
    anthropic:
      model: "claude-3-5-sonnet"
      api_key_env: "ANTHROPIC_API_KEY"
      rate_limit: 50
```

### Security Considerations:
- API keys stored only in environment variables
- No API keys in logs or configuration files
- Rate limiting to prevent quota exhaustion
- Input sanitization for all prompts
- Response validation to prevent code injection

## Related ADRs
- ADR-0003: Prompt Template Management Strategy
- ADR-0004: Error Handling and Retry Mechanisms