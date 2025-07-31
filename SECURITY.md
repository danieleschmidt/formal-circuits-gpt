# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ Current development |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow responsible disclosure:

### How to Report

1. **Email**: Send details to security@terragonlabs.com
2. **Encrypted**: Use our PGP key for sensitive reports
3. **GitHub**: For non-sensitive issues, create a private security advisory

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)

### Response Timeline

- **24 hours**: Initial acknowledgment
- **72 hours**: Preliminary assessment
- **7 days**: Detailed response with timeline
- **30 days**: Security patch release (if confirmed)

### Our Commitment

- We will not take legal action against good-faith security researchers
- We will acknowledge your contribution (unless you prefer anonymity)
- We will keep you informed throughout the resolution process

## Security Considerations

### LLM API Keys
- Store API keys in environment variables only
- Never commit API keys to version control
- Use least-privilege API key permissions
- Rotate keys regularly

### Theorem Prover Execution
- Theorem provers run in sandboxed environments
- Resource limits applied to prevent DoS
- No arbitrary code execution from LLM responses
- Input validation for all HDL sources

### Data Handling
- No circuit designs stored without explicit consent
- Proof attempts logged with privacy controls
- API calls to LLM services use TLS encryption
- Local data encrypted at rest when possible

### Dependencies
- Regular security scanning of dependencies
- Automated vulnerability alerts enabled
- Minimum required permissions for all packages
- Lock file security validation

## Security Features

### Input Validation
- All HDL inputs sanitized before parsing
- Property specifications validated against schema
- File path traversal prevention
- Resource consumption limits

### Authentication
- API key validation before requests
- Token expiration handling
- Rate limiting on API calls
- Audit logging for security events

### Sandboxing
- Theorem provers isolated from file system
- Network access restricted during verification
- Temporary file cleanup after processing
- Process resource limits enforced

## Best Practices for Users

### API Key Management
```bash
# Use environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Never hardcode in scripts
# ❌ Don't do this
verifier = CircuitVerifier(api_key="sk-...")

# ✅ Do this instead  
verifier = CircuitVerifier()  # Reads from environment
```

### Secure Configuration
```yaml
# ~/.formal-circuits-gpt/config.yaml
api:
  timeout: 30
  max_retries: 3
  rate_limit: 10

security:
  sandbox_provers: true
  log_level: INFO
  audit_trail: true
```

### Circuit Code Review
- Review all circuit code before verification
- Validate property specifications
- Monitor resource usage during verification
- Use version control for all designs

## Threat Model

### In Scope
- Circuit design confidentiality
- API key compromise
- Malicious HDL input processing
- Theorem prover exploitation
- Supply chain attacks

### Out of Scope
- Physical security of systems running the tool
- Security of external theorem prover installations
- LLM service provider security (OpenAI, Anthropic)
- Network-level attacks (use standard protections)

## Compliance

This project follows security best practices including:
- OWASP Top 10 guidelines
- NIST Cybersecurity Framework
- Supply chain security (SLSA)
- Dependency vulnerability management

## Security Tooling

We use automated security scanning:
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability scanning  
- **CodeQL**: Static analysis security testing
- **Semgrep**: Custom security rule scanning

## Incident Response

In case of confirmed security incidents:

1. **Immediate**: Contain the issue and assess impact
2. **Communication**: Notify affected users within 24 hours
3. **Resolution**: Release patches with clear upgrade guidance
4. **Follow-up**: Post-incident review and process improvements

## Security Contact

- **General Security**: security@terragonlabs.com
- **Emergency**: +1-xxx-xxx-xxxx (24/7 security hotline)
- **PGP Key**: Available at https://terragonlabs.com/pgp

For general questions about this security policy, please open an issue or contact our security team.