# ADR-0001: Architecture Decision Record Template

## Status
Template

## Context
Architecture Decision Records (ADRs) help capture important architectural decisions along with their context and consequences. This template establishes a consistent format for all ADRs in this project.

## Decision
We will use this lightweight ADR template for documenting architectural decisions in the formal-circuits-gpt project.

## Consequences

### Positive
- Consistent documentation format across all architectural decisions
- Easy to track decision history and evolution
- Clear structure helps capture essential information
- Supports decision review and retrospectives

### Negative
- Requires discipline to maintain and update records
- Additional overhead for architectural changes

## Template Structure

```markdown
# ADR-XXXX: [Decision Title]

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-YYYY]

## Context
[Describe the architectural issue that we're addressing]

## Decision
[Describe the architectural decision and rationale]

## Consequences
### Positive
- [List positive outcomes]

### Negative
- [List negative outcomes and tradeoffs]
```

## Usage Guidelines

1. **Numbering**: Use sequential numbering starting from 0001
2. **Status**: Update status as decisions evolve
3. **Context**: Provide sufficient background for future readers
4. **Decision**: Be clear and concise about what was decided
5. **Consequences**: Be honest about both positive and negative impacts

## Review Process

All ADRs should be:
1. Created as part of the design discussion process
2. Reviewed by the development team
3. Updated when implementations change the decision
4. Referenced in code comments where relevant