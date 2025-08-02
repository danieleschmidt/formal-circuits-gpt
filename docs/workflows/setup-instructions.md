# GitHub Workflows Setup Instructions

## Overview

This document provides step-by-step instructions for setting up GitHub Actions workflows for formal-circuits-gpt. Since GitHub Apps may not have permissions to create workflow files directly, these must be set up manually by repository administrators.

## Prerequisites

### Repository Settings
1. **Actions Enabled**: Ensure GitHub Actions are enabled in repository settings
2. **Permissions**: Configure appropriate workflow permissions
3. **Branch Protection**: Set up branch protection rules for main branch

### Required Secrets
Configure the following secrets in repository settings (`Settings > Secrets and variables > Actions`):

#### Required Secrets
```bash
# PyPI Publishing
PYPI_API_TOKEN=pypi-...          # PyPI API token for package publishing
TEST_PYPI_API_TOKEN=pypi-...     # TestPyPI API token for testing

# Docker Registry
DOCKER_USERNAME=your_username     # Docker Hub username
DOCKER_PASSWORD=your_token        # Docker Hub access token

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...  # Slack webhook for notifications

# Security Scanning (Optional)
SNYK_TOKEN=your_snyk_token       # Snyk security scanning token
```

#### Optional Secrets for Advanced Features
```bash
# Code Coverage
CODECOV_TOKEN=your_codecov_token

# Additional Integrations
DISCORD_WEBHOOK=your_discord_webhook
TEAMS_WEBHOOK=your_teams_webhook
```

## Step 1: Create Workflow Directory

Create the `.github/workflows/` directory in your repository root:

```bash
mkdir -p .github/workflows/
```

## Step 2: Copy Workflow Files

Copy the example workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Copy main CI workflow
cp docs/workflows/examples/ci.yml .github/workflows/

# Copy security scanning workflow
cp docs/workflows/examples/security.yml .github/workflows/

# Copy release workflow
cp docs/workflows/examples/release.yml .github/workflows/

# Optional: Copy additional workflows
cp docs/workflows/examples/docs.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

## Step 3: Configure Additional Files

### CodeQL Configuration
Create `.github/codeql/codeql-config.yml`:

```yaml
name: "CodeQL Config"

queries:
  - uses: security-and-quality
  - uses: security-extended

paths-ignore:
  - "tests/**"
  - "docs/**"
  - "examples/**"

paths:
  - "src/**"
```

### Dependabot Configuration
Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "@core-team"
    assignees:
      - "@maintainer"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
```

### Issue Templates
Create `.github/ISSUE_TEMPLATE/bug_report.yml`:

```yaml
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!

  - type: input
    id: version
    attributes:
      label: Version
      description: What version of formal-circuits-gpt are you using?
      placeholder: e.g., 0.1.0
    validations:
      required: true

  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true

  - type: textarea
    id: circuit-code
    attributes:
      label: Circuit Code
      description: Please provide the Verilog/VHDL code that caused the issue
      render: verilog
    validations:
      required: false

  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output
      render: shell
    validations:
      required: false
```

Create `.github/ISSUE_TEMPLATE/feature_request.yml`:

```yaml
name: Feature Request
description: Suggest an idea for this project
title: "[Feature]: "
labels: ["enhancement", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!

  - type: textarea
    id: problem
    attributes:
      label: Problem Description
      description: Is your feature request related to a problem? Please describe.
      placeholder: A clear and concise description of what the problem is.
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe the solution you'd like
      placeholder: A clear and concise description of what you want to happen.
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Describe alternatives you've considered
      placeholder: A clear and concise description of any alternative solutions.
    validations:
      required: false
```

### Pull Request Template
Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description
Brief description of the changes in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested this change with real circuits
- [ ] Performance impact has been evaluated

## Circuit Examples
If applicable, provide examples of circuits this change affects:

```verilog
// Example circuit code
```

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Related Issues
Closes #(issue_number)
```

## Step 4: Configure Branch Protection

Set up branch protection rules for the main branch:

1. Go to `Settings > Branches`
2. Add rule for `main` branch
3. Configure the following settings:

```yaml
Branch Protection Settings:
✅ Require a pull request before merging
✅ Require approvals (minimum 1)
✅ Dismiss stale PR approvals when new commits are pushed
✅ Require review from code owners
✅ Require status checks to pass before merging
✅ Require branches to be up to date before merging

Required Status Checks:
- lint
- test (ubuntu-latest, 3.11)
- build
- docker
- dependency-check

✅ Require conversation resolution before merging
✅ Restrict pushes that create files over 100MB
✅ Do not allow bypassing the above settings
```

## Step 5: Configure Repository Settings

### General Settings
```yaml
Repository Settings:
- Default branch: main
- Allow merge commits: ✅
- Allow squash merging: ✅ (default)
- Allow rebase merging: ❌
- Automatically delete head branches: ✅
```

### Security Settings
```yaml
Security Settings:
✅ Enable dependency graph
✅ Enable Dependabot alerts
✅ Enable Dependabot security updates
✅ Enable secret scanning
✅ Enable push protection for secrets
```

### Actions Settings
```yaml
Actions Permissions:
- Allow all actions and reusable workflows: ✅

Workflow Permissions:
- Read repository contents and packages permissions: ✅
- Write repository contents and packages permissions: ✅

Fork Pull Request Workflows:
- Run workflows from fork pull requests: ✅
- Send write tokens to workflows from fork pull requests: ❌
- Send secrets to workflows from fork pull requests: ❌
```

## Step 6: Set Up External Integrations

### Codecov
1. Sign up at [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Add `CODECOV_TOKEN` to repository secrets
4. Configure `.codecov.yml` in repository root:

```yaml
coverage:
  status:
    project:
      default:
        target: 80%
        threshold: 5%
    patch:
      default:
        target: 80%
        threshold: 10%

ignore:
  - "tests/"
  - "docs/"
  - "examples/"

comment:
  layout: "reach,diff,flags,tree"
  behavior: default
  require_changes: false
```

### Snyk (Optional)
1. Sign up at [snyk.io](https://snyk.io)
2. Connect your GitHub repository
3. Add `SNYK_TOKEN` to repository secrets

### Docker Hub
1. Create Docker Hub account
2. Generate access token
3. Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets

## Step 7: Test Workflows

### Test CI Workflow
1. Create a test branch
2. Make a small change
3. Create pull request
4. Verify all checks pass

### Test Security Workflow
1. Trigger manually or wait for scheduled run
2. Check Security tab for results
3. Verify SARIF uploads work

### Test Release Workflow
1. Create a test tag: `git tag v0.1.0-test`
2. Push tag: `git push origin v0.1.0-test`
3. Verify release is created
4. Test PyPI publishing to TestPyPI

## Step 8: Configure Notifications

### Slack Integration
1. Create Slack app with incoming webhooks
2. Add webhook URL to `SLACK_WEBHOOK_URL` secret
3. Configure desired channels

### Email Notifications
Configure in GitHub notification settings for:
- Workflow failures
- Security alerts
- Dependabot notifications

## Troubleshooting

### Common Issues

#### Workflow Fails to Start
- Check Actions are enabled
- Verify workflow syntax with GitHub Actions validator
- Check repository permissions

#### Secret Not Available
- Verify secret name matches exactly
- Check secret is added to correct repository/organization
- Ensure secret is not empty

#### Permission Denied
- Check workflow permissions configuration
- Verify GitHub token has required scopes
- Check repository collaborator permissions

#### Tests Failing
- Run tests locally first
- Check system dependencies are installed
- Verify test data is available

### Getting Help

1. **Check Workflow Logs**: Detailed information in Actions tab
2. **GitHub Docs**: [GitHub Actions Documentation](https://docs.github.com/en/actions)
3. **Community**: Ask in project discussions
4. **Issues**: Create issue with `workflow` label

## Maintenance

### Regular Updates
- Review Dependabot PRs weekly
- Update workflow actions monthly
- Monitor security scan results
- Update secrets before expiration

### Performance Monitoring
- Track workflow duration trends
- Monitor resource usage
- Optimize slow jobs
- Consider workflow parallelization

---

**Last Updated:** August 2025  
**Next Review:** September 2025  
**Owner:** DevOps Team