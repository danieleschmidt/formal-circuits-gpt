# Manual Setup Required

## Overview

Due to GitHub App permission limitations, the following setup steps require manual action by repository administrators. This document provides step-by-step instructions for completing the SDLC implementation.

## Required Permissions

The following GitHub permissions are needed but cannot be set automatically:

### Repository Permissions
- **Actions**: Write (to create workflow files)
- **Administration**: Write (to configure branch protection)
- **Issues**: Write (to create issue templates)
- **Pull Requests**: Write (to create PR templates)
- **Secrets**: Write (to configure repository secrets)

### Organization Permissions (if applicable)
- **Organization Administration**: Read (to access organization-level settings)
- **Organization Secrets**: Write (to use organization-level secrets)

## Step 1: GitHub Workflows Setup

### 1.1 Create Workflow Files
Copy the workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows/
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
```

### 1.2 Configure Repository Secrets
Go to `Settings > Secrets and variables > Actions` and add:

#### Required Secrets
```
PYPI_API_TOKEN=pypi-...                    # PyPI publishing
TEST_PYPI_API_TOKEN=pypi-...               # TestPyPI for testing
DOCKER_USERNAME=your_username              # Docker Hub
DOCKER_PASSWORD=your_token                 # Docker Hub token
SLACK_WEBHOOK_URL=https://hooks.slack.com/... # Notifications
```

#### Optional Secrets
```
CODECOV_TOKEN=your_codecov_token           # Code coverage
SNYK_TOKEN=your_snyk_token                 # Security scanning
```

### 1.3 Enable GitHub Actions
1. Go to `Settings > Actions > General`
2. Select "Allow all actions and reusable workflows"
3. Set workflow permissions to "Read and write permissions"

## Step 2: Branch Protection Rules

Configure branch protection for the `main` branch:

1. Go to `Settings > Branches`
2. Add rule for `main` branch
3. Configure these settings:

```yaml
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

## Step 3: Repository Configuration

### 3.1 General Settings
Go to `Settings > General`:

```yaml
✅ Allow merge commits
✅ Allow squash merging (set as default)
❌ Allow rebase merging
✅ Automatically delete head branches
✅ Allow auto-merge
✅ Allow update branch
```

### 3.2 Security Settings
Go to `Settings > Security`:

```yaml
✅ Enable dependency graph
✅ Enable Dependabot alerts
✅ Enable Dependabot security updates
✅ Enable secret scanning
✅ Enable push protection for secrets
```

### 3.3 Issue and PR Templates
Create the following files:

#### `.github/ISSUE_TEMPLATE/bug_report.yml`
```yaml
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "needs-triage"]
body:
  - type: input
    id: version
    attributes:
      label: Version
      description: What version are you using?
    validations:
      required: true
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: Describe the bug
    validations:
      required: true
```

#### `.github/PULL_REQUEST_TEMPLATE.md`
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## Step 4: External Integrations

### 4.1 Codecov Integration
1. Sign up at [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Add `CODECOV_TOKEN` to repository secrets
4. Create `.codecov.yml`:

```yaml
coverage:
  status:
    project:
      default:
        target: 80%
    patch:
      default:
        target: 80%
```

### 4.2 Dependabot Configuration
Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 4.3 CodeQL Configuration
Create `.github/codeql/codeql-config.yml`:

```yaml
name: "CodeQL Config"
queries:
  - uses: security-and-quality
paths-ignore:
  - "tests/**"
  - "docs/**"
```

## Step 5: Additional Configurations

### 5.1 CODEOWNERS File
Create `.github/CODEOWNERS`:

```
# Global owners
* @core-team

# Specific areas
/docs/ @docs-team
/tests/ @qa-team
/.github/ @devops-team
```

### 5.2 Repository Topics
Go to `Settings > General` and add topics:
- `formal-verification`
- `hardware-verification`
- `llm`
- `verilog`
- `vhdl`
- `theorem-proving`
- `python`

### 5.3 Repository Description
Update the repository description to:
"LLM-Assisted Hardware Verification for Verilog/VHDL circuits using formal methods"

## Step 6: Verification and Testing

### 6.1 Test Workflows
1. Create a test branch
2. Make a small change
3. Create a pull request
4. Verify all checks run and pass

### 6.2 Test Releases
1. Create a test tag: `git tag v0.1.0-test`
2. Push the tag: `git push origin v0.1.0-test`
3. Verify release workflow runs
4. Check that release is created on GitHub

### 6.3 Test Security Scans
1. Go to `Actions` tab
2. Run the Security workflow manually
3. Check that results appear in Security tab

## Step 7: Documentation Updates

### 7.1 Update README
Ensure the main README.md reflects all new features and setup instructions.

### 7.2 Contributing Guidelines
Review and update CONTRIBUTING.md with the new workflow information.

### 7.3 Developer Documentation
Update docs/guides/developer-guide.md with workflow and setup information.

## Step 8: Monitoring Setup

### 8.1 GitHub Insights
Enable GitHub Insights to track:
- Code frequency
- Commit activity
- Contributors
- Traffic

### 8.2 Workflow Monitoring
Set up monitoring for:
- Workflow success rates
- Build times
- Test execution times
- Deployment frequency

## Troubleshooting

### Common Issues

#### Workflows Not Running
- Check that Actions are enabled
- Verify workflow syntax
- Check repository permissions

#### Secret Access Issues
- Verify secret names match exactly
- Check secret values are correct
- Ensure secrets are available to workflows

#### Branch Protection Conflicts
- Check that required status checks exist
- Verify branch names are correct
- Ensure team permissions are set

### Getting Help

1. Check workflow logs in Actions tab
2. Review GitHub documentation
3. Contact repository administrators
4. Create an issue with the `setup` label

## Success Criteria

The setup is complete when:

✅ All workflows run successfully on PRs
✅ Branch protection rules are enforced
✅ Security scans execute and report results
✅ Release workflow can create releases
✅ External integrations are working
✅ All team members can contribute following the new process

## Next Steps

After completing this setup:

1. **Team Training**: Ensure all team members understand the new workflow
2. **Process Documentation**: Update team processes to reflect new procedures
3. **Regular Reviews**: Schedule periodic reviews of the SDLC implementation
4. **Continuous Improvement**: Monitor metrics and optimize processes

---

**Completion Checklist:**
- [ ] GitHub workflows created and tested
- [ ] Branch protection rules configured
- [ ] Repository secrets added
- [ ] External integrations configured
- [ ] Issue and PR templates created
- [ ] CODEOWNERS file created
- [ ] All workflows tested and working
- [ ] Team trained on new processes
- [ ] Documentation updated

**Estimated Setup Time:** 2-4 hours
**Required Role:** Repository Administrator
**Dependencies:** GitHub repository, PyPI account, Docker Hub account (optional)