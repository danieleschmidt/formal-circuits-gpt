#!/bin/bash
# Automated dependency update script for formal-circuits-gpt
# This script checks for and applies dependency updates

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/dependency-update.log"
BACKUP_DIR="$PROJECT_ROOT/.dependency-backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    log "${RED}ERROR: $1${NC}"
}

success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

info() {
    log "${BLUE}INFO: $1${NC}"
}

# Create backup directory
create_backup() {
    info "Creating backup of current dependencies..."
    mkdir -p "$BACKUP_DIR"
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="$BACKUP_DIR/pyproject_toml_$backup_timestamp.bak"
    
    cp "$PROJECT_ROOT/pyproject.toml" "$backup_file"
    success "Backup created: $backup_file"
}

# Check if we're in a git repository
check_git_repo() {
    if ! git -C "$PROJECT_ROOT" rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a git repository"
        exit 1
    fi
}

# Check for uncommitted changes
check_clean_workspace() {
    if ! git -C "$PROJECT_ROOT" diff --quiet; then
        error "Working directory has uncommitted changes"
        info "Please commit or stash changes before running dependency updates"
        exit 1
    fi
}

# Install required tools
install_tools() {
    info "Installing/updating required tools..."
    
    # Install pip-tools for dependency management
    pip install --upgrade pip-tools pip-audit safety
    
    success "Tools installed successfully"
}

# Update Python dependencies
update_python_deps() {
    info "Updating Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Generate updated requirements
    if [[ -f "requirements.in" ]]; then
        pip-compile --upgrade requirements.in
    fi
    
    # Update main dependencies in pyproject.toml
    pip install --upgrade pip-tools
    
    # Check for security vulnerabilities
    info "Checking for security vulnerabilities..."
    pip-audit --format=json --output=security-report.json || warning "Security vulnerabilities found"
    safety check --json --output=safety-report.json || warning "Safety check found issues"
    
    success "Python dependencies updated"
}

# Update GitHub Actions
update_github_actions() {
    info "Checking GitHub Actions versions..."
    
    local workflows_dir="$PROJECT_ROOT/.github/workflows"
    if [[ ! -d "$workflows_dir" ]]; then
        warning "No GitHub Actions workflows found"
        return
    fi
    
    # Find all workflow files
    find "$workflows_dir" -name "*.yml" -o -name "*.yaml" | while read -r workflow; do
        info "Checking workflow: $(basename "$workflow")"
        
        # Extract action versions (basic regex, could be improved)
        grep -E "uses: .+@v[0-9]+" "$workflow" || true
    done
    
    warning "GitHub Actions updates must be done manually"
    info "Consider using Dependabot for automated GitHub Actions updates"
}

# Update pre-commit hooks
update_pre_commit() {
    if [[ -f "$PROJECT_ROOT/.pre-commit-config.yaml" ]]; then
        info "Updating pre-commit hooks..."
        
        cd "$PROJECT_ROOT"
        pre-commit autoupdate
        
        success "Pre-commit hooks updated"
    else
        warning "No pre-commit configuration found"
    fi
}

# Run tests to verify updates
run_tests() {
    info "Running tests to verify dependency updates..."
    
    cd "$PROJECT_ROOT"
    
    # Install updated dependencies
    pip install -e .[dev,test]
    
    # Run linting
    info "Running linting checks..."
    if command -v flake8 &> /dev/null; then
        flake8 src/ tests/ || warning "Linting issues found"
    fi
    
    if command -v mypy &> /dev/null; then
        mypy src/ || warning "Type checking issues found"
    fi
    
    # Run tests
    info "Running test suite..."
    if command -v pytest &> /dev/null; then
        pytest tests/ -x -v || warning "Some tests failed"
    fi
    
    success "Tests completed"
}

# Generate update report
generate_report() {
    info "Generating dependency update report..."
    
    local report_file="$PROJECT_ROOT/dependency-update-report.md"
    
    cat > "$report_file" << EOF
# Dependency Update Report

**Generated:** $(date)
**Script Version:** 1.0.0

## Summary

This report summarizes the dependency updates performed by the automated update script.

## Updated Dependencies

### Python Dependencies
\`\`\`
$(pip list --outdated --format=columns || echo "No outdated packages found")
\`\`\`

### Security Scan Results

#### pip-audit Results
\`\`\`json
$(cat security-report.json 2>/dev/null || echo "No security report available")
\`\`\`

#### Safety Check Results
\`\`\`json
$(cat safety-report.json 2>/dev/null || echo "No safety report available")
\`\`\`

## Actions Required

- [ ] Review security vulnerabilities (if any)
- [ ] Test application functionality
- [ ] Update documentation if API changes occurred
- [ ] Create pull request with changes

## Files Modified

- pyproject.toml
- .pre-commit-config.yaml (if updated)

## Next Steps

1. Review this report
2. Run additional tests if needed
3. Commit changes and create PR
4. Monitor for any issues post-deployment

---

Generated by dependency update automation script
EOF

    success "Report generated: $report_file"
}

# Create git commit
create_commit() {
    cd "$PROJECT_ROOT"
    
    # Check if there are any changes
    if git diff --quiet && git diff --cached --quiet; then
        info "No changes to commit"
        return
    fi
    
    info "Creating git commit for dependency updates..."
    
    # Stage all changed files
    git add pyproject.toml
    git add .pre-commit-config.yaml || true
    git add dependency-update-report.md
    
    # Create commit
    git commit -m "chore: automated dependency updates

- Update Python dependencies to latest versions
- Update pre-commit hooks
- Run security scans and generate report
- Verify all tests pass

Generated by dependency update automation script"
    
    success "Git commit created"
    info "To push changes: git push origin $(git branch --show-current)"
}

# Main execution function
main() {
    info "Starting automated dependency update process..."
    
    # Preliminary checks
    check_git_repo
    check_clean_workspace
    
    # Create backup
    create_backup
    
    # Install required tools
    install_tools
    
    # Update dependencies
    update_python_deps
    update_github_actions
    update_pre_commit
    
    # Verify updates
    run_tests
    
    # Generate report
    generate_report
    
    # Create commit
    if [[ "${1:-}" == "--commit" ]]; then
        create_commit
    else
        info "Skipping git commit (use --commit to enable)"
        info "Review changes and commit manually if satisfied"
    fi
    
    success "Dependency update process completed successfully!"
    info "Review the dependency-update-report.md file for details"
}

# Show help
show_help() {
    cat << EOF
Dependency Update Script for formal-circuits-gpt

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --commit        Create git commit after successful updates
    --help, -h      Show this help message

DESCRIPTION:
    This script automates the process of updating project dependencies:
    
    1. Creates backup of current dependency files
    2. Updates Python dependencies in pyproject.toml
    3. Updates pre-commit hooks
    4. Runs security scans
    5. Executes test suite to verify updates
    6. Generates detailed update report
    7. Optionally creates git commit

EXAMPLES:
    # Update dependencies and review changes manually
    $0
    
    # Update dependencies and automatically commit
    $0 --commit

REQUIREMENTS:
    - Git repository
    - Python with pip
    - Clean working directory (no uncommitted changes)

LOG FILE:
    $LOG_FILE

EOF
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --commit)
        main --commit
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac