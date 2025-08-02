#!/bin/bash

# Development container setup script for formal-circuits-gpt

set -e

echo "🚀 Setting up Formal-Circuits-GPT development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "/opt/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv /opt/venv
fi

# Activate virtual environment
source /opt/venv/bin/activate

# Upgrade pip and essential tools
echo "⬆️ Upgrading pip and essential tools..."
pip install --upgrade pip setuptools wheel

# Install the project in development mode with all extras
echo "📚 Installing formal-circuits-gpt in development mode..."
pip install -e .[dev,test,docs]

# Install theorem provers
echo "🔧 Installing theorem provers..."

# Install Isabelle (if not already installed)
if ! command -v isabelle &> /dev/null; then
    echo "Installing Isabelle/HOL..."
    sudo apt-get update
    sudo apt-get install -y wget default-jdk
    
    # Download and install Isabelle
    ISABELLE_VERSION="Isabelle2024"
    cd /tmp
    wget -q "https://isabelle.in.tum.de/dist/${ISABELLE_VERSION}.tar.gz"
    tar -xzf "${ISABELLE_VERSION}.tar.gz"
    sudo mv "${ISABELLE_VERSION}" /opt/
    sudo ln -sf "/opt/${ISABELLE_VERSION}/bin/isabelle" /usr/local/bin/isabelle
    rm "${ISABELLE_VERSION}.tar.gz"
    
    echo "✅ Isabelle installed successfully"
else
    echo "✅ Isabelle already installed"
fi

# Install Coq (if not already installed)
if ! command -v coq &> /dev/null; then
    echo "Installing Coq..."
    sudo apt-get update
    sudo apt-get install -y coq
    echo "✅ Coq installed successfully"
else
    echo "✅ Coq already installed"
fi

# Install additional development tools
echo "🛠️ Installing additional development tools..."
sudo apt-get install -y \
    make \
    git \
    curl \
    jq \
    tree \
    ripgrep \
    fd-find

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create useful aliases
echo "🔗 Setting up useful aliases..."
cat >> ~/.bashrc << 'EOF'

# Formal-Circuits-GPT development aliases
alias fcg='formal-circuits-gpt'
alias pytest-cov='pytest --cov=formal_circuits_gpt --cov-report=html --cov-report=term'
alias lint-check='flake8 src/ tests/ && mypy src/ && black --check src/ tests/ && isort --check-only src/ tests/'
alias lint-fix='black src/ tests/ && isort src/ tests/'
alias docs-serve='cd docs && python -m http.server 8000'

# Python development helpers
alias venv-activate='source /opt/venv/bin/activate'
alias pip-list='pip list --format=columns'
alias pytest-fast='pytest -x -v'

# Git helpers
alias git-clean='git clean -fd && git reset --hard'
alias git-branches='git branch -av'

# Docker helpers (if needed)
alias docker-clean='docker system prune -f'
alias docker-ps='docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
EOF

# Create project-specific directories
echo "📁 Creating project directories..."
mkdir -p \
    ~/.formal-circuits-gpt \
    /tmp/formal-circuits-verification \
    /workspaces/formal-circuits-gpt/examples/output \
    /workspaces/formal-circuits-gpt/benchmarks/results

# Set up development configuration
echo "⚙️ Setting up development configuration..."
cat > ~/.formal-circuits-gpt/config.yaml << 'EOF'
# Development configuration for formal-circuits-gpt
default:
  prover: "isabelle"
  model: "gpt-4-turbo"
  timeout: 1800
  debug: true

# LLM providers configuration
llm:
  openai:
    model: "gpt-4-turbo"
    temperature: 0.1
    max_tokens: 4000
  anthropic:
    model: "claude-3-sonnet"
    temperature: 0.1
    max_tokens: 4000

# Development settings
development:
  save_intermediate_files: true
  verbose_logging: true
  cache_enabled: true
  
# Caching configuration  
cache:
  enabled: true
  directory: "/tmp/formal-circuits-verification"
  max_size: "1GB"
  ttl: 3600
EOF

# Set up example environment file
echo "🔧 Setting up environment template..."
cat > /workspaces/formal-circuits-gpt/.env.example << 'EOF'
# LLM API Keys (required for verification)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Local LLM configuration
LOCAL_LLM_URL=http://localhost:8080/v1
LOCAL_LLM_MODEL=codellama

# Development settings
FORMAL_CIRCUITS_DEBUG=true
FORMAL_CIRCUITS_LOG_LEVEL=DEBUG
FORMAL_CIRCUITS_CACHE_DIR=/tmp/formal-circuits-verification

# Theorem prover paths (auto-detected if installed)
ISABELLE_PATH=/usr/local/bin/isabelle
COQ_PATH=/usr/bin/coq

# Performance settings
FORMAL_CIRCUITS_MAX_WORKERS=4
FORMAL_CIRCUITS_TIMEOUT=1800

# Testing configuration
PYTEST_DISABLE_WARNINGS=true
PYTEST_TIMEOUT=60
EOF

# Verify installation
echo "🔍 Verifying installation..."
python -c "import formal_circuits_gpt; print('✅ formal-circuits-gpt imported successfully')"

# Check theorem provers
if command -v isabelle &> /dev/null; then
    echo "✅ Isabelle available at $(which isabelle)"
else
    echo "⚠️ Isabelle not found in PATH"
fi

if command -v coq &> /dev/null; then
    echo "✅ Coq available at $(which coq)"
else
    echo "⚠️ Coq not found in PATH"
fi

# Run a quick test
echo "🧪 Running quick validation test..."
cd /workspaces/formal-circuits-gpt
python -c "
try:
    from formal_circuits_gpt.cli import main
    print('✅ CLI module loaded successfully')
except ImportError as e:
    print(f'⚠️ CLI import issue: {e}')
"

echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Copy .env.example to .env and add your API keys"
echo "   2. Run 'pytest' to verify everything works"
echo "   3. Try 'formal-circuits-gpt --help' to see available commands"
echo "   4. Check out examples/ directory for sample circuits"
echo ""
echo "🔗 Useful commands:"
echo "   - fcg --help           # Show CLI help"
echo "   - pytest-cov          # Run tests with coverage"
echo "   - lint-check          # Check code style"
echo "   - lint-fix            # Fix code style"
echo "   - docs-serve          # Serve documentation locally"
echo ""
echo "Happy coding! 🚀"