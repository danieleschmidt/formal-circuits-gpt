#!/bin/bash

# Post-create script for Formal-Circuits-GPT development container
set -e

echo "🚀 Setting up Formal-Circuits-GPT development environment..."

# Install theorem provers
echo "📦 Installing theorem provers..."
sudo apt-get update

# Install Isabelle/HOL
echo "Installing Isabelle/HOL..."
wget -q https://isabelle.in.tum.de/dist/Isabelle2023_linux.tar.gz
tar -xzf Isabelle2023_linux.tar.gz
sudo mv Isabelle2023 /opt/isabelle
sudo ln -sf /opt/isabelle/bin/isabelle /usr/local/bin/isabelle
rm Isabelle2023_linux.tar.gz

# Install Coq
echo "Installing Coq..."
sudo apt-get install -y coq coq-theories

# Install additional development tools
echo "🛠️ Installing development tools..."
sudo apt-get install -y \
    make \
    build-essential \
    curl \
    wget \
    git \
    tree \
    jq \
    ripgrep \
    fd-find

# Install Python development dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create useful aliases
echo "⚡ Setting up aliases..."
cat >> ~/.bashrc << 'EOF'

# Formal-Circuits-GPT aliases
alias fcg="formal-circuits-gpt"
alias pytest="python -m pytest"
alias mypy="python -m mypy"
alias black="python -m black"
alias isort="python -m isort"
alias flake8="python -m flake8"

# Quick test commands
alias test-unit="pytest tests/ -m 'not integration' -v"
alias test-integration="pytest tests/ -m integration -v"
alias test-all="pytest tests/ -v"
alias test-cov="pytest --cov=src --cov-report=html --cov-report=term"

# Code quality
alias lint="flake8 src tests"
alias typecheck="mypy src"
alias format="black src tests && isort src tests"
alias quality="lint && typecheck && format"

# Git shortcuts
alias gst="git status"
alias gco="git checkout"
alias gcb="git checkout -b"
alias gpl="git pull"
alias gps="git push"
alias gcm="git commit -m"

# Development shortcuts
alias serve-docs="cd docs && python -m http.server 8000"
alias clean="find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
EOF

# Set up Git configuration
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Create development directories
echo "📁 Creating development directories..."
mkdir -p ~/.formal-circuits-gpt
mkdir -p ~/examples
mkdir -p ~/benchmarks

# Create example configuration
echo "⚙️ Creating example configuration..."
cat > ~/.formal-circuits-gpt/config.yaml << 'EOF'
# Formal-Circuits-GPT Development Configuration
llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4-turbo"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.1
  max_tokens: 4000

theorem_provers:
  isabelle:
    path: "/usr/local/bin/isabelle"
    timeout: 300
  coq:
    path: "/usr/bin/coq"
    timeout: 300

verification:
  default_prover: "isabelle"
  max_refinement_rounds: 5
  parallel_workers: 4
  cache_dir: "~/.formal-circuits-gpt/cache"

logging:
  level: "INFO"
  file: "~/.formal-circuits-gpt/logs/fcg.log"
EOF

# Create sample circuits for testing
echo "📄 Creating sample circuits..."
mkdir -p ~/examples/circuits

cat > ~/examples/circuits/simple_adder.v << 'EOF'
module simple_adder(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
EOF

cat > ~/examples/circuits/counter.v << 'EOF'
module counter #(
    parameter WIDTH = 8
)(
    input clk,
    input reset,
    input enable,
    output reg [WIDTH-1:0] count
);
    always @(posedge clk) begin
        if (reset)
            count <= 0;
        else if (enable)
            count <= count + 1;
    end
endmodule
EOF

# Test the installation
echo "🧪 Testing installation..."
if command -v formal-circuits-gpt &> /dev/null; then
    echo "✅ formal-circuits-gpt CLI is available"
    formal-circuits-gpt --version
else
    echo "❌ formal-circuits-gpt CLI not found"
fi

if command -v isabelle &> /dev/null; then
    echo "✅ Isabelle/HOL is available"
    isabelle version
else
    echo "❌ Isabelle/HOL not found"
fi

if command -v coq &> /dev/null; then
    echo "✅ Coq is available"
    coq --version
else
    echo "❌ Coq not found"
fi

# Check Python packages
echo "🐍 Checking Python packages..."
python -c "import formal_circuits_gpt; print('✅ formal_circuits_gpt package imported successfully')" 2>/dev/null || echo "❌ formal_circuits_gpt package not found"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Quick start commands:"
echo "  fcg --help                    # Show help"
echo "  fcg verify ~/examples/circuits/simple_adder.v  # Verify sample circuit"
echo "  test-unit                     # Run unit tests"
echo "  quality                       # Run code quality checks"
echo "  serve-docs                    # Serve documentation"
echo ""
echo "Configuration file: ~/.formal-circuits-gpt/config.yaml"
echo "Sample circuits: ~/examples/circuits/"
echo ""
echo "Don't forget to set your LLM API key:"
echo "  export OPENAI_API_KEY='your-api-key-here'"
echo "  # or"
echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
echo ""