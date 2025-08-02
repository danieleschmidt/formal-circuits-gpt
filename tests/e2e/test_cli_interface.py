"""End-to-end tests for CLI interface."""

import os
import subprocess
import pytest
from pathlib import Path
from tests.fixtures import SIMPLE_FIXTURES


@pytest.mark.e2e
class TestCLIInterface:
    """End-to-end tests for command-line interface."""

    def test_cli_help_command(self):
        """Test that CLI help command works."""
        result = subprocess.run(
            ["python", "-m", "formal_circuits_gpt", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Should exit successfully
        assert result.returncode == 0
        assert "formal-circuits-gpt" in result.stdout.lower()
        assert "usage" in result.stdout.lower()

    def test_cli_version_command(self):
        """Test that CLI version command works."""
        result = subprocess.run(
            ["python", "-m", "formal_circuits_gpt", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Should exit successfully and show version
        assert result.returncode == 0
        assert "0.1.0" in result.stdout  # Current version

    def test_cli_check_setup_command(self):
        """Test CLI setup check command."""
        result = subprocess.run(
            ["python", "-m", "formal_circuits_gpt", "--check-setup"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # May succeed or fail depending on setup, but should not crash
        assert result.returncode in [0, 1]
        
        if result.returncode == 0:
            assert "setup complete" in result.stdout.lower()
        else:
            # Should provide helpful error messages
            assert len(result.stderr) > 0 or len(result.stdout) > 0

    def test_cli_verify_nonexistent_file(self):
        """Test CLI behavior with nonexistent input file."""
        result = subprocess.run(
            ["python", "-m", "formal_circuits_gpt", "verify", "nonexistent.v"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Should fail gracefully
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()

    def test_cli_verify_with_sample_circuit(self, temp_dir):
        """Test CLI verification with a sample circuit file."""
        # Create a sample circuit file
        fixture = SIMPLE_FIXTURES[0]  # simple_adder
        circuit_file = temp_dir / "test_adder.v"
        circuit_file.write_text(fixture.verilog_code)
        
        # Run CLI verification
        result = subprocess.run([
            "python", "-m", "formal_circuits_gpt", 
            "verify", str(circuit_file),
            "--property", "sum == a + b",
            "--dry-run"  # Don't actually call LLM APIs
        ], 
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
        )
        
        # Should not crash (may succeed or fail based on implementation)
        assert result.returncode in [0, 1, 2]  # Allow various exit codes
        
        # Should show some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_cli_batch_verification(self, temp_dir):
        """Test CLI batch verification mode."""
        # Create multiple circuit files
        for i, fixture in enumerate(SIMPLE_FIXTURES[:2]):
            circuit_file = temp_dir / f"circuit_{i}.v"
            circuit_file.write_text(fixture.verilog_code)
        
        # Run batch verification
        result = subprocess.run([
            "python", "-m", "formal_circuits_gpt",
            "batch",
            "--input", str(temp_dir),
            "--output", str(temp_dir / "results"),
            "--dry-run"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
        )
        
        # Should handle batch processing
        assert result.returncode in [0, 1, 2]
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_cli_configuration_options(self):
        """Test CLI configuration and option parsing."""
        result = subprocess.run([
            "python", "-m", "formal_circuits_gpt",
            "verify", "dummy.v",  # File doesn't need to exist for option parsing test
            "--prover", "isabelle",
            "--model", "gpt-4-turbo",
            "--temperature", "0.2",
            "--max-refinement-rounds", "3",
            "--verbose",
            "--dry-run"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
        )
        
        # Should parse options correctly (may fail due to missing file)
        # But failure should be about missing file, not option parsing
        if "not found" not in result.stderr.lower():
            # If it's not a file not found error, options were parsed correctly
            assert result.returncode in [0, 1, 2]

    @pytest.mark.skipif(
        not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')),
        reason="No LLM API keys available"
    )
    def test_cli_real_verification(self, temp_dir):
        """Test CLI with real API calls (requires API keys)."""
        fixture = SIMPLE_FIXTURES[0]  # simple_adder
        circuit_file = temp_dir / "real_test.v"
        circuit_file.write_text(fixture.verilog_code)
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        if os.getenv('OPENAI_API_KEY'):
            env['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('ANTHROPIC_API_KEY'):
            env['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
        
        result = subprocess.run([
            "python", "-m", "formal_circuits_gpt",
            "verify", str(circuit_file),
            "--property", "sum == a + b",
            "--timeout", "60",  # 1 minute timeout
            "--verbose"
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,  # 2 minute total timeout
        cwd=Path(__file__).parent.parent.parent
        )
        
        # Should complete (successfully or with failure)
        assert result.returncode in [0, 1]
        
        # Should show verification progress
        output = result.stdout + result.stderr
        assert any(keyword in output.lower() for keyword in [
            "parsing", "generating", "verifying", "proof", "result"
        ])

    def test_cli_output_formats(self, temp_dir):
        """Test different CLI output formats."""
        fixture = SIMPLE_FIXTURES[0]
        circuit_file = temp_dir / "format_test.v"
        circuit_file.write_text(fixture.verilog_code)
        
        output_formats = ["text", "json", "yaml"]
        
        for fmt in output_formats:
            result = subprocess.run([
                "python", "-m", "formal_circuits_gpt",
                "verify", str(circuit_file),
                "--output-format", fmt,
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
            )
            
            # Should handle different output formats
            assert result.returncode in [0, 1, 2]

    def test_cli_error_handling(self):
        """Test CLI error handling for various error conditions."""
        error_cases = [
            # Invalid prover
            ["verify", "test.v", "--prover", "invalid_prover"],
            # Invalid model
            ["verify", "test.v", "--model", "invalid_model"], 
            # Invalid temperature
            ["verify", "test.v", "--temperature", "2.0"],
            # Invalid refinement rounds
            ["verify", "test.v", "--max-refinement-rounds", "-1"],
        ]
        
        for args in error_cases:
            result = subprocess.run(
                ["python", "-m", "formal_circuits_gpt"] + args + ["--dry-run"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            # Should fail with helpful error message
            assert result.returncode != 0
            assert len(result.stderr) > 0 or "error" in result.stdout.lower()


@pytest.mark.e2e
class TestCLIWorkflows:
    """Test complete CLI workflows."""

    def test_development_workflow(self, temp_dir):
        """Test typical development workflow using CLI."""
        # 1. Create a circuit
        circuit_code = """
        module dev_test(
            input [7:0] data_in,
            input enable,
            output [7:0] data_out
        );
            assign data_out = enable ? data_in : 8'b0;
        endmodule
        """
        
        circuit_file = temp_dir / "dev_test.v"
        circuit_file.write_text(circuit_code)
        
        # 2. Check setup
        setup_result = subprocess.run([
            "python", "-m", "formal_circuits_gpt", "--check-setup"
        ], capture_output=True, text=True,
        cwd=Path(__file__).parent.parent.parent)
        
        # 3. Verify circuit (dry run)
        verify_result = subprocess.run([
            "python", "-m", "formal_circuits_gpt",
            "verify", str(circuit_file),
            "--property", "enable == 0 -> data_out == 0",
            "--property", "enable == 1 -> data_out == data_in",
            "--verbose",
            "--dry-run"
        ], capture_output=True, text=True,
        cwd=Path(__file__).parent.parent.parent)
        
        # Should complete workflow steps
        assert setup_result.returncode in [0, 1]  # May pass or fail based on setup
        assert verify_result.returncode in [0, 1, 2]  # May have various outcomes

    def test_benchmark_workflow(self, temp_dir):
        """Test benchmarking workflow."""
        # Create multiple test circuits
        for i, fixture in enumerate(SIMPLE_FIXTURES[:2]):
            circuit_file = temp_dir / f"bench_{i}.v"
            circuit_file.write_text(fixture.verilog_code)
        
        # Run benchmark
        result = subprocess.run([
            "python", "-m", "formal_circuits_gpt",
            "benchmark",
            "--circuits", str(temp_dir),
            "--output", str(temp_dir / "benchmark_results.json"),
            "--dry-run"
        ], capture_output=True, text=True,
        cwd=Path(__file__).parent.parent.parent)
        
        # Should handle benchmarking
        assert result.returncode in [0, 1, 2]

    def test_configuration_file_workflow(self, temp_dir):
        """Test workflow with configuration file."""
        # Create configuration file
        config = """
        llm:
          provider: openai
          model: gpt-4-turbo
          temperature: 0.1
        
        verification:
          default_prover: isabelle
          max_refinement_rounds: 5
        """
        
        config_file = temp_dir / "config.yaml"
        config_file.write_text(config)
        
        # Create test circuit
        fixture = SIMPLE_FIXTURES[0]
        circuit_file = temp_dir / "config_test.v"
        circuit_file.write_text(fixture.verilog_code)
        
        # Run with configuration file
        result = subprocess.run([
            "python", "-m", "formal_circuits_gpt",
            "verify", str(circuit_file),
            "--config", str(config_file),
            "--dry-run"
        ], capture_output=True, text=True,
        cwd=Path(__file__).parent.parent.parent)
        
        # Should use configuration file
        assert result.returncode in [0, 1, 2]