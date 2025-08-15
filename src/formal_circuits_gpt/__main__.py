#!/usr/bin/env python3
"""Command-line interface for formal-circuits-gpt."""

import argparse
import sys
import json
import yaml
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from formal_circuits_gpt import CircuitVerifier
from formal_circuits_gpt.core import ProofResult


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Setup logging configuration."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    config = {}

    if config_path and Path(config_path).exists():
        config_file = Path(config_path)
        try:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                import yaml

                with open(config_file) as f:
                    config = yaml.safe_load(f)
            elif config_file.suffix.lower() == ".json":
                with open(config_file) as f:
                    config = json.load(f)
        except Exception as e:
            print(
                f"Warning: Failed to load config file {config_path}: {e}",
                file=sys.stderr,
            )

    return config


def check_setup() -> int:
    """Check if the system setup is complete."""
    print("Checking formal-circuits-gpt setup...")

    issues = []

    # Check Python version
    if sys.version_info < (3, 9):
        issues.append(
            f"Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    else:
        print("✓ Python version: OK")

    # Check theorem provers
    prover_found = False

    # Check for Isabelle
    isabelle_paths = [
        "/opt/Isabelle2023/bin/isabelle",
        "/usr/local/bin/isabelle",
        "/usr/bin/isabelle",
    ]

    for path in isabelle_paths:
        if Path(path).exists():
            print(f"✓ Found Isabelle at: {path}")
            prover_found = True
            break

    if not prover_found:
        issues.append("No theorem prover found. Please install Isabelle or Coq.")
        print("✗ No theorem prover found")
        print("  Install Isabelle: https://isabelle.in.tum.de/")
        print("  Or install Coq: https://coq.inria.fr/")

    # Check LLM API keys
    llm_found = False
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API key found")
        llm_found = True
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✓ Anthropic API key found")
        llm_found = True

    if not llm_found:
        issues.append("No LLM API keys found")
        print("✗ No LLM API keys found")
        print("  Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")

    # Check optional dependencies
    optional_deps = [
        ("openai", "OpenAI API client"),
        ("anthropic", "Anthropic API client"),
        ("yaml", "YAML configuration support"),
    ]

    for module_name, description in optional_deps:
        try:
            __import__(module_name)
            print(f"✓ {description}: installed")
        except ImportError:
            print(f"○ {description}: not installed (optional)")

    if issues:
        print(f"\n❌ Setup incomplete. Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("\n✅ Setup complete! Ready to verify circuits.")
        return 0


def verify_circuit(args) -> int:
    """Verify a single circuit."""
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        return 1

    try:
        # Load configuration
        config = load_config(args.config)

        # Create verifier
        verifier_config = config.get("verification", {})
        if args.prover:
            verifier_config["prover"] = args.prover
        if args.model:
            verifier_config["llm_model"] = args.model
        if args.temperature is not None:
            verifier_config["temperature"] = args.temperature

        verifier = CircuitVerifier(**verifier_config)

        # Read circuit file
        with open(args.input_file) as f:
            circuit_code = f.read()

        # Prepare properties
        properties = args.properties or []

        if args.verbose:
            print(f"Verifying circuit: {args.input_file}")
            if properties:
                print(f"Properties to verify: {properties}")

        if args.dry_run:
            print("Dry run mode: Would verify circuit but not calling LLM APIs")
            result_data = {
                "status": "dry_run",
                "circuit": args.input_file,
                "properties": properties,
                "message": "Dry run completed successfully",
            }
        else:
            # Actually verify the circuit
            result = verifier.verify(circuit_code, properties=properties)

            result_data = {
                "status": result.status,
                "circuit": args.input_file,
                "properties": properties,
                "proof": result.proof_code,
                "errors": result.errors,
                "duration_ms": result.duration_ms,
                "metadata": getattr(result, "metadata", {}),
            }

        # Output results
        if args.output_format == "json":
            print(json.dumps(result_data, indent=2))
        elif args.output_format == "yaml":
            print(yaml.dump(result_data, default_flow_style=False))
        else:  # text
            status = result_data["status"]
            print(f"Verification result: {status}")
            if status == "VERIFIED":
                print("✅ Circuit verification PASSED")
            elif status == "FAILED":
                print("❌ Circuit verification FAILED")
                if result_data.get("errors"):
                    print("Errors:")
                    for error in result_data["errors"]:
                        print(f"  - {error}")
            else:
                print(f"○ Verification completed with status: {status}")

        return 0 if result_data["status"] in ["VERIFIED", "dry_run"] else 1

    except Exception as e:
        print(f"Error during verification: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def batch_verify(args) -> int:
    """Verify multiple circuits in batch mode."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' not found", file=sys.stderr)
        return 1

    # Find circuit files
    circuit_files = []
    if input_path.is_dir():
        for ext in ["*.v", "*.vhdl", "*.vhd"]:
            circuit_files.extend(input_path.glob(f"**/{ext}"))
    else:
        circuit_files = [input_path]

    if not circuit_files:
        print("No circuit files found", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(circuit_files)} circuit files")

    results = []
    success_count = 0

    for circuit_file in circuit_files:
        if args.verbose:
            print(f"Processing: {circuit_file}")

        # Create args for individual verification
        file_args = argparse.Namespace(**vars(args))
        file_args.input_file = str(circuit_file)

        # Verify individual circuit
        result_code = verify_circuit(file_args)

        results.append({"file": str(circuit_file), "success": result_code == 0})

        if result_code == 0:
            success_count += 1

    # Output batch results
    batch_result = {
        "total": len(circuit_files),
        "successful": success_count,
        "failed": len(circuit_files) - success_count,
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(batch_result, f, indent=2)
        print(f"Results written to: {output_path}")

    print(
        f"Batch verification completed: {success_count}/{len(circuit_files)} successful"
    )

    return 0 if success_count == len(circuit_files) else 1


def benchmark_circuits(args) -> int:
    """Run benchmark tests on circuits."""
    print("Benchmarking not yet implemented in this version")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="formal-circuits-gpt",
        description="Formal verification of digital circuits using LLMs and theorem provers",
    )

    parser.add_argument(
        "--version", action="version", version="formal-circuits-gpt 0.1.0"
    )
    parser.add_argument(
        "--check-setup", action="store_true", help="Check if system setup is complete"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a single circuit")
    verify_parser.add_argument("input_file", help="Input circuit file")
    verify_parser.add_argument(
        "--property",
        "-p",
        dest="properties",
        action="append",
        help="Property to verify (can be used multiple times)",
    )
    verify_parser.add_argument(
        "--prover", choices=["isabelle", "coq"], help="Theorem prover to use"
    )
    verify_parser.add_argument(
        "--model",
        type=str,
        help="LLM model to use (e.g., gpt-4-turbo, claude-3-sonnet)",
    )
    verify_parser.add_argument(
        "--temperature", type=float, help="LLM temperature (0.0-1.0)"
    )
    verify_parser.add_argument(
        "--max-refinement-rounds", type=int, help="Maximum refinement rounds"
    )
    verify_parser.add_argument(
        "--output-format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format",
    )
    verify_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without calling LLM APIs",
    )
    verify_parser.add_argument("--timeout", type=int, help="Timeout in seconds")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Verify multiple circuits")
    batch_parser.add_argument(
        "--input", "-i", required=True, help="Input directory or file pattern"
    )
    batch_parser.add_argument("--output", "-o", help="Output results file")
    batch_parser.add_argument(
        "--prover", choices=["isabelle", "coq"], help="Theorem prover to use"
    )
    batch_parser.add_argument("--model", type=str, help="LLM model to use")
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without calling LLM APIs",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark tests")
    benchmark_parser.add_argument(
        "--circuits", required=True, help="Directory containing test circuits"
    )
    benchmark_parser.add_argument(
        "--output", "-o", help="Output benchmark results file"
    )
    benchmark_parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run"
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Handle special commands
    if args.check_setup:
        return check_setup()

    # Validate temperature
    if hasattr(args, "temperature") and args.temperature is not None:
        if not 0.0 <= args.temperature <= 1.0:
            print("Error: Temperature must be between 0.0 and 1.0", file=sys.stderr)
            return 1

    # Validate refinement rounds
    if (
        hasattr(args, "max_refinement_rounds")
        and args.max_refinement_rounds is not None
    ):
        if args.max_refinement_rounds < 0:
            print("Error: Max refinement rounds must be non-negative", file=sys.stderr)
            return 1

    # Handle commands
    if args.command == "verify":
        return verify_circuit(args)
    elif args.command == "batch":
        return batch_verify(args)
    elif args.command == "benchmark":
        return benchmark_circuits(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
