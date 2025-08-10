"""Command-line interface for formal-circuits-gpt."""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from .core import CircuitVerifier, ProofResult
from .exceptions import VerificationError
from .provers import IsabelleInterface, CoqInterface
from .database import DatabaseManager, run_migrations
from .cache import CacheManager

console = Console()


@click.group()
@click.version_option(version="1.0.0")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.pass_context
def main(ctx, debug):
    """Formal-Circuits-GPT: LLM-Assisted Hardware Verification"""
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug


@main.command()
@click.option("--prover", default="isabelle", type=click.Choice(["isabelle", "coq"]), 
              help="Theorem prover to use")
@click.option("--model", default="gpt-4-turbo", help="LLM model to use")
@click.option("--temperature", default=0.1, type=float, help="LLM temperature (0.0-2.0)")
@click.option("--timeout", default=300, type=int, help="Verification timeout (seconds)")
@click.option("--properties", multiple=True, help="Properties to verify")
@click.option("--output", type=click.Path(), help="Output file for proof")
@click.option("--format", "output_format", default="text", 
              type=click.Choice(["text", "latex", "sva"]), help="Output format")
@click.argument("hdl_file", type=click.Path(exists=True))
@click.pass_context
def verify(ctx, prover, model, temperature, timeout, properties, output, output_format, hdl_file):
    """Verify a hardware design file."""
    debug = ctx.obj.get('debug', False)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize verifier
            task = progress.add_task("Initializing verifier...", total=None)
            verifier = CircuitVerifier(
                prover=prover,
                model=model,
                temperature=temperature,
                debug_mode=debug
            )
            
            # Perform verification
            progress.update(task, description=f"Verifying {Path(hdl_file).name}...")
            
            properties_list = list(properties) if properties else None
            result = verifier.verify_file(
                hdl_file=hdl_file,
                properties=properties_list,
                timeout=timeout
            )
            
            progress.update(task, description="Verification complete", completed=True)
        
        # Display results
        _display_verification_result(result, hdl_file)
        
        # Save output if requested
        if output:
            _save_verification_output(result, output, output_format)
            console.print(f"[green]✓[/green] Results saved to {output}")
        
        # Exit with appropriate code
        sys.exit(0 if result.status == "VERIFIED" else 1)
        
    except VerificationError as e:
        console.print(f"[red]✗ Verification Error:[/red] {e}")
        if debug:
            console.print_exception()
        sys.exit(1)
    
    except FileNotFoundError:
        console.print(f"[red]✗ Error:[/red] File {hdl_file} not found")
        sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]✗ Unexpected Error:[/red] {e}")
        if debug:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option("--prover", type=click.Choice(["isabelle", "coq", "all"]), default="all",
              help="Check specific prover or all")
def check_setup(prover):
    """Check if theorem provers are properly installed."""
    console.print("[blue]🔍 Checking theorem prover setup...[/blue]\n")
    
    status_table = Table(title="System Status")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Version/Path", style="yellow")
    
    all_ok = True
    
    # Check Python environment
    status_table.add_row("Python", "✓ Available", f"{sys.version.split()[0]}")
    
    # Check theorem provers
    if prover in ["isabelle", "all"]:
        isabelle = IsabelleInterface()
        if isabelle.check_installation():
            version = isabelle.get_version()
            status_table.add_row("Isabelle", "✓ Available", version)
        else:
            status_table.add_row("Isabelle", "✗ Not Found", "Not installed")
            all_ok = False
    
    if prover in ["coq", "all"]:
        coq = CoqInterface()
        if coq.check_installation():
            version = coq.get_version()
            status_table.add_row("Coq", "✓ Available", version)
        else:
            status_table.add_row("Coq", "✗ Not Found", "Not installed")
            all_ok = False
    
    # Check environment variables
    if os.getenv("OPENAI_API_KEY"):
        status_table.add_row("OpenAI API", "✓ Configured", "Key present")
    else:
        status_table.add_row("OpenAI API", "⚠ Not configured", "Set OPENAI_API_KEY")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        status_table.add_row("Anthropic API", "✓ Configured", "Key present")
    else:
        status_table.add_row("Anthropic API", "⚠ Not configured", "Set ANTHROPIC_API_KEY")
    
    console.print(status_table)
    
    if all_ok:
        console.print("\n[green]✓ All required components are available![/green]")
    else:
        console.print("\n[yellow]⚠ Some components are missing. See installation guide.[/yellow]")
        sys.exit(1)


@main.command()
@click.option("--suite", default="builtin", help="Benchmark suite to run")
@click.option("--prover", type=click.Choice(["isabelle", "coq", "both"]), default="both")
@click.option("--timeout", default=300, type=int, help="Timeout per circuit")
@click.option("--output", type=click.Path(), help="Output results to file")
def benchmark(suite, prover, timeout, output):
    """Run verification benchmarks."""
    console.print(f"[blue]🏃 Running {suite} benchmarks...[/blue]\n")
    
    # Get benchmark circuits
    benchmark_dir = Path("benchmarks")
    if not benchmark_dir.exists():
        console.print("[red]✗ Benchmark directory not found[/red]")
        console.print("Run setup script to create benchmark data")
        sys.exit(1)
    
    config_file = benchmark_dir / "config.json"
    if not config_file.exists():
        console.print("[red]✗ Benchmark configuration not found[/red]")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    circuits = config.get("circuits", [])
    provers_to_test = ["isabelle", "coq"] if prover == "both" else [prover]
    
    results = []
    
    with Progress(console=console) as progress:
        total_tests = len(circuits) * len(provers_to_test)
        task = progress.add_task("Running benchmarks...", total=total_tests)
        
        for circuit_config in circuits:
            circuit_file = benchmark_dir / circuit_config["file"]
            circuit_name = circuit_config["name"]
            circuit_properties = config.get("properties", {}).get(circuit_name, [])
            
            for prover_name in provers_to_test:
                try:
                    verifier = CircuitVerifier(prover=prover_name)
                    result = verifier.verify_file(
                        hdl_file=str(circuit_file),
                        properties=circuit_properties,
                        timeout=timeout
                    )
                    
                    results.append({
                        "circuit": circuit_name,
                        "prover": prover_name,
                        "status": result.status,
                        "properties_verified": len(result.properties_verified),
                        "errors": len(result.errors)
                    })
                    
                except Exception as e:
                    results.append({
                        "circuit": circuit_name,
                        "prover": prover_name,
                        "status": "ERROR",
                        "error": str(e)
                    })
                
                progress.advance(task)
    
    # Display results
    _display_benchmark_results(results)
    
    # Save results if requested
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]✓[/green] Results saved to {output}")


@main.command()
@click.option("--port", default=5000, type=int, help="Port to run API server")
@click.option("--host", default="127.0.0.1", help="Host to bind server")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
def serve(port, host, debug):
    """Start the REST API server."""
    try:
        from .api import create_app
        
        console.print(f"[blue]🚀 Starting API server on {host}:{port}...[/blue]")
        
        app = create_app()
        app.run(
            host=host,
            port=port,
            debug=debug
        )
        
    except ImportError:
        console.print("[red]✗ Flask not installed. Install with: pip install flask flask-cors[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Failed to start server:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Experiment configuration file")
@click.option("--circuits", type=click.Path(exists=True), multiple=True, help="Circuit files to test")
@click.option("--provers", multiple=True, default=["isabelle"], help="Provers to compare")
@click.option("--models", multiple=True, default=["gpt-4-turbo"], help="LLM models to test")
@click.option("--repetitions", default=3, type=int, help="Repetitions for statistical significance")
@click.option("--timeout", default=300, type=int, help="Timeout per verification")
@click.option("--output-dir", type=click.Path(), help="Output directory for results")
def research(config, circuits, provers, models, repetitions, timeout, output_dir):
    """Run research experiments and comparative studies."""
    try:
        from .research.experiment_runner import ExperimentRunner, ExperimentConfig
        from pathlib import Path
        
        if config:
            # Load experiment from config file
            import json
            with open(config) as f:
                config_data = json.load(f)
            exp_config = ExperimentConfig(**config_data)
        else:
            # Create config from CLI parameters
            if not circuits:
                console.print("[red]✗ No circuits specified. Use --circuits or --config[/red]")
                sys.exit(1)
            
            exp_config = ExperimentConfig(
                name=f"cli_experiment_{int(time.time())}",
                description="CLI-generated experiment",
                circuits=list(circuits),
                provers=list(provers),
                models=list(models),
                repetitions=repetitions,
                timeout=timeout
            )
        
        # Set up results directory
        results_dir = Path(output_dir) if output_dir else Path("research_results")
        runner = ExperimentRunner(results_dir)
        
        # Run experiment
        console.print(f"[blue]🔬 Starting research experiment: {exp_config.name}[/blue]")
        results = runner.run_experiment(exp_config)
        
        # Display summary
        stats = runner._calculate_statistics(results)
        console.print(f"[green]✅ Experiment completed![/green]")
        console.print(f"Success rate: {stats['overall']['success_rate']:.1%}")
        console.print(f"Average duration: {stats['overall']['avg_duration_ms']:.1f}ms")
        console.print(f"Results saved to: {results_dir}")
        
    except ImportError as e:
        console.print(f"[red]✗ Research modules not available: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Research experiment failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--stats", is_flag=True, help="Show cache statistics")
@click.option("--clear", is_flag=True, help="Clear all caches")
@click.option("--cleanup", type=int, metavar="DAYS", help="Cleanup entries older than DAYS")
def cache(stats, clear, cleanup):
    """Manage proof and lemma caches."""
    try:
        cache_manager = CacheManager()
        
        if stats:
            cache_stats = cache_manager.get_cache_stats()
            _display_cache_stats(cache_stats)
        
        if clear:
            if click.confirm("This will clear ALL cached proofs and lemmas. Continue?"):
                cache_manager.clear_all_caches()
                console.print("[green]✓ All caches cleared[/green]")
        
        if cleanup:
            result = cache_manager.cleanup_cache(cleanup)
            console.print(f"[green]✓ Cleanup complete:[/green]")
            for key, count in result.items():
                console.print(f"  {key}: {count}")
        
        if not any([stats, clear, cleanup]):
            console.print("Use --stats, --clear, or --cleanup. See --help for details.")
    
    except Exception as e:
        console.print(f"[red]✗ Cache operation failed:[/red] {e}")
        sys.exit(1)


def _display_verification_result(result: ProofResult, hdl_file: str):
    """Display verification results in a formatted way."""
    status_color = "green" if result.status == "VERIFIED" else "red"
    status_icon = "✓" if result.status == "VERIFIED" else "✗"
    
    console.print(f"\n[{status_color}]{status_icon} Verification {result.status}[/{status_color}]")
    
    # Create results panel
    info_lines = [
        f"File: {Path(hdl_file).name}",
        f"Properties verified: {len(result.properties_verified)}",
    ]
    
    if result.errors:
        info_lines.append(f"Errors: {len(result.errors)}")
    
    console.print(Panel("\n".join(info_lines), title="Results"))
    
    # Show properties
    if result.properties_verified:
        console.print("\n[cyan]Properties Verified:[/cyan]")
        for i, prop in enumerate(result.properties_verified, 1):
            console.print(f"  {i}. {prop}")
    
    # Show errors
    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for i, error in enumerate(result.errors, 1):
            console.print(f"  {i}. {error}")
    
    # Show proof preview
    if result.proof_code and len(result.proof_code) < 2000:
        console.print("\n[cyan]Generated Proof:[/cyan]")
        syntax = Syntax(result.proof_code, "isabelle" if "theory" in result.proof_code else "coq", 
                       theme="monokai", line_numbers=True)
        console.print(syntax)


def _save_verification_output(result: ProofResult, output_path: str, format_type: str):
    """Save verification output to file."""
    if format_type == "latex":
        result.export_latex(output_path)
    elif format_type == "sva":
        result.export_systemverilog_assertions(output_path)
    else:  # text
        with open(output_path, 'w') as f:
            f.write(f"Verification Status: {result.status}\n\n")
            
            if result.properties_verified:
                f.write("Properties Verified:\n")
                for i, prop in enumerate(result.properties_verified, 1):
                    f.write(f"  {i}. {prop}\n")
                f.write("\n")
            
            if result.errors:
                f.write("Errors:\n")
                for i, error in enumerate(result.errors, 1):
                    f.write(f"  {i}. {error}\n")
                f.write("\n")
            
            if result.proof_code:
                f.write("Generated Proof:\n")
                f.write("=" * 50 + "\n")
                f.write(result.proof_code)


def _display_benchmark_results(results: List[dict]):
    """Display benchmark results in a table."""
    table = Table(title="Benchmark Results")
    table.add_column("Circuit", style="cyan")
    table.add_column("Prover", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Properties", style="blue")
    table.add_column("Errors", style="red")
    
    verified_count = 0
    total_count = len(results)
    
    for result in results:
        status = result["status"]
        status_color = "green" if status == "VERIFIED" else "red" if status == "FAILED" else "yellow"
        
        if status == "VERIFIED":
            verified_count += 1
        
        table.add_row(
            result["circuit"],
            result["prover"],
            f"[{status_color}]{status}[/{status_color}]",
            str(result.get("properties_verified", "N/A")),
            str(result.get("errors", result.get("error", "N/A")))
        )
    
    console.print(table)
    console.print(f"\n[green]Success Rate:[/green] {verified_count}/{total_count} ({verified_count/total_count*100:.1f}%)")


def _display_cache_stats(stats: dict):
    """Display cache statistics."""
    table = Table(title="Cache Statistics")
    table.add_column("Cache Type", style="cyan")
    table.add_column("Entries", style="yellow")
    table.add_column("Size/Usage", style="green")
    
    # Database cache
    db_stats = stats.get("database", {})
    table.add_row(
        "Database Proofs",
        str(db_stats.get("total_entries", 0)),
        f"{db_stats.get('verified_count', 0)} verified"
    )
    
    # Memory cache
    mem_stats = stats.get("memory", {})
    table.add_row(
        "Memory Cache",
        str(mem_stats.get("memory_entries", 0)),
        f"{mem_stats.get('memory_usage_percent', 0):.1f}% full"
    )
    
    # File cache
    file_stats = stats.get("files", {})
    table.add_row(
        "File Cache",
        str(file_stats.get("file_count", 0)),
        f"{file_stats.get('total_size_mb', 0):.1f} MB"
    )
    
    console.print(table)
    console.print(f"\nCache Directory: {stats.get('cache_dir', 'Unknown')}")


if __name__ == "__main__":
    main()