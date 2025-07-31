"""Command-line interface for formal-circuits-gpt."""

import click
from rich.console import Console
from . import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx):
    """Formal-Circuits-GPT: LLM-Assisted Hardware Verification"""
    ctx.ensure_object(dict)


@main.command()
@click.option("--prover", default="isabelle", help="Theorem prover (isabelle/coq)")
@click.option("--model", default="gpt-4-turbo", help="LLM model to use")
@click.option("--timeout", default=300, help="Verification timeout (seconds)")
@click.argument("hdl_file", type=click.Path(exists=True))
def verify(prover, model, timeout, hdl_file):
    """Verify a hardware design file."""
    console.print(f"[blue]Verifying {hdl_file} with {prover}...[/blue]")
    console.print("[yellow]Note: Core verification logic not yet implemented[/yellow]")


@main.command()
def check_setup():
    """Check if theorem provers are properly installed."""
    console.print("[blue]Checking setup...[/blue]")
    console.print("[yellow]Setup check not yet implemented[/yellow]")


@main.command()
@click.option("--suite", default="aiger", help="Benchmark suite to run")
def benchmark(suite):
    """Run verification benchmarks."""
    console.print(f"[blue]Running {suite} benchmarks...[/blue]")
    console.print("[yellow]Benchmark system not yet implemented[/yellow]")


if __name__ == "__main__":
    main()