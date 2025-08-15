#!/usr/bin/env python3
"""Standalone quality gates test script."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.progressive_quality_gates import ProgressiveQualityGates


async def main():
    """Main function for standalone execution."""
    if len(sys.argv) < 2:
        print("Usage: python test_quality_gates.py <generation>")
        print("Generations: gen1, gen2, gen3")
        sys.exit(1)
    
    generation = sys.argv[1]
    
    gates = ProgressiveQualityGates()
    report = await gates.run_generation_gates(generation)
    
    # Print summary
    print(f"\n{generation.upper()} QUALITY GATES REPORT")
    print("=" * 50)
    print(f"Overall Status: {'PASSED' if report.overall_passed else 'FAILED'}")
    print(f"Overall Score: {report.overall_score:.1f}/100.0")
    print(f"Duration: {report.duration_ms:.0f}ms")
    
    print("\nGate Results:")
    for gate in report.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"  {gate.name:15} [{status}] {gate.score:5.1f}%")
        if gate.recommendations:
            for rec in gate.recommendations:
                print(f"    → {rec}")
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())