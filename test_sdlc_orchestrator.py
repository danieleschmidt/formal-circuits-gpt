#!/usr/bin/env python3
"""Test autonomous SDLC orchestrator."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from formal_circuits_gpt.autonomous_sdlc_orchestrator import AutonomousSDLCOrchestrator


async def main():
    """Main function for standalone SDLC execution."""
    target_generation = sys.argv[1] if len(sys.argv) > 1 else "gen3"

    orchestrator = AutonomousSDLCOrchestrator()
    report = await orchestrator.execute_autonomous_sdlc(target_generation)

    # Print summary
    print(f"\nAUTONOMOUS SDLC EXECUTION REPORT")
    print("=" * 60)
    print(f"Overall Success: {'YES' if report.overall_success else 'NO'}")
    print(f"Duration: {report.total_duration_ms:.0f}ms")
    print(f"Stages Completed: {', '.join(report.stages_completed)}")
    print(f"Average Quality Score: {report.metrics['average_quality_score']:.1f}%")

    print(f"\nEnhancements Implemented ({len(report.enhancements_implemented)}):")
    for enhancement in report.enhancements_implemented:
        print(f"  ✓ {enhancement}")

    print(f"\nNext Recommendations ({len(report.next_recommendations)}):")
    for rec in report.next_recommendations:
        print(f"  [{rec.priority.upper()}] {rec.description}")

    # Exit with appropriate code
    sys.exit(0 if report.overall_success else 1)


if __name__ == "__main__":
    asyncio.run(main())