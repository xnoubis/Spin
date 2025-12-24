"""
Unified Genie CLI
=================

Command-line interface for the unified genie system.

Usage:
    python -m unified_genie.cli.genie [command] [options]

Commands:
    run         Run the genie nexus for N cycles
    evolve      Evolve capabilities to specified depth
    query       Query memory systems
    gift        Enter gift mode and synthesize insights
    export      Export system state
    demo        Run demonstration
"""

from unified_genie.cli.genie import main

__all__ = ['main']
