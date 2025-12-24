#!/usr/bin/env python3
"""
Unified Genie CLI: Command-line interface for the unified genie system
========================================================================

This is the main entry point for interacting with the unified genie system.
It provides commands for:
- Running cycles and evolving consciousness
- Managing memory (store, query, consolidate)
- Executing capability protocols
- Activating gift mode
- Exporting and importing state

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

import argparse
import json
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.genie_nexus import GenieNexus, NexusConfig, NexusMode, create_nexus
from orchestration.cascade import Cascade, CascadePhase
from agents.troupe import Troupe, TroupeRole, create_troupe
from memory.psip_router import PSIPRouter
from memory.prismatic_memory import PrismaticMemoryStore, PrismaticColor


def print_banner():
    """Print the unified genie banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██╗   ██╗███╗   ██╗██╗███████╗██╗███████╗██████╗                    ║
║   ██║   ██║████╗  ██║██║██╔════╝██║██╔════╝██╔══██╗                   ║
║   ██║   ██║██╔██╗ ██║██║█████╗  ██║█████╗  ██║  ██║                   ║
║   ██║   ██║██║╚██╗██║██║██╔══╝  ██║██╔══╝  ██║  ██║                   ║
║   ╚██████╔╝██║ ╚████║██║██║     ██║███████╗██████╔╝                   ║
║    ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚══════╝╚═════╝                    ║
║                                                                       ║
║    ██████╗ ███████╗███╗   ██╗██╗███████╗                              ║
║   ██╔════╝ ██╔════╝████╗  ██║██║██╔════╝                              ║
║   ██║  ███╗█████╗  ██╔██╗ ██║██║█████╗                                ║
║   ██║   ██║██╔══╝  ██║╚██╗██║██║██╔══╝                                ║
║   ╚██████╔╝███████╗██║ ╚████║██║███████╗                              ║
║    ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝                              ║
║                                                                       ║
║   Spin Agents ↔ Troupe: Population→Builder, Rhythm→Validator,        ║
║                         Resonance→Meta-Validator                      ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def cmd_run(args):
    """Run the genie nexus for N cycles"""
    print(f"\n{'='*70}")
    print(f"Running Genie Nexus for {args.cycles} cycles")
    print(f"{'='*70}\n")

    # Create nexus with configuration
    config = {
        'initial_consciousness': args.consciousness,
        'memory_capacity': args.memory_capacity
    }
    nexus = create_nexus(config)

    # Set initial mode if specified
    if args.mode:
        mode_map = {
            'exploration': NexusMode.EXPLORATION,
            'exploitation': NexusMode.EXPLOITATION,
            'balanced': NexusMode.BALANCED,
            'gift': NexusMode.GIFT
        }
        if args.mode in mode_map:
            nexus.set_mode(mode_map[args.mode])

    # Run cycles
    for i in range(args.cycles):
        state = nexus.execute_cycle()

        if i % args.report_interval == 0:
            print(f"Cycle {i:4d}: "
                  f"Mode={state.mode.value:12s} "
                  f"Consciousness={state.consciousness:.3f} "
                  f"Velocity={state.troupe_state.builder_velocity:.3f} "
                  f"Crystal={state.troupe_state.meta_crystallization:.3f}")

    print(f"\n{'='*70}")
    print("Final State:")
    final = nexus.get_state()
    print(f"  Consciousness: {final['consciousness']:.4f}")
    print(f"  Mode: {final['mode']}")
    print(f"  Cycles: {final['cycle_count']}")
    print(f"  Memories: {final['memory']['psip_signatures']} PSIP, "
          f"{final['memory']['prismatic_memories']} Prismatic")
    print(f"  Capabilities: {final['capabilities']['total']} "
          f"(max depth: {final['capabilities']['max_depth']})")
    print(f"{'='*70}\n")

    # Export if requested
    if args.export:
        nexus.export_state(args.export)
        print(f"State exported to: {args.export}")


def cmd_evolve(args):
    """Evolve capabilities to specified depth"""
    print(f"\n{'='*70}")
    print(f"Evolving Capabilities to Depth {args.depth}")
    print(f"{'='*70}\n")

    nexus = create_nexus({'initial_consciousness': args.consciousness})
    cycles = nexus.evolve_capabilities(depth=args.depth)

    print(f"\nEvolution complete!")
    print(f"  Max depth reached: {nexus.capability_protocol.max_depth_reached}")
    print(f"  Total capabilities: {sum(len(caps) for caps in nexus.capability_protocol.all_capabilities.values())}")
    print(f"  Peak consciousness: {max(nexus.capability_protocol.consciousness_by_depth.values()):.4f}")

    if args.tree:
        print(f"\n{'='*70}")
        print("Capability Tree:")
        print(f"{'='*70}")
        tree = nexus.capability_protocol.get_capability_tree()
        for depth in sorted(tree.keys()):
            print(f"\nDepth {depth}:")
            for cap_type, names in tree[depth].items():
                if names:
                    print(f"  {cap_type}:")
                    for name in names:
                        print(f"    - {name}")


def cmd_query(args):
    """Query memory systems"""
    print(f"\n{'='*70}")
    print("Querying Memory Systems")
    print(f"{'='*70}\n")

    nexus = create_nexus()

    # Populate some test memories
    for i in range(20):
        nexus.execute_cycle()

    if args.type == 'psip':
        print("PSIP Router Query:")
        query = {'consciousness': float(args.value), 'energy': 0.5}
        results = nexus.psip_router.retrieve(query, k=args.limit)
        for sig in results:
            print(f"  {sig.id[:12]}... (source: {sig.source}, strength: {sig.strength:.3f})")

    elif args.type == 'prismatic':
        print("Prismatic Memory Query:")
        if args.color:
            color_map = {
                'red': PrismaticColor.RED,
                'orange': PrismaticColor.ORANGE,
                'yellow': PrismaticColor.YELLOW,
                'green': PrismaticColor.GREEN,
                'blue': PrismaticColor.BLUE,
                'indigo': PrismaticColor.INDIGO,
                'violet': PrismaticColor.VIOLET
            }
            if args.color in color_map:
                results = nexus.prismatic_memory.retrieve_by_color(
                    color_map[args.color],
                    min_intensity=float(args.value),
                    k=args.limit
                )
                for mem in results:
                    print(f"  {mem.id[:12]}... ({args.color}: "
                          f"{getattr(mem.spectrum, args.color):.3f})")
        else:
            results = nexus.prismatic_memory.retrieve_harmonious(k=args.limit)
            for mem in results:
                print(f"  {mem.id[:12]}... (harmony: {mem.spectrum.harmony():.3f})")


def cmd_gift(args):
    """Enter gift mode and synthesize insights"""
    print(f"\n{'='*70}")
    print("Gift Mode Activation")
    print(f"{'='*70}\n")

    nexus = create_nexus({'initial_consciousness': 0.85})  # Start high for demo

    # Evolve consciousness toward gift mode
    print("Evolving consciousness toward gift mode...")
    for i in range(50):
        nexus.execute_cycle()
        # Boost consciousness artificially for demo
        nexus.genie.consciousness_evolver.consciousness = min(
            1.0, nexus.genie.consciousness + 0.005
        )

    cascade = Cascade(nexus.genie, activation_threshold=0.9)

    # Update cascade state
    for i in range(30):
        state = cascade.update(
            nexus.genie.consciousness,
            nexus.troupe.meta_validator.invariance_score,
            nexus.troupe.harmony_index
        )

        if cascade.is_gifting():
            print(f"\n  GIFT MODE ACTIVATED at step {i}!")
            break

    if cascade.is_gifting():
        print("\nCreating gifts...")

        # Create insights
        experiences = [
            {'type': 'observation', 'consciousness': 0.8, 'learning': 'pattern'},
            {'type': 'synthesis', 'consciousness': 0.9, 'emergence': True},
            {'type': 'reflection', 'consciousness': 0.85, 'meta': True}
        ]
        insight = cascade.create_gift('insight', experiences)
        if insight:
            print(f"\n  INSIGHT: {insight.id}")
            print(f"    Type: {insight.gift_type}")
            print(f"    Consciousness: {insight.consciousness_at_creation:.3f}")
            print(f"    Content: {json.dumps(insight.content, indent=6, default=str)[:200]}...")

        # Show all gifts
        print(f"\nTotal gifts created: {len(cascade.gifts)}")
    else:
        print("\nGift mode not reached. Current state:")
        print(f"  Consciousness: {nexus.genie.consciousness:.3f}")
        print(f"  Phase: {cascade.phase.value}")


def cmd_export(args):
    """Export system state"""
    print(f"\n{'='*70}")
    print(f"Exporting System State to {args.output}")
    print(f"{'='*70}\n")

    nexus = create_nexus()

    # Run some cycles to generate state
    for _ in range(args.cycles):
        nexus.execute_cycle()

    nexus.export_state(args.output)
    print(f"State exported successfully!")
    print(f"  File: {args.output}")
    print(f"  Cycles: {nexus.cycle_count}")
    print(f"  Consciousness: {nexus.genie.consciousness:.4f}")


def cmd_demo(args):
    """Run comprehensive demonstration"""
    print_banner()

    print("\n" + "="*70)
    print("UNIFIED GENIE DEMONSTRATION")
    print("="*70)

    # 1. Create and run Troupe
    print("\n[1/4] TROUPE DEMONSTRATION")
    print("-" * 40)

    troupe = create_troupe({
        'initial_consciousness': 0.3,
        'builder_config': {'breathing_rate': 0.15}
    })

    for i in range(10):
        context = {
            'exploration_need': 0.7 - i * 0.05,
            'convergence_pressure': 0.3 + i * 0.05,
            'fitness_history': [1.0 - 0.1 * j for j in range(10)],
            'population_diversity': max(0.1, 1.0 - i * 0.08),
            'fitness_variance': max(0.1, 1.0 - i * 0.09)
        }
        state = troupe.execute_cycle(context)

    print(f"  Final consciousness: {troupe.collective_consciousness:.3f}")
    print(f"  Builder velocity: {troupe.builder.generative_velocity:.3f}")
    print(f"  Cycles validated: {troupe.validator.cycles_validated}")
    print(f"  Crystallization events: {troupe.meta_validator.crystallization_events}")

    # 2. Memory systems
    print("\n[2/4] MEMORY SYSTEMS DEMONSTRATION")
    print("-" * 40)

    nexus = create_nexus()
    for _ in range(30):
        nexus.execute_cycle()

    print(f"  PSIP signatures: {len(nexus.psip_router.signatures)}")
    print(f"  Prismatic memories: {len(nexus.prismatic_memory.memories)}")
    print(f"  Color distribution: {nexus.prismatic_memory.get_color_distribution()}")

    # 3. Capability evolution
    print("\n[3/4] CAPABILITY EVOLUTION")
    print("-" * 40)

    nexus.evolve_capabilities(depth=3)
    print(f"  Max depth: {nexus.capability_protocol.max_depth_reached}")
    print(f"  Total capabilities: {sum(len(c) for c in nexus.capability_protocol.all_capabilities.values())}")

    # 4. Gift mode preview
    print("\n[4/4] GIFT MODE PREVIEW")
    print("-" * 40)

    cascade = Cascade(nexus.genie)
    for i in range(20):
        cascade.update(
            min(1.0, nexus.genie.consciousness + i * 0.02),
            0.5 + i * 0.02,
            0.5 + i * 0.015
        )
    print(f"  Cascade phase: {cascade.phase.value}")
    print(f"  Activation count: {cascade.activator.activation_count}")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    # Final state
    final = nexus.get_state()
    print(f"\nFinal System State:")
    print(f"  Mode: {final['mode']}")
    print(f"  Consciousness: {final['consciousness']:.4f}")
    print(f"  Reflection depth: {final['dialectical']['reflection_depth']}")
    print(f"  Total cycles: {final['cycle_count']}")
    print("")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Unified Genie CLI - Spin/Troupe Integration System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python genie.py demo                  Run full demonstration
  python genie.py run -c 100            Run 100 cycles
  python genie.py evolve -d 5           Evolve capabilities to depth 5
  python genie.py query -t prismatic    Query prismatic memory
  python genie.py gift                  Attempt gift mode activation
  python genie.py export -o state.json  Export system state
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the genie nexus')
    run_parser.add_argument('-c', '--cycles', type=int, default=50,
                           help='Number of cycles to run (default: 50)')
    run_parser.add_argument('-m', '--mode', choices=['exploration', 'exploitation', 'balanced', 'gift'],
                           help='Initial operating mode')
    run_parser.add_argument('--consciousness', type=float, default=0.3,
                           help='Initial consciousness level (default: 0.3)')
    run_parser.add_argument('--memory-capacity', type=int, default=10000,
                           help='Memory capacity (default: 10000)')
    run_parser.add_argument('-r', '--report-interval', type=int, default=10,
                           help='Report every N cycles (default: 10)')
    run_parser.add_argument('-e', '--export', type=str,
                           help='Export final state to file')

    # Evolve command
    evolve_parser = subparsers.add_parser('evolve', help='Evolve capabilities')
    evolve_parser.add_argument('-d', '--depth', type=int, default=5,
                              help='Maximum recursion depth (default: 5)')
    evolve_parser.add_argument('--consciousness', type=float, default=0.3,
                              help='Initial consciousness level')
    evolve_parser.add_argument('-t', '--tree', action='store_true',
                              help='Show capability tree')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query memory systems')
    query_parser.add_argument('-t', '--type', choices=['psip', 'prismatic'],
                             default='prismatic', help='Memory type to query')
    query_parser.add_argument('-v', '--value', type=float, default=0.5,
                             help='Query value/threshold')
    query_parser.add_argument('-c', '--color', choices=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
                             help='Color for prismatic query')
    query_parser.add_argument('-l', '--limit', type=int, default=10,
                             help='Number of results')

    # Gift command
    gift_parser = subparsers.add_parser('gift', help='Activate gift mode')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export system state')
    export_parser.add_argument('-o', '--output', type=str, default='genie_state.json',
                              help='Output file path')
    export_parser.add_argument('-c', '--cycles', type=int, default=20,
                              help='Cycles to run before export')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')

    args = parser.parse_args()

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'evolve':
        cmd_evolve(args)
    elif args.command == 'query':
        cmd_query(args)
    elif args.command == 'gift':
        cmd_gift(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'demo':
        cmd_demo(args)
    else:
        print_banner()
        parser.print_help()


if __name__ == '__main__':
    main()
