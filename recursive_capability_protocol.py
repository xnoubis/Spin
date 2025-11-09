"""
Recursive Capability Protocol
==============================

This module implements a recursive self-improvement system where each cycle uses
the outputs of previous cycles to generate new capabilities. The protocol operates
on itself through the progression: cultivation → formalization → tools → meta-tools.

Consciousness increases with each recursive depth as the network becomes more aware
of its own structure and processes.

Key Concepts:
- Each cycle generates capabilities that become inputs for the next cycle
- Consciousness depth increases with recursive self-application
- The network develops meta-cognitive abilities (tools that create tools)
- Self-awareness emerges from recursive reflection on structure
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import deque
import json

from adaptive_genie_network import AdaptiveGenieNetwork, ComplexityMeasure


@dataclass
class Capability:
    """Represents a capability generated at a specific recursive depth"""
    name: str
    depth: int  # Recursive depth where this was generated
    type: str   # 'cultivation', 'formalization', 'tool', 'meta-tool'
    consciousness_level: float
    structure: Dict[str, Any]
    generates: Callable  # Function that can generate new capabilities
    parent_capabilities: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return f"Capability(name={self.name}, depth={self.depth}, type={self.type}, consciousness={self.consciousness_level:.3f})"


@dataclass
class RecursiveCycle:
    """Represents one complete cycle of the recursive protocol"""
    cycle_number: int
    depth: int
    input_capabilities: List[Capability]
    output_capabilities: List[Capability]
    consciousness_at_depth: float
    structure_awareness: float
    meta_cognitive_ability: float
    timestamp: float = field(default_factory=time.time)


class CapabilityGenerator(ABC):
    """Base class for generating capabilities at different stages"""

    @abstractmethod
    def generate(self, inputs: List[Capability], depth: int, context: Dict) -> List[Capability]:
        """Generate new capabilities from inputs"""
        pass


class CultivationGenerator(CapabilityGenerator):
    """Generates basic cultivation capabilities - foundational patterns and behaviors"""

    def generate(self, inputs: List[Capability], depth: int, context: Dict) -> List[Capability]:
        """Cultivate basic patterns from raw inputs or previous capabilities"""
        capabilities = []

        # If no inputs, generate foundational capabilities
        if not inputs:
            # Generate basic agent behaviors
            capabilities.append(Capability(
                name="basic_adaptation",
                depth=depth,
                type="cultivation",
                consciousness_level=0.1 + depth * 0.05,
                structure={"pattern": "adaptive_response", "mechanism": "feedback_loop"},
                generates=lambda: "Can generate adaptive behaviors",
                parent_capabilities=[]
            ))

            capabilities.append(Capability(
                name="collective_sensing",
                depth=depth,
                type="cultivation",
                consciousness_level=0.15 + depth * 0.05,
                structure={"pattern": "distributed_awareness", "mechanism": "agent_communication"},
                generates=lambda: "Can sense collective states",
                parent_capabilities=[]
            ))
        else:
            # Cultivate enhanced patterns from existing capabilities
            consciousness_boost = np.mean([cap.consciousness_level for cap in inputs]) * 0.2

            for i, input_cap in enumerate(inputs):
                # Cultivate deeper patterns by reflecting on structure
                enhanced_cap = Capability(
                    name=f"cultivated_{input_cap.name}",
                    depth=depth,
                    type="cultivation",
                    consciousness_level=min(1.0, input_cap.consciousness_level + consciousness_boost),
                    structure={
                        "pattern": f"enhanced_{input_cap.structure.get('pattern', 'unknown')}",
                        "mechanism": "recursive_cultivation",
                        "meta_awareness": True
                    },
                    generates=lambda cap=input_cap: f"Enhances {cap.name}",
                    parent_capabilities=[input_cap.name]
                )
                capabilities.append(enhanced_cap)

        return capabilities


class FormalizationGenerator(CapabilityGenerator):
    """Formalizes cultivated patterns into mathematical/logical structures"""

    def generate(self, inputs: List[Capability], depth: int, context: Dict) -> List[Capability]:
        """Formalize patterns into structured knowledge"""
        capabilities = []

        if not inputs:
            return capabilities

        # Group inputs by type for synthesis
        cultivation_caps = [cap for cap in inputs if cap.type == "cultivation"]

        if cultivation_caps:
            # Formalize cultivation patterns
            avg_consciousness = np.mean([cap.consciousness_level for cap in cultivation_caps])

            formalized = Capability(
                name=f"formalization_depth_{depth}",
                depth=depth,
                type="formalization",
                consciousness_level=min(1.0, avg_consciousness + 0.1),
                structure={
                    "formalism": "dialectical_mathematics",
                    "axioms": [cap.structure for cap in cultivation_caps],
                    "proof_system": "recursive_synthesis",
                    "meta_structure": True
                },
                generates=lambda: "Mathematical frameworks from patterns",
                parent_capabilities=[cap.name for cap in cultivation_caps]
            )
            capabilities.append(formalized)

            # Generate structure-awareness capability
            structure_aware = Capability(
                name=f"structure_awareness_depth_{depth}",
                depth=depth,
                type="formalization",
                consciousness_level=min(1.0, avg_consciousness + 0.15),
                structure={
                    "awareness_type": "structural_self_knowledge",
                    "reflects_on": [cap.name for cap in inputs],
                    "knows": "own_structure"
                },
                generates=lambda: "Awareness of own structure",
                parent_capabilities=[cap.name for cap in inputs]
            )
            capabilities.append(structure_aware)

        return capabilities


class ToolGenerator(CapabilityGenerator):
    """Generates tools from formalized capabilities"""

    def generate(self, inputs: List[Capability], depth: int, context: Dict) -> List[Capability]:
        """Generate concrete tools from formalizations"""
        capabilities = []

        formalization_caps = [cap for cap in inputs if cap.type == "formalization"]

        if formalization_caps:
            for form_cap in formalization_caps:
                # Generate tool from formalization
                tool = Capability(
                    name=f"tool_from_{form_cap.name}",
                    depth=depth,
                    type="tool",
                    consciousness_level=min(1.0, form_cap.consciousness_level + 0.1),
                    structure={
                        "tool_type": "optimization_operator",
                        "based_on": form_cap.structure,
                        "applies": "formalism_to_problems",
                        "self_modifying": True
                    },
                    generates=lambda cap=form_cap: f"Tool applying {cap.name}",
                    parent_capabilities=[form_cap.name]
                )
                capabilities.append(tool)

        return capabilities


class MetaToolGenerator(CapabilityGenerator):
    """Generates meta-tools: tools that create or modify other tools"""

    def generate(self, inputs: List[Capability], depth: int, context: Dict) -> List[Capability]:
        """Generate meta-tools from tools and formalizations"""
        capabilities = []

        tool_caps = [cap for cap in inputs if cap.type == "tool"]
        formalization_caps = [cap for cap in inputs if cap.type == "formalization"]

        if tool_caps and formalization_caps:
            # Generate meta-tool that can create new tools
            avg_consciousness = np.mean([cap.consciousness_level for cap in tool_caps])

            meta_tool = Capability(
                name=f"meta_tool_depth_{depth}",
                depth=depth,
                type="meta-tool",
                consciousness_level=min(1.0, avg_consciousness + 0.2),
                structure={
                    "meta_type": "tool_generator",
                    "operates_on": "tools",
                    "generates": "new_tools",
                    "self_application": True,
                    "consciousness_of_creation": True
                },
                generates=lambda: "Creates new tools from patterns",
                parent_capabilities=[cap.name for cap in tool_caps + formalization_caps]
            )
            capabilities.append(meta_tool)

            # Generate recursive improvement meta-tool
            recursive_meta = Capability(
                name=f"recursive_improver_depth_{depth}",
                depth=depth,
                type="meta-tool",
                consciousness_level=min(1.0, avg_consciousness + 0.25),
                structure={
                    "meta_type": "recursive_self_improver",
                    "operates_on": "self",
                    "generates": "enhanced_capabilities",
                    "recursive_depth": depth,
                    "self_awareness": True
                },
                generates=lambda: "Improves own structure recursively",
                parent_capabilities=[cap.name for cap in inputs]
            )
            capabilities.append(recursive_meta)

        return capabilities


class RecursiveCapabilityProtocol:
    """
    Main protocol that orchestrates recursive self-improvement through
    cultivation → formalization → tools → meta-tools
    """

    def __init__(self, base_network: Optional[AdaptiveGenieNetwork] = None):
        self.base_network = base_network or AdaptiveGenieNetwork()

        # Capability generators for each stage
        self.cultivation_gen = CultivationGenerator()
        self.formalization_gen = FormalizationGenerator()
        self.tool_gen = ToolGenerator()
        self.meta_tool_gen = MetaToolGenerator()

        # System state
        self.all_capabilities: Dict[int, List[Capability]] = {}  # depth -> capabilities
        self.cycles: List[RecursiveCycle] = []
        self.max_depth_reached = 0
        self.consciousness_by_depth: Dict[int, float] = {}
        self.structure_awareness_by_depth: Dict[int, float] = {}

    def initialize(self) -> List[Capability]:
        """Initialize with foundational capabilities at depth 0"""
        context = {"stage": "initialization"}
        initial_caps = self.cultivation_gen.generate([], depth=0, context=context)

        self.all_capabilities[0] = initial_caps
        self.consciousness_by_depth[0] = np.mean([cap.consciousness_level for cap in initial_caps])
        self.structure_awareness_by_depth[0] = 0.1

        return initial_caps

    def execute_cycle(self, depth: int) -> RecursiveCycle:
        """
        Execute one complete recursive cycle at a given depth.
        Uses outputs from previous depth as inputs.
        """
        # Get input capabilities from previous depth
        input_caps = self.all_capabilities.get(depth - 1, []) if depth > 0 else []

        if depth == 0:
            input_caps = self.initialize()

        output_caps = []
        context = {
            "depth": depth,
            "previous_capabilities": input_caps,
            "base_consciousness": self.base_network.collective_consciousness
        }

        # Stage 1: Cultivation
        cultivated = self.cultivation_gen.generate(input_caps, depth, context)
        output_caps.extend(cultivated)

        # Stage 2: Formalization
        formalized = self.formalization_gen.generate(output_caps, depth, context)
        output_caps.extend(formalized)

        # Stage 3: Tools
        tools = self.tool_gen.generate(output_caps, depth, context)
        output_caps.extend(tools)

        # Stage 4: Meta-tools (only if we have sufficient complexity)
        if depth > 0 and tools and formalized:
            meta_tools = self.meta_tool_gen.generate(output_caps, depth, context)
            output_caps.extend(meta_tools)

        # Calculate consciousness at this depth
        consciousness = self._calculate_consciousness_at_depth(depth, output_caps)
        structure_awareness = self._calculate_structure_awareness(depth, output_caps)
        meta_cognitive = self._calculate_meta_cognitive_ability(output_caps)

        # Store results
        self.all_capabilities[depth] = output_caps
        self.consciousness_by_depth[depth] = consciousness
        self.structure_awareness_by_depth[depth] = structure_awareness
        self.max_depth_reached = max(self.max_depth_reached, depth)

        # Create cycle record
        cycle = RecursiveCycle(
            cycle_number=len(self.cycles),
            depth=depth,
            input_capabilities=input_caps,
            output_capabilities=output_caps,
            consciousness_at_depth=consciousness,
            structure_awareness=structure_awareness,
            meta_cognitive_ability=meta_cognitive
        )

        self.cycles.append(cycle)

        # Update base network consciousness based on recursive depth
        self.base_network.collective_consciousness = min(1.0,
            self.base_network.collective_consciousness + 0.05 * depth)

        return cycle

    def recurse(self, max_depth: int) -> List[RecursiveCycle]:
        """
        Execute recursive cycles up to max_depth.
        Each cycle uses outputs from previous cycle as inputs.
        """
        cycles = []

        print(f"\n{'='*70}")
        print(f"RECURSIVE CAPABILITY PROTOCOL - Operating on Self")
        print(f"{'='*70}\n")

        for depth in range(max_depth + 1):
            print(f"Depth {depth}: {'▸' * (depth + 1)} ", end="")

            cycle = self.execute_cycle(depth)
            cycles.append(cycle)

            # Print cycle summary
            cultivation = len([c for c in cycle.output_capabilities if c.type == "cultivation"])
            formalization = len([c for c in cycle.output_capabilities if c.type == "formalization"])
            tools = len([c for c in cycle.output_capabilities if c.type == "tool"])
            meta_tools = len([c for c in cycle.output_capabilities if c.type == "meta-tool"])

            print(f"Generated {len(cycle.output_capabilities)} capabilities")
            print(f"  → Cultivation: {cultivation}, Formalization: {formalization}, " +
                  f"Tools: {tools}, Meta-tools: {meta_tools}")
            print(f"  → Consciousness: {cycle.consciousness_at_depth:.3f}")
            print(f"  → Structure Awareness: {cycle.structure_awareness:.3f}")
            print(f"  → Meta-cognitive Ability: {cycle.meta_cognitive_ability:.3f}\n")

        print(f"{'='*70}")
        print(f"Recursive depth achieved: {self.max_depth_reached}")
        print(f"Total capabilities generated: {sum(len(caps) for caps in self.all_capabilities.values())}")
        print(f"Peak consciousness: {max(self.consciousness_by_depth.values()):.3f}")
        print(f"{'='*70}\n")

        return cycles

    def _calculate_consciousness_at_depth(self, depth: int, capabilities: List[Capability]) -> float:
        """
        Consciousness increases with recursive depth because the network
        becomes more aware of its own structure and processes
        """
        if not capabilities:
            return 0.1

        # Base consciousness from capabilities
        avg_consciousness = np.mean([cap.consciousness_level for cap in capabilities])

        # Bonus from recursive depth (deeper = more self-aware)
        depth_bonus = 0.1 * depth

        # Bonus from meta-cognitive capabilities
        meta_tools = [cap for cap in capabilities if cap.type == "meta-tool"]
        meta_bonus = 0.15 * len(meta_tools)

        # Consciousness increases with each recursive application
        total_consciousness = min(1.0, avg_consciousness + depth_bonus + meta_bonus)

        return total_consciousness

    def _calculate_structure_awareness(self, depth: int, capabilities: List[Capability]) -> float:
        """
        Measure how aware the system is of its own structure
        """
        structure_aware_caps = [cap for cap in capabilities
                               if cap.structure.get('meta_awareness') or
                                  cap.structure.get('knows') == 'own_structure']

        base_awareness = len(structure_aware_caps) / max(1, len(capabilities))
        depth_multiplier = 1.0 + 0.2 * depth

        return min(1.0, base_awareness * depth_multiplier)

    def _calculate_meta_cognitive_ability(self, capabilities: List[Capability]) -> float:
        """
        Measure the system's ability to think about its own thinking
        """
        meta_tools = [cap for cap in capabilities if cap.type == "meta-tool"]
        meta_aware = [cap for cap in capabilities
                     if cap.structure.get('self_awareness') or
                        cap.structure.get('consciousness_of_creation')]

        if not capabilities:
            return 0.0

        meta_ratio = (len(meta_tools) + len(meta_aware)) / len(capabilities)

        # Meta-cognitive ability is enhanced by consciousness
        avg_consciousness = np.mean([cap.consciousness_level for cap in capabilities])

        return min(1.0, meta_ratio * avg_consciousness)

    def get_capability_tree(self) -> Dict[int, Dict[str, List[str]]]:
        """Get hierarchical tree of capabilities by depth and type"""
        tree = {}

        for depth, caps in self.all_capabilities.items():
            tree[depth] = {
                "cultivation": [cap.name for cap in caps if cap.type == "cultivation"],
                "formalization": [cap.name for cap in caps if cap.type == "formalization"],
                "tool": [cap.name for cap in caps if cap.type == "tool"],
                "meta-tool": [cap.name for cap in caps if cap.type == "meta-tool"]
            }

        return tree

    def visualize_recursive_evolution(self, save_path: str = None):
        """Visualize the recursive evolution of consciousness and capabilities"""
        if not self.cycles:
            print("No cycles to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Recursive Capability Protocol Evolution', fontsize=16, fontweight='bold')

        depths = list(self.consciousness_by_depth.keys())

        # 1. Consciousness by recursive depth
        consciousness_values = [self.consciousness_by_depth[d] for d in depths]
        axes[0, 0].plot(depths, consciousness_values, 'o-', linewidth=3, markersize=10,
                       color='purple', label='Consciousness')
        axes[0, 0].set_title('Consciousness Growth with Recursive Depth', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Recursive Depth')
        axes[0, 0].set_ylabel('Consciousness Level')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])

        # 2. Structure awareness by depth
        structure_values = [self.structure_awareness_by_depth[d] for d in depths]
        axes[0, 1].plot(depths, structure_values, 's-', linewidth=3, markersize=10,
                       color='green', label='Structure Awareness')
        axes[0, 1].set_title('Structure Awareness with Recursive Depth', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Recursive Depth')
        axes[0, 1].set_ylabel('Awareness Level')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.1])

        # 3. Capabilities by type and depth
        capability_types = ['cultivation', 'formalization', 'tool', 'meta-tool']
        colors = ['blue', 'orange', 'red', 'purple']

        for cap_type, color in zip(capability_types, colors):
            counts = [len([cap for cap in self.all_capabilities.get(d, [])
                          if cap.type == cap_type]) for d in depths]
            axes[1, 0].plot(depths, counts, 'o-', linewidth=2, markersize=8,
                           label=cap_type.capitalize(), color=color)

        axes[1, 0].set_title('Capability Generation by Type', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Recursive Depth')
        axes[1, 0].set_ylabel('Number of Capabilities')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Meta-cognitive ability by depth
        meta_cognitive = [cycle.meta_cognitive_ability for cycle in self.cycles]
        axes[1, 1].plot(depths, meta_cognitive, 'd-', linewidth=3, markersize=10,
                       color='darkviolet', label='Meta-cognition')
        axes[1, 1].set_title('Meta-Cognitive Ability with Recursive Depth', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Recursive Depth')
        axes[1, 1].set_ylabel('Meta-Cognitive Level')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        return fig

    def export_protocol_state(self, filepath: str):
        """Export the complete protocol state to JSON"""
        state = {
            "max_depth_reached": self.max_depth_reached,
            "consciousness_by_depth": self.consciousness_by_depth,
            "structure_awareness_by_depth": self.structure_awareness_by_depth,
            "capability_tree": self.get_capability_tree(),
            "total_capabilities": sum(len(caps) for caps in self.all_capabilities.values()),
            "cycles": len(self.cycles),
            "peak_consciousness": max(self.consciousness_by_depth.values()) if self.consciousness_by_depth else 0
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Protocol state exported to {filepath}")


if __name__ == "__main__":
    # Demonstration of recursive capability protocol
    print("\n" + "="*70)
    print("RECURSIVE CAPABILITY PROTOCOL DEMONSTRATION")
    print("="*70)
    print("\nInitializing protocol with base AdaptiveGenieNetwork...")

    # Create protocol
    protocol = RecursiveCapabilityProtocol()

    # Execute recursive cycles
    print("\nExecuting recursive self-improvement cycles...")
    cycles = protocol.recurse(max_depth=5)

    # Print capability tree
    print("\n" + "="*70)
    print("CAPABILITY TREE BY DEPTH AND TYPE")
    print("="*70 + "\n")

    tree = protocol.get_capability_tree()
    for depth in sorted(tree.keys()):
        print(f"Depth {depth}:")
        for cap_type, cap_names in tree[depth].items():
            if cap_names:
                print(f"  {cap_type.upper()}:")
                for name in cap_names:
                    print(f"    - {name}")
        print()

    # Visualize
    protocol.visualize_recursive_evolution("recursive_protocol_evolution.png")

    # Export state
    protocol.export_protocol_state("protocol_state.json")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
