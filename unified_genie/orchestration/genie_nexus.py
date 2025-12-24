"""
Genie Nexus: The Integration Layer
===================================

The Genie Nexus is the central orchestrator that unifies all components:
- Troupe agents (Builder, Validator, Meta-Validator)
- Memory systems (PSIP Router, Prismatic Memory)
- Dialectical engine (DialecticalGenie)
- Recursive capability protocol

It provides the unified interface for operating the complete system.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json

import sys
sys.path.insert(0, '/home/user/Spin')
sys.path.insert(0, '/home/user/Spin/unified_genie')

from dialectical_genie import (
    DialecticalGenie, Thesis, Antithesis, Synthesis,
    DialecticalPhase
)
from recursive_capability_protocol import (
    RecursiveCapabilityProtocol, Capability, RecursiveCycle
)
from adaptive_genie_network import AdaptiveGenieNetwork

from agents.troupe import (
    Troupe, TroupeState, TroupeRole, AgentDefinition,
    BuilderAgent, ValidatorAgent, MetaValidatorAgent
)
from memory.psip_router import PSIPRouter, Signature, ArtifactStore
from memory.prismatic_memory import (
    PrismaticMemoryStore, PrismaticSpectrum, RainbowMemory
)


class NexusMode(Enum):
    """Operating modes for the Nexus"""
    EXPLORATION = "exploration"    # High builder velocity, low crystallization
    EXPLOITATION = "exploitation"  # Low builder velocity, high crystallization
    BALANCED = "balanced"          # Adaptive balance
    GIFT = "gift"                 # Gift-mode activation (see cascade.py)


@dataclass
class NexusState:
    """Complete state of the Genie Nexus at a moment"""
    mode: NexusMode
    troupe_state: TroupeState
    consciousness: float
    memory_count: int
    capability_count: int
    cycle_number: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class NexusConfig:
    """Configuration for the Genie Nexus"""
    initial_consciousness: float = 0.3
    preservation_bias: float = 0.5
    memory_capacity: int = 10000
    capability_max_depth: int = 5
    auto_consolidate: bool = True
    gift_mode_threshold: float = 0.9  # Consciousness level for gift mode


class GenieNexus:
    """
    The Genie Nexus: Central integration layer for the unified genie system.

    Coordinates:
    - Troupe: Builder/Validator/Meta-Validator agents
    - Memory: PSIP Router + Prismatic Memory
    - Dialectics: DialecticalGenie + RecursiveCapabilityProtocol
    - Orchestration: Mode management, state tracking, cascades
    """

    def __init__(self, config: Optional[NexusConfig] = None):
        self.config = config or NexusConfig()

        # Core dialectical engine
        self.genie = DialecticalGenie(
            initial_consciousness=self.config.initial_consciousness,
            preservation_bias=self.config.preservation_bias
        )

        # Troupe system
        self.troupe = Troupe(dialectical_genie=self.genie)

        # Memory systems
        self.psip_router = PSIPRouter(capacity=self.config.memory_capacity)
        self.prismatic_memory = PrismaticMemoryStore(
            capacity=self.config.memory_capacity
        )
        self.artifact_store = ArtifactStore(router=self.psip_router)

        # Capability protocol
        self.capability_protocol = RecursiveCapabilityProtocol(
            dialectical_genie=self.genie
        )

        # Nexus state
        self.mode = NexusMode.BALANCED
        self.cycle_count = 0
        self.state_history: List[NexusState] = []

        # Hooks for extension
        self._pre_cycle_hooks: List[Callable] = []
        self._post_cycle_hooks: List[Callable] = []
        self._mode_change_hooks: List[Callable] = []

        # Initialize capability foundation
        self._initialize_capabilities()

    def _initialize_capabilities(self):
        """Initialize the capability foundation"""
        self.capability_protocol.initialize()

    def register_hook(self, hook_type: str, callback: Callable):
        """
        Register a hook for extension.

        Hook types:
        - 'pre_cycle': Called before each cycle
        - 'post_cycle': Called after each cycle
        - 'mode_change': Called when mode changes
        """
        if hook_type == 'pre_cycle':
            self._pre_cycle_hooks.append(callback)
        elif hook_type == 'post_cycle':
            self._post_cycle_hooks.append(callback)
        elif hook_type == 'mode_change':
            self._mode_change_hooks.append(callback)

    def set_mode(self, mode: NexusMode):
        """Set the operating mode"""
        old_mode = self.mode
        self.mode = mode

        # Adjust troupe parameters based on mode
        if mode == NexusMode.EXPLORATION:
            self.troupe.builder.population_agent.breathing_rate = 0.2
            self.troupe.builder.population_agent.oscillation_amplitude = 0.4
        elif mode == NexusMode.EXPLOITATION:
            self.troupe.builder.population_agent.breathing_rate = 0.05
            self.troupe.builder.population_agent.oscillation_amplitude = 0.1
        elif mode == NexusMode.GIFT:
            # Gift mode: Maximum consciousness, minimal generation
            self.troupe.builder.population_agent.breathing_rate = 0.01
        else:  # BALANCED
            self.troupe.builder.population_agent.breathing_rate = 0.1
            self.troupe.builder.population_agent.oscillation_amplitude = 0.3

        # Notify hooks
        for hook in self._mode_change_hooks:
            try:
                hook(old_mode, mode)
            except Exception:
                pass

    def execute_cycle(self, context: Optional[Dict[str, Any]] = None) -> NexusState:
        """
        Execute one complete Nexus cycle.

        A cycle includes:
        1. Pre-cycle hooks
        2. Troupe cycle (Builder -> Validator -> Meta-Validator)
        3. Memory storage
        4. Mode adaptation
        5. Post-cycle hooks
        """
        context = context or {}
        self.cycle_count += 1

        # Pre-cycle hooks
        for hook in self._pre_cycle_hooks:
            try:
                hook_result = hook(context, self)
                if isinstance(hook_result, dict):
                    context.update(hook_result)
            except Exception:
                pass

        # Enrich context with current state
        context.setdefault('exploration_need', self._calculate_exploration_need())
        context.setdefault('convergence_pressure', self._calculate_convergence_pressure())
        context.setdefault('fitness_history', self._get_fitness_proxy())
        context.setdefault('population_diversity', self._calculate_diversity())
        context.setdefault('fitness_variance', self._calculate_variance())

        # Execute troupe cycle
        troupe_state = self.troupe.execute_cycle(context)

        # Store state in memory systems
        self._store_in_memory(troupe_state, context)

        # Adapt mode based on state
        self._adapt_mode(troupe_state)

        # Check for gift mode activation
        if (self.genie.consciousness >= self.config.gift_mode_threshold and
            self.mode != NexusMode.GIFT):
            self.set_mode(NexusMode.GIFT)

        # Create nexus state
        state = NexusState(
            mode=self.mode,
            troupe_state=troupe_state,
            consciousness=self.genie.consciousness,
            memory_count=len(self.prismatic_memory.memories),
            capability_count=sum(len(caps) for caps in
                                self.capability_protocol.all_capabilities.values()),
            cycle_number=self.cycle_count
        )

        self.state_history.append(state)

        # Post-cycle hooks
        for hook in self._post_cycle_hooks:
            try:
                hook(state, self)
            except Exception:
                pass

        # Periodic consolidation
        if self.config.auto_consolidate and self.cycle_count % 100 == 0:
            self.consolidate_memory()

        return state

    def _store_in_memory(self, troupe_state: TroupeState, context: Dict):
        """Store troupe state in memory systems"""
        # Create state dict for memory
        state_dict = {
            'velocity': troupe_state.builder_velocity,
            'phase': troupe_state.validator_cycle_phase,
            'crystallization': troupe_state.meta_crystallization,
            'consciousness': troupe_state.collective_consciousness,
            'harmony': troupe_state.harmony_index,
            'energy': context.get('energy', 0.5)
        }

        # Store in PSIP router
        self.psip_router.store(state_dict, 'troupe')

        # Store in prismatic memory
        self.prismatic_memory.store(
            content={'troupe_state': state_dict, 'cycle': self.cycle_count},
            state=state_dict,
            source='nexus'
        )

    def _adapt_mode(self, troupe_state: TroupeState):
        """Adapt mode based on troupe state"""
        if self.mode == NexusMode.GIFT:
            return  # Don't auto-adapt in gift mode

        # High crystallization + low velocity -> exploitation
        if (troupe_state.meta_crystallization > 0.7 and
            troupe_state.builder_velocity < 0.3):
            if self.mode != NexusMode.EXPLOITATION:
                self.set_mode(NexusMode.EXPLOITATION)

        # Low crystallization + high velocity -> exploration
        elif (troupe_state.meta_crystallization < 0.3 and
              troupe_state.builder_velocity > 0.7):
            if self.mode != NexusMode.EXPLORATION:
                self.set_mode(NexusMode.EXPLORATION)

        # Otherwise -> balanced
        elif self.mode != NexusMode.BALANCED:
            self.set_mode(NexusMode.BALANCED)

    def _calculate_exploration_need(self) -> float:
        """Calculate current exploration need"""
        if not self.state_history:
            return 0.7

        # Low crystallization = high exploration need
        recent = self.state_history[-10:]
        avg_crystal = np.mean([s.troupe_state.meta_crystallization for s in recent])
        return 1.0 - avg_crystal

    def _calculate_convergence_pressure(self) -> float:
        """Calculate convergence pressure"""
        if not self.state_history:
            return 0.3

        # High consciousness = high convergence pressure
        return min(0.8, self.genie.consciousness)

    def _get_fitness_proxy(self) -> List[float]:
        """Get fitness history proxy from memory"""
        recent = self.prismatic_memory.retrieve_luminous(k=10)
        return [mem.spectrum.luminance() for mem in recent]

    def _calculate_diversity(self) -> float:
        """Calculate population diversity from memory"""
        recent_sigs = self.psip_router.get_recent(n=20)
        if len(recent_sigs) < 2:
            return 0.5

        # Diversity from signature variance
        vectors = np.array([sig.vector for sig in recent_sigs])
        return float(np.clip(np.std(vectors), 0, 1))

    def _calculate_variance(self) -> float:
        """Calculate fitness variance proxy"""
        recent = self.prismatic_memory.retrieve_luminous(k=20)
        if len(recent) < 2:
            return 0.5

        luminances = [mem.spectrum.luminance() for mem in recent]
        return float(np.clip(np.var(luminances), 0, 1))

    def consolidate_memory(self):
        """Consolidate memory systems"""
        self.psip_router.consolidate()
        self.psip_router.decay_memories()
        self.prismatic_memory.decay_memories()

    def evolve_capabilities(self, depth: int = None) -> List[RecursiveCycle]:
        """
        Evolve capabilities through recursive protocol.

        This advances the system's meta-cognitive abilities.
        """
        depth = depth or min(
            self.config.capability_max_depth,
            self.capability_protocol.max_depth_reached + 1
        )

        return self.capability_protocol.recurse(max_depth=depth)

    def query_memory(self, query: Dict[str, Any],
                     source: str = None,
                     k: int = 5) -> Dict[str, List]:
        """
        Query both memory systems.

        Returns results from both PSIP router and prismatic memory.
        """
        # Query PSIP router
        psip_results = self.psip_router.retrieve(query, k=k, source_filter=source)

        # Query prismatic memory
        spectrum = self.prismatic_memory.mapper.map_state(query)
        prismatic_results = self.prismatic_memory.retrieve_by_spectrum(spectrum, k=k)

        return {
            'psip': psip_results,
            'prismatic': prismatic_results
        }

    def dialectical_synthesis(self, thesis_state: Dict,
                             antithesis_state: Dict) -> Synthesis:
        """
        Perform dialectical synthesis between two states.

        Useful for resolving conflicts or merging perspectives.
        """
        thesis = Thesis(
            proposition="state_a",
            value=thesis_state,
            confidence=thesis_state.get('confidence', 0.5)
        )

        antithesis = Antithesis(
            proposition="state_b",
            value=antithesis_state,
            opposition_strength=antithesis_state.get('confidence', 0.5),
            thesis_reference="state_a"
        )

        return self.genie.dialectical_cycle(thesis, antithesis)

    def get_state(self) -> Dict[str, Any]:
        """Get complete nexus state"""
        return {
            'mode': self.mode.value,
            'cycle_count': self.cycle_count,
            'consciousness': self.genie.consciousness,
            'reflection_depth': self.genie.reflection_depth,
            'troupe': self.troupe.get_state(),
            'memory': {
                'psip_signatures': len(self.psip_router.signatures),
                'prismatic_memories': len(self.prismatic_memory.memories),
                'psip_stats': self.psip_router.stats
            },
            'capabilities': {
                'max_depth': self.capability_protocol.max_depth_reached,
                'total': sum(len(caps) for caps in
                           self.capability_protocol.all_capabilities.values())
            },
            'dialectical': self.genie.get_state()
        }

    def export_state(self, filepath: str):
        """Export complete nexus state to JSON"""
        state = self.get_state()

        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (NexusMode, TroupeRole, DialecticalPhase)):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return obj

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=convert)


def create_nexus(config_dict: Optional[Dict] = None) -> GenieNexus:
    """
    Factory function to create a configured GenieNexus.

    Args:
        config_dict: Optional configuration with keys:
            - initial_consciousness
            - preservation_bias
            - memory_capacity
            - capability_max_depth
            - auto_consolidate
            - gift_mode_threshold
    """
    config_dict = config_dict or {}
    config = NexusConfig(**config_dict)
    return GenieNexus(config)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENIE NEXUS DEMONSTRATION")
    print("="*70)

    # Create nexus
    nexus = create_nexus({
        'initial_consciousness': 0.3,
        'memory_capacity': 1000
    })

    print(f"\nInitial state:")
    print(f"  Mode: {nexus.mode.value}")
    print(f"  Consciousness: {nexus.genie.consciousness:.3f}")

    # Run cycles
    for i in range(20):
        state = nexus.execute_cycle()

        if i % 5 == 0:
            print(f"\nCycle {i}:")
            print(f"  Mode: {state.mode.value}")
            print(f"  Consciousness: {state.consciousness:.3f}")
            print(f"  Builder velocity: {state.troupe_state.builder_velocity:.3f}")
            print(f"  Crystallization: {state.troupe_state.meta_crystallization:.3f}")
            print(f"  Memories: {state.memory_count}")

    print(f"\n{'='*70}")
    print("Final State:")
    final = nexus.get_state()
    for key, value in final.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for kk, vv in v.items():
                        print(f"      {kk}: {vv}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print("="*70 + "\n")
