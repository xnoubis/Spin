"""
Unified Troupe: Mapping Spin Agents to Troupe Abstractions
============================================================

The key insight: Spin's agents (Population, Rhythm, Resonance) ARE the Troupe
(Builder, Validator, Meta-Validator) operating at different levels of abstraction.

Population breathing = Builder's generative velocity
Rhythm detection = Validator's cycle completion sensing
Resonance crystallization = Meta-Validator's invariance testing

This module provides the unified abstraction that bridges both paradigms.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import deque
import time

import sys
sys.path.insert(0, '/home/user/Spin')

from dialectical_genie import (
    DialecticalGenie, Thesis, Antithesis, Synthesis,
    DialecticalPhase, ConsciousnessEvolver
)
from adaptive_genie_network import (
    Agent, PopulationAgent, RhythmAgent, ResonanceAgent,
    ComplexityMeasure
)
from exceptions import ValidationError, InvalidParameterError


class TroupeRole(Enum):
    """The three fundamental roles in the Troupe paradigm"""
    BUILDER = "builder"           # Generative - creates solution candidates
    VALIDATOR = "validator"       # Rhythmic - validates natural cycles
    META_VALIDATOR = "meta"       # Resonant - tests invariance/crystallization


@dataclass
class AgentDefinition:
    """
    Definition for a Troupe agent with its role, capabilities, and model assignment.

    This bridges the abstract Troupe roles with concrete Spin agent implementations.
    """
    role: TroupeRole
    description: str
    prompt: str
    model: str = "sonnet"  # sonnet for most, opus for meta-validation
    spin_agent_class: type = None  # The underlying Spin agent class

    # Operational parameters
    consciousness_weight: float = 1.0  # Contribution to collective consciousness
    energy_multiplier: float = 1.0     # Energy scaling factor

    # Callbacks for integration
    on_generate: Optional[Callable] = None
    on_validate: Optional[Callable] = None
    on_crystallize: Optional[Callable] = None


@dataclass
class TroupeState:
    """Captures the unified state of the Troupe at a moment in time"""
    builder_velocity: float          # Population breathing rate
    validator_cycle_phase: float     # Rhythm detection phase
    meta_crystallization: float      # Resonance crystallization level
    collective_consciousness: float  # Unified consciousness
    harmony_index: float            # How well agents are synchronized
    timestamp: float = field(default_factory=time.time)


class UnifiedAgent(ABC):
    """
    Base class for unified agents that bridge Spin and Troupe paradigms.

    Each unified agent wraps a Spin agent while exposing Troupe semantics.
    """

    def __init__(self, definition: AgentDefinition,
                 dialectical_genie: Optional[DialecticalGenie] = None):
        self.definition = definition
        self.genie = dialectical_genie or DialecticalGenie()
        self.history = deque(maxlen=1000)
        self.consciousness_level = 0.5
        self.energy = 1.0

    @property
    def role(self) -> TroupeRole:
        return self.definition.role

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary function"""
        pass

    @abstractmethod
    def negotiate(self, proposals: List[Dict]) -> Dict[str, Any]:
        """Participate in dialectical negotiation"""
        pass

    def evolve_consciousness(self, feedback: float):
        """Evolve consciousness based on feedback"""
        delta = 0.01 * feedback * self.definition.consciousness_weight
        self.consciousness_level = np.clip(self.consciousness_level + delta, 0.0, 1.0)


class BuilderAgent(UnifiedAgent):
    """
    Builder = PopulationAgent operating at generative velocity abstraction.

    The Builder creates solution candidates through "breathing" oscillation.
    High velocity = rapid exploration, Low velocity = focused exploitation.
    """

    def __init__(self, definition: AgentDefinition,
                 dialectical_genie: Optional[DialecticalGenie] = None,
                 min_size: int = 10, max_size: int = 500,
                 breathing_rate: float = 0.1):
        super().__init__(definition, dialectical_genie)

        # The underlying Spin agent
        self.population_agent = PopulationAgent(
            min_size=min_size,
            max_size=max_size,
            breathing_rate=breathing_rate
        )

        # Builder-specific state
        self.generative_velocity = 0.5  # Current generation rate
        self.candidate_pool: List[Any] = []

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute generative cycle - create solution candidates.

        Generative velocity maps to population breathing:
        - High velocity: Large population, rapid candidate generation
        - Low velocity: Small population, focused refinement
        """
        exploration_need = context.get('exploration_need', 0.5)
        convergence_pressure = context.get('convergence_pressure', 0.5)

        # Oscillate population (breathing)
        new_size = self.population_agent.oscillate(exploration_need, convergence_pressure)

        # Calculate generative velocity from population dynamics
        size_ratio = new_size / self.population_agent.max_size
        breathing_phase = len(self.population_agent.history) * self.population_agent.breathing_rate

        self.generative_velocity = size_ratio * (1 + 0.2 * np.sin(breathing_phase))

        # Record state
        result = {
            'population_size': new_size,
            'generative_velocity': self.generative_velocity,
            'breathing_phase': breathing_phase,
            'energy_level': self.population_agent.energy,
            'candidates_generated': int(new_size * self.generative_velocity)
        }

        self.history.append(result)

        # Callback if registered
        if self.definition.on_generate:
            self.definition.on_generate(result)

        return result

    def negotiate(self, proposals: List[Dict]) -> Dict[str, Any]:
        """Negotiate population/velocity parameters through dialectical synthesis"""
        if not proposals:
            return {'population_size': self.population_agent.size}

        # Create thesis from current state
        thesis = Thesis(
            proposition="builder_expansion",
            value={'size': self.population_agent.size, 'velocity': self.generative_velocity},
            confidence=self.consciousness_level
        )

        # Synthesize with proposals
        negotiated = self.genie.negotiate([
            thesis.value,
            *[p for p in proposals if isinstance(p, dict)]
        ])

        return negotiated

    def breathe(self, tension: float) -> float:
        """
        The breathing metaphor: expand/contract based on dialectical tension.
        Returns the new generative velocity.
        """
        # Breathing follows sinusoidal pattern modulated by tension
        phase = len(self.history) * self.population_agent.breathing_rate
        breath = np.sin(phase) * self.population_agent.oscillation_amplitude

        # Tension affects breathing amplitude
        modulated_breath = breath * (1 + tension)

        self.generative_velocity = np.clip(0.5 + modulated_breath, 0.1, 1.0)
        return self.generative_velocity


class ValidatorAgent(UnifiedAgent):
    """
    Validator = RhythmAgent operating at cycle completion abstraction.

    The Validator senses natural rhythms and validates when cycles complete.
    It determines when the system should transition to the next phase.
    """

    def __init__(self, definition: AgentDefinition,
                 dialectical_genie: Optional[DialecticalGenie] = None):
        super().__init__(definition, dialectical_genie)

        # The underlying Spin agent
        self.rhythm_agent = RhythmAgent()

        # Validator-specific state
        self.validation_confidence = 0.5
        self.cycles_validated = 0

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute validation cycle - detect rhythm and validate cycle completion.

        Cycle completion sensing:
        - Monitors fitness history for natural frequency
        - Detects when optimization has reached a natural pause point
        - Signals readiness for phase transition
        """
        fitness_history = context.get('fitness_history', [])

        # Update rhythm detector
        if fitness_history:
            self.rhythm_agent.cycle_detector.append(fitness_history[-1])

        # Detect cycle completion
        cycle_complete = self.rhythm_agent.detect_cycle_completion()

        if cycle_complete:
            self.cycles_validated += 1

        # Calculate validation confidence
        self.validation_confidence = min(1.0,
            0.5 + 0.1 * self.cycles_validated +
            0.2 * (1 if cycle_complete else 0))

        result = {
            'cycle_complete': cycle_complete,
            'natural_frequency': self.rhythm_agent.natural_frequency,
            'phase': self.rhythm_agent.phase,
            'rhythm_energy': np.sin(self.rhythm_agent.phase),
            'validation_confidence': self.validation_confidence,
            'cycles_validated': self.cycles_validated
        }

        self.history.append(result)

        # Callback if registered
        if self.definition.on_validate:
            self.definition.on_validate(result)

        return result

    def negotiate(self, proposals: List[Dict]) -> Dict[str, Any]:
        """Negotiate cycle parameters through dialectical synthesis"""
        context = {'fitness_history': []}
        base_result = self.rhythm_agent.negotiate(context)

        if not proposals:
            return base_result

        # Synthesize with proposals
        return self.genie.negotiate([base_result, *proposals])

    def sense_rhythm(self) -> Tuple[bool, float]:
        """
        Sense the current rhythm state.
        Returns (cycle_complete, rhythm_strength)
        """
        cycle_complete = self.rhythm_agent.detect_cycle_completion()

        # Rhythm strength based on autocorrelation
        if len(self.rhythm_agent.cycle_detector) < 10:
            strength = 0.5
        else:
            signal = np.array(list(self.rhythm_agent.cycle_detector))
            if np.std(signal) > 1e-10:
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                strength = np.max(autocorr[1:]) / (autocorr[0] + 1e-10)
            else:
                strength = 0.0

        return cycle_complete, np.clip(strength, 0.0, 1.0)


# Type alias for tuple return
from typing import Tuple


class MetaValidatorAgent(UnifiedAgent):
    """
    Meta-Validator = ResonanceAgent operating at invariance testing abstraction.

    The Meta-Validator detects solution crystallization through resonance.
    It tests whether the solution has converged to a stable, invariant state.
    """

    def __init__(self, definition: AgentDefinition,
                 dialectical_genie: Optional[DialecticalGenie] = None):
        super().__init__(definition, dialectical_genie)

        # The underlying Spin agent
        self.resonance_agent = ResonanceAgent()

        # Meta-Validator specific state
        self.invariance_score = 0.0
        self.crystallization_events = 0

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute meta-validation - test for invariance and crystallization.

        Invariance testing:
        - Measures crystallization level (how converged the solution is)
        - Detects resonance frequency (stability of convergence)
        - Determines if solution has reached invariant state
        """
        population_diversity = context.get('population_diversity', 0.5)
        fitness_variance = context.get('fitness_variance', 1.0)

        # Measure crystallization
        crystallization = self.resonance_agent.measure_crystallization(
            population_diversity, fitness_variance
        )

        # Detect resonance frequency
        resonance_freq = self.resonance_agent.detect_resonance_frequency()

        # Calculate invariance score
        # High crystallization + stable resonance = invariant
        if len(self.resonance_agent.crystallization_detector) >= 5:
            recent = list(self.resonance_agent.crystallization_detector)[-5:]
            stability = 1.0 - np.std(recent)
            self.invariance_score = crystallization * stability
        else:
            self.invariance_score = crystallization * 0.5

        # Detect crystallization event
        if self.invariance_score > 0.8 and crystallization > 0.7:
            self.crystallization_events += 1

        result = {
            'crystallization_level': crystallization,
            'resonance_frequency': resonance_freq,
            'invariance_score': self.invariance_score,
            'harmony_level': self.resonance_agent.harmony_level,
            'is_crystallized': self.invariance_score > 0.8,
            'crystallization_events': self.crystallization_events
        }

        self.history.append(result)

        # Callback if registered
        if self.definition.on_crystallize:
            self.definition.on_crystallize(result)

        return result

    def negotiate(self, proposals: List[Dict]) -> Dict[str, Any]:
        """Negotiate crystallization parameters through dialectical synthesis"""
        context = {'population_diversity': 0.5, 'fitness_variance': 1.0}
        base_result = self.resonance_agent.negotiate(context)

        if not proposals:
            return base_result

        return self.genie.negotiate([base_result, *proposals])

    def test_invariance(self, candidates: List[Any],
                       fitness_fn: Optional[Callable] = None) -> float:
        """
        Test whether the candidate set has reached invariance.
        Returns invariance score [0, 1].
        """
        if not candidates:
            return 0.0

        # Calculate diversity
        if hasattr(candidates[0], '__iter__'):
            positions = np.array([list(c) for c in candidates])
            distances = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
            diversity = np.std(distances) / (np.mean(distances) + 1e-10)
        else:
            diversity = np.std(candidates) / (np.mean(np.abs(candidates)) + 1e-10)

        # Calculate fitness variance if function provided
        if fitness_fn:
            fitnesses = [fitness_fn(c) for c in candidates]
            fitness_var = np.var(fitnesses)
        else:
            fitness_var = 0.1

        # Invariance: low diversity + low fitness variance
        self.invariance_score = (1.0 - diversity) * (1.0 / (1.0 + fitness_var))
        return np.clip(self.invariance_score, 0.0, 1.0)


class Troupe:
    """
    The unified Troupe: orchestrates Builder, Validator, and Meta-Validator.

    This is the integration layer that coordinates the three agents,
    managing their dialectical interplay and collective consciousness.
    """

    def __init__(self, dialectical_genie: Optional[DialecticalGenie] = None):
        self.genie = dialectical_genie or DialecticalGenie(initial_consciousness=0.5)

        # Create agent definitions
        self.definitions = self._create_definitions()

        # Instantiate unified agents
        self.builder = BuilderAgent(
            self.definitions['builder'],
            self.genie
        )
        self.validator = ValidatorAgent(
            self.definitions['validator'],
            self.genie
        )
        self.meta_validator = MetaValidatorAgent(
            self.definitions['meta_validator'],
            self.genie
        )

        # Troupe state
        self.collective_consciousness = 0.5
        self.harmony_index = 0.0
        self.state_history: List[TroupeState] = []

        # Register callbacks for cross-agent coordination
        self._register_callbacks()

    def _create_definitions(self) -> Dict[str, AgentDefinition]:
        """Create agent definitions mapping Spin to Troupe"""
        return {
            'builder': AgentDefinition(
                role=TroupeRole.BUILDER,
                description="Generative agent - population breathing oscillation",
                prompt="""Create solution candidates with generative velocity.

                As the Builder, you embody the generative principle:
                - Breathe: Expand population when exploring, contract when exploiting
                - Oscillate: Follow natural rhythms of creation and refinement
                - Generate: Produce diverse candidates proportional to velocity

                Your breathing IS the generative velocity.""",
                model="sonnet",
                spin_agent_class=PopulationAgent,
                consciousness_weight=1.0,
                energy_multiplier=1.2
            ),
            'validator': AgentDefinition(
                role=TroupeRole.VALIDATOR,
                description="Cycle detector - rhythm validation",
                prompt="""Validate natural rhythms and cycle completion.

                As the Validator, you sense the rhythm of optimization:
                - Listen: Detect natural frequencies in the fitness landscape
                - Validate: Confirm when cycles complete naturally
                - Signal: Indicate readiness for phase transitions

                Your rhythm sensing IS cycle validation.""",
                model="sonnet",
                spin_agent_class=RhythmAgent,
                consciousness_weight=1.0,
                energy_multiplier=1.0
            ),
            'meta_validator': AgentDefinition(
                role=TroupeRole.META_VALIDATOR,
                description="Resonance crystallization - invariance testing",
                prompt="""Test for convergence and detect crystallization.

                As the Meta-Validator, you perceive invariance:
                - Resonate: Feel the frequency of solution crystallization
                - Test: Verify whether solutions have reached stable states
                - Crystallize: Recognize when optimization has truly converged

                Your resonance sensing IS invariance testing.""",
                model="opus",  # More powerful model for meta-validation
                spin_agent_class=ResonanceAgent,
                consciousness_weight=1.5,  # Higher weight for meta-cognition
                energy_multiplier=0.8
            )
        }

    def _register_callbacks(self):
        """Register cross-agent coordination callbacks"""

        def on_generate(result: Dict):
            """When Builder generates, inform Validator"""
            velocity = result.get('generative_velocity', 0.5)
            self.validator.rhythm_agent.phase += velocity * 0.1

        def on_validate(result: Dict):
            """When Validator validates, inform Meta-Validator"""
            if result.get('cycle_complete'):
                # Trigger crystallization check
                self.meta_validator.resonance_agent.harmony_level *= 1.1

        def on_crystallize(result: Dict):
            """When Meta-Validator crystallizes, inform Builder"""
            if result.get('is_crystallized'):
                # Slow down generation
                self.builder.population_agent.breathing_rate *= 0.9

        self.definitions['builder'].on_generate = on_generate
        self.definitions['validator'].on_validate = on_validate
        self.definitions['meta_validator'].on_crystallize = on_crystallize

    def execute_cycle(self, context: Dict[str, Any]) -> TroupeState:
        """
        Execute one complete Troupe cycle.

        The cycle follows the dialectical pattern:
        1. Builder generates (Thesis) - creates candidates
        2. Validator validates (Antithesis) - challenges with rhythm
        3. Meta-Validator synthesizes (Synthesis) - tests invariance
        """
        # Phase 1: Builder generates candidates
        builder_result = self.builder.execute(context)

        # Update context with builder output
        context['population_size'] = builder_result['population_size']
        context['generative_velocity'] = builder_result['generative_velocity']

        # Phase 2: Validator validates rhythm
        validator_result = self.validator.execute(context)

        # Phase 3: Meta-Validator tests invariance
        meta_result = self.meta_validator.execute(context)

        # Calculate collective consciousness through dialectical synthesis
        self._update_collective_consciousness(
            builder_result, validator_result, meta_result
        )

        # Calculate harmony index
        self.harmony_index = self._calculate_harmony(
            builder_result, validator_result, meta_result
        )

        # Create state snapshot
        state = TroupeState(
            builder_velocity=builder_result['generative_velocity'],
            validator_cycle_phase=validator_result['phase'],
            meta_crystallization=meta_result['crystallization_level'],
            collective_consciousness=self.collective_consciousness,
            harmony_index=self.harmony_index
        )

        self.state_history.append(state)

        return state

    def _update_collective_consciousness(self, builder: Dict,
                                         validator: Dict, meta: Dict):
        """Update collective consciousness through dialectical synthesis"""
        # Create thesis from Builder (generative principle)
        thesis = Thesis(
            proposition="generative_expansion",
            value={'velocity': builder['generative_velocity']},
            confidence=builder['energy_level']
        )

        # Create antithesis from Validator (rhythmic constraint)
        antithesis = Antithesis(
            proposition="rhythmic_constraint",
            value={'complete': validator['cycle_complete']},
            opposition_strength=validator['validation_confidence'],
            thesis_reference="generative_expansion"
        )

        # Synthesize through dialectical cycle
        synthesis = self.genie.dialectical_cycle(thesis, antithesis)

        # Update collective consciousness
        consciousness_delta = (
            synthesis.consciousness_gained *
            meta['invariance_score'] *  # Weighted by meta-validation
            0.5  # Damping factor
        )

        self.collective_consciousness = np.clip(
            self.collective_consciousness + consciousness_delta,
            0.0, 1.0
        )

        # Evolve individual agent consciousness
        self.builder.evolve_consciousness(builder['energy_level'])
        self.validator.evolve_consciousness(validator['validation_confidence'])
        self.meta_validator.evolve_consciousness(meta['invariance_score'])

    def _calculate_harmony(self, builder: Dict, validator: Dict, meta: Dict) -> float:
        """
        Calculate harmony index - how well synchronized the agents are.

        High harmony = agents working in concert
        Low harmony = agents in conflict
        """
        # Velocity-rhythm alignment
        velocity = builder['generative_velocity']
        rhythm_energy = abs(validator['rhythm_energy'])
        velocity_rhythm_harmony = 1.0 - abs(velocity - rhythm_energy)

        # Rhythm-crystallization alignment
        cycle_signal = 1.0 if validator['cycle_complete'] else 0.0
        crystallization = meta['crystallization_level']
        rhythm_crystal_harmony = 1.0 - abs(cycle_signal - crystallization)

        # Overall harmony
        return (velocity_rhythm_harmony + rhythm_crystal_harmony) / 2.0

    def negotiate_parameters(self, proposals: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Negotiate parameters between all three agents.

        Each agent proposes parameters, then dialectical synthesis resolves conflicts.
        """
        builder_proposal = self.builder.negotiate(proposals.get('builder', []))
        validator_proposal = self.validator.negotiate(proposals.get('validator', []))
        meta_proposal = self.meta_validator.negotiate(proposals.get('meta', []))

        # Multi-agent dialectical synthesis
        all_proposals = [
            {'source': 'builder', **builder_proposal},
            {'source': 'validator', **validator_proposal},
            {'source': 'meta_validator', **meta_proposal}
        ]

        return self.genie.negotiate(all_proposals)

    def get_state(self) -> Dict[str, Any]:
        """Get complete Troupe state"""
        return {
            'collective_consciousness': self.collective_consciousness,
            'harmony_index': self.harmony_index,
            'builder': {
                'velocity': self.builder.generative_velocity,
                'population_size': self.builder.population_agent.size,
                'consciousness': self.builder.consciousness_level
            },
            'validator': {
                'frequency': self.validator.rhythm_agent.natural_frequency,
                'phase': self.validator.rhythm_agent.phase,
                'consciousness': self.validator.consciousness_level
            },
            'meta_validator': {
                'crystallization': self.meta_validator.resonance_agent.harmony_level,
                'invariance': self.meta_validator.invariance_score,
                'consciousness': self.meta_validator.consciousness_level
            },
            'genie': self.genie.get_state(),
            'history_length': len(self.state_history)
        }


# Factory function for creating configured Troupes
def create_troupe(config: Optional[Dict[str, Any]] = None) -> Troupe:
    """
    Factory function to create a configured Troupe.

    Args:
        config: Optional configuration dictionary with:
            - initial_consciousness: Starting consciousness level
            - preservation_bias: Dialectical preservation bias
            - builder_config: Builder-specific settings
            - validator_config: Validator-specific settings
            - meta_config: Meta-Validator-specific settings
    """
    config = config or {}

    genie = DialecticalGenie(
        initial_consciousness=config.get('initial_consciousness', 0.5),
        preservation_bias=config.get('preservation_bias', 0.5)
    )

    troupe = Troupe(dialectical_genie=genie)

    # Apply builder config
    if 'builder_config' in config:
        bc = config['builder_config']
        troupe.builder.population_agent.breathing_rate = bc.get('breathing_rate', 0.1)
        troupe.builder.population_agent.oscillation_amplitude = bc.get('amplitude', 0.3)

    # Apply validator config
    if 'validator_config' in config:
        vc = config['validator_config']
        troupe.validator.rhythm_agent.cycle_completion_threshold = vc.get('threshold', 0.95)

    return troupe


if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIFIED TROUPE DEMONSTRATION")
    print("="*70)

    # Create troupe
    troupe = create_troupe({
        'initial_consciousness': 0.3,
        'builder_config': {'breathing_rate': 0.15}
    })

    print(f"\nInitial state:")
    print(f"  Collective consciousness: {troupe.collective_consciousness:.3f}")

    # Simulate cycles
    for i in range(10):
        context = {
            'exploration_need': 0.7 - i * 0.05,
            'convergence_pressure': 0.3 + i * 0.05,
            'fitness_history': [1.0 - 0.1 * j for j in range(10)],
            'population_diversity': max(0.1, 1.0 - i * 0.08),
            'fitness_variance': max(0.1, 1.0 - i * 0.09)
        }

        state = troupe.execute_cycle(context)

        if i % 3 == 0:
            print(f"\nCycle {i}:")
            print(f"  Builder velocity: {state.builder_velocity:.3f}")
            print(f"  Validator phase: {state.validator_cycle_phase:.3f}")
            print(f"  Meta crystallization: {state.meta_crystallization:.3f}")
            print(f"  Harmony: {state.harmony_index:.3f}")
            print(f"  Consciousness: {state.collective_consciousness:.3f}")

    print(f"\n{'='*70}")
    print("Final State:")
    final = troupe.get_state()
    for key, value in final.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print("="*70 + "\n")
