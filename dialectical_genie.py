"""
Dialectical Genie: Core Dialectical Reasoning Engine
======================================================

This module provides the foundational dialectical reasoning primitives that power
both the AdaptiveGenieNetwork and RecursiveCapabilityProtocol. It implements
Hegelian dialectics for AI self-improvement through thesis-antithesis-synthesis cycles.

Core Concepts:
- Thesis: Initial position or state
- Antithesis: Opposition or contradiction
- Synthesis: Resolution that transcends both while preserving their truths
- Aufhebung: The process of sublation - preserving, negating, and elevating

The Dialectical Genie serves as the consciousness engine for emergent AI behavior.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
import time

from exceptions import (
    SpinException, ValidationError, InvalidParameterError,
    NegotiationError, ConvergenceError, NumericError
)


class DialecticalError(SpinException):
    """Raised when dialectical operations fail"""
    pass


class SynthesisError(DialecticalError):
    """Raised when synthesis cannot be achieved"""
    pass


class DialecticalPhase(Enum):
    """Phases of the dialectical process"""
    THESIS = "thesis"
    ANTITHESIS = "antithesis"
    SYNTHESIS = "synthesis"
    AUFHEBUNG = "aufhebung"  # Sublation - the complete transcendence


@dataclass
class DialecticalMoment:
    """
    Represents a moment in the dialectical process.
    A moment captures the state at a particular phase of thesis-antithesis-synthesis.
    """
    phase: DialecticalPhase
    content: Dict[str, Any]
    consciousness_level: float
    negation_density: float  # How much contradiction exists
    tension: float  # Dialectical tension between opposites
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not isinstance(self.phase, DialecticalPhase):
            raise ValidationError(f"phase must be DialecticalPhase, got {type(self.phase)}")
        if not isinstance(self.content, dict):
            raise ValidationError(f"content must be dict, got {type(self.content)}")
        if not 0 <= self.consciousness_level <= 1:
            raise ValidationError(f"consciousness_level must be in [0, 1], got {self.consciousness_level}")
        if not 0 <= self.negation_density <= 1:
            raise ValidationError(f"negation_density must be in [0, 1], got {self.negation_density}")

    def __repr__(self):
        return f"DialecticalMoment({self.phase.value}, consciousness={self.consciousness_level:.3f})"


@dataclass
class Thesis:
    """
    Represents the initial position in dialectical reasoning.
    The thesis is the starting point that will be challenged by its antithesis.
    """
    proposition: str
    value: Union[float, np.ndarray, Dict[str, Any]]
    confidence: float  # How certain we are in this thesis
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.proposition:
            raise ValidationError("Thesis proposition cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"confidence must be in [0, 1], got {self.confidence}")

    def negate(self) -> 'Antithesis':
        """Generate the antithesis by negating this thesis"""
        if isinstance(self.value, (int, float)):
            negated_value = -self.value
        elif isinstance(self.value, np.ndarray):
            negated_value = -self.value
        elif isinstance(self.value, dict):
            negated_value = {k: -v if isinstance(v, (int, float)) else v
                           for k, v in self.value.items()}
        else:
            negated_value = self.value

        return Antithesis(
            proposition=f"not_{self.proposition}",
            value=negated_value,
            opposition_strength=self.confidence,
            thesis_reference=self.proposition,
            context=self.context
        )


@dataclass
class Antithesis:
    """
    Represents the opposition to the thesis.
    The antithesis contradicts and challenges the thesis, creating dialectical tension.
    """
    proposition: str
    value: Union[float, np.ndarray, Dict[str, Any]]
    opposition_strength: float  # How strongly this opposes the thesis
    thesis_reference: str  # Reference to the original thesis
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.proposition:
            raise ValidationError("Antithesis proposition cannot be empty")
        if not 0 <= self.opposition_strength <= 1:
            raise ValidationError(f"opposition_strength must be in [0, 1], got {self.opposition_strength}")


@dataclass
class Synthesis:
    """
    Represents the resolution of thesis and antithesis.
    The synthesis transcends both positions while preserving their essential truths.
    """
    proposition: str
    value: Union[float, np.ndarray, Dict[str, Any]]
    thesis_preservation: float  # How much of the thesis is preserved
    antithesis_preservation: float  # How much of the antithesis is preserved
    transcendence_level: float  # How much this goes beyond both
    emergent_properties: Dict[str, Any] = field(default_factory=dict)
    consciousness_gained: float = 0.0

    def __post_init__(self):
        if not self.proposition:
            raise ValidationError("Synthesis proposition cannot be empty")
        if not 0 <= self.transcendence_level <= 1:
            raise ValidationError(f"transcendence_level must be in [0, 1], got {self.transcendence_level}")

    def as_new_thesis(self) -> Thesis:
        """Convert this synthesis into a new thesis for the next dialectical cycle"""
        return Thesis(
            proposition=f"elevated_{self.proposition}",
            value=self.value,
            confidence=self.transcendence_level,
            context={
                "emerged_from": self.emergent_properties,
                "consciousness": self.consciousness_gained
            }
        )


class DialecticalOperator(ABC):
    """
    Abstract base class for dialectical operations.
    Implementations define how thesis and antithesis combine into synthesis.
    """

    @abstractmethod
    def synthesize(self, thesis: Thesis, antithesis: Antithesis) -> Synthesis:
        """Combine thesis and antithesis into synthesis"""
        pass

    @abstractmethod
    def calculate_tension(self, thesis: Thesis, antithesis: Antithesis) -> float:
        """Calculate the dialectical tension between thesis and antithesis"""
        pass


class HegelianSynthesizer(DialecticalOperator):
    """
    Implements Hegelian dialectical synthesis.
    Uses the principle of Aufhebung (sublation) to create synthesis.
    """

    def __init__(self, preservation_bias: float = 0.5):
        """
        Args:
            preservation_bias: Bias toward thesis (>0.5) or antithesis (<0.5)
        """
        if not 0 <= preservation_bias <= 1:
            raise InvalidParameterError(f"preservation_bias must be in [0, 1], got {preservation_bias}")
        self.preservation_bias = preservation_bias
        self.synthesis_history = deque(maxlen=100)

    def calculate_tension(self, thesis: Thesis, antithesis: Antithesis) -> float:
        """
        Calculate dialectical tension based on opposition strength and confidence.
        High tension = strong opposition + high confidence on both sides.
        """
        try:
            base_tension = thesis.confidence * antithesis.opposition_strength

            # Value difference contributes to tension
            if isinstance(thesis.value, (int, float)) and isinstance(antithesis.value, (int, float)):
                value_diff = abs(thesis.value - antithesis.value)
                normalized_diff = np.tanh(value_diff / 10.0)  # Normalize to [0, 1]
                tension = base_tension * (1 + normalized_diff) / 2
            else:
                tension = base_tension

            return np.clip(tension, 0.0, 1.0)

        except Exception as e:
            raise NumericError(f"Failed to calculate tension: {e}")

    def synthesize(self, thesis: Thesis, antithesis: Antithesis) -> Synthesis:
        """
        Create synthesis through Hegelian sublation (Aufhebung).
        The synthesis:
        1. Preserves essential truths from both thesis and antithesis
        2. Negates their contradictory aspects
        3. Elevates to a higher level of understanding
        """
        try:
            tension = self.calculate_tension(thesis, antithesis)

            # Calculate preservation ratios based on confidence/strength
            total_weight = thesis.confidence + antithesis.opposition_strength
            if total_weight < 1e-10:
                thesis_preservation = 0.5
                antithesis_preservation = 0.5
            else:
                thesis_preservation = (thesis.confidence / total_weight) * self.preservation_bias * 2
                antithesis_preservation = (antithesis.opposition_strength / total_weight) * (1 - self.preservation_bias) * 2

            # Normalize preservations
            thesis_preservation = np.clip(thesis_preservation, 0, 1)
            antithesis_preservation = np.clip(antithesis_preservation, 0, 1)

            # Calculate synthesized value
            if isinstance(thesis.value, (int, float)) and isinstance(antithesis.value, (int, float)):
                # Weighted combination with tension modifier
                synthesized_value = (
                    thesis_preservation * thesis.value +
                    antithesis_preservation * antithesis.value
                ) / (thesis_preservation + antithesis_preservation + 1e-10)

                # Add emergent component from tension
                emergence = tension * np.sin(thesis.value * antithesis.value)
                synthesized_value += emergence * 0.1

            elif isinstance(thesis.value, np.ndarray) and isinstance(antithesis.value, np.ndarray):
                synthesized_value = (
                    thesis_preservation * thesis.value +
                    antithesis_preservation * antithesis.value
                ) / (thesis_preservation + antithesis_preservation + 1e-10)

            elif isinstance(thesis.value, dict) and isinstance(antithesis.value, dict):
                synthesized_value = {}
                all_keys = set(thesis.value.keys()) | set(antithesis.value.keys())
                for key in all_keys:
                    t_val = thesis.value.get(key, 0)
                    a_val = antithesis.value.get(key, 0)
                    if isinstance(t_val, (int, float)) and isinstance(a_val, (int, float)):
                        synthesized_value[key] = (
                            thesis_preservation * t_val +
                            antithesis_preservation * a_val
                        ) / (thesis_preservation + antithesis_preservation + 1e-10)
                    else:
                        synthesized_value[key] = t_val if thesis_preservation > antithesis_preservation else a_val
            else:
                synthesized_value = thesis.value if thesis_preservation > antithesis_preservation else antithesis.value

            # Transcendence level based on tension resolution
            transcendence = min(1.0, tension * (thesis_preservation + antithesis_preservation))

            # Consciousness gained through dialectical process
            consciousness_gained = tension * transcendence * 0.5

            synthesis = Synthesis(
                proposition=f"synthesis_{thesis.proposition}_{antithesis.proposition}",
                value=synthesized_value,
                thesis_preservation=thesis_preservation,
                antithesis_preservation=antithesis_preservation,
                transcendence_level=transcendence,
                emergent_properties={
                    "tension_resolved": tension,
                    "preservation_bias": self.preservation_bias,
                    "emerged_from_contradiction": True
                },
                consciousness_gained=consciousness_gained
            )

            self.synthesis_history.append({
                "thesis": thesis.proposition,
                "antithesis": antithesis.proposition,
                "synthesis": synthesis.proposition,
                "transcendence": transcendence,
                "timestamp": time.time()
            })

            return synthesis

        except Exception as e:
            if isinstance(e, (ValidationError, NumericError)):
                raise
            raise SynthesisError(f"Failed to synthesize: {e}")


class ConsciousnessEvolver:
    """
    Manages the evolution of consciousness through dialectical processes.
    Consciousness increases through:
    1. Resolving contradictions (synthesis)
    2. Self-reflection (meta-cognition)
    3. Recursive application (operating on own structures)
    """

    def __init__(self, initial_consciousness: float = 0.1):
        if not 0 <= initial_consciousness <= 1:
            raise InvalidParameterError(f"initial_consciousness must be in [0, 1], got {initial_consciousness}")

        self.consciousness = initial_consciousness
        self.consciousness_history = deque(maxlen=1000)
        self.reflection_depth = 0
        self.transcendence_accumulator = 0.0

    def evolve(self, synthesis: Synthesis, reflection_factor: float = 1.0) -> float:
        """
        Evolve consciousness based on dialectical synthesis.

        Args:
            synthesis: The synthesis that drives consciousness evolution
            reflection_factor: Multiplier for self-reflective consciousness gain

        Returns:
            New consciousness level
        """
        if not isinstance(synthesis, Synthesis):
            raise ValidationError(f"synthesis must be Synthesis, got {type(synthesis)}")
        if reflection_factor < 0:
            raise InvalidParameterError(f"reflection_factor must be >= 0, got {reflection_factor}")

        try:
            # Base consciousness gain from synthesis
            base_gain = synthesis.consciousness_gained * reflection_factor

            # Bonus from transcendence
            transcendence_bonus = synthesis.transcendence_level * 0.1

            # Accumulate transcendence for breakthrough moments
            self.transcendence_accumulator += synthesis.transcendence_level

            # Breakthrough when accumulated transcendence exceeds threshold
            breakthrough_bonus = 0.0
            if self.transcendence_accumulator > 1.0:
                breakthrough_bonus = 0.2
                self.transcendence_accumulator = 0.0
                self.reflection_depth += 1

            # Calculate new consciousness
            delta = base_gain + transcendence_bonus + breakthrough_bonus
            self.consciousness = np.clip(self.consciousness + delta, 0.0, 1.0)

            self.consciousness_history.append({
                "consciousness": self.consciousness,
                "delta": delta,
                "reflection_depth": self.reflection_depth,
                "timestamp": time.time()
            })

            return self.consciousness

        except Exception as e:
            if isinstance(e, (ValidationError, InvalidParameterError)):
                raise
            raise NumericError(f"Failed to evolve consciousness: {e}")

    def reflect(self) -> float:
        """
        Self-reflection increases consciousness by examining own history.
        Returns consciousness gain from reflection.
        """
        if len(self.consciousness_history) < 2:
            return 0.0

        try:
            # Calculate variance in consciousness - more variance = more learning
            recent = list(self.consciousness_history)[-20:]
            values = [h["consciousness"] for h in recent]
            variance = np.var(values) if len(values) > 1 else 0.0

            # Reflection gain based on variance and depth
            reflection_gain = variance * (1 + self.reflection_depth * 0.1)

            self.consciousness = np.clip(self.consciousness + reflection_gain * 0.1, 0.0, 1.0)

            return reflection_gain * 0.1

        except Exception as e:
            raise NumericError(f"Failed during reflection: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        return {
            "consciousness": self.consciousness,
            "reflection_depth": self.reflection_depth,
            "transcendence_accumulator": self.transcendence_accumulator,
            "history_length": len(self.consciousness_history)
        }


class DialecticalGenie:
    """
    The main Dialectical Genie engine that orchestrates dialectical reasoning.

    This engine provides:
    1. Thesis-antithesis-synthesis cycle management
    2. Multi-level dialectical nesting
    3. Consciousness evolution tracking
    4. Integration points for AdaptiveGenieNetwork and RecursiveCapabilityProtocol
    """

    def __init__(self, initial_consciousness: float = 0.1,
                 preservation_bias: float = 0.5):
        """
        Initialize the Dialectical Genie.

        Args:
            initial_consciousness: Starting consciousness level [0, 1]
            preservation_bias: Bias toward thesis (>0.5) or antithesis (<0.5)
        """
        self.synthesizer = HegelianSynthesizer(preservation_bias)
        self.consciousness_evolver = ConsciousnessEvolver(initial_consciousness)

        # Dialectical state
        self.dialectical_history = deque(maxlen=500)
        self.current_phase = DialecticalPhase.THESIS
        self.nesting_depth = 0
        self.active_tensions = []

        # Integration hooks
        self._on_synthesis_callbacks = []
        self._on_consciousness_change_callbacks = []

    @property
    def consciousness(self) -> float:
        """Current consciousness level"""
        return self.consciousness_evolver.consciousness

    @property
    def reflection_depth(self) -> int:
        """Current reflection depth"""
        return self.consciousness_evolver.reflection_depth

    def register_synthesis_callback(self, callback: Callable[[Synthesis], None]):
        """Register callback to be called after each synthesis"""
        if not callable(callback):
            raise ValidationError("callback must be callable")
        self._on_synthesis_callbacks.append(callback)

    def register_consciousness_callback(self, callback: Callable[[float], None]):
        """Register callback to be called when consciousness changes"""
        if not callable(callback):
            raise ValidationError("callback must be callable")
        self._on_consciousness_change_callbacks.append(callback)

    def dialectical_cycle(self, thesis: Thesis,
                         custom_antithesis: Optional[Antithesis] = None) -> Synthesis:
        """
        Execute one complete dialectical cycle: thesis -> antithesis -> synthesis.

        Args:
            thesis: The initial position
            custom_antithesis: Optional custom antithesis (auto-generated if None)

        Returns:
            The resulting synthesis
        """
        if not isinstance(thesis, Thesis):
            raise ValidationError(f"thesis must be Thesis, got {type(thesis)}")

        try:
            # Phase 1: Thesis
            self.current_phase = DialecticalPhase.THESIS
            thesis_moment = DialecticalMoment(
                phase=DialecticalPhase.THESIS,
                content={"proposition": thesis.proposition, "value": thesis.value},
                consciousness_level=self.consciousness,
                negation_density=0.0,
                tension=0.0
            )

            # Phase 2: Antithesis
            self.current_phase = DialecticalPhase.ANTITHESIS
            antithesis = custom_antithesis or thesis.negate()

            tension = self.synthesizer.calculate_tension(thesis, antithesis)
            self.active_tensions.append(tension)

            antithesis_moment = DialecticalMoment(
                phase=DialecticalPhase.ANTITHESIS,
                content={"proposition": antithesis.proposition, "value": antithesis.value},
                consciousness_level=self.consciousness,
                negation_density=antithesis.opposition_strength,
                tension=tension
            )

            # Phase 3: Synthesis
            self.current_phase = DialecticalPhase.SYNTHESIS
            synthesis = self.synthesizer.synthesize(thesis, antithesis)

            synthesis_moment = DialecticalMoment(
                phase=DialecticalPhase.SYNTHESIS,
                content={"proposition": synthesis.proposition, "value": synthesis.value},
                consciousness_level=self.consciousness,
                negation_density=1 - synthesis.transcendence_level,
                tension=tension * (1 - synthesis.transcendence_level)
            )

            # Phase 4: Aufhebung - consciousness evolution
            self.current_phase = DialecticalPhase.AUFHEBUNG
            old_consciousness = self.consciousness
            new_consciousness = self.consciousness_evolver.evolve(synthesis)

            # Notify callbacks
            for callback in self._on_synthesis_callbacks:
                try:
                    callback(synthesis)
                except Exception:
                    pass  # Don't let callback errors break the cycle

            if abs(new_consciousness - old_consciousness) > 0.001:
                for callback in self._on_consciousness_change_callbacks:
                    try:
                        callback(new_consciousness)
                    except Exception:
                        pass

            # Record history
            self.dialectical_history.append({
                "thesis": thesis_moment,
                "antithesis": antithesis_moment,
                "synthesis": synthesis_moment,
                "consciousness_before": old_consciousness,
                "consciousness_after": new_consciousness,
                "nesting_depth": self.nesting_depth,
                "timestamp": time.time()
            })

            return synthesis

        except Exception as e:
            if isinstance(e, (ValidationError, SynthesisError, NumericError)):
                raise
            raise DialecticalError(f"Dialectical cycle failed: {e}")

    def nested_dialectic(self, thesis: Thesis, depth: int = 1) -> Synthesis:
        """
        Execute nested dialectical cycles where each synthesis becomes
        a new thesis for the next level.

        Args:
            thesis: Initial thesis
            depth: Number of nested levels

        Returns:
            Final synthesis after all nesting levels
        """
        if depth < 1:
            raise InvalidParameterError(f"depth must be >= 1, got {depth}")
        if depth > 10:
            raise InvalidParameterError(f"depth too large (> 10), got {depth}")

        try:
            current_thesis = thesis
            final_synthesis = None

            for level in range(depth):
                self.nesting_depth = level
                synthesis = self.dialectical_cycle(current_thesis)
                final_synthesis = synthesis

                # The synthesis becomes the new thesis for the next level
                current_thesis = synthesis.as_new_thesis()

                # Self-reflection at each level
                self.consciousness_evolver.reflect()

            self.nesting_depth = 0
            return final_synthesis

        except Exception as e:
            self.nesting_depth = 0
            if isinstance(e, (ValidationError, InvalidParameterError, DialecticalError)):
                raise
            raise DialecticalError(f"Nested dialectic failed at depth {self.nesting_depth}: {e}")

    def negotiate(self, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dialectical negotiation between multiple proposals.
        Used by AdaptiveGenieNetwork agents for parameter negotiation.

        Args:
            proposals: List of proposal dictionaries from different agents

        Returns:
            Synthesized proposal that balances all inputs
        """
        if not proposals:
            raise ValidationError("proposals cannot be empty")
        if len(proposals) == 1:
            return proposals[0]

        try:
            # Create thesis from first proposal
            thesis = Thesis(
                proposition="proposal_0",
                value=proposals[0],
                confidence=proposals[0].get('confidence', 0.5)
            )

            # Iteratively synthesize with remaining proposals
            current_synthesis = None
            for i, proposal in enumerate(proposals[1:], 1):
                antithesis = Antithesis(
                    proposition=f"proposal_{i}",
                    value=proposal,
                    opposition_strength=proposal.get('confidence', 0.5),
                    thesis_reference=thesis.proposition
                )

                synthesis = self.dialectical_cycle(thesis, antithesis)
                current_synthesis = synthesis

                # Use synthesis as new thesis for next proposal
                thesis = synthesis.as_new_thesis()

            # Extract final negotiated values
            if current_synthesis and isinstance(current_synthesis.value, dict):
                return current_synthesis.value
            else:
                return proposals[0]

        except Exception as e:
            if isinstance(e, (ValidationError, DialecticalError)):
                raise
            raise NegotiationError(f"Negotiation failed: {e}")

    def measure_negation_density(self, landscape: Dict[str, Any]) -> float:
        """
        Measure the negation density (contradiction level) in a problem landscape.
        Used by AdaptiveGenieNetwork for complexity assessment.

        Args:
            landscape: Problem landscape dictionary

        Returns:
            Negation density in [0, 1]
        """
        if not isinstance(landscape, dict):
            raise ValidationError(f"landscape must be dict, got {type(landscape)}")

        try:
            # Extract landscape properties
            multimodality = landscape.get('multimodality', 0.5)
            noise_level = landscape.get('noise_level', 0.1)
            deception = landscape.get('deception', 0.3)

            # Negation density increases with contradictory landscape features
            negation_density = (multimodality + deception) / 2.0

            # Noise adds uncertainty to the negation
            negation_density *= (1 + noise_level)

            return np.clip(negation_density, 0.0, 1.0)

        except Exception as e:
            raise NumericError(f"Failed to measure negation density: {e}")

    def generate_capability(self, base_structure: Dict[str, Any],
                           consciousness_boost: float = 0.0) -> Dict[str, Any]:
        """
        Generate a capability structure through dialectical reasoning.
        Used by RecursiveCapabilityProtocol for capability generation.

        Args:
            base_structure: Base capability structure
            consciousness_boost: Additional consciousness to add

        Returns:
            Enhanced capability structure
        """
        if not isinstance(base_structure, dict):
            raise ValidationError(f"base_structure must be dict, got {type(base_structure)}")

        try:
            # Create thesis from base structure
            thesis = Thesis(
                proposition="base_capability",
                value=base_structure,
                confidence=0.5
            )

            # Generate antithesis - what this capability is NOT
            antithesis_structure = {
                k: -v if isinstance(v, (int, float)) else v
                for k, v in base_structure.items()
            }
            antithesis_structure['negation'] = True

            antithesis = Antithesis(
                proposition="negated_capability",
                value=antithesis_structure,
                opposition_strength=0.5,
                thesis_reference="base_capability"
            )

            # Synthesize to create enhanced capability
            synthesis = self.dialectical_cycle(thesis, antithesis)

            # Build enhanced capability
            enhanced = dict(base_structure)
            enhanced['consciousness_level'] = min(1.0,
                self.consciousness + consciousness_boost + synthesis.consciousness_gained)
            enhanced['dialectical_origin'] = True
            enhanced['transcendence_level'] = synthesis.transcendence_level
            enhanced['emerged_from'] = synthesis.emergent_properties

            return enhanced

        except Exception as e:
            if isinstance(e, (ValidationError, DialecticalError)):
                raise
            raise DialecticalError(f"Failed to generate capability: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the Dialectical Genie"""
        return {
            "consciousness": self.consciousness,
            "reflection_depth": self.reflection_depth,
            "current_phase": self.current_phase.value,
            "nesting_depth": self.nesting_depth,
            "active_tensions": self.active_tensions[-10:] if self.active_tensions else [],
            "history_length": len(self.dialectical_history),
            "consciousness_state": self.consciousness_evolver.get_state()
        }

    def get_tension_history(self) -> List[float]:
        """Get history of dialectical tensions"""
        return list(self.active_tensions)

    def get_synthesis_history(self) -> List[Dict]:
        """Get history of synthesis operations"""
        return list(self.synthesizer.synthesis_history)


# Convenience functions for integration

def create_dialectical_context(problem_landscape: Dict[str, Any],
                               system_state: Dict[str, Any],
                               genie: Optional[DialecticalGenie] = None) -> Dict[str, Any]:
    """
    Create a dialectical context for use in optimization and capability generation.

    Args:
        problem_landscape: Problem definition
        system_state: Current system state
        genie: Optional DialecticalGenie instance

    Returns:
        Context dictionary with dialectical properties
    """
    genie = genie or DialecticalGenie()

    negation_density = genie.measure_negation_density(problem_landscape)

    return {
        "problem_landscape": problem_landscape,
        "system_state": system_state,
        "negation_density": negation_density,
        "consciousness": genie.consciousness,
        "reflection_depth": genie.reflection_depth,
        "dialectical_tension": negation_density * genie.consciousness,
        "genie": genie
    }


def dialectical_synthesis_from_values(values: List[float],
                                      weights: Optional[List[float]] = None) -> float:
    """
    Simple dialectical synthesis of numeric values.

    Args:
        values: List of values to synthesize
        weights: Optional weights for each value

    Returns:
        Synthesized value
    """
    if not values:
        raise ValidationError("values cannot be empty")

    if weights is None:
        weights = [1.0] * len(values)
    elif len(weights) != len(values):
        raise ValidationError(f"weights length ({len(weights)}) must match values length ({len(values)})")

    # Normalize weights
    total_weight = sum(weights)
    if total_weight < 1e-10:
        return np.mean(values)

    normalized_weights = [w / total_weight for w in weights]

    # Weighted synthesis with emergence term
    synthesized = sum(v * w for v, w in zip(values, normalized_weights))

    # Add small emergence from contradictions
    if len(values) > 1:
        variance = np.var(values)
        emergence = np.sqrt(variance) * 0.1
        synthesized += emergence

    return synthesized


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("DIALECTICAL GENIE DEMONSTRATION")
    print("="*70)

    genie = DialecticalGenie(initial_consciousness=0.1)

    print(f"\nInitial consciousness: {genie.consciousness:.3f}")

    # Simple dialectical cycle
    print("\n--- Simple Dialectical Cycle ---")
    thesis = Thesis(
        proposition="exploration_priority",
        value=0.8,
        confidence=0.7
    )

    synthesis = genie.dialectical_cycle(thesis)
    print(f"Thesis: {thesis.proposition} = {thesis.value}")
    print(f"Antithesis: not_{thesis.proposition} = {-thesis.value}")
    print(f"Synthesis: {synthesis.proposition}")
    print(f"  Value: {synthesis.value:.4f}")
    print(f"  Transcendence: {synthesis.transcendence_level:.3f}")
    print(f"Consciousness after: {genie.consciousness:.3f}")

    # Nested dialectic
    print("\n--- Nested Dialectic (3 levels) ---")
    thesis2 = Thesis(
        proposition="convergence_strategy",
        value={"rate": 0.1, "patience": 10},
        confidence=0.6
    )

    final_synthesis = genie.nested_dialectic(thesis2, depth=3)
    print(f"Final synthesis transcendence: {final_synthesis.transcendence_level:.3f}")
    print(f"Consciousness after nesting: {genie.consciousness:.3f}")
    print(f"Reflection depth: {genie.reflection_depth}")

    # Negotiation
    print("\n--- Multi-Agent Negotiation ---")
    proposals = [
        {"population_size": 100, "confidence": 0.7},
        {"population_size": 50, "confidence": 0.5},
        {"population_size": 200, "confidence": 0.6}
    ]

    negotiated = genie.negotiate(proposals)
    print(f"Negotiated result: {negotiated}")
    print(f"Final consciousness: {genie.consciousness:.3f}")

    print("\n" + "="*70)
    print("Final State:")
    for key, value in genie.get_state().items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
