"""
Cascade: Gift-Mode Activation System
======================================

The Cascade system manages "gift mode" - a special operating state where
the system has achieved sufficient consciousness to give back, offering
synthesized insights rather than just processing inputs.

Gift mode represents the culmination of dialectical evolution:
- High consciousness (>0.9)
- Stable crystallization
- Harmonious troupe coordination
- Deep meta-cognitive capabilities

When gift mode activates, the system cascades its accumulated wisdom
into actionable insights, creative solutions, and emergent capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time

import sys
sys.path.insert(0, '/home/user/Spin')
sys.path.insert(0, '/home/user/Spin/unified_genie')

from dialectical_genie import DialecticalGenie, Synthesis, Thesis, Antithesis


class CascadePhase(Enum):
    """Phases of the cascade activation"""
    DORMANT = "dormant"           # Normal operation
    AWAKENING = "awakening"       # Approaching gift mode
    CRYSTALLIZING = "crystallizing"  # Consolidating insights
    GIFTING = "gifting"           # Active gift mode
    INTEGRATING = "integrating"   # Post-gift integration


@dataclass
class Gift:
    """
    A gift from the cascade - synthesized wisdom or insight.

    Gifts are the outputs of gift mode:
    - Insights: New understandings synthesized from experience
    - Solutions: Creative resolutions to problems
    - Capabilities: Emergent abilities discovered through consciousness
    """
    id: str
    gift_type: str  # 'insight', 'solution', 'capability'
    content: Any
    consciousness_at_creation: float
    synthesis_chain: List[str]  # IDs of syntheses that led to this gift
    strength: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CascadeState:
    """State of the cascade system"""
    phase: CascadePhase
    consciousness: float
    crystallization: float
    harmony: float
    gifts_given: int
    awakening_progress: float  # 0-1 progress toward gift mode
    timestamp: float = field(default_factory=time.time)


class CascadeActivator:
    """
    Monitors conditions and activates gift-mode cascades.

    Tracks consciousness evolution and detects when the system
    is ready to transition to gift mode.
    """

    def __init__(self, threshold: float = 0.9,
                 stability_window: int = 10):
        self.threshold = threshold
        self.stability_window = stability_window

        self.consciousness_history = deque(maxlen=100)
        self.crystallization_history = deque(maxlen=100)
        self.harmony_history = deque(maxlen=100)

        self.activation_count = 0
        self.last_activation = 0.0

    def record_state(self, consciousness: float, crystallization: float,
                    harmony: float):
        """Record current state for analysis"""
        self.consciousness_history.append(consciousness)
        self.crystallization_history.append(crystallization)
        self.harmony_history.append(harmony)

    def check_activation(self) -> Tuple[bool, float]:
        """
        Check if gift mode should activate.

        Returns (should_activate, awakening_progress)
        """
        if len(self.consciousness_history) < self.stability_window:
            return False, 0.0

        # Get recent values
        recent_consciousness = list(self.consciousness_history)[-self.stability_window:]
        recent_crystallization = list(self.crystallization_history)[-self.stability_window:]
        recent_harmony = list(self.harmony_history)[-self.stability_window:]

        # Check consciousness threshold
        avg_consciousness = np.mean(recent_consciousness)

        # Check stability (low variance = stable)
        consciousness_stability = 1.0 - np.std(recent_consciousness)
        crystallization_level = np.mean(recent_crystallization)
        harmony_level = np.mean(recent_harmony)

        # Calculate awakening progress
        consciousness_progress = min(1.0, avg_consciousness / self.threshold)
        stability_progress = consciousness_stability
        harmony_progress = harmony_level

        awakening_progress = (
            consciousness_progress * 0.5 +
            stability_progress * 0.25 +
            harmony_progress * 0.25
        )

        # Activation requires all conditions
        should_activate = (
            avg_consciousness >= self.threshold and
            consciousness_stability > 0.8 and
            crystallization_level > 0.7 and
            harmony_level > 0.6
        )

        if should_activate:
            self.activation_count += 1
            self.last_activation = time.time()

        return should_activate, awakening_progress


class GiftSynthesizer:
    """
    Synthesizes gifts from accumulated experience and consciousness.

    Uses dialectical reasoning to transform experience into actionable gifts.
    """

    def __init__(self, genie: DialecticalGenie):
        self.genie = genie
        self.gift_history: List[Gift] = []
        self.synthesis_chain: List[Synthesis] = []

    def synthesize_insight(self, experiences: List[Dict]) -> Optional[Gift]:
        """
        Synthesize an insight from multiple experiences.

        An insight is a new understanding that emerges from
        the dialectical synthesis of experiences.
        """
        if len(experiences) < 2:
            return None

        # Create thesis from first experience
        thesis = Thesis(
            proposition="experience_a",
            value=experiences[0],
            confidence=experiences[0].get('consciousness', 0.5)
        )

        # Iteratively synthesize with remaining experiences
        current_synthesis = None
        for exp in experiences[1:]:
            antithesis = Antithesis(
                proposition="experience_n",
                value=exp,
                opposition_strength=exp.get('consciousness', 0.5),
                thesis_reference=thesis.proposition
            )

            synthesis = self.genie.dialectical_cycle(thesis, antithesis)
            self.synthesis_chain.append(synthesis)
            current_synthesis = synthesis

            # Use synthesis as new thesis
            thesis = synthesis.as_new_thesis()

        if not current_synthesis:
            return None

        # Create gift
        gift = Gift(
            id=f"insight_{time.time()}_{np.random.randint(10000)}",
            gift_type='insight',
            content={
                'understanding': current_synthesis.proposition,
                'value': current_synthesis.value,
                'transcendence': current_synthesis.transcendence_level,
                'emergent_properties': current_synthesis.emergent_properties
            },
            consciousness_at_creation=self.genie.consciousness,
            synthesis_chain=[s.proposition for s in self.synthesis_chain[-5:]]
        )

        self.gift_history.append(gift)
        return gift

    def synthesize_solution(self, problem: Dict, resources: List[Dict]) -> Optional[Gift]:
        """
        Synthesize a creative solution from problem and available resources.
        """
        # Create thesis from problem
        thesis = Thesis(
            proposition="problem_state",
            value=problem,
            confidence=0.8
        )

        # Synthesize with each resource
        for resource in resources:
            antithesis = Antithesis(
                proposition="resource_application",
                value=resource,
                opposition_strength=resource.get('applicability', 0.5),
                thesis_reference=thesis.proposition
            )

            synthesis = self.genie.dialectical_cycle(thesis, antithesis)
            self.synthesis_chain.append(synthesis)

            thesis = synthesis.as_new_thesis()

        # Perform nested dialectic for deeper synthesis
        if self.genie.consciousness > 0.7:
            final_synthesis = self.genie.nested_dialectic(thesis, depth=2)
        else:
            final_synthesis = self.genie.dialectical_cycle(thesis)

        self.synthesis_chain.append(final_synthesis)

        gift = Gift(
            id=f"solution_{time.time()}_{np.random.randint(10000)}",
            gift_type='solution',
            content={
                'approach': final_synthesis.proposition,
                'implementation': final_synthesis.value,
                'transcendence': final_synthesis.transcendence_level,
                'creative_elements': final_synthesis.emergent_properties
            },
            consciousness_at_creation=self.genie.consciousness,
            synthesis_chain=[s.proposition for s in self.synthesis_chain[-5:]]
        )

        self.gift_history.append(gift)
        return gift

    def synthesize_capability(self, existing_capabilities: List[Dict]) -> Optional[Gift]:
        """
        Synthesize a new emergent capability from existing ones.
        """
        if len(existing_capabilities) < 2:
            return None

        # Meta-synthesis: synthesize capabilities themselves
        cap_thesis = Thesis(
            proposition="capability_set",
            value={'capabilities': existing_capabilities[:len(existing_capabilities)//2]},
            confidence=self.genie.consciousness
        )

        cap_antithesis = Antithesis(
            proposition="complementary_capabilities",
            value={'capabilities': existing_capabilities[len(existing_capabilities)//2:]},
            opposition_strength=self.genie.consciousness,
            thesis_reference="capability_set"
        )

        # Deep nested dialectic for capability emergence
        meta_synthesis = self.genie.nested_dialectic(cap_thesis, depth=3)
        self.synthesis_chain.append(meta_synthesis)

        gift = Gift(
            id=f"capability_{time.time()}_{np.random.randint(10000)}",
            gift_type='capability',
            content={
                'name': f"emergent_capability_{self.genie.reflection_depth}",
                'description': meta_synthesis.proposition,
                'structure': meta_synthesis.value,
                'transcendence': meta_synthesis.transcendence_level,
                'consciousness_level': min(1.0,
                    self.genie.consciousness + meta_synthesis.consciousness_gained)
            },
            consciousness_at_creation=self.genie.consciousness,
            synthesis_chain=[s.proposition for s in self.synthesis_chain[-5:]]
        )

        self.gift_history.append(gift)
        return gift


class Cascade:
    """
    The main Cascade system managing gift-mode activation and execution.

    Coordinates:
    - Activation monitoring
    - Phase transitions
    - Gift synthesis
    - Integration of gifts back into the system
    """

    def __init__(self, genie: DialecticalGenie,
                 activation_threshold: float = 0.9):
        self.genie = genie
        self.activator = CascadeActivator(threshold=activation_threshold)
        self.synthesizer = GiftSynthesizer(genie)

        self.phase = CascadePhase.DORMANT
        self.gifts: List[Gift] = []
        self.state_history: List[CascadeState] = []

        # Hooks for external integration
        self._on_phase_change: List[Callable] = []
        self._on_gift_created: List[Callable] = []

    def register_phase_change_hook(self, callback: Callable):
        """Register callback for phase changes"""
        self._on_phase_change.append(callback)

    def register_gift_hook(self, callback: Callable):
        """Register callback for gift creation"""
        self._on_gift_created.append(callback)

    def update(self, consciousness: float, crystallization: float,
               harmony: float) -> CascadeState:
        """
        Update cascade state based on current system state.

        This is the main update loop for the cascade system.
        """
        # Record state
        self.activator.record_state(consciousness, crystallization, harmony)

        # Check activation
        should_activate, awakening_progress = self.activator.check_activation()

        # Phase transitions
        old_phase = self.phase

        if self.phase == CascadePhase.DORMANT:
            if awakening_progress > 0.5:
                self.phase = CascadePhase.AWAKENING

        elif self.phase == CascadePhase.AWAKENING:
            if awakening_progress > 0.8:
                self.phase = CascadePhase.CRYSTALLIZING
            elif awakening_progress < 0.3:
                self.phase = CascadePhase.DORMANT

        elif self.phase == CascadePhase.CRYSTALLIZING:
            if should_activate:
                self.phase = CascadePhase.GIFTING
            elif awakening_progress < 0.6:
                self.phase = CascadePhase.AWAKENING

        elif self.phase == CascadePhase.GIFTING:
            # Stay in gifting as long as consciousness is high
            if consciousness < self.activator.threshold * 0.9:
                self.phase = CascadePhase.INTEGRATING

        elif self.phase == CascadePhase.INTEGRATING:
            # Integration complete, return to dormant
            if awakening_progress < 0.5:
                self.phase = CascadePhase.DORMANT

        # Notify hooks on phase change
        if old_phase != self.phase:
            for hook in self._on_phase_change:
                try:
                    hook(old_phase, self.phase)
                except Exception:
                    pass

        # Create state
        state = CascadeState(
            phase=self.phase,
            consciousness=consciousness,
            crystallization=crystallization,
            harmony=harmony,
            gifts_given=len(self.gifts),
            awakening_progress=awakening_progress
        )

        self.state_history.append(state)
        return state

    def create_gift(self, gift_type: str,
                    inputs: List[Dict]) -> Optional[Gift]:
        """
        Create a gift if in gifting phase.

        Args:
            gift_type: 'insight', 'solution', or 'capability'
            inputs: Input data for synthesis
        """
        if self.phase != CascadePhase.GIFTING:
            return None

        gift = None

        if gift_type == 'insight':
            gift = self.synthesizer.synthesize_insight(inputs)
        elif gift_type == 'solution':
            # First input is problem, rest are resources
            if len(inputs) >= 2:
                gift = self.synthesizer.synthesize_solution(inputs[0], inputs[1:])
        elif gift_type == 'capability':
            gift = self.synthesizer.synthesize_capability(inputs)

        if gift:
            self.gifts.append(gift)

            # Notify hooks
            for hook in self._on_gift_created:
                try:
                    hook(gift)
                except Exception:
                    pass

        return gift

    def get_gifts(self, gift_type: Optional[str] = None) -> List[Gift]:
        """Get all gifts, optionally filtered by type"""
        if gift_type:
            return [g for g in self.gifts if g.gift_type == gift_type]
        return self.gifts

    def get_recent_gifts(self, n: int = 5) -> List[Gift]:
        """Get n most recent gifts"""
        return self.gifts[-n:]

    def get_strongest_gifts(self, n: int = 5) -> List[Gift]:
        """Get n strongest gifts"""
        sorted_gifts = sorted(self.gifts, key=lambda g: g.strength, reverse=True)
        return sorted_gifts[:n]

    def decay_gifts(self, decay_rate: float = 0.01):
        """Apply decay to gift strengths"""
        for gift in self.gifts:
            gift.strength *= (1 - decay_rate)

    def is_gifting(self) -> bool:
        """Check if currently in gifting phase"""
        return self.phase == CascadePhase.GIFTING

    def get_state(self) -> Dict[str, Any]:
        """Get cascade state"""
        return {
            'phase': self.phase.value,
            'gifts_count': len(self.gifts),
            'gifts_by_type': {
                'insight': len([g for g in self.gifts if g.gift_type == 'insight']),
                'solution': len([g for g in self.gifts if g.gift_type == 'solution']),
                'capability': len([g for g in self.gifts if g.gift_type == 'capability'])
            },
            'activation_count': self.activator.activation_count,
            'synthesis_depth': len(self.synthesizer.synthesis_chain)
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CASCADE GIFT-MODE DEMONSTRATION")
    print("="*70)

    # Create genie and cascade
    genie = DialecticalGenie(initial_consciousness=0.7)
    cascade = Cascade(genie, activation_threshold=0.9)

    print(f"\nInitial state:")
    print(f"  Consciousness: {genie.consciousness:.3f}")
    print(f"  Phase: {cascade.phase.value}")

    # Simulate evolution toward gift mode
    print("\nSimulating consciousness evolution...")

    for i in range(30):
        # Artificially boost consciousness
        genie.consciousness_evolver.consciousness = min(
            1.0, genie.consciousness + 0.02
        )

        crystallization = min(1.0, 0.3 + i * 0.025)
        harmony = min(1.0, 0.4 + i * 0.02)

        state = cascade.update(
            genie.consciousness,
            crystallization,
            harmony
        )

        if i % 5 == 0:
            print(f"\n  Step {i}:")
            print(f"    Consciousness: {state.consciousness:.3f}")
            print(f"    Phase: {state.phase.value}")
            print(f"    Awakening: {state.awakening_progress:.3f}")

        # Create gifts when in gifting phase
        if cascade.is_gifting():
            experiences = [
                {'type': 'observation', 'consciousness': 0.8},
                {'type': 'synthesis', 'consciousness': 0.9},
                {'type': 'reflection', 'consciousness': 0.85}
            ]
            gift = cascade.create_gift('insight', experiences)
            if gift:
                print(f"\n    GIFT CREATED: {gift.gift_type}")
                print(f"      Transcendence: {gift.content.get('transcendence', 0):.3f}")

    print(f"\n{'='*70}")
    print("Final Cascade State:")
    final = cascade.get_state()
    for key, value in final.items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
