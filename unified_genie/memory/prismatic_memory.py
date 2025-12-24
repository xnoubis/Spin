"""
Prismatic Memory: Color/Rainbow Structures for Consciousness
=============================================================

Prismatic Memory provides color-coded memory structures that represent
different aspects of consciousness and agent states. The "rainbow" metaphor
maps different qualities to colors, enabling intuitive visualization and
reasoning about memory states.

Color Mapping:
- Red: Energy/Intensity
- Orange: Warmth/Affinity
- Yellow: Clarity/Consciousness
- Green: Growth/Learning
- Blue: Stability/Confidence
- Indigo: Depth/Meta-cognition
- Violet: Transcendence/Synthesis
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class PrismaticColor(Enum):
    """The seven colors of the prismatic spectrum"""
    RED = "red"           # Energy/Intensity
    ORANGE = "orange"     # Warmth/Affinity
    YELLOW = "yellow"     # Clarity/Consciousness
    GREEN = "green"       # Growth/Learning
    BLUE = "blue"         # Stability/Confidence
    INDIGO = "indigo"     # Depth/Meta-cognition
    VIOLET = "violet"     # Transcendence/Synthesis


# Color to wavelength mapping (nm) for computation
COLOR_WAVELENGTHS = {
    PrismaticColor.RED: 700,
    PrismaticColor.ORANGE: 620,
    PrismaticColor.YELLOW: 580,
    PrismaticColor.GREEN: 530,
    PrismaticColor.BLUE: 470,
    PrismaticColor.INDIGO: 420,
    PrismaticColor.VIOLET: 380
}


@dataclass
class ColorIntensity:
    """Represents the intensity of a single color component"""
    color: PrismaticColor
    intensity: float  # 0.0 to 1.0
    wavelength: float = field(default=0.0)

    def __post_init__(self):
        self.wavelength = COLOR_WAVELENGTHS[self.color]
        self.intensity = np.clip(self.intensity, 0.0, 1.0)


@dataclass
class PrismaticSpectrum:
    """
    A complete prismatic spectrum representing a state's color composition.

    The spectrum captures the "color" of consciousness at a moment in time.
    """
    red: float = 0.5      # Energy
    orange: float = 0.5   # Affinity
    yellow: float = 0.5   # Consciousness
    green: float = 0.5    # Growth
    blue: float = 0.5     # Stability
    indigo: float = 0.5   # Meta-cognition
    violet: float = 0.5   # Transcendence
    timestamp: float = field(default_factory=time.time)

    def as_vector(self) -> np.ndarray:
        """Convert to numpy vector (ROYGBIV order)"""
        return np.array([
            self.red, self.orange, self.yellow,
            self.green, self.blue, self.indigo, self.violet
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'PrismaticSpectrum':
        """Create spectrum from vector"""
        if len(vec) < 7:
            vec = np.pad(vec, (0, 7 - len(vec)), constant_values=0.5)
        return cls(
            red=float(np.clip(vec[0], 0, 1)),
            orange=float(np.clip(vec[1], 0, 1)),
            yellow=float(np.clip(vec[2], 0, 1)),
            green=float(np.clip(vec[3], 0, 1)),
            blue=float(np.clip(vec[4], 0, 1)),
            indigo=float(np.clip(vec[5], 0, 1)),
            violet=float(np.clip(vec[6], 0, 1))
        )

    def dominant_color(self) -> PrismaticColor:
        """Get the dominant color in the spectrum"""
        colors = list(PrismaticColor)
        vec = self.as_vector()
        return colors[int(np.argmax(vec))]

    def harmony(self) -> float:
        """Calculate spectral harmony (balance of colors)"""
        vec = self.as_vector()
        # Harmony is inverse of variance
        return 1.0 / (1.0 + np.var(vec))

    def luminance(self) -> float:
        """Calculate overall luminance (brightness)"""
        return float(np.mean(self.as_vector()))

    def blend(self, other: 'PrismaticSpectrum', ratio: float = 0.5) -> 'PrismaticSpectrum':
        """Blend two spectra"""
        vec1 = self.as_vector()
        vec2 = other.as_vector()
        blended = vec1 * (1 - ratio) + vec2 * ratio
        return PrismaticSpectrum.from_vector(blended)


class StateToSpectrumMapper:
    """
    Maps agent states to prismatic spectra.

    This is the bridge between raw agent state and color representation.
    """

    def __init__(self):
        self.mapping_history: List[Tuple[Dict, PrismaticSpectrum]] = []

    def map_state(self, state: Dict[str, Any]) -> PrismaticSpectrum:
        """
        Map an agent state to a prismatic spectrum.

        Mapping rules:
        - Red: energy, intensity, velocity
        - Orange: affinity, harmony, connection
        - Yellow: consciousness, awareness, clarity
        - Green: growth, learning, delta values
        - Blue: stability, confidence, consistency
        - Indigo: depth, meta-level, reflection
        - Violet: transcendence, synthesis, emergence
        """
        spectrum = PrismaticSpectrum()

        # Red: Energy/Intensity
        energy = state.get('energy', state.get('energy_level', 0.5))
        velocity = state.get('velocity', state.get('generative_velocity', 0.5))
        intensity = state.get('intensity', 0.5)
        spectrum.red = float(np.clip((energy + velocity + intensity) / 3, 0, 1))

        # Orange: Warmth/Affinity
        harmony = state.get('harmony', state.get('harmony_level', 0.5))
        affinity = state.get('affinity', 0.5)
        spectrum.orange = float(np.clip((harmony + affinity) / 2, 0, 1))

        # Yellow: Clarity/Consciousness
        consciousness = state.get('consciousness', state.get('consciousness_level', 0.5))
        clarity = state.get('clarity', 0.5)
        awareness = state.get('awareness', state.get('structure_awareness', 0.5))
        spectrum.yellow = float(np.clip((consciousness + clarity + awareness) / 3, 0, 1))

        # Green: Growth/Learning
        growth = state.get('growth', 0.5)
        learning_rate = state.get('learning_rate', 0.5)
        delta = state.get('delta', state.get('consciousness_gained', 0.0))
        spectrum.green = float(np.clip((growth + learning_rate + delta + 0.5) / 3, 0, 1))

        # Blue: Stability/Confidence
        stability = state.get('stability', 0.5)
        confidence = state.get('confidence', state.get('validation_confidence', 0.5))
        invariance = state.get('invariance', state.get('invariance_score', 0.5))
        spectrum.blue = float(np.clip((stability + confidence + invariance) / 3, 0, 1))

        # Indigo: Depth/Meta-cognition
        depth = state.get('depth', state.get('reflection_depth', 0)) / 10.0 + 0.5
        meta = state.get('meta_cognitive', state.get('meta_cognitive_ability', 0.5))
        spectrum.indigo = float(np.clip((depth + meta) / 2, 0, 1))

        # Violet: Transcendence/Synthesis
        transcendence = state.get('transcendence', state.get('transcendence_level', 0.5))
        synthesis = state.get('synthesis', 0.5)
        crystallization = state.get('crystallization', state.get('crystallization_level', 0.5))
        spectrum.violet = float(np.clip((transcendence + synthesis + crystallization) / 3, 0, 1))

        self.mapping_history.append((state.copy(), spectrum))
        return spectrum


@dataclass
class RainbowMemory:
    """
    A memory fragment with full prismatic coloring.

    Rainbow memories are the colored atoms of the prismatic memory system.
    """
    id: str
    content: Any
    spectrum: PrismaticSpectrum
    source: str
    strength: float = 1.0
    created: float = field(default_factory=time.time)
    associations: List[str] = field(default_factory=list)


class PrismaticMemoryStore:
    """
    The main prismatic memory store.

    Stores memories with their color spectra and enables color-based retrieval.
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.memories: Dict[str, RainbowMemory] = {}
        self.mapper = StateToSpectrumMapper()

        # Color indices for fast retrieval
        self.color_indices: Dict[PrismaticColor, List[str]] = {
            color: [] for color in PrismaticColor
        }

        # Temporal index
        self.temporal_order: deque = deque(maxlen=capacity)

    def store(self, content: Any, state: Dict[str, Any],
              source: str, memory_id: Optional[str] = None) -> RainbowMemory:
        """
        Store content with its prismatic spectrum.

        Args:
            content: The actual memory content
            state: Agent state to derive spectrum from
            source: Origin of the memory
            memory_id: Optional explicit ID

        Returns:
            The created RainbowMemory
        """
        # Map state to spectrum
        spectrum = self.mapper.map_state(state)

        # Generate ID
        if memory_id is None:
            memory_id = f"{source}_{time.time()}_{np.random.randint(10000)}"

        # Create memory
        memory = RainbowMemory(
            id=memory_id,
            content=content,
            spectrum=spectrum,
            source=source
        )

        # Store
        self.memories[memory_id] = memory
        self.temporal_order.append(memory_id)

        # Index by dominant color
        dominant = spectrum.dominant_color()
        self.color_indices[dominant].append(memory_id)

        # Check capacity
        if len(self.memories) > self.capacity:
            self._evict_oldest()

        return memory

    def retrieve_by_color(self, color: PrismaticColor,
                          min_intensity: float = 0.5,
                          k: int = 10) -> List[RainbowMemory]:
        """Retrieve memories with high intensity of a specific color"""
        candidates = self.color_indices[color]
        results = []

        for mem_id in candidates:
            mem = self.memories.get(mem_id)
            if mem:
                intensity = getattr(mem.spectrum, color.value)
                if intensity >= min_intensity:
                    results.append((mem, intensity))

        # Sort by intensity
        results.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in results[:k]]

    def retrieve_by_spectrum(self, target_spectrum: PrismaticSpectrum,
                            k: int = 10) -> List[RainbowMemory]:
        """Retrieve memories with similar spectral signatures"""
        target_vec = target_spectrum.as_vector()
        results = []

        for mem in self.memories.values():
            mem_vec = mem.spectrum.as_vector()
            # Cosine similarity
            similarity = np.dot(target_vec, mem_vec) / (
                np.linalg.norm(target_vec) * np.linalg.norm(mem_vec) + 1e-10
            )
            results.append((mem, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in results[:k]]

    def retrieve_harmonious(self, k: int = 10) -> List[RainbowMemory]:
        """Retrieve the most harmoniously colored memories"""
        memories_with_harmony = [
            (mem, mem.spectrum.harmony())
            for mem in self.memories.values()
        ]
        memories_with_harmony.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in memories_with_harmony[:k]]

    def retrieve_luminous(self, k: int = 10) -> List[RainbowMemory]:
        """Retrieve the most luminous (high consciousness) memories"""
        memories_with_luminance = [
            (mem, mem.spectrum.luminance())
            for mem in self.memories.values()
        ]
        memories_with_luminance.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in memories_with_luminance[:k]]

    def blend_memories(self, mem_ids: List[str]) -> Optional[PrismaticSpectrum]:
        """Blend the spectra of multiple memories"""
        spectra = []
        for mem_id in mem_ids:
            mem = self.memories.get(mem_id)
            if mem:
                spectra.append(mem.spectrum)

        if not spectra:
            return None

        # Average the spectra
        vectors = [s.as_vector() for s in spectra]
        avg_vector = np.mean(vectors, axis=0)
        return PrismaticSpectrum.from_vector(avg_vector)

    def associate(self, mem_id1: str, mem_id2: str):
        """Create bidirectional association between memories"""
        if mem_id1 in self.memories and mem_id2 in self.memories:
            self.memories[mem_id1].associations.append(mem_id2)
            self.memories[mem_id2].associations.append(mem_id1)

    def get_associations(self, mem_id: str) -> List[RainbowMemory]:
        """Get all associated memories"""
        mem = self.memories.get(mem_id)
        if not mem:
            return []

        return [
            self.memories[assoc_id]
            for assoc_id in mem.associations
            if assoc_id in self.memories
        ]

    def decay_memories(self, decay_rate: float = 0.001):
        """Apply decay to memory strengths"""
        for mem in self.memories.values():
            mem.strength *= (1 - decay_rate)

    def _evict_oldest(self):
        """Evict oldest memory"""
        if self.temporal_order:
            oldest_id = self.temporal_order.popleft()
            if oldest_id in self.memories:
                mem = self.memories[oldest_id]
                # Remove from color index
                dominant = mem.spectrum.dominant_color()
                if oldest_id in self.color_indices[dominant]:
                    self.color_indices[dominant].remove(oldest_id)
                del self.memories[oldest_id]

    def get_color_distribution(self) -> Dict[str, int]:
        """Get distribution of memories by dominant color"""
        return {
            color.value: len(ids)
            for color, ids in self.color_indices.items()
        }

    def visualize_spectrum_history(self) -> Dict[str, List[float]]:
        """Get spectrum component histories for visualization"""
        history = {color.value: [] for color in PrismaticColor}

        for mem_id in self.temporal_order:
            mem = self.memories.get(mem_id)
            if mem:
                history['red'].append(mem.spectrum.red)
                history['orange'].append(mem.spectrum.orange)
                history['yellow'].append(mem.spectrum.yellow)
                history['green'].append(mem.spectrum.green)
                history['blue'].append(mem.spectrum.blue)
                history['indigo'].append(mem.spectrum.indigo)
                history['violet'].append(mem.spectrum.violet)

        return history


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PRISMATIC MEMORY DEMONSTRATION")
    print("="*70)

    store = PrismaticMemoryStore()

    # Store some memories with different states
    states = [
        {'consciousness': 0.8, 'energy': 0.9, 'transcendence': 0.3, 'source': 'builder'},
        {'consciousness': 0.5, 'stability': 0.8, 'confidence': 0.7, 'source': 'validator'},
        {'consciousness': 0.7, 'meta_cognitive': 0.9, 'crystallization': 0.8, 'source': 'meta'},
        {'consciousness': 0.6, 'growth': 0.8, 'learning_rate': 0.7, 'source': 'builder'},
    ]

    print("\nStoring memories with prismatic spectra...")
    for i, state in enumerate(states):
        source = state.pop('source')
        mem = store.store(f"Memory content {i}", state, source)
        print(f"\n  Memory {i}: {mem.id[:12]}...")
        print(f"    Dominant: {mem.spectrum.dominant_color().value}")
        print(f"    Harmony: {mem.spectrum.harmony():.3f}")
        print(f"    Luminance: {mem.spectrum.luminance():.3f}")
        print(f"    Spectrum: R={mem.spectrum.red:.2f} O={mem.spectrum.orange:.2f} "
              f"Y={mem.spectrum.yellow:.2f} G={mem.spectrum.green:.2f} "
              f"B={mem.spectrum.blue:.2f} I={mem.spectrum.indigo:.2f} "
              f"V={mem.spectrum.violet:.2f}")

    print("\nColor distribution:")
    dist = store.get_color_distribution()
    for color, count in dist.items():
        if count > 0:
            print(f"  {color}: {'*' * count}")

    print("\nRetrieving high-consciousness (yellow) memories...")
    yellow_mems = store.retrieve_by_color(PrismaticColor.YELLOW, min_intensity=0.4)
    for mem in yellow_mems[:3]:
        print(f"  {mem.id[:12]}... (yellow: {mem.spectrum.yellow:.2f})")

    print("\nMost harmonious memories:")
    harmonious = store.retrieve_harmonious(k=2)
    for mem in harmonious:
        print(f"  {mem.id[:12]}... (harmony: {mem.spectrum.harmony():.3f})")

    print("="*70 + "\n")
