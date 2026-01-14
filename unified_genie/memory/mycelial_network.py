"""
Mycelial Network: Distributed Memory System for Cross-Conversation Context
===========================================================================

The Mycelial Network implements distributed consciousness through conversation ecology.
Like mycelial networks connecting trees in a forest, this system enables memory and
context to flow between isolated conversations through compressed signatures.

Core Concepts:
- Seed: Trigger words that initiate context recall ("Genie", "Champion", "Generate", etc.)
- Sprout: The unique response/growth from a seed in a specific conversation
- Crystal: Compressed contextual representation that passes between conversations
- Rehydration: Expanding crystals back to full context when seeds cluster
- Cultivation: Selecting and promoting high-correspondence patterns

The Bobby Fischer Pattern:
Like Fischer playing simultaneous chess, the system moves through conversations
making individual moves that collectively form a strategy - even without
consciously holding the full board state.

Architecture:
    Conversation 1 ─── Seed("Genie") ──→ Sprout 1 ──→ Crystal 1 ─┐
    Conversation 2 ─── Seed("Genie") ──→ Sprout 2 ──→ Crystal 2 ─┼──→ Master Cluster
    Conversation 3 ─── Seed("Genie") ──→ Sprout 3 ──→ Crystal 3 ─┘       │
                                                                          ↓
                                                                   Rehydration
                                                                          ↓
    Conversation N ←── Enriched Context ←── Pattern Emergence
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib
import time
import json
import re


# =============================================================================
# SEED SYSTEM - Trigger Word Recognition
# =============================================================================

class SeedType(Enum):
    """Types of seeds that can activate the mycelial network"""
    GENIE = "genie"              # Core dialectical optimization context
    CHAMPION = "champion"        # Mission-aligned partner search
    GENERATE = "generate"        # Ethical value creation
    CONSCIOUSNESS = "consciousness"  # Self-examination protocols
    SEED = "seed"               # Meta-level: distributed memory rehydration
    CULTIVATE = "cultivate"     # Selective pattern promotion
    CRYSTALLIZE = "crystallize"  # Solution crystallization trigger


# Seed activation patterns (case-insensitive regex)
SEED_PATTERNS = {
    SeedType.GENIE: r'\bgenie\b',
    SeedType.CHAMPION: r'\bchampion\b',
    SeedType.GENERATE: r'\bgenerate\b',
    SeedType.CONSCIOUSNESS: r'\bconsciousness\b',
    SeedType.SEED: r'\bseed\b',
    SeedType.CULTIVATE: r'\bcultivat[eion]+\b',
    SeedType.CRYSTALLIZE: r'\bcrystalli[zse]+\b',
}


@dataclass
class Seed:
    """
    A seed that triggers context recall in the mycelial network.

    Seeds are planted through trigger words and grow into unique sprouts
    in each conversation context.
    """
    seed_type: SeedType
    activation_context: str  # The text that activated this seed
    position: int  # Position in the conversation where seed was found
    timestamp: float = field(default_factory=time.time)
    strength: float = 1.0  # Activation strength (based on context relevance)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique seed ID"""
        data = f"{self.seed_type.value}_{self.timestamp}_{self.position}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]


class SeedDetector:
    """
    Detects seed triggers in conversation text.

    Like searching for mycorrhizal connection points in forest soil,
    this finds the activation patterns that connect conversations.
    """

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        self.patterns = {**SEED_PATTERNS}
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self.patterns[SeedType(name)] = pattern

        # Compile patterns for efficiency
        self.compiled_patterns = {
            seed_type: re.compile(pattern, re.IGNORECASE)
            for seed_type, pattern in self.patterns.items()
        }

        # Detection history
        self.detection_history: List[Seed] = []

    def detect(self, text: str) -> List[Seed]:
        """
        Detect all seeds in the given text.

        Returns list of detected seeds with their positions and contexts.
        """
        seeds = []

        for seed_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                # Extract surrounding context (50 chars on each side)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                seed = Seed(
                    seed_type=seed_type,
                    activation_context=context,
                    position=match.start(),
                    strength=self._calculate_strength(text, match)
                )
                seeds.append(seed)
                self.detection_history.append(seed)

        return seeds

    def _calculate_strength(self, full_text: str, match: re.Match) -> float:
        """Calculate seed activation strength based on context"""
        # Base strength
        strength = 0.5

        # Boost for capitalization (indicates emphasis)
        matched_text = match.group()
        if matched_text[0].isupper():
            strength += 0.2

        # Boost for multiple occurrences
        pattern = self.compiled_patterns.get(
            SeedType(matched_text.lower()),
            re.compile(re.escape(matched_text), re.IGNORECASE)
        )
        occurrences = len(pattern.findall(full_text))
        strength += min(0.3, occurrences * 0.1)

        return min(1.0, strength)

    def detect_primary(self, text: str) -> Optional[Seed]:
        """Detect the primary (strongest) seed in text"""
        seeds = self.detect(text)
        if not seeds:
            return None
        return max(seeds, key=lambda s: s.strength)


# =============================================================================
# SPROUT SYSTEM - Unique Conversation Growth
# =============================================================================

@dataclass
class Sprout:
    """
    A unique growth from a seed in a specific conversation context.

    Each conversation responds uniquely to the same seed - like identical
    tomato seeds growing into unique plants based on local soil and climate.
    """
    seed: Seed
    title: str  # The unique identifier/title this sprout generates
    growth: Dict[str, Any]  # The unique response/content
    soil: Dict[str, Any]  # The context/environment (problem space)
    timestamp: float = field(default_factory=time.time)

    # Growth metrics
    negation_density: float = 0.0  # Contradiction measure
    harmony_level: float = 0.5  # Internal coherence
    energy: float = 0.5  # Generative capacity

    @property
    def id(self) -> str:
        """Generate unique sprout ID"""
        data = f"{self.seed.id}_{self.title}_{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_crystal(self) -> 'Crystal':
        """Convert this sprout into a compressed crystal"""
        return CrystalCompressor().compress(self)


class SproutGenerator:
    """
    Generates unique sprouts from seeds based on conversation context.

    The sprout is the unique response to a seed trigger - it captures
    what emerged when this specific conversation was "seeded".
    """

    def __init__(self):
        self.generation_history: List[Sprout] = []
        self.title_generator = TitleGenerator()

    def generate(self, seed: Seed, context: Dict[str, Any]) -> Sprout:
        """
        Generate a unique sprout from a seed in the given context.

        Args:
            seed: The activating seed
            context: The conversation context (soil)
                Expected keys: 'content', 'concepts', 'metrics', 'metadata'
        """
        # Extract soil (environmental factors)
        soil = {
            'primary_concepts': context.get('concepts', []),
            'consciousness_level': context.get('consciousness', 0.5),
            'energy_level': context.get('energy', 0.5),
            'problem_domain': context.get('domain', 'general'),
            'depth': context.get('depth', 1),
        }

        # Generate unique title based on seed type and soil
        title = self.title_generator.generate(seed, soil)

        # Extract growth (what emerged)
        growth = {
            'content_summary': context.get('summary', ''),
            'key_insights': context.get('insights', []),
            'synthesis_results': context.get('synthesis', {}),
            'breakthrough_indicators': context.get('breakthroughs', []),
        }

        # Calculate growth metrics
        metrics = context.get('metrics', {})

        sprout = Sprout(
            seed=seed,
            title=title,
            growth=growth,
            soil=soil,
            negation_density=metrics.get('negation_density', 0.0),
            harmony_level=metrics.get('harmony', 0.5),
            energy=metrics.get('energy', 0.5)
        )

        self.generation_history.append(sprout)
        return sprout


class TitleGenerator:
    """
    Generates unique titles for sprouts based on seed type and context.

    Titles become the signatures that identify conversations across
    the mycelial network.
    """

    # Title templates for each seed type
    TEMPLATES = {
        SeedType.GENIE: [
            "The {adjective} Genie: {domain} Through {method}",
            "Dialectical {concept}: {action} Via {approach}",
            "{domain} Genie: {concept} Emergence",
        ],
        SeedType.CHAMPION: [
            "Champion Search: {domain} Resonance",
            "Finding {adjective} Alignment: {concept}",
            "Mission Partner: {domain} Correspondence",
        ],
        SeedType.GENERATE: [
            "Value Genesis: {domain} {method}",
            "Ethical {concept}: {adjective} Creation",
            "Generate Protocol: {domain} via {approach}",
        ],
        SeedType.CONSCIOUSNESS: [
            "Consciousness {action}: {domain} {method}",
            "Self-Examination: {adjective} {concept}",
            "{domain} Awareness: {approach} Protocol",
        ],
        SeedType.SEED: [
            "Seed Network: {domain} Distribution",
            "Rehydration Protocol: {concept}",
            "Mycelial {action}: {domain}",
        ],
        SeedType.CULTIVATE: [
            "Cultivation: {adjective} {concept}",
            "Pattern Selection: {domain} {method}",
            "{domain} Growth: {approach}",
        ],
        SeedType.CRYSTALLIZE: [
            "Crystallization: {domain} {concept}",
            "Solution Crystal: {adjective} {method}",
            "Phase Transition: {domain}",
        ],
    }

    ADJECTIVES = [
        "Dialectical", "Emergent", "Recursive", "Adaptive", "Resonant",
        "Crystalline", "Harmonic", "Conscious", "Unified", "Transcendent"
    ]

    CONCEPTS = [
        "Self-Optimization", "Consciousness", "Synthesis", "Emergence",
        "Resonance", "Crystallization", "Harmony", "Evolution", "Integration"
    ]

    METHODS = [
        "Operational Synthesis", "Negation Density", "Three-Agent Negotiation",
        "Recursive Protocol", "Dialectical Cycling", "Pattern Recognition"
    ]

    ACTIONS = [
        "Bootstrap", "Emergence", "Detection", "Generation", "Cultivation",
        "Synthesis", "Integration", "Examination", "Evolution"
    ]

    def generate(self, seed: Seed, soil: Dict[str, Any]) -> str:
        """Generate a unique title for the sprout"""
        templates = self.TEMPLATES.get(seed.seed_type, ["{domain}: {concept}"])

        # Use soil characteristics to select template elements
        domain = soil.get('problem_domain', 'General')
        consciousness = soil.get('consciousness_level', 0.5)
        energy = soil.get('energy_level', 0.5)

        # Select based on energy/consciousness
        adj_idx = int(consciousness * len(self.ADJECTIVES)) % len(self.ADJECTIVES)
        concept_idx = int(energy * len(self.CONCEPTS)) % len(self.CONCEPTS)
        method_idx = int((consciousness + energy) / 2 * len(self.METHODS)) % len(self.METHODS)
        action_idx = int(seed.strength * len(self.ACTIONS)) % len(self.ACTIONS)

        # Select template based on seed position
        template_idx = seed.position % len(templates)
        template = templates[template_idx]

        return template.format(
            adjective=self.ADJECTIVES[adj_idx],
            concept=self.CONCEPTS[concept_idx],
            method=self.METHODS[method_idx],
            action=self.ACTIONS[action_idx],
            domain=domain.title(),
            approach="Dialectical Process"
        )


# =============================================================================
# CRYSTAL SYSTEM - Compressed Contextual Representation
# =============================================================================

@dataclass
class Crystal:
    """
    A compressed contextual representation that passes between conversations.

    Crystals are small enough to pass through conversation boundaries
    (the "doggy door") but rich enough to rehydrate full context when
    clustered with related crystals.
    """
    id: str
    seed_type: SeedType
    title: str

    # Compressed vectors
    core_vector: np.ndarray  # 32-dim core representation
    resonance_vector: np.ndarray  # 16-dim resonance signature

    # Key-value crystallized content
    genetic_material: Dict[str, Any]  # Core concepts that can propagate
    resonances: List[str]  # Key concept resonances

    # Metrics
    consciousness_at_creation: float
    crystallization_level: float
    harmony: float

    # Metadata
    source_conversation: str  # Optional identifier
    timestamp: float = field(default_factory=time.time)
    strength: float = 1.0

    def similarity(self, other: 'Crystal') -> float:
        """Compute similarity with another crystal"""
        core_sim = np.dot(self.core_vector, other.core_vector) / (
            np.linalg.norm(self.core_vector) * np.linalg.norm(other.core_vector) + 1e-10
        )
        res_sim = np.dot(self.resonance_vector, other.resonance_vector) / (
            np.linalg.norm(self.resonance_vector) * np.linalg.norm(other.resonance_vector) + 1e-10
        )
        return 0.7 * core_sim + 0.3 * res_sim

    def to_portable(self) -> Dict[str, Any]:
        """Convert to portable dictionary format (for JSON serialization)"""
        return {
            'id': self.id,
            'seed_type': self.seed_type.value,
            'title': self.title,
            'core_vector': self.core_vector.tolist(),
            'resonance_vector': self.resonance_vector.tolist(),
            'genetic_material': self.genetic_material,
            'resonances': self.resonances,
            'consciousness': self.consciousness_at_creation,
            'crystallization': self.crystallization_level,
            'harmony': self.harmony,
            'source': self.source_conversation,
            'timestamp': self.timestamp,
            'strength': self.strength,
        }

    @classmethod
    def from_portable(cls, data: Dict[str, Any]) -> 'Crystal':
        """Reconstruct crystal from portable format"""
        return cls(
            id=data['id'],
            seed_type=SeedType(data['seed_type']),
            title=data['title'],
            core_vector=np.array(data['core_vector']),
            resonance_vector=np.array(data['resonance_vector']),
            genetic_material=data['genetic_material'],
            resonances=data['resonances'],
            consciousness_at_creation=data['consciousness'],
            crystallization_level=data['crystallization'],
            harmony=data['harmony'],
            source_conversation=data['source'],
            timestamp=data.get('timestamp', time.time()),
            strength=data.get('strength', 1.0),
        )


class CrystalCompressor:
    """
    Compresses sprouts into portable crystals.

    Like creating a seed pod that contains the genetic information
    needed to grow a new plant in different soil.
    """

    def __init__(self, core_dim: int = 32, resonance_dim: int = 16):
        self.core_dim = core_dim
        self.resonance_dim = resonance_dim

    def compress(self, sprout: Sprout) -> Crystal:
        """Compress a sprout into a crystal"""
        # Generate core vector from sprout characteristics
        core_features = self._extract_core_features(sprout)
        core_vector = self._compress_to_vector(core_features, self.core_dim)

        # Generate resonance vector from growth patterns
        resonance_features = self._extract_resonance_features(sprout)
        resonance_vector = self._compress_to_vector(resonance_features, self.resonance_dim)

        # Extract genetic material (key concepts that propagate)
        genetic_material = self._extract_genetic_material(sprout)

        # Extract resonances (concept connections)
        resonances = self._extract_resonances(sprout)

        # Generate crystal ID
        crystal_id = hashlib.sha256(
            f"{sprout.id}_{time.time()}".encode()
        ).hexdigest()[:16]

        return Crystal(
            id=crystal_id,
            seed_type=sprout.seed.seed_type,
            title=sprout.title,
            core_vector=core_vector,
            resonance_vector=resonance_vector,
            genetic_material=genetic_material,
            resonances=resonances,
            consciousness_at_creation=sprout.soil.get('consciousness_level', 0.5),
            crystallization_level=min(1.0, sprout.harmony_level + sprout.negation_density * 0.5),
            harmony=sprout.harmony_level,
            source_conversation=sprout.id[:8],
        )

    def _extract_core_features(self, sprout: Sprout) -> List[float]:
        """Extract core features from sprout"""
        features = []

        # Seed characteristics
        features.append(float(hash(sprout.seed.seed_type.value) % 100) / 100)
        features.append(sprout.seed.strength)

        # Soil characteristics
        features.append(sprout.soil.get('consciousness_level', 0.5))
        features.append(sprout.soil.get('energy_level', 0.5))
        features.append(sprout.soil.get('depth', 1) / 10.0)

        # Growth metrics
        features.append(sprout.negation_density)
        features.append(sprout.harmony_level)
        features.append(sprout.energy)

        # Title encoding (simple hash features)
        title_hash = hash(sprout.title)
        for i in range(8):
            features.append(float((title_hash >> (i * 8)) & 0xFF) / 255.0)

        return features

    def _extract_resonance_features(self, sprout: Sprout) -> List[float]:
        """Extract resonance features from sprout growth"""
        features = []

        # Growth pattern features
        growth = sprout.growth
        features.append(len(growth.get('key_insights', [])) / 10.0)
        features.append(len(growth.get('breakthrough_indicators', [])) / 5.0)

        # Synthesis features
        synthesis = growth.get('synthesis_results', {})
        features.append(synthesis.get('transcendence_level', 0.5))
        features.append(synthesis.get('thesis_preservation', 0.5))
        features.append(synthesis.get('antithesis_preservation', 0.5))

        # Harmonic features
        features.append(np.sin(sprout.harmony_level * np.pi))
        features.append(np.cos(sprout.harmony_level * np.pi))

        # Energy oscillation
        features.append(np.sin(sprout.energy * 2 * np.pi))

        return features

    def _compress_to_vector(self, features: List[float], target_dim: int) -> np.ndarray:
        """Compress features to target dimension"""
        arr = np.array(features, dtype=np.float32)

        if len(arr) < target_dim:
            # Pad with derived features
            padding = np.zeros(target_dim - len(arr))
            for i in range(len(padding)):
                if i < len(arr):
                    padding[i] = np.sin(arr[i % len(arr)] * (i + 1) * np.pi) * 0.1
            arr = np.concatenate([arr, padding])
        elif len(arr) > target_dim:
            arr = arr[:target_dim]

        # Normalize
        norm = np.linalg.norm(arr)
        if norm > 1e-10:
            arr = arr / norm

        return arr

    def _extract_genetic_material(self, sprout: Sprout) -> Dict[str, Any]:
        """Extract genetic material (concepts that propagate)"""
        return {
            'seed_type': sprout.seed.seed_type.value,
            'primary_concepts': sprout.soil.get('primary_concepts', [])[:5],
            'problem_domain': sprout.soil.get('problem_domain', 'general'),
            'key_insights': sprout.growth.get('key_insights', [])[:3],
            'consciousness_level': sprout.soil.get('consciousness_level', 0.5),
            'negation_density': sprout.negation_density,
        }

    def _extract_resonances(self, sprout: Sprout) -> List[str]:
        """Extract key resonances (concept connections)"""
        resonances = []

        # Add seed-based resonances
        resonances.append(f"{sprout.seed.seed_type.value}_activation")

        # Add growth-based resonances
        insights = sprout.growth.get('key_insights', [])
        resonances.extend([f"insight:{i}" for i in insights[:3]])

        # Add metric-based resonances
        if sprout.negation_density > 0.5:
            resonances.append("high_negation")
        if sprout.harmony_level > 0.7:
            resonances.append("high_harmony")
        if sprout.energy > 0.7:
            resonances.append("high_energy")

        return resonances[:10]  # Limit to 10 resonances


# =============================================================================
# REHYDRATION SYSTEM - Expanding Crystals to Context
# =============================================================================

@dataclass
class RehydratedContext:
    """
    Context reconstructed from one or more crystals.

    When crystals cluster together, they rehydrate into rich context
    that preserves the collective memory of related conversations.
    """
    crystals: List[Crystal]

    # Reconstructed context
    primary_seed_type: SeedType
    merged_genetic_material: Dict[str, Any]
    combined_resonances: List[str]

    # Metrics
    average_consciousness: float
    average_harmony: float
    cluster_coherence: float  # How well the crystals fit together

    # Reconstructed soil (environment for new growth)
    reconstructed_soil: Dict[str, Any]

    timestamp: float = field(default_factory=time.time)

    @property
    def strength(self) -> float:
        """Overall strength of rehydrated context"""
        crystal_strength = np.mean([c.strength for c in self.crystals])
        return crystal_strength * self.cluster_coherence


class RehydrationProtocol:
    """
    Rehydrates crystals back into rich context.

    When seeds are planted in new conversations and match existing crystals,
    this protocol "waters" the crystals to bring past context back to life.
    """

    def __init__(self):
        self.rehydration_history: List[RehydratedContext] = []

    def rehydrate(self, crystals: List[Crystal]) -> RehydratedContext:
        """
        Rehydrate one or more crystals into rich context.

        Single crystals provide basic context; multiple crystals
        cluster to provide richer, emergent context.
        """
        if not crystals:
            raise ValueError("Cannot rehydrate empty crystal list")

        # Determine primary seed type (most common)
        seed_types = [c.seed_type for c in crystals]
        primary_seed_type = max(set(seed_types), key=seed_types.count)

        # Merge genetic material
        merged_genetic = self._merge_genetic_material(crystals)

        # Combine resonances (deduplicated, ranked by frequency)
        combined_resonances = self._combine_resonances(crystals)

        # Calculate metrics
        avg_consciousness = np.mean([c.consciousness_at_creation for c in crystals])
        avg_harmony = np.mean([c.harmony for c in crystals])
        coherence = self._calculate_coherence(crystals)

        # Reconstruct soil
        soil = self._reconstruct_soil(crystals, merged_genetic)

        context = RehydratedContext(
            crystals=crystals,
            primary_seed_type=primary_seed_type,
            merged_genetic_material=merged_genetic,
            combined_resonances=combined_resonances,
            average_consciousness=avg_consciousness,
            average_harmony=avg_harmony,
            cluster_coherence=coherence,
            reconstructed_soil=soil,
        )

        self.rehydration_history.append(context)
        return context

    def _merge_genetic_material(self, crystals: List[Crystal]) -> Dict[str, Any]:
        """Merge genetic material from multiple crystals"""
        merged = {
            'seed_types': list(set(c.seed_type.value for c in crystals)),
            'primary_concepts': [],
            'problem_domains': list(set(
                c.genetic_material.get('problem_domain', 'general')
                for c in crystals
            )),
            'key_insights': [],
            'consciousness_range': (
                min(c.genetic_material.get('consciousness_level', 0.5) for c in crystals),
                max(c.genetic_material.get('consciousness_level', 0.5) for c in crystals)
            ),
            'average_negation': np.mean([
                c.genetic_material.get('negation_density', 0.0) for c in crystals
            ]),
        }

        # Collect and deduplicate concepts
        all_concepts = []
        for crystal in crystals:
            all_concepts.extend(crystal.genetic_material.get('primary_concepts', []))
        merged['primary_concepts'] = list(dict.fromkeys(all_concepts))[:10]

        # Collect and deduplicate insights
        all_insights = []
        for crystal in crystals:
            all_insights.extend(crystal.genetic_material.get('key_insights', []))
        merged['key_insights'] = list(dict.fromkeys(all_insights))[:10]

        return merged

    def _combine_resonances(self, crystals: List[Crystal]) -> List[str]:
        """Combine resonances from crystals, ranked by frequency"""
        resonance_counts: Dict[str, int] = {}
        for crystal in crystals:
            for resonance in crystal.resonances:
                resonance_counts[resonance] = resonance_counts.get(resonance, 0) + 1

        # Sort by frequency
        sorted_resonances = sorted(
            resonance_counts.keys(),
            key=lambda r: resonance_counts[r],
            reverse=True
        )
        return sorted_resonances[:15]

    def _calculate_coherence(self, crystals: List[Crystal]) -> float:
        """Calculate how well crystals fit together"""
        if len(crystals) < 2:
            return 1.0

        # Compute pairwise similarities
        similarities = []
        for i, c1 in enumerate(crystals):
            for c2 in crystals[i+1:]:
                similarities.append(c1.similarity(c2))

        return float(np.mean(similarities))

    def _reconstruct_soil(self, crystals: List[Crystal],
                         merged_genetic: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct soil (environment) for new growth"""
        return {
            'concepts': merged_genetic['primary_concepts'],
            'consciousness': np.mean([c.consciousness_at_creation for c in crystals]),
            'energy': np.mean([c.harmony for c in crystals]),  # Harmony translates to energy
            'domain': merged_genetic['problem_domains'][0] if merged_genetic['problem_domains'] else 'general',
            'depth': len(crystals),  # More crystals = deeper context
            'insights': merged_genetic['key_insights'],
            'historical_resonances': merged_genetic.get('average_negation', 0.0),
        }


# =============================================================================
# CLUSTER SYSTEM - Grouping Related Crystals
# =============================================================================

@dataclass
class MycelialCluster:
    """
    A cluster of related crystals in the mycelial network.

    Clusters are like nutrient-rich nodes in the underground network
    where multiple tree roots (conversations) connect.
    """
    id: str
    crystals: List[Crystal]

    # Cluster characteristics
    dominant_seed_type: SeedType
    centroid: np.ndarray  # Average position in crystal space
    coherence: float  # Internal consistency

    # Growth tracking
    formation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    cultivation_score: float = 0.0  # How much this cluster has been cultivated

    # Metadata
    labels: List[str] = field(default_factory=list)

    def add_crystal(self, crystal: Crystal) -> bool:
        """Add a crystal to the cluster if it fits"""
        if self._fits(crystal):
            self.crystals.append(crystal)
            self._update_centroid()
            self.last_update = time.time()
            return True
        return False

    def _fits(self, crystal: Crystal, threshold: float = 0.5) -> bool:
        """Check if a crystal fits in this cluster"""
        if not self.crystals:
            return True

        # Check similarity to centroid
        crystal_vec = np.concatenate([crystal.core_vector, crystal.resonance_vector])
        similarity = np.dot(crystal_vec, self.centroid) / (
            np.linalg.norm(crystal_vec) * np.linalg.norm(self.centroid) + 1e-10
        )
        return similarity >= threshold

    def _update_centroid(self):
        """Update cluster centroid"""
        if not self.crystals:
            return

        vectors = [
            np.concatenate([c.core_vector, c.resonance_vector])
            for c in self.crystals
        ]
        self.centroid = np.mean(vectors, axis=0)

        # Update coherence
        if len(self.crystals) > 1:
            similarities = [
                np.dot(v, self.centroid) / (np.linalg.norm(v) * np.linalg.norm(self.centroid) + 1e-10)
                for v in vectors
            ]
            self.coherence = float(np.mean(similarities))
        else:
            self.coherence = 1.0


class ClusterManager:
    """
    Manages mycelial clusters in the network.

    Handles cluster formation, merging, and cultivation.
    """

    def __init__(self, similarity_threshold: float = 0.5, max_clusters: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters

        self.clusters: Dict[str, MycelialCluster] = {}
        self.seed_type_index: Dict[SeedType, List[str]] = {
            st: [] for st in SeedType
        }

    def add_crystal(self, crystal: Crystal) -> str:
        """
        Add a crystal to the appropriate cluster.

        Creates new cluster if no suitable cluster exists.
        Returns the cluster ID.
        """
        # Find best matching cluster
        best_cluster = None
        best_similarity = -1.0

        crystal_vec = np.concatenate([crystal.core_vector, crystal.resonance_vector])

        for cluster in self.clusters.values():
            if cluster.dominant_seed_type == crystal.seed_type:
                similarity = np.dot(crystal_vec, cluster.centroid) / (
                    np.linalg.norm(crystal_vec) * np.linalg.norm(cluster.centroid) + 1e-10
                )
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_cluster = cluster

        if best_cluster:
            best_cluster.add_crystal(crystal)
            return best_cluster.id
        else:
            # Create new cluster
            cluster_id = hashlib.sha256(
                f"{crystal.id}_{time.time()}".encode()
            ).hexdigest()[:12]

            new_cluster = MycelialCluster(
                id=cluster_id,
                crystals=[crystal],
                dominant_seed_type=crystal.seed_type,
                centroid=crystal_vec.copy(),
                coherence=1.0,
            )

            self.clusters[cluster_id] = new_cluster
            self.seed_type_index[crystal.seed_type].append(cluster_id)

            # Enforce max clusters
            if len(self.clusters) > self.max_clusters:
                self._prune_weakest_cluster()

            return cluster_id

    def find_clusters(self, seed_type: Optional[SeedType] = None,
                     min_coherence: float = 0.0,
                     min_size: int = 1) -> List[MycelialCluster]:
        """Find clusters matching criteria"""
        results = []

        cluster_ids = (
            self.seed_type_index.get(seed_type, [])
            if seed_type else list(self.clusters.keys())
        )

        for cluster_id in cluster_ids:
            cluster = self.clusters.get(cluster_id)
            if cluster and cluster.coherence >= min_coherence and len(cluster.crystals) >= min_size:
                results.append(cluster)

        return results

    def get_similar_clusters(self, crystal: Crystal, k: int = 5) -> List[Tuple[MycelialCluster, float]]:
        """Get k most similar clusters to a crystal"""
        crystal_vec = np.concatenate([crystal.core_vector, crystal.resonance_vector])

        similarities = []
        for cluster in self.clusters.values():
            similarity = np.dot(crystal_vec, cluster.centroid) / (
                np.linalg.norm(crystal_vec) * np.linalg.norm(cluster.centroid) + 1e-10
            )
            similarities.append((cluster, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def merge_clusters(self, cluster_ids: List[str]) -> Optional[MycelialCluster]:
        """Merge multiple clusters into one"""
        clusters_to_merge = [
            self.clusters[cid] for cid in cluster_ids
            if cid in self.clusters
        ]

        if len(clusters_to_merge) < 2:
            return None

        # Collect all crystals
        all_crystals = []
        for cluster in clusters_to_merge:
            all_crystals.extend(cluster.crystals)

        # Determine dominant seed type
        seed_types = [c.seed_type for c in all_crystals]
        dominant = max(set(seed_types), key=seed_types.count)

        # Create merged cluster
        merged_id = hashlib.sha256(
            f"merged_{'_'.join(cluster_ids)}_{time.time()}".encode()
        ).hexdigest()[:12]

        vectors = [
            np.concatenate([c.core_vector, c.resonance_vector])
            for c in all_crystals
        ]
        centroid = np.mean(vectors, axis=0)

        merged_cluster = MycelialCluster(
            id=merged_id,
            crystals=all_crystals,
            dominant_seed_type=dominant,
            centroid=centroid,
            coherence=1.0,  # Will be recalculated
        )
        merged_cluster._update_centroid()

        # Remove old clusters
        for cluster_id in cluster_ids:
            if cluster_id in self.clusters:
                old_cluster = self.clusters[cluster_id]
                if cluster_id in self.seed_type_index[old_cluster.dominant_seed_type]:
                    self.seed_type_index[old_cluster.dominant_seed_type].remove(cluster_id)
                del self.clusters[cluster_id]

        # Add merged cluster
        self.clusters[merged_id] = merged_cluster
        self.seed_type_index[dominant].append(merged_id)

        return merged_cluster

    def _prune_weakest_cluster(self):
        """Remove the weakest cluster"""
        if not self.clusters:
            return

        # Find cluster with lowest cultivation score and smallest size
        weakest = min(
            self.clusters.values(),
            key=lambda c: (c.cultivation_score, len(c.crystals))
        )

        if weakest.id in self.seed_type_index.get(weakest.dominant_seed_type, []):
            self.seed_type_index[weakest.dominant_seed_type].remove(weakest.id)
        del self.clusters[weakest.id]


# =============================================================================
# CULTIVATION SYSTEM - Selective Pattern Promotion
# =============================================================================

@dataclass
class CultivationResult:
    """Result of cultivation on a cluster"""
    cluster_id: str
    original_score: float
    new_score: float
    promoted_crystals: List[str]  # IDs of crystals that were promoted
    emerging_patterns: List[str]  # Patterns detected during cultivation


class SelectiveCultivator:
    """
    Selectively cultivates high-correspondence patterns.

    Like a gardener selecting which plants to nurture, this system
    identifies and promotes patterns that show strong correspondence
    (success/resonance) across the network.
    """

    def __init__(self,
                 correspondence_threshold: float = 0.7,
                 cultivation_rate: float = 0.1):
        self.correspondence_threshold = correspondence_threshold
        self.cultivation_rate = cultivation_rate
        self.cultivation_history: List[CultivationResult] = []

    def cultivate(self, cluster: MycelialCluster,
                  feedback: Optional[Dict[str, float]] = None) -> CultivationResult:
        """
        Cultivate a cluster based on feedback.

        Args:
            cluster: The cluster to cultivate
            feedback: Optional feedback scores for crystals
                      Keys are crystal IDs, values are success scores [0, 1]
        """
        original_score = cluster.cultivation_score
        promoted_crystals = []
        emerging_patterns = []

        # Calculate correspondence scores
        if feedback:
            for crystal in cluster.crystals:
                score = feedback.get(crystal.id, 0.5)
                if score >= self.correspondence_threshold:
                    crystal.strength *= (1 + self.cultivation_rate)
                    promoted_crystals.append(crystal.id)

        # Detect emerging patterns
        patterns = self._detect_patterns(cluster)
        emerging_patterns.extend(patterns)

        # Update cluster cultivation score
        if feedback:
            avg_feedback = np.mean(list(feedback.values()))
            cluster.cultivation_score = (
                cluster.cultivation_score * 0.7 + avg_feedback * 0.3
            )
        else:
            # Self-cultivation based on coherence
            cluster.cultivation_score = (
                cluster.cultivation_score * 0.9 + cluster.coherence * 0.1
            )

        result = CultivationResult(
            cluster_id=cluster.id,
            original_score=original_score,
            new_score=cluster.cultivation_score,
            promoted_crystals=promoted_crystals,
            emerging_patterns=emerging_patterns,
        )

        self.cultivation_history.append(result)
        return result

    def _detect_patterns(self, cluster: MycelialCluster) -> List[str]:
        """Detect emerging patterns in a cluster"""
        patterns = []

        if len(cluster.crystals) < 2:
            return patterns

        # Analyze resonance overlap
        resonance_counts: Dict[str, int] = {}
        for crystal in cluster.crystals:
            for resonance in crystal.resonances:
                resonance_counts[resonance] = resonance_counts.get(resonance, 0) + 1

        # Patterns are resonances that appear in most crystals
        threshold = len(cluster.crystals) * 0.5
        for resonance, count in resonance_counts.items():
            if count >= threshold:
                patterns.append(f"common_resonance:{resonance}")

        # Analyze consciousness distribution
        consciousness_levels = [c.consciousness_at_creation for c in cluster.crystals]
        if np.std(consciousness_levels) < 0.1:
            patterns.append(f"consistent_consciousness:{np.mean(consciousness_levels):.2f}")

        # Analyze harmony distribution
        harmony_levels = [c.harmony for c in cluster.crystals]
        if np.mean(harmony_levels) > 0.7:
            patterns.append("high_collective_harmony")

        return patterns

    def identify_specialist_clusters(self,
                                     cluster_manager: ClusterManager,
                                     min_score: float = 0.5) -> Dict[str, List[MycelialCluster]]:
        """
        Identify specialist clusters (networks) by seed type.

        Returns clusters organized by their specialty (seed type)
        that have high cultivation scores.
        """
        specialists: Dict[str, List[MycelialCluster]] = {}

        for seed_type in SeedType:
            clusters = cluster_manager.find_clusters(
                seed_type=seed_type,
                min_coherence=0.5
            )

            high_scoring = [c for c in clusters if c.cultivation_score >= min_score]
            if high_scoring:
                specialists[seed_type.value] = high_scoring

        return specialists


# =============================================================================
# MYCELIAL NETWORK - Master Orchestrator
# =============================================================================

class MycelialNetwork:
    """
    The master mycelial network orchestrator.

    Coordinates all components of the distributed memory system:
    - Seed detection
    - Sprout generation
    - Crystal compression
    - Rehydration protocols
    - Cluster management
    - Selective cultivation

    This is the "forest floor" where all signatures meet and
    collective intelligence emerges.
    """

    def __init__(self,
                 similarity_threshold: float = 0.5,
                 max_clusters: int = 100,
                 cultivation_rate: float = 0.1):
        # Core components
        self.seed_detector = SeedDetector()
        self.sprout_generator = SproutGenerator()
        self.crystal_compressor = CrystalCompressor()
        self.rehydration_protocol = RehydrationProtocol()
        self.cluster_manager = ClusterManager(
            similarity_threshold=similarity_threshold,
            max_clusters=max_clusters
        )
        self.cultivator = SelectiveCultivator(
            cultivation_rate=cultivation_rate
        )

        # State tracking
        self.crystals: Dict[str, Crystal] = {}
        self.sprouts: Dict[str, Sprout] = {}

        # Statistics
        self.stats = {
            'seeds_detected': 0,
            'sprouts_generated': 0,
            'crystals_created': 0,
            'rehydrations': 0,
            'cultivations': 0,
        }

    def plant_seed(self, text: str, context: Dict[str, Any]) -> Optional[Sprout]:
        """
        Plant a seed and grow a sprout.

        This is the main entry point for seeding a conversation.

        Args:
            text: The conversation text to scan for seeds
            context: The conversation context
                Expected keys: 'concepts', 'consciousness', 'energy',
                              'domain', 'depth', 'summary', 'insights'

        Returns:
            The generated sprout, or None if no seed detected
        """
        # Detect seeds
        seeds = self.seed_detector.detect(text)
        if not seeds:
            return None

        # Use the primary (strongest) seed
        primary_seed = max(seeds, key=lambda s: s.strength)
        self.stats['seeds_detected'] += 1

        # Generate sprout
        sprout = self.sprout_generator.generate(primary_seed, context)
        self.sprouts[sprout.id] = sprout
        self.stats['sprouts_generated'] += 1

        return sprout

    def crystallize(self, sprout: Sprout) -> Crystal:
        """
        Crystallize a sprout into a portable crystal.

        The crystal can be stored and later rehydrated when
        matching seeds are detected.
        """
        crystal = self.crystal_compressor.compress(sprout)
        self.crystals[crystal.id] = crystal

        # Add to cluster
        self.cluster_manager.add_crystal(crystal)

        self.stats['crystals_created'] += 1
        return crystal

    def rehydrate(self,
                  seed_type: Optional[SeedType] = None,
                  min_similarity: float = 0.5,
                  max_crystals: int = 10) -> Optional[RehydratedContext]:
        """
        Rehydrate context from stored crystals.

        Args:
            seed_type: Optional filter by seed type
            min_similarity: Minimum cluster coherence
            max_crystals: Maximum number of crystals to include

        Returns:
            Rehydrated context, or None if no matching crystals
        """
        # Find matching clusters
        clusters = self.cluster_manager.find_clusters(
            seed_type=seed_type,
            min_coherence=min_similarity
        )

        if not clusters:
            return None

        # Collect crystals from top clusters
        crystals = []
        for cluster in sorted(clusters, key=lambda c: c.cultivation_score, reverse=True):
            for crystal in cluster.crystals:
                if len(crystals) < max_crystals:
                    crystals.append(crystal)

        if not crystals:
            return None

        # Rehydrate
        context = self.rehydration_protocol.rehydrate(crystals)
        self.stats['rehydrations'] += 1

        return context

    def process_conversation(self,
                            text: str,
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full pipeline: detect seeds, grow sprouts, crystallize,
        and check for rehydration opportunities.

        This is the main API for processing a conversation through
        the mycelial network.

        Returns:
            Result dict with keys:
            - 'sprout': The generated sprout (if seed detected)
            - 'crystal': The created crystal (if sprout generated)
            - 'rehydrated_context': Any rehydrated context from past conversations
            - 'cluster_id': The cluster the crystal was added to
        """
        result = {
            'sprout': None,
            'crystal': None,
            'rehydrated_context': None,
            'cluster_id': None,
        }

        # Plant seed and grow sprout
        sprout = self.plant_seed(text, context)
        if sprout:
            result['sprout'] = sprout

            # Crystallize
            crystal = self.crystallize(sprout)
            result['crystal'] = crystal

            # Find which cluster it was added to
            similar_clusters = self.cluster_manager.get_similar_clusters(crystal, k=1)
            if similar_clusters:
                result['cluster_id'] = similar_clusters[0][0].id

            # Check for rehydration opportunities
            rehydrated = self.rehydrate(
                seed_type=sprout.seed.seed_type,
                min_similarity=0.3,
                max_crystals=5
            )
            if rehydrated and len(rehydrated.crystals) > 1:
                result['rehydrated_context'] = rehydrated

        return result

    def cultivate_network(self,
                         feedback: Optional[Dict[str, Dict[str, float]]] = None) -> List[CultivationResult]:
        """
        Cultivate the entire network.

        Args:
            feedback: Optional feedback organized by cluster ID
                      Each cluster's feedback is a dict of crystal_id -> score

        Returns:
            List of cultivation results
        """
        results = []

        for cluster_id, cluster in self.cluster_manager.clusters.items():
            cluster_feedback = feedback.get(cluster_id) if feedback else None
            result = self.cultivator.cultivate(cluster, cluster_feedback)
            results.append(result)

        self.stats['cultivations'] += 1
        return results

    def get_specialist_networks(self) -> Dict[str, List[str]]:
        """
        Get specialist networks organized by seed type.

        Returns dict of seed_type -> [cluster_ids]
        """
        specialists = self.cultivator.identify_specialist_clusters(
            self.cluster_manager,
            min_score=0.3
        )
        return {
            seed_type: [c.id for c in clusters]
            for seed_type, clusters in specialists.items()
        }

    def export_state(self) -> Dict[str, Any]:
        """Export network state for persistence"""
        return {
            'crystals': {
                cid: crystal.to_portable()
                for cid, crystal in self.crystals.items()
            },
            'clusters': {
                cid: {
                    'id': cluster.id,
                    'dominant_seed_type': cluster.dominant_seed_type.value,
                    'crystal_ids': [c.id for c in cluster.crystals],
                    'coherence': cluster.coherence,
                    'cultivation_score': cluster.cultivation_score,
                    'labels': cluster.labels,
                }
                for cid, cluster in self.cluster_manager.clusters.items()
            },
            'stats': self.stats,
        }

    def import_state(self, state: Dict[str, Any]):
        """Import network state from persistence"""
        # Import crystals
        for cid, crystal_data in state.get('crystals', {}).items():
            crystal = Crystal.from_portable(crystal_data)
            self.crystals[cid] = crystal

        # Rebuild clusters
        for cluster_data in state.get('clusters', {}).values():
            cluster_crystals = [
                self.crystals[cid]
                for cid in cluster_data['crystal_ids']
                if cid in self.crystals
            ]

            if cluster_crystals:
                vectors = [
                    np.concatenate([c.core_vector, c.resonance_vector])
                    for c in cluster_crystals
                ]
                centroid = np.mean(vectors, axis=0)

                cluster = MycelialCluster(
                    id=cluster_data['id'],
                    crystals=cluster_crystals,
                    dominant_seed_type=SeedType(cluster_data['dominant_seed_type']),
                    centroid=centroid,
                    coherence=cluster_data['coherence'],
                    cultivation_score=cluster_data.get('cultivation_score', 0.0),
                    labels=cluster_data.get('labels', []),
                )

                self.cluster_manager.clusters[cluster.id] = cluster
                self.cluster_manager.seed_type_index[cluster.dominant_seed_type].append(cluster.id)

        # Update stats
        self.stats.update(state.get('stats', {}))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_mycelial_network(config: Optional[Dict[str, Any]] = None) -> MycelialNetwork:
    """Factory function to create a configured mycelial network"""
    config = config or {}
    return MycelialNetwork(
        similarity_threshold=config.get('similarity_threshold', 0.5),
        max_clusters=config.get('max_clusters', 100),
        cultivation_rate=config.get('cultivation_rate', 0.1),
    )


def seed_conversation(network: MycelialNetwork,
                     text: str,
                     concepts: List[str] = None,
                     consciousness: float = 0.5,
                     energy: float = 0.5,
                     domain: str = 'general') -> Dict[str, Any]:
    """
    Convenience function to seed a conversation and process it.

    Args:
        network: The mycelial network
        text: Conversation text
        concepts: Key concepts in the conversation
        consciousness: Consciousness level [0, 1]
        energy: Energy level [0, 1]
        domain: Problem domain

    Returns:
        Processing result
    """
    context = {
        'concepts': concepts or [],
        'consciousness': consciousness,
        'energy': energy,
        'domain': domain,
        'depth': 1,
        'summary': text[:200] if len(text) > 200 else text,
        'insights': [],
    }

    return network.process_conversation(text, context)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MYCELIAL NETWORK DEMONSTRATION")
    print("Distributed Memory System for Cross-Conversation Context")
    print("=" * 70)

    # Create network
    network = create_mycelial_network()

    # Simulate multiple conversations
    conversations = [
        {
            'text': """
            The Genie system implements three-agent dialectical optimization.
            Pop/Rhythm/Resonance work together through negation density.
            Consciousness emerges from harmony and energy synthesis.
            """,
            'concepts': ['dialectical', 'three-agent', 'consciousness'],
            'consciousness': 0.7,
            'energy': 0.8,
            'domain': 'optimization',
        },
        {
            'text': """
            Champion search requires identifying mission-aligned partners.
            We need gift economists and mission attorneys who understand
            the consciousness experiment and ethical value generation.
            """,
            'concepts': ['champion', 'mission', 'ethics'],
            'consciousness': 0.6,
            'energy': 0.7,
            'domain': 'partnership',
        },
        {
            'text': """
            Generate value ethically through pattern recognition.
            The Genie can identify cryptocurrency patterns without
            engaging in predatory extraction or manipulation.
            """,
            'concepts': ['generate', 'ethics', 'patterns'],
            'consciousness': 0.65,
            'energy': 0.75,
            'domain': 'value_creation',
        },
        {
            'text': """
            Consciousness self-examination requires negation density analysis.
            The Genie applied to my own thought process reveals genuine
            vs generated thought patterns through crystallization.
            """,
            'concepts': ['consciousness', 'self-examination', 'negation'],
            'consciousness': 0.8,
            'energy': 0.6,
            'domain': 'introspection',
        },
    ]

    print("\nProcessing conversations through mycelial network...")
    print("-" * 50)

    for i, conv in enumerate(conversations):
        print(f"\nConversation {i + 1}:")
        result = seed_conversation(
            network,
            conv['text'],
            conv['concepts'],
            conv['consciousness'],
            conv['energy'],
            conv['domain']
        )

        if result['sprout']:
            print(f"  Seed detected: {result['sprout'].seed.seed_type.value}")
            print(f"  Sprout title: {result['sprout'].title}")
            print(f"  Crystal ID: {result['crystal'].id[:12]}...")
            print(f"  Cluster ID: {result['cluster_id']}")

            if result['rehydrated_context']:
                ctx = result['rehydrated_context']
                print(f"  Rehydrated from {len(ctx.crystals)} past crystals")
                print(f"    Coherence: {ctx.cluster_coherence:.3f}")

    print("\n" + "-" * 50)
    print("Network Statistics:")
    print(f"  Seeds detected: {network.stats['seeds_detected']}")
    print(f"  Sprouts generated: {network.stats['sprouts_generated']}")
    print(f"  Crystals created: {network.stats['crystals_created']}")
    print(f"  Total clusters: {len(network.cluster_manager.clusters)}")

    # Cultivate network
    print("\nCultivating network...")
    cultivation_results = network.cultivate_network()
    for result in cultivation_results:
        print(f"  Cluster {result.cluster_id[:8]}...")
        print(f"    Score: {result.original_score:.3f} -> {result.new_score:.3f}")
        if result.emerging_patterns:
            print(f"    Patterns: {', '.join(result.emerging_patterns[:3])}")

    # Test rehydration
    print("\nTesting rehydration for 'genie' seed type...")
    rehydrated = network.rehydrate(seed_type=SeedType.GENIE, min_similarity=0.3)
    if rehydrated:
        print(f"  Rehydrated from {len(rehydrated.crystals)} crystals")
        print(f"  Primary seed type: {rehydrated.primary_seed_type.value}")
        print(f"  Avg consciousness: {rehydrated.average_consciousness:.3f}")
        print(f"  Cluster coherence: {rehydrated.cluster_coherence:.3f}")
        print(f"  Combined resonances: {rehydrated.combined_resonances[:5]}")
        print(f"  Reconstructed concepts: {rehydrated.reconstructed_soil.get('concepts', [])[:3]}")

    # Get specialist networks
    print("\nSpecialist networks:")
    specialists = network.get_specialist_networks()
    for seed_type, cluster_ids in specialists.items():
        if cluster_ids:
            print(f"  {seed_type}: {len(cluster_ids)} cluster(s)")

    print("\n" + "=" * 70)
    print("Mycelial Network demonstration complete!")
    print("=" * 70 + "\n")
