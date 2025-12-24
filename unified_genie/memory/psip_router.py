"""
PSIP Router: Perceptual Signature Compression and Retrieval
============================================================

The PSIP (Perceptual Signature Indexing Protocol) Router manages the compression
and retrieval of agent signatures and memory artifacts. It provides the memory
backbone for the unified genie system.

Core Concepts:
- Signatures: Compressed representations of agent states
- Routing: Efficient retrieval of relevant memories
- Compression: Lossy encoding that preserves essential structure
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import hashlib
import time
import json


@dataclass
class Signature:
    """
    A compressed perceptual signature representing an agent state or memory.

    Signatures are the atoms of memory - small, comparable, and composable.
    """
    id: str
    vector: np.ndarray  # Compressed feature vector
    source: str  # Origin agent (builder, validator, meta_validator)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0  # Memory strength (decays over time)

    def __hash__(self):
        return hash(self.id)

    def similarity(self, other: 'Signature') -> float:
        """Compute cosine similarity with another signature"""
        dot = np.dot(self.vector, other.vector)
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self < 1e-10 or norm_other < 1e-10:
            return 0.0
        return dot / (norm_self * norm_other)


@dataclass
class Route:
    """A route connecting signatures through association"""
    source_id: str
    target_id: str
    weight: float  # Association strength
    route_type: str  # 'causal', 'temporal', 'semantic'
    created: float = field(default_factory=time.time)


class SignatureCompressor:
    """
    Compresses rich agent states into compact signature vectors.

    Uses dimensionality reduction while preserving perceptual structure.
    """

    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.compression_history: List[Dict] = []

    def compress(self, state: Dict[str, Any], source: str) -> Signature:
        """
        Compress an agent state into a signature.

        Preserves:
        - Consciousness level (scaled)
        - Energy patterns
        - Phase information
        - Structural fingerprint
        """
        # Extract key features
        features = []

        # Consciousness features
        consciousness = state.get('consciousness', state.get('consciousness_level', 0.5))
        features.extend([
            consciousness,
            np.sin(consciousness * np.pi),
            np.cos(consciousness * np.pi)
        ])

        # Energy features
        energy = state.get('energy', state.get('energy_level', 0.5))
        features.extend([
            energy,
            energy ** 2,
            np.sqrt(max(0, energy))
        ])

        # Phase features (if available)
        phase = state.get('phase', state.get('breathing_phase', 0.0))
        features.extend([
            np.sin(phase),
            np.cos(phase),
            np.sin(2 * phase),
            np.cos(2 * phase)
        ])

        # Velocity/frequency features
        velocity = state.get('velocity', state.get('generative_velocity', 0.5))
        frequency = state.get('frequency', state.get('natural_frequency', 1.0))
        features.extend([velocity, frequency, velocity * frequency])

        # Crystallization features
        crystal = state.get('crystallization', state.get('crystallization_level', 0.0))
        harmony = state.get('harmony', state.get('harmony_level', 0.5))
        features.extend([crystal, harmony, crystal * harmony])

        # Pad or truncate to target dimension
        features = np.array(features, dtype=np.float32)
        if len(features) < self.target_dim:
            padding = np.random.randn(self.target_dim - len(features)) * 0.01
            features = np.concatenate([features, padding])
        elif len(features) > self.target_dim:
            features = features[:self.target_dim]

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 1e-10:
            features = features / norm

        # Generate ID
        sig_id = self._generate_id(source, features)

        # Record compression
        self.compression_history.append({
            'id': sig_id,
            'source': source,
            'original_keys': list(state.keys()),
            'timestamp': time.time()
        })

        return Signature(
            id=sig_id,
            vector=features,
            source=source,
            metadata={'original_state': state}
        )

    def _generate_id(self, source: str, vector: np.ndarray) -> str:
        """Generate unique signature ID"""
        data = f"{source}_{time.time()}_{vector.tobytes().hex()[:16]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class PSIPRouter:
    """
    The main PSIP Router - manages signature storage, routing, and retrieval.

    Provides:
    - Signature storage with automatic decay
    - Similarity-based retrieval
    - Association routing between signatures
    - Memory consolidation
    """

    def __init__(self, capacity: int = 10000, decay_rate: float = 0.001):
        self.capacity = capacity
        self.decay_rate = decay_rate

        # Core storage
        self.signatures: Dict[str, Signature] = {}
        self.routes: List[Route] = []

        # Indices for fast retrieval
        self.source_index: Dict[str, List[str]] = {}  # source -> [sig_ids]
        self.temporal_index: deque = deque(maxlen=capacity)

        # Compressor
        self.compressor = SignatureCompressor()

        # Statistics
        self.stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'consolidations': 0
        }

    def store(self, state: Dict[str, Any], source: str) -> Signature:
        """
        Store an agent state as a compressed signature.

        Returns the created signature.
        """
        # Compress state
        signature = self.compressor.compress(state, source)

        # Store signature
        self.signatures[signature.id] = signature

        # Update indices
        if source not in self.source_index:
            self.source_index[source] = []
        self.source_index[source].append(signature.id)

        self.temporal_index.append(signature.id)

        # Update stats
        self.stats['total_stored'] += 1

        # Check capacity
        if len(self.signatures) > self.capacity:
            self._evict_weakest()

        return signature

    def retrieve(self, query: Union[Signature, Dict[str, Any], np.ndarray],
                 k: int = 5, source_filter: Optional[str] = None) -> List[Signature]:
        """
        Retrieve k most similar signatures to the query.

        Args:
            query: Signature, state dict, or raw vector
            k: Number of signatures to retrieve
            source_filter: Optional filter by source agent
        """
        # Convert query to vector
        if isinstance(query, Signature):
            query_vector = query.vector
        elif isinstance(query, dict):
            temp_sig = self.compressor.compress(query, 'query')
            query_vector = temp_sig.vector
        else:
            query_vector = np.array(query)

        # Filter candidates
        if source_filter:
            candidate_ids = self.source_index.get(source_filter, [])
        else:
            candidate_ids = list(self.signatures.keys())

        # Compute similarities
        similarities = []
        for sig_id in candidate_ids:
            sig = self.signatures.get(sig_id)
            if sig:
                sim = np.dot(query_vector, sig.vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(sig.vector) + 1e-10
                )
                similarities.append((sig, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        self.stats['total_retrieved'] += 1
        return [sig for sig, _ in similarities[:k]]

    def route(self, source_sig: Signature, target_sig: Signature,
              route_type: str = 'semantic', weight: float = 1.0):
        """Create a route (association) between two signatures"""
        route = Route(
            source_id=source_sig.id,
            target_id=target_sig.id,
            weight=weight,
            route_type=route_type
        )
        self.routes.append(route)

    def follow_routes(self, sig_id: str, depth: int = 2) -> List[Signature]:
        """Follow routes from a signature to find associated memories"""
        visited = set()
        result = []

        def _follow(current_id: str, remaining_depth: int):
            if remaining_depth <= 0 or current_id in visited:
                return
            visited.add(current_id)

            for route in self.routes:
                if route.source_id == current_id:
                    target = self.signatures.get(route.target_id)
                    if target and target.id not in visited:
                        result.append(target)
                        _follow(target.id, remaining_depth - 1)

        _follow(sig_id, depth)
        return result

    def decay_memories(self):
        """Apply decay to all memory strengths"""
        for sig in self.signatures.values():
            sig.strength *= (1 - self.decay_rate)

    def consolidate(self):
        """
        Consolidate similar memories to reduce storage.

        Merges very similar signatures into representative prototypes.
        """
        self.stats['consolidations'] += 1

        # Group signatures by source
        for source, sig_ids in self.source_index.items():
            if len(sig_ids) < 10:
                continue

            # Find similar pairs
            sigs = [self.signatures[sid] for sid in sig_ids if sid in self.signatures]
            to_merge = []

            for i, sig1 in enumerate(sigs):
                for sig2 in sigs[i+1:]:
                    if sig1.similarity(sig2) > 0.95:
                        to_merge.append((sig1, sig2))

            # Merge pairs
            for sig1, sig2 in to_merge[:10]:  # Limit merges per consolidation
                self._merge_signatures(sig1, sig2)

    def _merge_signatures(self, sig1: Signature, sig2: Signature):
        """Merge two similar signatures"""
        # Create merged vector
        total_strength = sig1.strength + sig2.strength
        if total_strength > 0:
            merged_vector = (
                sig1.vector * sig1.strength +
                sig2.vector * sig2.strength
            ) / total_strength
        else:
            merged_vector = (sig1.vector + sig2.vector) / 2

        # Keep the stronger signature
        if sig1.strength >= sig2.strength:
            sig1.vector = merged_vector
            sig1.strength = total_strength
            del self.signatures[sig2.id]
        else:
            sig2.vector = merged_vector
            sig2.strength = total_strength
            del self.signatures[sig1.id]

    def _evict_weakest(self):
        """Evict the weakest signature to make room"""
        if not self.signatures:
            return

        weakest = min(self.signatures.values(), key=lambda s: s.strength)
        source = weakest.source

        del self.signatures[weakest.id]

        if source in self.source_index and weakest.id in self.source_index[source]:
            self.source_index[source].remove(weakest.id)

    def get_recent(self, n: int = 10) -> List[Signature]:
        """Get n most recent signatures"""
        recent_ids = list(self.temporal_index)[-n:]
        return [self.signatures[sid] for sid in recent_ids if sid in self.signatures]

    def export_state(self) -> Dict[str, Any]:
        """Export router state for persistence"""
        return {
            'signatures': {
                sid: {
                    'vector': sig.vector.tolist(),
                    'source': sig.source,
                    'timestamp': sig.timestamp,
                    'strength': sig.strength,
                    'metadata': {k: v for k, v in sig.metadata.items()
                                if k != 'original_state'}
                }
                for sid, sig in self.signatures.items()
            },
            'routes': [
                {
                    'source': r.source_id,
                    'target': r.target_id,
                    'weight': r.weight,
                    'type': r.route_type
                }
                for r in self.routes
            ],
            'stats': self.stats
        }


class ArtifactStore:
    """
    Store for memory artifacts (from HTML/JSON sources).

    Manages structured artifacts that can be indexed by the router.
    """

    def __init__(self, router: Optional[PSIPRouter] = None):
        self.router = router or PSIPRouter()
        self.artifacts: Dict[str, Dict] = {}

    def store_artifact(self, artifact: Dict[str, Any],
                       artifact_type: str = 'general') -> str:
        """
        Store an artifact and create router signatures.

        Returns artifact ID.
        """
        # Generate ID
        artifact_id = hashlib.sha256(
            json.dumps(artifact, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # Store artifact
        self.artifacts[artifact_id] = {
            'content': artifact,
            'type': artifact_type,
            'created': time.time()
        }

        # Create signature for routing
        signature_state = {
            'artifact_id': artifact_id,
            'artifact_type': artifact_type,
            **{k: v for k, v in artifact.items()
               if isinstance(v, (int, float))}
        }
        self.router.store(signature_state, f'artifact_{artifact_type}')

        return artifact_id

    def retrieve_artifact(self, artifact_id: str) -> Optional[Dict]:
        """Retrieve an artifact by ID"""
        entry = self.artifacts.get(artifact_id)
        return entry['content'] if entry else None

    def search_artifacts(self, query: Dict[str, Any], k: int = 5) -> List[Dict]:
        """Search for similar artifacts using the router"""
        signatures = self.router.retrieve(query, k=k)
        results = []

        for sig in signatures:
            artifact_id = sig.metadata.get('original_state', {}).get('artifact_id')
            if artifact_id and artifact_id in self.artifacts:
                results.append(self.artifacts[artifact_id]['content'])

        return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PSIP ROUTER DEMONSTRATION")
    print("="*70)

    router = PSIPRouter()

    # Store some agent states
    states = [
        {'consciousness': 0.3, 'energy': 0.8, 'phase': 0.5, 'source': 'builder'},
        {'consciousness': 0.5, 'energy': 0.6, 'phase': 1.0, 'source': 'validator'},
        {'consciousness': 0.7, 'energy': 0.4, 'crystallization': 0.8, 'source': 'meta'},
        {'consciousness': 0.4, 'energy': 0.7, 'phase': 0.6, 'source': 'builder'},
    ]

    print("\nStoring signatures...")
    stored = []
    for state in states:
        source = state.pop('source')
        sig = router.store(state, source)
        stored.append(sig)
        print(f"  Stored: {sig.id[:8]}... from {source}")

    # Create routes
    print("\nCreating routes...")
    router.route(stored[0], stored[3], 'temporal')
    router.route(stored[1], stored[2], 'causal')

    # Retrieve similar
    print("\nRetrieving similar to first signature...")
    query = {'consciousness': 0.35, 'energy': 0.75, 'phase': 0.55}
    similar = router.retrieve(query, k=3)
    for sig in similar:
        print(f"  Found: {sig.id[:8]}... (source: {sig.source})")

    # Follow routes
    print("\nFollowing routes from first signature...")
    associated = router.follow_routes(stored[0].id)
    for sig in associated:
        print(f"  Associated: {sig.id[:8]}... (source: {sig.source})")

    print(f"\nRouter stats: {router.stats}")
    print("="*70 + "\n")
