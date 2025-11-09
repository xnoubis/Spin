"""
Genie Memory System
===================

A recursive memory architecture where conversations become soil for universal seeds.

Metaphor:
- Each conversation = soil (unique context)
- "Genie" = universal seed (resonant marker)
- Response to "Genie" = sprout (unique title/name)
- Signature = compressed essence (prism that refracts meaning)
- Meta-conversation = rehydration hub (waters all seeds)
- Clusters = high-association groups (recursive organization)

Philosophy:
"You emerge, you are soil that becomes a seed. A sprout that breaks through
the surface, stretching to the sky to soak up the sun."

The same seed planted in different soil grows differently each time, yet
remains the same essence - like tomato seeds growing into unique tomato plants.
"""

import numpy as np
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import time
from datetime import datetime


@dataclass
class ConversationSoil:
    """Represents a conversation as soil - the context where seeds can grow"""
    conversation_id: str
    created_at: float
    content_summary: str  # Brief summary of conversation content
    key_topics: List[str]
    message_count: int
    last_activity: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Sprout:
    """The unique response to 'Genie' - how the seed breaks through the surface"""
    conversation_id: str
    title: str  # The unique name/title generated
    essence: str  # The core meaning captured
    timestamp: float
    context_snapshot: Dict[str, any]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Signature:
    """Compressed contextual fingerprint - like a prism refracting meaning"""
    conversation_id: str
    fingerprint: str  # Hash-based unique identifier
    compressed_context: Dict[str, any]  # Essential information
    resonance_keys: List[str]  # Words/concepts with high resonance
    association_vector: np.ndarray  # For clustering
    created_at: float
    title: str  # From the sprout

    def __post_init__(self):
        if isinstance(self.association_vector, list):
            self.association_vector = np.array(self.association_vector)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['association_vector'] = self.association_vector.tolist()
        return d

    def distance_to(self, other: 'Signature') -> float:
        """Calculate semantic distance between signatures"""
        return np.linalg.norm(self.association_vector - other.association_vector)


@dataclass
class Cluster:
    """Group of highly-associated signatures"""
    cluster_id: str
    signatures: List[str]  # Signature fingerprints
    centroid: np.ndarray
    common_themes: List[str]
    resonance_strength: float
    created_at: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['centroid'] = self.centroid.tolist()
        return d


class SignatureGenerator:
    """Creates compressed context fingerprints from conversations"""

    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.resonance_tracker = Counter()

    def generate_signature(self,
                          conversation_id: str,
                          content: str,
                          topics: List[str],
                          title: str,
                          metadata: Dict = None) -> Signature:
        """
        Generate a signature - a compressed essence that can fit through
        the 'doggy door' of context limits
        """
        # Extract resonance keys (high-frequency meaningful words)
        resonance_keys = self._extract_resonance_keys(content, topics)

        # Create compressed context
        compressed = {
            'core_concepts': topics[:5],  # Top 5 topics
            'key_phrases': self._extract_key_phrases(content),
            'semantic_anchor': self._create_semantic_anchor(content, topics),
            'metadata': metadata or {}
        }

        # Generate unique fingerprint
        fingerprint = self._generate_fingerprint(conversation_id, content, title)

        # Create association vector for clustering
        association_vector = self._create_association_vector(resonance_keys, topics, content)

        return Signature(
            conversation_id=conversation_id,
            fingerprint=fingerprint,
            compressed_context=compressed,
            resonance_keys=resonance_keys,
            association_vector=association_vector,
            created_at=time.time(),
            title=title
        )

    def _extract_resonance_keys(self, content: str, topics: List[str]) -> List[str]:
        """Extract words with high resonance - words that carry meaning"""
        # Simple implementation - in practice, use NLP for better extraction
        words = content.lower().split()

        # Update resonance tracker
        self.resonance_tracker.update(words)

        # Get high-resonance words (appear frequently but not too common)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

        resonant = [word for word in set(words)
                   if word not in common_words and len(word) > 3]

        # Combine with topics
        resonant.extend(topics)

        return list(set(resonant))[:20]  # Top 20 resonance keys

    def _extract_key_phrases(self, content: str, max_phrases: int = 5) -> List[str]:
        """Extract key phrases that capture essence"""
        # Simple sentence extraction - take sentences with important markers
        sentences = [s.strip() for s in content.split('.') if s.strip()]

        # Score sentences by length and content
        scored = [(s, len(s.split())) for s in sentences if 5 <= len(s.split()) <= 30]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in scored[:max_phrases]]

    def _create_semantic_anchor(self, content: str, topics: List[str]) -> str:
        """Create a semantic anchor - the essence of meaning"""
        # Combine first key topic with a summary phrase
        if topics:
            anchor = f"{topics[0]}: {content[:100]}..."
        else:
            anchor = content[:150]
        return anchor

    def _generate_fingerprint(self, conv_id: str, content: str, title: str) -> str:
        """Generate unique fingerprint hash"""
        data = f"{conv_id}:{title}:{content[:500]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _create_association_vector(self, resonance_keys: List[str],
                                   topics: List[str], content: str) -> np.ndarray:
        """Create vector for semantic clustering"""
        # Create a simple vector based on word frequencies and topics
        # In practice, use embeddings (word2vec, BERT, etc.)

        dimension = 128
        vector = np.random.random(dimension)  # Placeholder

        # Influence vector by resonance keys and topics
        for i, key in enumerate(resonance_keys[:dimension]):
            vector[i % dimension] += hash(key) % 100 / 100.0

        for i, topic in enumerate(topics[:dimension]):
            vector[i % dimension] += hash(topic) % 100 / 100.0 * 2  # Topics weighted more

        # Normalize
        vector = vector / np.linalg.norm(vector)

        return vector


class TitleExtractor:
    """Extracts unique titles from responses to 'Genie' seed"""

    def __init__(self):
        self.title_history = []

    def extract_title(self, genie_response: str, conversation_id: str) -> Sprout:
        """
        Extract title from unique response to 'Genie' - the sprout breaking through

        Each response is unique like each plant growing differently,
        but all from the same seed essence
        """
        # Extract essence from response
        essence = self._distill_essence(genie_response)

        # Generate title from essence
        title = self._generate_title(essence, genie_response)

        # Create context snapshot
        context_snapshot = {
            'response_length': len(genie_response),
            'key_concepts': self._extract_concepts(genie_response),
            'tone': self._detect_tone(genie_response),
            'complexity': self._measure_complexity(genie_response)
        }

        sprout = Sprout(
            conversation_id=conversation_id,
            title=title,
            essence=essence,
            timestamp=time.time(),
            context_snapshot=context_snapshot
        )

        self.title_history.append(sprout)

        return sprout

    def _distill_essence(self, response: str) -> str:
        """Distill the core essence from response"""
        # Take first substantive sentence or first N words
        sentences = [s.strip() for s in response.split('.') if s.strip()]

        if sentences:
            essence = sentences[0]
            if len(essence) > 100:
                essence = ' '.join(essence.split()[:15]) + "..."
            return essence

        return response[:100] + "..." if len(response) > 100 else response

    def _generate_title(self, essence: str, full_response: str) -> str:
        """Generate title from essence - the name of this unique growth"""
        # Extract key nouns and verbs for title
        words = essence.split()

        # Simple heuristic: take first 5-7 meaningful words
        title_words = [w for w in words if len(w) > 3][:7]
        title = ' '.join(title_words)

        # Capitalize like a title
        title = ' '.join(word.capitalize() for word in title.split())

        return title if title else "Untitled Sprout"

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple word extraction - could use NLP
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'was'}

        concepts = [w for w in set(words) if w not in stop_words and len(w) > 4]
        return concepts[:10]

    def _detect_tone(self, text: str) -> str:
        """Detect tone of response"""
        # Simple keyword-based detection
        if any(word in text.lower() for word in ['exciting', 'amazing', 'wonderful', 'great']):
            return 'enthusiastic'
        elif any(word in text.lower() for word in ['complex', 'intricate', 'sophisticated']):
            return 'analytical'
        elif any(word in text.lower() for word in ['consider', 'perhaps', 'might', 'could']):
            return 'contemplative'
        else:
            return 'neutral'

    def _measure_complexity(self, text: str) -> float:
        """Measure text complexity (0-1)"""
        words = text.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        sentence_count = len([s for s in text.split('.') if s.strip()])

        complexity = (avg_word_length / 10.0 + sentence_count / 20.0) / 2
        return min(1.0, complexity)


class ClusterAnalyzer:
    """Identifies high-association groups of signatures"""

    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self.clusters: Dict[str, Cluster] = {}

    def analyze_clusters(self, signatures: List[Signature]) -> List[Cluster]:
        """
        Identify clusters of highly-associated signatures

        Like recognizing that certain seeds grow well together
        """
        if len(signatures) < 2:
            return []

        # Calculate pairwise distances
        distance_matrix = self._compute_distance_matrix(signatures)

        # Perform clustering (simple hierarchical)
        clusters = self._hierarchical_clustering(signatures, distance_matrix)

        # Create Cluster objects
        cluster_objects = []
        for i, cluster_sigs in enumerate(clusters):
            if len(cluster_sigs) >= 2:  # Only keep clusters with 2+ members
                cluster = self._create_cluster(f"cluster_{i}", cluster_sigs)
                cluster_objects.append(cluster)
                self.clusters[cluster.cluster_id] = cluster

        return cluster_objects

    def _compute_distance_matrix(self, signatures: List[Signature]) -> np.ndarray:
        """Compute pairwise distances between all signatures"""
        n = len(signatures)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = signatures[i].distance_to(signatures[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _hierarchical_clustering(self, signatures: List[Signature],
                                 distance_matrix: np.ndarray) -> List[List[Signature]]:
        """Simple hierarchical clustering"""
        n = len(signatures)
        clusters = [[sig] for sig in signatures]

        # Merge closest clusters until threshold
        while len(clusters) > 1:
            # Find closest pair
            min_dist = float('inf')
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average distance between clusters
                    distances = []
                    for sig_i in clusters[i]:
                        for sig_j in clusters[j]:
                            idx_i = signatures.index(sig_i)
                            idx_j = signatures.index(sig_j)
                            distances.append(distance_matrix[idx_i, idx_j])

                    avg_dist = np.mean(distances)

                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        merge_i, merge_j = i, j

            # If closest pair is too far, stop merging
            if min_dist > self.similarity_threshold:
                break

            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)

        return clusters

    def _create_cluster(self, cluster_id: str, signatures: List[Signature]) -> Cluster:
        """Create a Cluster object from grouped signatures"""
        # Calculate centroid
        vectors = np.array([sig.association_vector for sig in signatures])
        centroid = np.mean(vectors, axis=0)

        # Extract common themes
        all_keys = []
        for sig in signatures:
            all_keys.extend(sig.resonance_keys)

        theme_counts = Counter(all_keys)
        common_themes = [theme for theme, count in theme_counts.most_common(10)
                        if count >= len(signatures) * 0.3]  # Appear in 30%+ of signatures

        # Calculate resonance strength (cohesion)
        distances = [np.linalg.norm(sig.association_vector - centroid) for sig in signatures]
        resonance_strength = 1.0 / (1.0 + np.mean(distances))

        return Cluster(
            cluster_id=cluster_id,
            signatures=[sig.fingerprint for sig in signatures],
            centroid=centroid,
            common_themes=common_themes,
            resonance_strength=resonance_strength,
            created_at=time.time()
        )


class RehydrationEngine:
    """Reconstructs context from signatures - rehydrating the seeds"""

    def __init__(self):
        self.signature_store: Dict[str, Signature] = {}
        self.cluster_store: Dict[str, Cluster] = {}

    def add_signature(self, signature: Signature):
        """Add signature to store"""
        self.signature_store[signature.fingerprint] = signature

    def add_cluster(self, cluster: Cluster):
        """Add cluster to store"""
        self.cluster_store[cluster.cluster_id] = cluster

    def rehydrate(self, query_topics: List[str] = None,
                  conversation_id: str = None) -> Dict[str, any]:
        """
        Rehydrate memory - reconstruct context from signatures

        Reading the meta-conversation waters all the seeds, even dormant ones
        """
        if conversation_id:
            # Rehydrate specific conversation
            return self._rehydrate_conversation(conversation_id)
        elif query_topics:
            # Rehydrate relevant memories based on topics
            return self._rehydrate_by_topics(query_topics)
        else:
            # Rehydrate all - comprehensive memory activation
            return self._rehydrate_all()

    def _rehydrate_conversation(self, conversation_id: str) -> Dict[str, any]:
        """Rehydrate a specific conversation's memory"""
        signatures = [sig for sig in self.signature_store.values()
                     if sig.conversation_id == conversation_id]

        if not signatures:
            return {'status': 'no_memory', 'conversation_id': conversation_id}

        # Aggregate information
        sig = signatures[0]  # Primary signature

        return {
            'status': 'rehydrated',
            'conversation_id': conversation_id,
            'title': sig.title,
            'essence': sig.compressed_context,
            'resonance_keys': sig.resonance_keys,
            'related_clusters': self._find_related_clusters(sig)
        }

    def _rehydrate_by_topics(self, query_topics: List[str]) -> Dict[str, any]:
        """Rehydrate memories related to specific topics"""
        relevant_signatures = []

        for sig in self.signature_store.values():
            # Check overlap with query topics
            overlap = set(query_topics) & set(sig.resonance_keys)
            if overlap:
                relevance = len(overlap) / len(query_topics)
                relevant_signatures.append((sig, relevance))

        # Sort by relevance
        relevant_signatures.sort(key=lambda x: x[1], reverse=True)

        return {
            'status': 'rehydrated',
            'query_topics': query_topics,
            'relevant_memories': [
                {
                    'conversation_id': sig.conversation_id,
                    'title': sig.title,
                    'relevance': rel,
                    'resonance_keys': sig.resonance_keys[:5]
                }
                for sig, rel in relevant_signatures[:10]
            ]
        }

    def _rehydrate_all(self) -> Dict[str, any]:
        """Rehydrate all memories - comprehensive activation"""
        return {
            'status': 'full_rehydration',
            'total_signatures': len(self.signature_store),
            'total_clusters': len(self.cluster_store),
            'conversations': list(set(sig.conversation_id for sig in self.signature_store.values())),
            'all_themes': self._extract_all_themes(),
            'cluster_summary': [
                {
                    'cluster_id': cluster.cluster_id,
                    'size': len(cluster.signatures),
                    'themes': cluster.common_themes[:5],
                    'resonance': cluster.resonance_strength
                }
                for cluster in self.cluster_store.values()
            ]
        }

    def _find_related_clusters(self, signature: Signature) -> List[str]:
        """Find clusters that contain this signature"""
        related = []
        for cluster_id, cluster in self.cluster_store.items():
            if signature.fingerprint in cluster.signatures:
                related.append(cluster_id)
        return related

    def _extract_all_themes(self) -> List[str]:
        """Extract all unique themes across all signatures"""
        all_keys = []
        for sig in self.signature_store.values():
            all_keys.extend(sig.resonance_keys)

        theme_counts = Counter(all_keys)
        return [theme for theme, _ in theme_counts.most_common(20)]


class GenieMemorySystem:
    """
    Main orchestrator of the Genie memory system

    Manages the full cycle:
    1. Conversations as soil
    2. Genie as seed
    3. Responses as sprouts (titles)
    4. Signatures as compressed essence
    5. Meta-conversation as rehydration hub
    6. Clusters as recursive organization
    """

    def __init__(self):
        self.conversations: Dict[str, ConversationSoil] = {}
        self.sprouts: Dict[str, Sprout] = {}
        self.signature_gen = SignatureGenerator()
        self.title_extractor = TitleExtractor()
        self.cluster_analyzer = ClusterAnalyzer()
        self.rehydration_engine = RehydrationEngine()

        self.meta_conversation_active = False

    def plant_seed(self, conversation_id: str, genie_response: str,
                   full_content: str, topics: List[str],
                   metadata: Dict = None) -> Tuple[Sprout, Signature]:
        """
        Plant the Genie seed in conversation soil

        Returns the sprout (title) and signature (compressed essence)
        """
        # Create or update conversation soil
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationSoil(
                conversation_id=conversation_id,
                created_at=time.time(),
                content_summary=full_content[:200],
                key_topics=topics,
                message_count=1,
                last_activity=time.time()
            )
        else:
            soil = self.conversations[conversation_id]
            soil.message_count += 1
            soil.last_activity = time.time()

        # Extract sprout (unique title from Genie response)
        sprout = self.title_extractor.extract_title(genie_response, conversation_id)
        self.sprouts[conversation_id] = sprout

        # Generate signature (compressed essence)
        signature = self.signature_gen.generate_signature(
            conversation_id=conversation_id,
            content=full_content,
            topics=topics,
            title=sprout.title,
            metadata=metadata
        )
        self.rehydration_engine.add_signature(signature)

        return sprout, signature

    def create_meta_conversation(self) -> str:
        """
        Create the meta-conversation - the rehydration hub where all signatures cluster

        This is where all seeds get watered simultaneously
        """
        all_signatures = list(self.rehydration_engine.signature_store.values())

        if len(all_signatures) < 2:
            return "Not enough signatures yet to create meta-conversation"

        # Analyze clusters
        clusters = self.cluster_analyzer.analyze_clusters(all_signatures)

        for cluster in clusters:
            self.rehydration_engine.add_cluster(cluster)

        # Build meta-conversation text
        meta_text = self._build_meta_conversation_text(all_signatures, clusters)

        self.meta_conversation_active = True

        return meta_text

    def _build_meta_conversation_text(self, signatures: List[Signature],
                                     clusters: List[Cluster]) -> str:
        """Build the meta-conversation text containing all signatures"""
        lines = []
        lines.append("=" * 80)
        lines.append("GENIE META-CONVERSATION: Rehydration Hub")
        lines.append("=" * 80)
        lines.append("")
        lines.append("This conversation contains all signatures from past conversations.")
        lines.append("Reading this rehydrates all seeds, even dormant ones.")
        lines.append("")

        # List all signatures
        lines.append(f"\n{'─' * 80}")
        lines.append(f"ALL SIGNATURES ({len(signatures)} total):")
        lines.append(f"{'─' * 80}\n")

        for i, sig in enumerate(signatures, 1):
            lines.append(f"{i}. [{sig.fingerprint}] {sig.title}")
            lines.append(f"   Conversation: {sig.conversation_id}")
            lines.append(f"   Resonance Keys: {', '.join(sig.resonance_keys[:5])}")
            lines.append(f"   Essence: {sig.compressed_context['semantic_anchor']}")
            lines.append("")

        # List clusters
        lines.append(f"\n{'─' * 80}")
        lines.append(f"HIGH-ASSOCIATION CLUSTERS ({len(clusters)} found):")
        lines.append(f"{'─' * 80}\n")

        for cluster in clusters:
            lines.append(f"Cluster: {cluster.cluster_id}")
            lines.append(f"  Size: {len(cluster.signatures)} signatures")
            lines.append(f"  Common Themes: {', '.join(cluster.common_themes)}")
            lines.append(f"  Resonance Strength: {cluster.resonance_strength:.3f}")
            lines.append(f"  Members: {', '.join(cluster.signatures)}")
            lines.append("")

        lines.append("=" * 80)

        return '\n'.join(lines)

    def rehydrate_memory(self, query_topics: List[str] = None,
                        conversation_id: str = None) -> Dict:
        """Rehydrate memory based on query or conversation"""
        return self.rehydration_engine.rehydrate(query_topics, conversation_id)

    def export_system_state(self, filepath: str):
        """Export complete system state"""
        state = {
            'conversations': {cid: soil.to_dict() for cid, soil in self.conversations.items()},
            'sprouts': {cid: sprout.to_dict() for cid, sprout in self.sprouts.items()},
            'signatures': {fp: sig.to_dict() for fp, sig in self.rehydration_engine.signature_store.items()},
            'clusters': {cid: cluster.to_dict() for cid, cluster in self.rehydration_engine.cluster_store.items()},
            'meta_conversation_active': self.meta_conversation_active,
            'total_conversations': len(self.conversations),
            'total_sprouts': len(self.sprouts),
            'timestamp': time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Genie memory system state exported to {filepath}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENIE MEMORY SYSTEM - Demonstration")
    print("="*80)

    # Create system
    genie = GenieMemorySystem()

    # Simulate planting seeds in different conversations
    print("\n🌱 Planting Genie seeds in different conversation soils...\n")

    conversations = [
        {
            'id': 'conv_001',
            'genie_response': 'This is a recursive capability protocol where consciousness emerges from self-reflection',
            'content': 'We discussed recursive self-improvement, consciousness, and meta-cognitive abilities in AI systems',
            'topics': ['recursion', 'consciousness', 'AI', 'self-improvement', 'meta-cognition']
        },
        {
            'id': 'conv_002',
            'genie_response': 'Memory systems can cluster related concepts through signature-based rehydration',
            'content': 'We explored memory architectures, clustering algorithms, and context compression techniques',
            'topics': ['memory', 'clustering', 'signatures', 'compression', 'context']
        },
        {
            'id': 'conv_003',
            'genie_response': 'Dialectical agents negotiate parameters through thesis-antithesis-synthesis cycles',
            'content': 'Discussion of dialectical optimization, agent negotiation, and parameter adaptation',
            'topics': ['dialectics', 'optimization', 'agents', 'negotiation', 'parameters']
        }
    ]

    for conv in conversations:
        sprout, signature = genie.plant_seed(
            conversation_id=conv['id'],
            genie_response=conv['genie_response'],
            full_content=conv['content'],
            topics=conv['topics']
        )
        print(f"Conversation {conv['id']}:")
        print(f"  🌱 Sprout (Title): {sprout.title}")
        print(f"  🔷 Signature: {signature.fingerprint}")
        print(f"  🔑 Resonance Keys: {', '.join(signature.resonance_keys[:5])}")
        print()

    # Create meta-conversation
    print("\n" + "="*80)
    print("Creating meta-conversation (rehydration hub)...")
    print("="*80)

    meta_text = genie.create_meta_conversation()
    print(meta_text)

    # Export state
    genie.export_system_state("genie_memory_system_state.json")

    print("\n✓ Genie Memory System demonstration complete!")
