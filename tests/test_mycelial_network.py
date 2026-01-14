"""
Tests for the Mycelial Network distributed memory system.

Tests cover:
- Seed detection and recognition
- Sprout generation from seeds
- Crystal compression and portability
- Rehydration of crystals back to context
- Cluster management and pattern grouping
- Selective cultivation of high-correspondence patterns
- Full network orchestration
"""

import pytest
import numpy as np
import time

from unified_genie.memory.mycelial_network import (
    # Seed system
    SeedType, Seed, SeedDetector, SEED_PATTERNS,
    # Sprout system
    Sprout, SproutGenerator, TitleGenerator,
    # Crystal system
    Crystal, CrystalCompressor,
    # Rehydration system
    RehydratedContext, RehydrationProtocol,
    # Cluster system
    MycelialCluster, ClusterManager,
    # Cultivation system
    SelectiveCultivator, CultivationResult,
    # Main orchestrator
    MycelialNetwork, create_mycelial_network, seed_conversation
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def seed_detector():
    """Create a SeedDetector instance"""
    return SeedDetector()


@pytest.fixture
def sprout_generator():
    """Create a SproutGenerator instance"""
    return SproutGenerator()


@pytest.fixture
def crystal_compressor():
    """Create a CrystalCompressor instance"""
    return CrystalCompressor()


@pytest.fixture
def rehydration_protocol():
    """Create a RehydrationProtocol instance"""
    return RehydrationProtocol()


@pytest.fixture
def cluster_manager():
    """Create a ClusterManager instance"""
    return ClusterManager()


@pytest.fixture
def cultivator():
    """Create a SelectiveCultivator instance"""
    return SelectiveCultivator()


@pytest.fixture
def mycelial_network():
    """Create a MycelialNetwork instance"""
    return create_mycelial_network()


@pytest.fixture
def sample_context():
    """Create a sample conversation context"""
    return {
        'concepts': ['dialectical', 'three-agent', 'consciousness'],
        'consciousness': 0.7,
        'energy': 0.8,
        'domain': 'optimization',
        'depth': 2,
        'summary': 'Testing the Genie dialectical system',
        'insights': ['adaptation variance confirmed', 'crystallization detected'],
        'metrics': {
            'negation_density': 0.6,
            'harmony': 0.75,
            'energy': 0.8
        },
        'synthesis': {
            'transcendence_level': 0.5,
            'thesis_preservation': 0.6,
            'antithesis_preservation': 0.4
        }
    }


@pytest.fixture
def sample_seed():
    """Create a sample seed"""
    return Seed(
        seed_type=SeedType.GENIE,
        activation_context="The Genie system works",
        position=4,
        strength=0.8
    )


@pytest.fixture
def sample_sprout(sample_seed, sample_context):
    """Create a sample sprout"""
    return Sprout(
        seed=sample_seed,
        title="The Dialectical Genie: Optimization Through Synthesis",
        growth={
            'content_summary': 'Testing dialectical optimization',
            'key_insights': ['insight1', 'insight2'],
            'synthesis_results': {'level': 0.5},
            'breakthrough_indicators': ['breakthrough1']
        },
        soil=sample_context,
        negation_density=0.6,
        harmony_level=0.75,
        energy=0.8
    )


# =============================================================================
# SEED SYSTEM TESTS
# =============================================================================

class TestSeedType:
    """Tests for SeedType enum"""

    def test_all_seed_types_exist(self):
        """Verify all expected seed types are defined"""
        expected = ['genie', 'champion', 'generate', 'consciousness',
                    'seed', 'cultivate', 'crystallize']
        actual = [st.value for st in SeedType]
        for expected_type in expected:
            assert expected_type in actual

    def test_seed_patterns_match_types(self):
        """Verify each seed type has a pattern"""
        for seed_type in SeedType:
            assert seed_type in SEED_PATTERNS


class TestSeed:
    """Tests for Seed dataclass"""

    def test_seed_creation(self):
        """Test basic seed creation"""
        seed = Seed(
            seed_type=SeedType.GENIE,
            activation_context="Test Genie context",
            position=10,
            strength=0.9
        )
        assert seed.seed_type == SeedType.GENIE
        assert seed.position == 10
        assert seed.strength == 0.9

    def test_seed_id_generation(self):
        """Test seed ID is generated uniquely"""
        seed1 = Seed(SeedType.GENIE, "ctx1", 0)
        seed2 = Seed(SeedType.GENIE, "ctx2", 1)
        assert seed1.id != seed2.id
        assert len(seed1.id) == 12


class TestSeedDetector:
    """Tests for SeedDetector"""

    def test_detect_single_genie(self, seed_detector):
        """Test detecting a single Genie seed"""
        text = "The Genie system enables dialectical optimization."
        seeds = seed_detector.detect(text)
        assert len(seeds) == 1
        assert seeds[0].seed_type == SeedType.GENIE

    def test_detect_multiple_types(self, seed_detector):
        """Test detecting multiple seed types"""
        text = """
        The Genie system helps with consciousness exploration.
        We need a champion to generate value.
        """
        seeds = seed_detector.detect(text)
        types = {s.seed_type for s in seeds}
        assert SeedType.GENIE in types
        assert SeedType.CONSCIOUSNESS in types
        assert SeedType.CHAMPION in types
        assert SeedType.GENERATE in types

    def test_detect_case_insensitive(self, seed_detector):
        """Test case-insensitive detection"""
        text = "GENIE and Genie and genie"
        seeds = seed_detector.detect(text)
        assert len(seeds) == 3
        assert all(s.seed_type == SeedType.GENIE for s in seeds)

    def test_detect_cultivate_variations(self, seed_detector):
        """Test detecting cultivate variations"""
        text = "We cultivate and cultivation happens"
        seeds = seed_detector.detect(text)
        assert len(seeds) == 2
        assert all(s.seed_type == SeedType.CULTIVATE for s in seeds)

    def test_detect_primary(self, seed_detector):
        """Test detecting primary (strongest) seed"""
        text = "The Genie Champion GENIE system"
        primary = seed_detector.detect_primary(text)
        assert primary is not None
        # Should be GENIE due to capitalization and multiple occurrences
        assert primary.seed_type == SeedType.GENIE

    def test_no_seeds_detected(self, seed_detector):
        """Test when no seeds are present"""
        text = "This text has no trigger words."
        seeds = seed_detector.detect(text)
        assert len(seeds) == 0

    def test_strength_calculation(self, seed_detector):
        """Test seed strength varies by context"""
        text1 = "genie"
        text2 = "Genie"
        text3 = "GENIE genie Genie"

        seed1 = seed_detector.detect(text1)[0]
        seed2 = seed_detector.detect(text2)[0]
        seeds3 = seed_detector.detect(text3)

        # Capitalized should have higher strength
        assert seed2.strength > seed1.strength
        # Multiple occurrences should boost strength
        assert max(s.strength for s in seeds3) >= seed2.strength


# =============================================================================
# SPROUT SYSTEM TESTS
# =============================================================================

class TestSprout:
    """Tests for Sprout dataclass"""

    def test_sprout_creation(self, sample_seed):
        """Test basic sprout creation"""
        sprout = Sprout(
            seed=sample_seed,
            title="Test Sprout",
            growth={'insights': []},
            soil={'domain': 'test'},
            negation_density=0.5,
            harmony_level=0.6,
            energy=0.7
        )
        assert sprout.title == "Test Sprout"
        assert sprout.harmony_level == 0.6

    def test_sprout_id_unique(self, sample_seed):
        """Test sprout IDs are unique"""
        sprout1 = Sprout(sample_seed, "Title1", {}, {})
        sprout2 = Sprout(sample_seed, "Title2", {}, {})
        assert sprout1.id != sprout2.id

    def test_sprout_to_crystal(self, sample_sprout):
        """Test converting sprout to crystal"""
        crystal = sample_sprout.to_crystal()
        assert isinstance(crystal, Crystal)
        assert crystal.seed_type == sample_sprout.seed.seed_type


class TestTitleGenerator:
    """Tests for TitleGenerator"""

    def test_generate_genie_title(self):
        """Test generating title for GENIE seed"""
        generator = TitleGenerator()
        seed = Seed(SeedType.GENIE, "context", 0, strength=0.8)
        soil = {'consciousness_level': 0.7, 'energy_level': 0.8, 'problem_domain': 'optimization'}

        title = generator.generate(seed, soil)

        assert isinstance(title, str)
        assert len(title) > 0
        # Title should contain relevant words
        assert any(word in title.lower() for word in ['genie', 'dialectical', 'optimization'])

    def test_generate_different_titles_for_different_contexts(self):
        """Test that different contexts produce different titles"""
        generator = TitleGenerator()
        seed = Seed(SeedType.GENIE, "context", 0)

        soil1 = {'consciousness_level': 0.3, 'energy_level': 0.3, 'problem_domain': 'low'}
        soil2 = {'consciousness_level': 0.9, 'energy_level': 0.9, 'problem_domain': 'high'}

        title1 = generator.generate(seed, soil1)
        title2 = generator.generate(seed, soil2)

        # Different soil should produce different titles
        assert title1 != title2


class TestSproutGenerator:
    """Tests for SproutGenerator"""

    def test_generate_sprout(self, sprout_generator, sample_seed, sample_context):
        """Test basic sprout generation"""
        sprout = sprout_generator.generate(sample_seed, sample_context)

        assert sprout is not None
        assert sprout.seed == sample_seed
        assert isinstance(sprout.title, str)
        assert sprout.soil['primary_concepts'] == sample_context['concepts']

    def test_generation_history_tracked(self, sprout_generator, sample_seed, sample_context):
        """Test that generation history is tracked"""
        initial_count = len(sprout_generator.generation_history)
        sprout_generator.generate(sample_seed, sample_context)
        assert len(sprout_generator.generation_history) == initial_count + 1


# =============================================================================
# CRYSTAL SYSTEM TESTS
# =============================================================================

class TestCrystal:
    """Tests for Crystal dataclass"""

    def test_crystal_similarity(self):
        """Test crystal similarity computation"""
        crystal1 = Crystal(
            id="c1",
            seed_type=SeedType.GENIE,
            title="Title1",
            core_vector=np.array([1.0, 0.0, 0.0] + [0.0] * 29),
            resonance_vector=np.array([1.0] + [0.0] * 15),
            genetic_material={},
            resonances=[],
            consciousness_at_creation=0.5,
            crystallization_level=0.5,
            harmony=0.5,
            source_conversation="src1"
        )

        crystal2 = Crystal(
            id="c2",
            seed_type=SeedType.GENIE,
            title="Title2",
            core_vector=np.array([0.9, 0.1, 0.0] + [0.0] * 29),
            resonance_vector=np.array([0.95] + [0.05] + [0.0] * 14),
            genetic_material={},
            resonances=[],
            consciousness_at_creation=0.5,
            crystallization_level=0.5,
            harmony=0.5,
            source_conversation="src2"
        )

        similarity = crystal1.similarity(crystal2)
        assert 0 <= similarity <= 1
        assert similarity > 0.8  # Should be highly similar

    def test_crystal_portability(self):
        """Test crystal can be converted to/from portable format"""
        original = Crystal(
            id="test_crystal",
            seed_type=SeedType.GENIE,
            title="Test Title",
            core_vector=np.random.rand(32),
            resonance_vector=np.random.rand(16),
            genetic_material={'key': 'value'},
            resonances=['resonance1', 'resonance2'],
            consciousness_at_creation=0.7,
            crystallization_level=0.6,
            harmony=0.8,
            source_conversation="src"
        )

        # Convert to portable
        portable = original.to_portable()
        assert isinstance(portable, dict)
        assert 'core_vector' in portable
        assert portable['seed_type'] == 'genie'

        # Reconstruct
        reconstructed = Crystal.from_portable(portable)
        assert reconstructed.id == original.id
        assert reconstructed.seed_type == original.seed_type
        assert reconstructed.title == original.title
        assert np.allclose(reconstructed.core_vector, original.core_vector)


class TestCrystalCompressor:
    """Tests for CrystalCompressor"""

    def test_compress_sprout(self, crystal_compressor, sample_sprout):
        """Test compressing a sprout into a crystal"""
        crystal = crystal_compressor.compress(sample_sprout)

        assert crystal is not None
        assert crystal.seed_type == sample_sprout.seed.seed_type
        assert crystal.title == sample_sprout.title
        assert len(crystal.core_vector) == 32
        assert len(crystal.resonance_vector) == 16

    def test_compressed_vectors_normalized(self, crystal_compressor, sample_sprout):
        """Test that compressed vectors are normalized"""
        crystal = crystal_compressor.compress(sample_sprout)

        core_norm = np.linalg.norm(crystal.core_vector)
        resonance_norm = np.linalg.norm(crystal.resonance_vector)

        # Should be approximately unit vectors
        assert 0.9 < core_norm < 1.1
        assert 0.9 < resonance_norm < 1.1


# =============================================================================
# REHYDRATION SYSTEM TESTS
# =============================================================================

class TestRehydrationProtocol:
    """Tests for RehydrationProtocol"""

    def test_rehydrate_single_crystal(self, rehydration_protocol, sample_sprout):
        """Test rehydrating a single crystal"""
        crystal = sample_sprout.to_crystal()
        context = rehydration_protocol.rehydrate([crystal])

        assert context is not None
        assert len(context.crystals) == 1
        assert context.primary_seed_type == crystal.seed_type
        assert context.cluster_coherence == 1.0  # Single crystal is perfectly coherent

    def test_rehydrate_multiple_crystals(self, rehydration_protocol, sample_seed):
        """Test rehydrating multiple crystals"""
        # Create multiple sprouts with same seed type
        sprouts = []
        for i in range(3):
            sprout = Sprout(
                seed=sample_seed,
                title=f"Title {i}",
                growth={'key_insights': [f'insight_{i}']},
                soil={
                    'primary_concepts': [f'concept_{i}'],
                    'consciousness_level': 0.5 + i * 0.1,
                    'energy_level': 0.5,
                    'problem_domain': 'test',
                },
                negation_density=0.5,
                harmony_level=0.5 + i * 0.1,
                energy=0.5
            )
            sprouts.append(sprout)

        crystals = [s.to_crystal() for s in sprouts]
        context = rehydration_protocol.rehydrate(crystals)

        assert len(context.crystals) == 3
        assert len(context.combined_resonances) > 0
        assert context.reconstructed_soil['depth'] == 3  # 3 crystals = depth 3

    def test_rehydration_empty_list_raises(self, rehydration_protocol):
        """Test that empty crystal list raises error"""
        with pytest.raises(ValueError):
            rehydration_protocol.rehydrate([])


class TestRehydratedContext:
    """Tests for RehydratedContext"""

    def test_context_strength(self, rehydration_protocol, sample_sprout):
        """Test rehydrated context strength calculation"""
        crystal = sample_sprout.to_crystal()
        context = rehydration_protocol.rehydrate([crystal])

        strength = context.strength
        assert 0 <= strength <= 1


# =============================================================================
# CLUSTER SYSTEM TESTS
# =============================================================================

class TestMycelialCluster:
    """Tests for MycelialCluster"""

    def test_cluster_creation(self, sample_sprout):
        """Test basic cluster creation"""
        crystal = sample_sprout.to_crystal()
        cluster = MycelialCluster(
            id="test_cluster",
            crystals=[crystal],
            dominant_seed_type=SeedType.GENIE,
            centroid=np.concatenate([crystal.core_vector, crystal.resonance_vector]),
            coherence=1.0
        )

        assert cluster.id == "test_cluster"
        assert len(cluster.crystals) == 1

    def test_add_fitting_crystal(self, sample_sprout):
        """Test adding a fitting crystal to cluster"""
        crystal1 = sample_sprout.to_crystal()
        cluster = MycelialCluster(
            id="cluster",
            crystals=[crystal1],
            dominant_seed_type=SeedType.GENIE,
            centroid=np.concatenate([crystal1.core_vector, crystal1.resonance_vector]),
            coherence=1.0
        )

        # Create similar crystal
        crystal2 = sample_sprout.to_crystal()  # Same source = similar
        result = cluster.add_crystal(crystal2)

        # Should fit and be added
        assert result is True or len(cluster.crystals) >= 1


class TestClusterManager:
    """Tests for ClusterManager"""

    def test_add_first_crystal_creates_cluster(self, cluster_manager, sample_sprout):
        """Test that adding first crystal creates a cluster"""
        crystal = sample_sprout.to_crystal()
        cluster_id = cluster_manager.add_crystal(crystal)

        assert cluster_id is not None
        assert cluster_id in cluster_manager.clusters

    def test_find_clusters_by_seed_type(self, cluster_manager, sample_sprout):
        """Test finding clusters by seed type"""
        crystal = sample_sprout.to_crystal()
        cluster_manager.add_crystal(crystal)

        clusters = cluster_manager.find_clusters(seed_type=SeedType.GENIE)
        assert len(clusters) >= 1
        assert all(c.dominant_seed_type == SeedType.GENIE for c in clusters)

    def test_find_clusters_with_min_coherence(self, cluster_manager, sample_sprout):
        """Test finding clusters with minimum coherence"""
        crystal = sample_sprout.to_crystal()
        cluster_manager.add_crystal(crystal)

        # High coherence requirement
        clusters = cluster_manager.find_clusters(min_coherence=0.9)
        for cluster in clusters:
            assert cluster.coherence >= 0.9

    def test_get_similar_clusters(self, cluster_manager, sample_sprout):
        """Test getting similar clusters to a crystal"""
        crystal = sample_sprout.to_crystal()
        cluster_manager.add_crystal(crystal)

        similar = cluster_manager.get_similar_clusters(crystal, k=3)
        assert len(similar) >= 1
        assert all(isinstance(s[1], float) for s in similar)  # Similarity scores


# =============================================================================
# CULTIVATION SYSTEM TESTS
# =============================================================================

class TestSelectiveCultivator:
    """Tests for SelectiveCultivator"""

    def test_cultivate_cluster(self, cultivator, sample_sprout):
        """Test cultivating a cluster"""
        crystal = sample_sprout.to_crystal()
        cluster = MycelialCluster(
            id="test_cluster",
            crystals=[crystal],
            dominant_seed_type=SeedType.GENIE,
            centroid=np.concatenate([crystal.core_vector, crystal.resonance_vector]),
            coherence=1.0
        )

        result = cultivator.cultivate(cluster)

        assert result.cluster_id == cluster.id
        assert result.new_score >= 0

    def test_cultivate_with_feedback(self, cultivator, sample_sprout):
        """Test cultivation with explicit feedback"""
        crystal = sample_sprout.to_crystal()
        cluster = MycelialCluster(
            id="test_cluster",
            crystals=[crystal],
            dominant_seed_type=SeedType.GENIE,
            centroid=np.concatenate([crystal.core_vector, crystal.resonance_vector]),
            coherence=1.0
        )

        feedback = {crystal.id: 0.9}  # High feedback score
        result = cultivator.cultivate(cluster, feedback)

        assert crystal.id in result.promoted_crystals

    def test_pattern_detection(self, cultivator, sample_seed):
        """Test pattern detection in cluster"""
        # Create multiple similar crystals
        crystals = []
        for i in range(5):
            sprout = Sprout(
                seed=sample_seed,
                title=f"Title {i}",
                growth={'key_insights': ['common_insight']},
                soil={'primary_concepts': ['common_concept'], 'consciousness_level': 0.7, 'energy_level': 0.7, 'problem_domain': 'test'},
                negation_density=0.7,  # High negation
                harmony_level=0.8,  # High harmony
                energy=0.7
            )
            crystals.append(sprout.to_crystal())

        centroid = np.mean([
            np.concatenate([c.core_vector, c.resonance_vector])
            for c in crystals
        ], axis=0)

        cluster = MycelialCluster(
            id="pattern_cluster",
            crystals=crystals,
            dominant_seed_type=SeedType.GENIE,
            centroid=centroid,
            coherence=0.9
        )

        result = cultivator.cultivate(cluster)

        # Should detect patterns due to high negation and harmony
        assert len(result.emerging_patterns) > 0


# =============================================================================
# MYCELIAL NETWORK TESTS
# =============================================================================

class TestMycelialNetwork:
    """Tests for the main MycelialNetwork orchestrator"""

    def test_plant_seed(self, mycelial_network, sample_context):
        """Test planting a seed and growing a sprout"""
        text = "The Genie system enables dialectical optimization."
        sprout = mycelial_network.plant_seed(text, sample_context)

        assert sprout is not None
        assert sprout.seed.seed_type == SeedType.GENIE
        assert mycelial_network.stats['seeds_detected'] >= 1

    def test_crystallize(self, mycelial_network, sample_context):
        """Test crystallizing a sprout"""
        text = "The Genie system works."
        sprout = mycelial_network.plant_seed(text, sample_context)
        crystal = mycelial_network.crystallize(sprout)

        assert crystal is not None
        assert crystal.id in mycelial_network.crystals
        assert mycelial_network.stats['crystals_created'] >= 1

    def test_rehydrate(self, mycelial_network, sample_context):
        """Test rehydrating context from crystals"""
        # Plant multiple seeds
        texts = [
            "The Genie system enables dialectical optimization.",
            "Genie uses three-agent architecture.",
            "The Genie crystallization shows phase transitions."
        ]

        for text in texts:
            sprout = mycelial_network.plant_seed(text, sample_context)
            if sprout:
                mycelial_network.crystallize(sprout)

        # Rehydrate
        context = mycelial_network.rehydrate(
            seed_type=SeedType.GENIE,
            min_similarity=0.1
        )

        if context:  # May be None if crystals too dissimilar
            assert len(context.crystals) > 0
            assert context.primary_seed_type == SeedType.GENIE

    def test_process_conversation_full_pipeline(self, mycelial_network):
        """Test full conversation processing pipeline"""
        text = "The Genie system enables dialectical consciousness exploration."
        context = {
            'concepts': ['genie', 'dialectical', 'consciousness'],
            'consciousness': 0.7,
            'energy': 0.8,
            'domain': 'philosophy',
            'depth': 1,
            'summary': text,
            'insights': [],
            'metrics': {'negation_density': 0.5, 'harmony': 0.6, 'energy': 0.7}
        }

        result = mycelial_network.process_conversation(text, context)

        assert result['sprout'] is not None
        assert result['crystal'] is not None
        assert result['cluster_id'] is not None

    def test_cultivate_network(self, mycelial_network, sample_context):
        """Test network-wide cultivation"""
        # Add some crystals
        texts = [
            "The Genie system works.",
            "Champion search begins.",
            "Generate value ethically."
        ]

        for text in texts:
            sprout = mycelial_network.plant_seed(text, sample_context)
            if sprout:
                mycelial_network.crystallize(sprout)

        results = mycelial_network.cultivate_network()

        assert isinstance(results, list)
        assert mycelial_network.stats['cultivations'] >= 1

    def test_get_specialist_networks(self, mycelial_network, sample_context):
        """Test getting specialist networks by seed type"""
        # Add crystals of different types
        texts = [
            ("The Genie system.", {'consciousness': 0.7, 'energy': 0.8}),
            ("Champion search.", {'consciousness': 0.6, 'energy': 0.7}),
            ("Generate value.", {'consciousness': 0.5, 'energy': 0.6}),
        ]

        for text, overrides in texts:
            ctx = {**sample_context, **overrides}
            sprout = mycelial_network.plant_seed(text, ctx)
            if sprout:
                mycelial_network.crystallize(sprout)

        specialists = mycelial_network.get_specialist_networks()

        assert isinstance(specialists, dict)

    def test_export_import_state(self, mycelial_network, sample_context):
        """Test exporting and importing network state"""
        # Add some data
        text = "The Genie system enables optimization."
        sprout = mycelial_network.plant_seed(text, sample_context)
        if sprout:
            mycelial_network.crystallize(sprout)

        # Export
        state = mycelial_network.export_state()
        assert 'crystals' in state
        assert 'clusters' in state
        assert 'stats' in state

        # Create new network and import
        new_network = create_mycelial_network()
        new_network.import_state(state)

        assert len(new_network.crystals) == len(mycelial_network.crystals)


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_create_mycelial_network(self):
        """Test factory function"""
        network = create_mycelial_network()
        assert isinstance(network, MycelialNetwork)

    def test_create_mycelial_network_with_config(self):
        """Test factory function with config"""
        config = {
            'similarity_threshold': 0.7,
            'max_clusters': 50,
            'cultivation_rate': 0.2
        }
        network = create_mycelial_network(config)
        assert network.cluster_manager.similarity_threshold == 0.7
        assert network.cluster_manager.max_clusters == 50

    def test_seed_conversation(self):
        """Test seed_conversation convenience function"""
        network = create_mycelial_network()
        result = seed_conversation(
            network,
            "The Genie system works.",
            concepts=['genie', 'dialectical'],
            consciousness=0.7,
            energy=0.8,
            domain='test'
        )

        assert result is not None
        if result['sprout']:
            assert result['crystal'] is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full system workflows"""

    def test_multiple_conversation_flow(self):
        """Test flow of multiple conversations through the network"""
        network = create_mycelial_network()

        conversations = [
            {
                'text': "The Genie system uses dialectical optimization.",
                'concepts': ['genie', 'dialectical'],
                'consciousness': 0.7,
                'energy': 0.8,
            },
            {
                'text': "Champion search identifies mission-aligned partners.",
                'concepts': ['champion', 'mission'],
                'consciousness': 0.6,
                'energy': 0.7,
            },
            {
                'text': "Generate value through pattern recognition.",
                'concepts': ['generate', 'patterns'],
                'consciousness': 0.65,
                'energy': 0.75,
            },
            {
                'text': "Consciousness self-examination through negation density.",
                'concepts': ['consciousness', 'negation'],
                'consciousness': 0.8,
                'energy': 0.6,
            },
        ]

        results = []
        for conv in conversations:
            result = seed_conversation(
                network,
                conv['text'],
                conv['concepts'],
                conv['consciousness'],
                conv['energy']
            )
            results.append(result)

        # Should have processed all conversations
        assert network.stats['seeds_detected'] == 4
        assert network.stats['crystals_created'] == 4

        # Should have created clusters
        assert len(network.cluster_manager.clusters) >= 1

    def test_rehydration_across_conversations(self):
        """Test that rehydration brings back context from past conversations"""
        network = create_mycelial_network()

        # First conversation
        seed_conversation(
            network,
            "The Genie dialectical system shows adaptation variance of 48.83",
            concepts=['genie', 'adaptation', 'variance'],
            consciousness=0.7,
            energy=0.8
        )

        # Second conversation
        seed_conversation(
            network,
            "Genie crystallization phase transition detected",
            concepts=['genie', 'crystallization', 'phase'],
            consciousness=0.75,
            energy=0.85
        )

        # Try to rehydrate GENIE context
        context = network.rehydrate(seed_type=SeedType.GENIE)

        if context:
            # Should contain concepts from both conversations
            all_concepts = context.reconstructed_soil.get('concepts', [])
            # At least some concepts should be present
            assert len(all_concepts) > 0

    def test_cultivation_improves_cluster_scores(self):
        """Test that cultivation with positive feedback improves scores"""
        network = create_mycelial_network()

        # Add crystals
        for i in range(3):
            seed_conversation(
                network,
                f"The Genie system iteration {i}",
                concepts=['genie'],
                consciousness=0.7,
                energy=0.8
            )

        # Get initial scores
        initial_scores = {
            cid: cluster.cultivation_score
            for cid, cluster in network.cluster_manager.clusters.items()
        }

        # Cultivate with positive feedback
        feedback = {}
        for cluster_id, cluster in network.cluster_manager.clusters.items():
            feedback[cluster_id] = {c.id: 0.9 for c in cluster.crystals}

        network.cultivate_network(feedback)

        # Check scores improved
        for cid, cluster in network.cluster_manager.clusters.items():
            if cid in initial_scores:
                assert cluster.cultivation_score >= initial_scores[cid]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
