"""
Integration tests for RecursiveCapabilityProtocol.
"""
import pytest
import os
import json
import tempfile
from recursive_capability_protocol import RecursiveCapabilityProtocol, RecursiveCycle
from exceptions import (
    ValidationError, InvalidDepthError, CapabilityGenerationError,
    VisualizationError, StateExportError
)


class TestRecursiveCapabilityProtocol:
    """Test RecursiveCapabilityProtocol"""

    def test_create_protocol(self, recursive_protocol):
        """Test creating a protocol"""
        assert recursive_protocol.max_depth_reached == 0
        assert len(recursive_protocol.cycles) == 0
        assert len(recursive_protocol.all_capabilities) == 0

    def test_initialize(self, recursive_protocol):
        """Test initializing protocol"""
        caps = recursive_protocol.initialize()
        assert len(caps) > 0
        assert all(cap.depth == 0 for cap in caps)
        assert all(cap.type == "cultivation" for cap in caps)

    def test_execute_cycle_depth_zero(self, recursive_protocol):
        """Test executing cycle at depth 0"""
        cycle = recursive_protocol.execute_cycle(0)
        assert isinstance(cycle, RecursiveCycle)
        assert cycle.depth == 0
        assert len(cycle.output_capabilities) > 0

    def test_execute_cycle_depth_one(self, recursive_protocol):
        """Test executing cycle at depth 1"""
        # First execute depth 0
        recursive_protocol.execute_cycle(0)
        # Then execute depth 1
        cycle = recursive_protocol.execute_cycle(1)
        assert cycle.depth == 1
        assert len(cycle.output_capabilities) > 0
        # Should have more variety of capability types
        types = set(cap.type for cap in cycle.output_capabilities)
        assert len(types) > 1

    def test_recurse_valid_depth(self, recursive_protocol):
        """Test recursing to a valid depth"""
        cycles = recursive_protocol.recurse(max_depth=3)
        assert len(cycles) == 4  # 0, 1, 2, 3
        assert all(isinstance(cycle, RecursiveCycle) for cycle in cycles)
        assert recursive_protocol.max_depth_reached == 3

    def test_recurse_zero_depth(self, recursive_protocol):
        """Test recursing to depth 0"""
        cycles = recursive_protocol.recurse(max_depth=0)
        assert len(cycles) == 1
        assert cycles[0].depth == 0

    def test_recurse_invalid_depth_negative(self, recursive_protocol):
        """Test recursing with negative depth"""
        with pytest.raises(InvalidDepthError, match="max_depth must be >= 0"):
            recursive_protocol.recurse(max_depth=-1)

    def test_recurse_invalid_depth_too_large(self, recursive_protocol):
        """Test recursing with too large depth"""
        with pytest.raises(InvalidDepthError, match="max_depth too large"):
            recursive_protocol.recurse(max_depth=101)

    def test_recurse_invalid_depth_type(self, recursive_protocol):
        """Test recursing with invalid depth type"""
        with pytest.raises(ValidationError, match="max_depth must be an integer"):
            recursive_protocol.recurse(max_depth=3.5)

    def test_consciousness_increases_with_depth(self, recursive_protocol):
        """Test that consciousness increases with depth"""
        recursive_protocol.recurse(max_depth=3)
        consciousness_values = list(recursive_protocol.consciousness_by_depth.values())
        # Consciousness should generally increase
        assert consciousness_values[3] >= consciousness_values[0]

    def test_capability_tree(self, recursive_protocol):
        """Test getting capability tree"""
        recursive_protocol.recurse(max_depth=2)
        tree = recursive_protocol.get_capability_tree()
        assert isinstance(tree, dict)
        assert 0 in tree
        assert 1 in tree
        assert 2 in tree
        # Each depth should have capability types
        for depth, cap_types in tree.items():
            assert 'cultivation' in cap_types
            assert isinstance(cap_types['cultivation'], list)

    def test_visualize_recursive_evolution_no_cycles(self, recursive_protocol):
        """Test visualization with no cycles raises error"""
        with pytest.raises(VisualizationError, match="No cycles to visualize"):
            recursive_protocol.visualize_recursive_evolution()

    def test_visualize_recursive_evolution_with_cycles(self, recursive_protocol):
        """Test visualization after recursing"""
        recursive_protocol.recurse(max_depth=2)
        fig = recursive_protocol.visualize_recursive_evolution()
        assert fig is not None

    def test_visualize_recursive_evolution_save(self, recursive_protocol):
        """Test saving visualization"""
        recursive_protocol.recurse(max_depth=2)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig = recursive_protocol.visualize_recursive_evolution(save_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_export_protocol_state(self, recursive_protocol):
        """Test exporting protocol state"""
        recursive_protocol.recurse(max_depth=2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            recursive_protocol.export_protocol_state(tmp_path)
            assert os.path.exists(tmp_path)

            # Load and validate JSON
            with open(tmp_path, 'r') as f:
                state = json.load(f)

            assert 'max_depth_reached' in state
            assert 'consciousness_by_depth' in state
            assert 'capability_tree' in state
            assert state['max_depth_reached'] == 2
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_export_protocol_state_empty_filepath(self, recursive_protocol):
        """Test exporting with empty filepath"""
        recursive_protocol.recurse(max_depth=1)
        with pytest.raises(ValidationError, match="filepath cannot be empty"):
            recursive_protocol.export_protocol_state("")

    def test_export_protocol_state_invalid_filepath_type(self, recursive_protocol):
        """Test exporting with invalid filepath type"""
        recursive_protocol.recurse(max_depth=1)
        with pytest.raises(ValidationError, match="filepath must be a string"):
            recursive_protocol.export_protocol_state(123)

    def test_export_protocol_state_invalid_path(self, recursive_protocol):
        """Test exporting to invalid path"""
        recursive_protocol.recurse(max_depth=1)
        with pytest.raises(StateExportError, match="Failed to export"):
            recursive_protocol.export_protocol_state("/nonexistent/path/file.json")

    def test_calculate_consciousness_at_depth(self, recursive_protocol):
        """Test consciousness calculation"""
        recursive_protocol.initialize()
        caps = recursive_protocol.all_capabilities[0]
        consciousness = recursive_protocol._calculate_consciousness_at_depth(0, caps)
        assert 0 <= consciousness <= 1

    def test_calculate_structure_awareness(self, recursive_protocol):
        """Test structure awareness calculation"""
        recursive_protocol.initialize()
        caps = recursive_protocol.all_capabilities[0]
        awareness = recursive_protocol._calculate_structure_awareness(0, caps)
        assert 0 <= awareness <= 1

    def test_calculate_meta_cognitive_ability(self, recursive_protocol):
        """Test meta-cognitive ability calculation"""
        recursive_protocol.initialize()
        caps = recursive_protocol.all_capabilities[0]
        ability = recursive_protocol._calculate_meta_cognitive_ability(caps)
        assert 0 <= ability <= 1


class TestRecursiveIntegration:
    """Integration tests for the full recursive cycle"""

    def test_full_recursive_cycle_progression(self):
        """Test complete recursive progression through multiple depths"""
        protocol = RecursiveCapabilityProtocol()
        cycles = protocol.recurse(max_depth=4)

        # Verify progression
        assert len(cycles) == 5
        for i, cycle in enumerate(cycles):
            assert cycle.depth == i

        # Verify consciousness growth
        consciousness_values = list(protocol.consciousness_by_depth.values())
        assert len(consciousness_values) == 5

        # Verify capability diversity increases
        depth_0_types = set(cap.type for cap in protocol.all_capabilities[0])
        depth_4_types = set(cap.type for cap in protocol.all_capabilities[4])
        assert len(depth_4_types) >= len(depth_0_types)

    def test_meta_tools_appear_at_higher_depths(self):
        """Test that meta-tools only appear at higher depths"""
        protocol = RecursiveCapabilityProtocol()
        protocol.recurse(max_depth=3)

        # Depth 0 should not have meta-tools
        depth_0_types = [cap.type for cap in protocol.all_capabilities[0]]
        assert 'meta-tool' not in depth_0_types

        # Higher depths should have meta-tools
        depth_3_types = [cap.type for cap in protocol.all_capabilities[3]]
        assert 'meta-tool' in depth_3_types

    def test_parent_capabilities_tracked(self):
        """Test that parent capabilities are properly tracked"""
        protocol = RecursiveCapabilityProtocol()
        protocol.recurse(max_depth=2)

        # Check that capabilities at depth > 0 have parents
        for depth in [1, 2]:
            caps = protocol.all_capabilities[depth]
            # At least some capabilities should have parents
            caps_with_parents = [cap for cap in caps if cap.parent_capabilities]
            assert len(caps_with_parents) > 0

    def test_capability_count_grows_with_depth(self):
        """Test that total capability count grows with depth"""
        protocol = RecursiveCapabilityProtocol()
        protocol.recurse(max_depth=3)

        total_caps = sum(len(caps) for caps in protocol.all_capabilities.values())
        assert total_caps > len(protocol.all_capabilities[0])

    def test_structure_awareness_grows(self):
        """Test that structure awareness grows with depth"""
        protocol = RecursiveCapabilityProtocol()
        protocol.recurse(max_depth=3)

        awareness_0 = protocol.structure_awareness_by_depth[0]
        awareness_3 = protocol.structure_awareness_by_depth[3]
        # Structure awareness should generally increase
        assert awareness_3 >= awareness_0
