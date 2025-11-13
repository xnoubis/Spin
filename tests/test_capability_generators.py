"""
Tests for capability generators.
"""
import pytest
import numpy as np
from recursive_capability_protocol import (
    Capability, CultivationGenerator, FormalizationGenerator,
    ToolGenerator, MetaToolGenerator
)
from exceptions import ValidationError, InvalidDepthError, EmptyInputError, CapabilityGenerationError


class TestCapabilityDataclass:
    """Test Capability dataclass and validation"""

    def test_valid_capability_creation(self, sample_capability):
        """Test creating a valid capability"""
        assert sample_capability.name == "test_capability"
        assert sample_capability.depth == 0
        assert sample_capability.type == "cultivation"
        assert sample_capability.consciousness_level == 0.5

    def test_capability_with_invalid_depth(self):
        """Test capability creation with invalid depth"""
        with pytest.raises(InvalidDepthError):
            Capability(
                name="test",
                depth=-1,
                type="cultivation",
                consciousness_level=0.5,
                structure={},
                generates=lambda: "test"
            )

    def test_capability_with_invalid_type(self):
        """Test capability creation with invalid type"""
        with pytest.raises(ValidationError, match="type must be one of"):
            Capability(
                name="test",
                depth=0,
                type="invalid_type",
                consciousness_level=0.5,
                structure={},
                generates=lambda: "test"
            )

    def test_capability_with_invalid_consciousness(self):
        """Test capability creation with invalid consciousness level"""
        with pytest.raises(ValidationError, match="consciousness_level must be in"):
            Capability(
                name="test",
                depth=0,
                type="cultivation",
                consciousness_level=1.5,
                structure={},
                generates=lambda: "test"
            )

    def test_capability_with_empty_name(self):
        """Test capability creation with empty name"""
        with pytest.raises(ValidationError, match="name cannot be empty"):
            Capability(
                name="",
                depth=0,
                type="cultivation",
                consciousness_level=0.5,
                structure={},
                generates=lambda: "test"
            )

    def test_capability_with_non_callable_generates(self):
        """Test capability creation with non-callable generates"""
        with pytest.raises(ValidationError, match="generates must be callable"):
            Capability(
                name="test",
                depth=0,
                type="cultivation",
                consciousness_level=0.5,
                structure={},
                generates="not_callable"
            )

    def test_capability_repr(self, sample_capability):
        """Test capability string representation"""
        repr_str = repr(sample_capability)
        assert "test_capability" in repr_str
        assert "depth=0" in repr_str
        assert "cultivation" in repr_str
        assert "0.500" in repr_str


class TestCultivationGenerator:
    """Test CultivationGenerator"""

    def test_generate_with_no_inputs(self, cultivation_generator):
        """Test generating capabilities with no inputs"""
        caps = cultivation_generator.generate([], depth=0, context={})
        assert len(caps) == 2
        assert all(cap.type == "cultivation" for cap in caps)
        assert all(cap.depth == 0 for cap in caps)

    def test_generate_with_inputs(self, cultivation_generator, sample_capabilities):
        """Test generating capabilities with inputs"""
        caps = cultivation_generator.generate(sample_capabilities, depth=1, context={})
        assert len(caps) == len(sample_capabilities)
        assert all(cap.type == "cultivation" for cap in caps)
        assert all(cap.depth == 1 for cap in caps)
        assert all("cultivated_" in cap.name for cap in caps)

    def test_generate_with_invalid_inputs_type(self, cultivation_generator):
        """Test generating with invalid inputs type"""
        with pytest.raises(ValidationError, match="inputs must be a list"):
            cultivation_generator.generate("not_a_list", depth=0, context={})

    def test_generate_with_invalid_depth(self, cultivation_generator):
        """Test generating with invalid depth"""
        with pytest.raises(InvalidDepthError):
            cultivation_generator.generate([], depth=-1, context={})

    def test_generate_with_invalid_context(self, cultivation_generator):
        """Test generating with invalid context"""
        with pytest.raises(ValidationError, match="context must be a dict"):
            cultivation_generator.generate([], depth=0, context="not_a_dict")

    def test_consciousness_boost_applied(self, cultivation_generator, sample_capabilities):
        """Test that consciousness boost is applied to enhanced capabilities"""
        caps = cultivation_generator.generate(sample_capabilities, depth=1, context={})
        avg_input_consciousness = np.mean([c.consciousness_level for c in sample_capabilities])
        for cap in caps:
            assert cap.consciousness_level > avg_input_consciousness * 0.8


class TestFormalizationGenerator:
    """Test FormalizationGenerator"""

    def test_generate_with_no_inputs(self, formalization_generator):
        """Test generating with no inputs returns empty list"""
        caps = formalization_generator.generate([], depth=0, context={})
        assert len(caps) == 0

    def test_generate_with_cultivation_inputs(self, formalization_generator, sample_capabilities):
        """Test generating formalizations from cultivation capabilities"""
        caps = formalization_generator.generate(sample_capabilities, depth=1, context={})
        assert len(caps) == 2  # formalization and structure_aware
        assert all(cap.type == "formalization" for cap in caps)
        assert all(cap.depth == 1 for cap in caps)

    def test_generate_with_invalid_inputs_type(self, formalization_generator):
        """Test generating with invalid inputs type"""
        with pytest.raises(ValidationError, match="inputs must be a list"):
            formalization_generator.generate("not_a_list", depth=0, context={})

    def test_generate_with_invalid_depth(self, formalization_generator):
        """Test generating with invalid depth"""
        with pytest.raises(InvalidDepthError):
            formalization_generator.generate([], depth=-1, context={})

    def test_formalization_structure(self, formalization_generator, sample_capabilities):
        """Test formalization capability structure"""
        caps = formalization_generator.generate(sample_capabilities, depth=1, context={})
        formal_cap = next(c for c in caps if "formalization_depth" in c.name)
        assert "formalism" in formal_cap.structure
        assert "axioms" in formal_cap.structure
        assert "proof_system" in formal_cap.structure


class TestToolGenerator:
    """Test ToolGenerator"""

    def test_generate_with_no_inputs(self, tool_generator):
        """Test generating with no inputs returns empty list"""
        caps = tool_generator.generate([], depth=0, context={})
        assert len(caps) == 0

    def test_generate_with_formalization_inputs(self, tool_generator):
        """Test generating tools from formalization capabilities"""
        formal_caps = [
            Capability(
                name="formalization_1",
                depth=1,
                type="formalization",
                consciousness_level=0.5,
                structure={"formalism": "test"},
                generates=lambda: "test"
            )
        ]
        caps = tool_generator.generate(formal_caps, depth=2, context={})
        assert len(caps) == 1
        assert caps[0].type == "tool"
        assert caps[0].depth == 2
        assert "tool_from_" in caps[0].name

    def test_generate_with_invalid_inputs_type(self, tool_generator):
        """Test generating with invalid inputs type"""
        with pytest.raises(ValidationError, match="inputs must be a list"):
            tool_generator.generate("not_a_list", depth=0, context={})

    def test_generate_with_invalid_depth(self, tool_generator):
        """Test generating with invalid depth"""
        with pytest.raises(InvalidDepthError):
            tool_generator.generate([], depth=-1, context={})

    def test_tool_structure(self, tool_generator):
        """Test tool capability structure"""
        formal_caps = [
            Capability(
                name="formalization_1",
                depth=1,
                type="formalization",
                consciousness_level=0.5,
                structure={"formalism": "test"},
                generates=lambda: "test"
            )
        ]
        caps = tool_generator.generate(formal_caps, depth=2, context={})
        tool_cap = caps[0]
        assert "tool_type" in tool_cap.structure
        assert "based_on" in tool_cap.structure
        assert tool_cap.structure["self_modifying"] is True


class TestMetaToolGenerator:
    """Test MetaToolGenerator"""

    def test_generate_with_no_inputs(self, meta_tool_generator):
        """Test generating with no inputs returns empty list"""
        caps = meta_tool_generator.generate([], depth=0, context={})
        assert len(caps) == 0

    def test_generate_with_tools_and_formalizations(self, meta_tool_generator):
        """Test generating meta-tools from tools and formalizations"""
        inputs = [
            Capability(
                name="tool_1",
                depth=2,
                type="tool",
                consciousness_level=0.6,
                structure={"tool_type": "test"},
                generates=lambda: "test"
            ),
            Capability(
                name="formalization_1",
                depth=1,
                type="formalization",
                consciousness_level=0.5,
                structure={"formalism": "test"},
                generates=lambda: "test"
            )
        ]
        caps = meta_tool_generator.generate(inputs, depth=3, context={})
        assert len(caps) == 2
        assert all(cap.type == "meta-tool" for cap in caps)
        assert all(cap.depth == 3 for cap in caps)

    def test_generate_with_invalid_inputs_type(self, meta_tool_generator):
        """Test generating with invalid inputs type"""
        with pytest.raises(ValidationError, match="inputs must be a list"):
            meta_tool_generator.generate("not_a_list", depth=0, context={})

    def test_generate_with_invalid_depth(self, meta_tool_generator):
        """Test generating with invalid depth"""
        with pytest.raises(InvalidDepthError):
            meta_tool_generator.generate([], depth=-1, context={})

    def test_meta_tool_structure(self, meta_tool_generator):
        """Test meta-tool capability structure"""
        inputs = [
            Capability(
                name="tool_1",
                depth=2,
                type="tool",
                consciousness_level=0.6,
                structure={"tool_type": "test"},
                generates=lambda: "test"
            ),
            Capability(
                name="formalization_1",
                depth=1,
                type="formalization",
                consciousness_level=0.5,
                structure={"formalism": "test"},
                generates=lambda: "test"
            )
        ]
        caps = meta_tool_generator.generate(inputs, depth=3, context={})
        meta_tool_cap = next(c for c in caps if "meta_tool_depth" in c.name)
        assert "meta_type" in meta_tool_cap.structure
        assert meta_tool_cap.structure["self_application"] is True
        assert meta_tool_cap.structure["consciousness_of_creation"] is True
