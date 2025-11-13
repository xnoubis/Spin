"""
Pytest configuration and fixtures for the test suite.
"""
import pytest
import numpy as np
from recursive_capability_protocol import (
    Capability, RecursiveCapabilityProtocol,
    CultivationGenerator, FormalizationGenerator,
    ToolGenerator, MetaToolGenerator
)
from adaptive_genie_network import (
    AdaptiveGenieNetwork, PopulationAgent,
    RhythmAgent, ResonanceAgent, GradientField
)


@pytest.fixture
def sample_capability():
    """Create a sample capability for testing"""
    return Capability(
        name="test_capability",
        depth=0,
        type="cultivation",
        consciousness_level=0.5,
        structure={"pattern": "test"},
        generates=lambda: "test",
        parent_capabilities=[]
    )


@pytest.fixture
def sample_capabilities():
    """Create a list of sample capabilities for testing"""
    return [
        Capability(
            name=f"capability_{i}",
            depth=0,
            type="cultivation",
            consciousness_level=0.3 + i * 0.1,
            structure={"pattern": f"pattern_{i}"},
            generates=lambda i=i: f"generator_{i}",
            parent_capabilities=[]
        )
        for i in range(3)
    ]


@pytest.fixture
def cultivation_generator():
    """Create a CultivationGenerator instance"""
    return CultivationGenerator()


@pytest.fixture
def formalization_generator():
    """Create a FormalizationGenerator instance"""
    return FormalizationGenerator()


@pytest.fixture
def tool_generator():
    """Create a ToolGenerator instance"""
    return ToolGenerator()


@pytest.fixture
def meta_tool_generator():
    """Create a MetaToolGenerator instance"""
    return MetaToolGenerator()


@pytest.fixture
def recursive_protocol():
    """Create a RecursiveCapabilityProtocol instance"""
    return RecursiveCapabilityProtocol()


@pytest.fixture
def adaptive_genie_network():
    """Create an AdaptiveGenieNetwork instance"""
    return AdaptiveGenieNetwork()


@pytest.fixture
def problem_landscape():
    """Create a sample problem landscape for testing"""
    return {
        'dimensions': 2,
        'bounds': [(-10, 10), (-5, 15)],
        'multimodality': 0.5,
        'noise_level': 0.1,
        'deception': 0.3
    }


@pytest.fixture
def system_state():
    """Create a sample system state for testing"""
    return {
        'fitness_history': [0.1, 0.2, 0.3, 0.4, 0.5],
        'population_diversity': 0.7,
        'fitness_variance': 0.2
    }
