"""Public API for the Spin recursive capability toolkit.

This module centralizes exports from the individual modules so consumers can
import documented classes and functions from a single namespace while keeping
alias names consistent with the README examples.
"""

from pathlib import Path
import sys

# Ensure local modules are importable when the package is used directly from source
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_genie_network import (
    AdaptiveGenieNetwork,
    PopulationAgent,
    RhythmAgent,
    ResonanceAgent,
    GradientField,
    ComplexityMeasure,
)
from recursive_capability_protocol import (
    RecursiveCapabilityProtocol,
    Capability,
    CapabilityGenerator,
    CultivationGenerator,
    FormalizationGenerator,
    ToolGenerator,
    MetaToolGenerator,
)
from example_applications import (
    OptimizationResult,
    DialecticalParticleSwarm,
    DialecticalGeneticAlgorithm,
    rastrigin,
    rosenbrock,
    ackley,
    schwefel,
    visualize_optimization_dynamics,
    run_comparative_study,
)
from mathematical_models import (
    DialecticalState,
    NegationDensityCalculator,
    ResonanceCalculator,
    GradientFieldMathematics,
    DialecticalSynthesis,
)
from visualization_tools import (
    VisualizationConfig,
    SystemDynamicsVisualizer,
    OptimizationLandscapeVisualizer,
    RealTimeMonitor,
)
from exceptions import (
    SpinException,
    ValidationError,
    InvalidDepthError,
    InvalidBoundsError,
    InvalidParameterError,
    EmptyInputError,
    CapabilityGenerationError,
    NegotiationError,
    ConvergenceError,
    VisualizationError,
    StateExportError,
    DimensionMismatchError,
    NumericError,
)

__all__ = [
    # Core network
    "AdaptiveGenieNetwork",
    "PopulationAgent",
    "RhythmAgent",
    "ResonanceAgent",
    "GradientField",
    "ComplexityMeasure",
    # Recursive capability protocol
    "RecursiveCapabilityProtocol",
    "Capability",
    "CapabilityGenerator",
    "CultivationGenerator",
    "FormalizationGenerator",
    "ToolGenerator",
    "MetaToolGenerator",
    # Applications and benchmarks
    "OptimizationResult",
    "DialecticalParticleSwarm",
    "DialecticalGeneticAlgorithm",
    "rastrigin",
    "rosenbrock",
    "ackley",
    "schwefel",
    "visualize_optimization_dynamics",
    "run_comparative_study",
    # Mathematical foundations
    "DialecticalState",
    "NegationDensityCalculator",
    "ResonanceCalculator",
    "GradientFieldMathematics",
    "DialecticalSynthesis",
    # Visualization suite
    "VisualizationConfig",
    "SystemDynamicsVisualizer",
    "OptimizationLandscapeVisualizer",
    "RealTimeMonitor",
    # Exceptions
    "SpinException",
    "ValidationError",
    "InvalidDepthError",
    "InvalidBoundsError",
    "InvalidParameterError",
    "EmptyInputError",
    "CapabilityGenerationError",
    "NegotiationError",
    "ConvergenceError",
    "VisualizationError",
    "StateExportError",
    "DimensionMismatchError",
    "NumericError",
]
