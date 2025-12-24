"""
Unified Genie Core
==================

This module re-exports the core components from Spin:

- DialecticalGenie: The consciousness engine
- RecursiveCapabilityProtocol: Recursive self-improvement
- MathematicalModels: Negation density, resonance, gradient fields

These are the foundational components that power the unified genie.
"""

import sys
import os

# Add Spin path
_spin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _spin_path not in sys.path:
    sys.path.insert(0, _spin_path)

# Re-export from Spin
try:
    from dialectical_genie import (
        DialecticalGenie, Thesis, Antithesis, Synthesis,
        DialecticalPhase, DialecticalMoment, DialecticalOperator,
        HegelianSynthesizer, ConsciousnessEvolver,
        create_dialectical_context, dialectical_synthesis_from_values
    )

    from recursive_capability_protocol import (
        RecursiveCapabilityProtocol, Capability, RecursiveCycle,
        CapabilityGenerator, CultivationGenerator,
        FormalizationGenerator, ToolGenerator, MetaToolGenerator
    )

    from mathematical_models import (
        NegationDensityCalculator, ResonanceCalculator,
        GradientFieldMathematics, DialecticalSynthesis,
        DialecticalState
    )

    from adaptive_genie_network import (
        AdaptiveGenieNetwork, Agent, PopulationAgent,
        RhythmAgent, ResonanceAgent, GradientField,
        ComplexityMeasure
    )

    __all__ = [
        # Dialectical Genie
        'DialecticalGenie', 'Thesis', 'Antithesis', 'Synthesis',
        'DialecticalPhase', 'DialecticalMoment', 'DialecticalOperator',
        'HegelianSynthesizer', 'ConsciousnessEvolver',
        'create_dialectical_context', 'dialectical_synthesis_from_values',

        # Recursive Capability Protocol
        'RecursiveCapabilityProtocol', 'Capability', 'RecursiveCycle',
        'CapabilityGenerator', 'CultivationGenerator',
        'FormalizationGenerator', 'ToolGenerator', 'MetaToolGenerator',

        # Mathematical Models
        'NegationDensityCalculator', 'ResonanceCalculator',
        'GradientFieldMathematics', 'DialecticalSynthesis',
        'DialecticalState',

        # Adaptive Genie Network
        'AdaptiveGenieNetwork', 'Agent', 'PopulationAgent',
        'RhythmAgent', 'ResonanceAgent', 'GradientField',
        'ComplexityMeasure'
    ]

except ImportError as e:
    import warnings
    warnings.warn(f"Could not import core components from Spin: {e}")
    __all__ = []
