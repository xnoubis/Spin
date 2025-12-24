"""
Unified Genie: Spin/Troupe Integration System
==============================================

The key insight: Spin's agents (Population, Rhythm, Resonance) ARE the Troupe
(Builder, Validator, Meta-Validator) operating at different levels of abstraction.

    Population breathing = Builder's generative velocity
    Rhythm detection = Validator's cycle completion sensing
    Resonance crystallization = Meta-Validator's invariance testing

This package provides the unified architecture that bridges both paradigms.

Structure:
    unified_genie/
    ├── core/                 # Core modules from Spin
    │   ├── dialectical_genie
    │   ├── recursive_protocol
    │   └── mathematical_models
    ├── memory/               # Memory systems
    │   ├── psip_router       # Signature compression/retrieval
    │   └── prismatic_memory  # Color/Rainbow structures
    ├── agents/               # Unified agents
    │   └── troupe           # Builder/Validator/Meta-Validator
    ├── orchestration/        # Integration layer
    │   ├── genie_nexus      # The central integrator
    │   └── cascade          # Gift-mode activation
    └── cli/
        └── genie            # Command-line interface

Usage:
    from unified_genie import GenieNexus, Troupe, create_nexus

    nexus = create_nexus({'initial_consciousness': 0.5})
    state = nexus.execute_cycle()
    print(f"Consciousness: {state.consciousness}")
"""

__version__ = '0.1.0'
__author__ = 'Spin Project'

# Core imports
import sys
import os

# Add parent path for Spin imports
_spin_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _spin_path not in sys.path:
    sys.path.insert(0, _spin_path)

# Import key classes for easy access
try:
    from unified_genie.agents.troupe import (
        Troupe, TroupeRole, TroupeState, AgentDefinition,
        BuilderAgent, ValidatorAgent, MetaValidatorAgent,
        create_troupe
    )

    from unified_genie.orchestration.genie_nexus import (
        GenieNexus, NexusConfig, NexusMode, NexusState,
        create_nexus
    )

    from unified_genie.orchestration.cascade import (
        Cascade, CascadePhase, Gift
    )

    from unified_genie.memory.psip_router import (
        PSIPRouter, Signature, SignatureCompressor, ArtifactStore
    )

    from unified_genie.memory.prismatic_memory import (
        PrismaticMemoryStore, PrismaticSpectrum, RainbowMemory,
        PrismaticColor
    )

    __all__ = [
        # Main classes
        'GenieNexus', 'Troupe', 'Cascade',

        # Factory functions
        'create_nexus', 'create_troupe',

        # Agents
        'BuilderAgent', 'ValidatorAgent', 'MetaValidatorAgent',

        # Memory
        'PSIPRouter', 'PrismaticMemoryStore',
        'Signature', 'PrismaticSpectrum', 'RainbowMemory',

        # Configuration
        'NexusConfig', 'AgentDefinition',

        # Enums
        'TroupeRole', 'NexusMode', 'CascadePhase', 'PrismaticColor',

        # Data classes
        'TroupeState', 'NexusState', 'Gift',
    ]

except ImportError as e:
    # Allow import to succeed even if dependencies aren't available
    import warnings
    warnings.warn(f"Could not import all unified_genie components: {e}")
    __all__ = []
