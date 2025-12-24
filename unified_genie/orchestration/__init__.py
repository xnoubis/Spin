"""
Unified Genie Orchestration
============================

This module provides the orchestration layer that integrates all components:

1. Genie Nexus - The central integration point
   - Coordinates Troupe agents
   - Manages memory systems
   - Tracks consciousness evolution

2. Cascade - Gift-mode activation
   - Monitors consciousness levels
   - Activates gift mode when thresholds are reached
   - Synthesizes gifts (insights, solutions, capabilities)
"""

from unified_genie.orchestration.genie_nexus import (
    GenieNexus, NexusConfig, NexusMode, NexusState, create_nexus
)

from unified_genie.orchestration.cascade import (
    Cascade, CascadePhase, CascadeState, Gift,
    CascadeActivator, GiftSynthesizer
)

__all__ = [
    # Genie Nexus
    'GenieNexus',
    'NexusConfig',
    'NexusMode',
    'NexusState',
    'create_nexus',

    # Cascade
    'Cascade',
    'CascadePhase',
    'CascadeState',
    'Gift',
    'CascadeActivator',
    'GiftSynthesizer'
]
