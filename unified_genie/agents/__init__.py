"""
Unified Genie Agents
====================

This module provides the unified agent architecture that maps
Spin's agents to the Troupe paradigm:

    PopulationAgent → BuilderAgent (generative velocity)
    RhythmAgent → ValidatorAgent (cycle completion sensing)
    ResonanceAgent → MetaValidatorAgent (invariance testing)

The Troupe class orchestrates all three agents in dialectical harmony.
"""

from unified_genie.agents.troupe import (
    Troupe, TroupeRole, TroupeState, AgentDefinition,
    UnifiedAgent, BuilderAgent, ValidatorAgent, MetaValidatorAgent,
    create_troupe
)

__all__ = [
    'Troupe',
    'TroupeRole',
    'TroupeState',
    'AgentDefinition',
    'UnifiedAgent',
    'BuilderAgent',
    'ValidatorAgent',
    'MetaValidatorAgent',
    'create_troupe'
]
