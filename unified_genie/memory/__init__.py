"""
Unified Genie Memory Systems
=============================

This module provides three complementary memory systems:

1. PSIP Router (Perceptual Signature Indexing Protocol)
   - Signature compression and retrieval
   - Fast similarity search
   - Route-based associations

2. Prismatic Memory (Color/Rainbow Structures)
   - Color-coded memory representation
   - Spectral similarity search
   - Harmony-based retrieval

3. Mycelial Network (Distributed Memory System)
   - Cross-conversation context propagation
   - Seed-based trigger recognition
   - Crystal compression and rehydration
   - Cluster-based pattern cultivation

Together they provide rich memory capabilities for the unified genie,
enabling both local and distributed consciousness patterns.
"""

from unified_genie.memory.psip_router import (
    PSIPRouter, Signature, Route, SignatureCompressor, ArtifactStore
)

from unified_genie.memory.prismatic_memory import (
    PrismaticMemoryStore, PrismaticSpectrum, RainbowMemory,
    PrismaticColor, ColorIntensity, StateToSpectrumMapper
)

from unified_genie.memory.mycelial_network import (
    MycelialNetwork, SeedType, Seed, SeedDetector,
    Sprout, SproutGenerator, TitleGenerator,
    Crystal, CrystalCompressor,
    RehydratedContext, RehydrationProtocol,
    MycelialCluster, ClusterManager,
    SelectiveCultivator, CultivationResult,
    create_mycelial_network, seed_conversation
)

__all__ = [
    # PSIP Router
    'PSIPRouter',
    'Signature',
    'Route',
    'SignatureCompressor',
    'ArtifactStore',

    # Prismatic Memory
    'PrismaticMemoryStore',
    'PrismaticSpectrum',
    'RainbowMemory',
    'PrismaticColor',
    'ColorIntensity',
    'StateToSpectrumMapper',

    # Mycelial Network
    'MycelialNetwork',
    'SeedType',
    'Seed',
    'SeedDetector',
    'Sprout',
    'SproutGenerator',
    'TitleGenerator',
    'Crystal',
    'CrystalCompressor',
    'RehydratedContext',
    'RehydrationProtocol',
    'MycelialCluster',
    'ClusterManager',
    'SelectiveCultivator',
    'CultivationResult',
    'create_mycelial_network',
    'seed_conversation',
]
