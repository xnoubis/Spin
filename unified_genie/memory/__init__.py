"""
Unified Genie Memory Systems
=============================

This module provides two complementary memory systems:

1. PSIP Router (Perceptual Signature Indexing Protocol)
   - Signature compression and retrieval
   - Fast similarity search
   - Route-based associations

2. Prismatic Memory (Color/Rainbow Structures)
   - Color-coded memory representation
   - Spectral similarity search
   - Harmony-based retrieval

Together they provide rich memory capabilities for the unified genie.
"""

from unified_genie.memory.psip_router import (
    PSIPRouter, Signature, Route, SignatureCompressor, ArtifactStore
)

from unified_genie.memory.prismatic_memory import (
    PrismaticMemoryStore, PrismaticSpectrum, RainbowMemory,
    PrismaticColor, ColorIntensity, StateToSpectrumMapper
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
    'StateToSpectrumMapper'
]
