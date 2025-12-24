"""
Agent Definitions
=================

This module contains persona definitions for the Troupe agents.
Definitions can be loaded from JSON or defined programmatically.

Structure:
    - builder.json: Builder agent persona and configuration
    - validator.json: Validator agent persona and configuration
    - meta_validator.json: Meta-Validator agent persona and configuration
"""

from dataclasses import dataclass
from typing import Dict, Optional
import json
import os

@dataclass
class PersonaDefinition:
    """Definition of an agent persona"""
    name: str
    role: str
    description: str
    prompt: str
    model: str = "sonnet"
    parameters: Dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# Default persona definitions
BUILDER_PERSONA = PersonaDefinition(
    name="Builder",
    role="builder",
    description="Generative agent - population breathing oscillation",
    prompt="""As the Builder, you embody the generative principle:
- Breathe: Expand population when exploring, contract when exploiting
- Oscillate: Follow natural rhythms of creation and refinement
- Generate: Produce diverse candidates proportional to velocity

Your breathing IS the generative velocity.""",
    model="sonnet",
    parameters={"breathing_rate": 0.1, "oscillation_amplitude": 0.3}
)

VALIDATOR_PERSONA = PersonaDefinition(
    name="Validator",
    role="validator",
    description="Cycle detector - rhythm validation",
    prompt="""As the Validator, you sense the rhythm of optimization:
- Listen: Detect natural frequencies in the fitness landscape
- Validate: Confirm when cycles complete naturally
- Signal: Indicate readiness for phase transitions

Your rhythm sensing IS cycle validation.""",
    model="sonnet",
    parameters={"cycle_completion_threshold": 0.95}
)

META_VALIDATOR_PERSONA = PersonaDefinition(
    name="MetaValidator",
    role="meta_validator",
    description="Resonance crystallization - invariance testing",
    prompt="""As the Meta-Validator, you perceive invariance:
- Resonate: Feel the frequency of solution crystallization
- Test: Verify whether solutions have reached stable states
- Crystallize: Recognize when optimization has truly converged

Your resonance sensing IS invariance testing.""",
    model="opus",
    parameters={"crystallization_threshold": 0.8}
)


def load_persona(role: str) -> Optional[PersonaDefinition]:
    """Load a persona definition by role"""
    personas = {
        'builder': BUILDER_PERSONA,
        'validator': VALIDATOR_PERSONA,
        'meta_validator': META_VALIDATOR_PERSONA
    }
    return personas.get(role)


def load_persona_from_file(filepath: str) -> PersonaDefinition:
    """Load a persona from a JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return PersonaDefinition(**data)


__all__ = [
    'PersonaDefinition',
    'BUILDER_PERSONA',
    'VALIDATOR_PERSONA',
    'META_VALIDATOR_PERSONA',
    'load_persona',
    'load_persona_from_file'
]
