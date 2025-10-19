"""
Spirit Angelus Framework

Modular framework for intention-driven guidance and symbolic computation.
"""

from .invocation.engine import InvocationEngine
from .resonance.oracle import ResonanceOracle
from .lattice.weaver import LatticeWeaver
from .framework import AngelusFramework, AngelusConfig
from .common.types import Intent, Guidance, SigilSpec

__all__ = [
    "InvocationEngine",
    "ResonanceOracle",
    "LatticeWeaver",
    "AngelusFramework",
    "AngelusConfig",
    "Intent",
    "Guidance",
    "SigilSpec",
]
