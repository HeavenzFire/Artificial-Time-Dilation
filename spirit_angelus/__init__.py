"""
Spirit Angelus Framework

A spiritual-tech hybrid framework that bridges ancient wisdom with modern AI,
creating personalized spiritual guidance systems through angelic archetypes,
quantum-inspired meditations, and machine learning.

The framework combines:
- Angelic archetypes and spiritual guidance
- ML-based intent matching and synchronicity detection  
- Quantum-inspired elements for "entanglement" meditations
- Ritual simulation and invocation systems
- No-code/low-code spiritual guidance tools
"""

__version__ = "0.1.0"
__author__ = "Spirit Angelus Framework Team"
__email__ = "contact@spiritangelus.dev"

# Core imports
from .angels import AngelicCore, GuardianAngel, Archangel
from .invocation import InvocationEngine, RitualSimulator
from .oracle import ResonanceOracle, SynchronicityDetector
from .lattice import LatticeWeaver, SpiritualNetwork
from .quantum import QuantumMeditation, EntanglementSimulator

# Utility imports
from .utils import SigilGenerator, SacredGeometry
from .config import SpiritConfig, AngelConfig

__all__ = [
    # Core Framework
    "AngelicCore",
    "GuardianAngel", 
    "Archangel",
    
    # Invocation System
    "InvocationEngine",
    "RitualSimulator",
    
    # Oracle & Resonance
    "ResonanceOracle",
    "SynchronicityDetector",
    
    # Lattice & Networks
    "LatticeWeaver",
    "SpiritualNetwork",
    
    # Quantum Spirituality
    "QuantumMeditation",
    "EntanglementSimulator",
    
    # Utilities
    "SigilGenerator",
    "SacredGeometry",
    
    # Configuration
    "SpiritConfig",
    "AngelConfig",
]