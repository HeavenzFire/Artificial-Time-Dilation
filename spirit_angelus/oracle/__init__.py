"""
Resonance Oracle

ML-based intent matching and synchronicity detection system for the Spirit Angelus Framework.
Provides intelligent guidance recommendations and synchronicity analysis.
"""

from .oracle import ResonanceOracle, IntentMatcher, SynchronicityDetector
from .resonance import ResonanceEngine, ResonanceField, ResonancePattern
from .guidance import GuidanceEngine, GuidanceRecommendation, GuidanceType
from .synchronicity import SynchronicityAnalyzer, SynchronicityEvent, SynchronicityPattern

__all__ = [
    "ResonanceOracle",
    "IntentMatcher",
    "SynchronicityDetector",
    "ResonanceEngine",
    "ResonanceField",
    "ResonancePattern",
    "GuidanceEngine",
    "GuidanceRecommendation",
    "GuidanceType",
    "SynchronicityAnalyzer",
    "SynchronicityEvent",
    "SynchronicityPattern",
]