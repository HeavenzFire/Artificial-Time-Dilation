"""
Resonance Oracle

The core oracle system that provides ML-based intent matching and synchronicity detection.
This is the intelligence layer of the Spirit Angelus Framework.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import re
from collections import defaultdict, Counter
import math

from ..angels.core import AngelicCore, AngelicEnergyType, DivineConnection


class IntentType(Enum):
    """Types of spiritual intentions"""
    HEALING = "healing"
    GUIDANCE = "guidance"
    PROTECTION = "protection"
    LOVE = "love"
    WISDOM = "wisdom"
    STRENGTH = "strength"
    PEACE = "peace"
    JOY = "joy"
    ABUNDANCE = "abundance"
    TRANSFORMATION = "transformation"
    FORGIVENESS = "forgiveness"
    CLARITY = "clarity"
    COURAGE = "courage"
    GRATITUDE = "gratitude"


class SynchronicityType(Enum):
    """Types of synchronicities"""
    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    RELATIONAL = "relational"
    DREAM = "dream"
    MEDIA = "media"
    NATURE = "nature"


@dataclass
class Intent:
    """Represents a spiritual intention"""
    intent_id: str
    intent_type: IntentType
    description: str
    keywords: List[str]
    energy_types: List[AngelicEnergyType]
    intensity: float  # 0.0 to 1.0
    created_at: datetime
    user_id: str
    context: Dict[str, Any]
    
    def get_keyword_frequency(self) -> Dict[str, int]:
        """Get frequency of keywords in the intent"""
        return Counter(self.keywords)
    
    def get_energy_signature(self) -> Dict[str, float]:
        """Get energy signature for this intent"""
        return {
            energy_type.value: self.intensity for energy_type in self.energy_types
        }


@dataclass
class SynchronicityEvent:
    """Represents a synchronicity event"""
    event_id: str
    event_type: SynchronicityType
    description: str
    significance: float  # 0.0 to 1.0
    timestamp: datetime
    location: str
    related_intents: List[str]
    patterns: List[str]
    user_id: str
    
    def get_importance_score(self) -> float:
        """Calculate importance score based on significance and patterns"""
        base_score = self.significance
        pattern_bonus = len(self.patterns) * 0.1
        return min(1.0, base_score + pattern_bonus)


@dataclass
class ResonanceMatch:
    """Represents a resonance match between intents and guidance"""
    match_id: str
    intent_id: str
    guidance_id: str
    resonance_score: float  # 0.0 to 1.0
    matching_keywords: List[str]
    energy_alignment: float
    synchronicity_boost: float
    timestamp: datetime
    
    def get_total_score(self) -> float:
        """Calculate total resonance score"""
        return (self.resonance_score * 0.4 + 
                self.energy_alignment * 0.3 + 
                self.synchronicity_boost * 0.3)


class IntentMatcher:
    """ML-based intent matching system"""
    
    def __init__(self):
        self.intent_vectors: Dict[str, np.ndarray] = {}
        self.keyword_vectors: Dict[str, np.ndarray] = {}
        self.energy_vectors: Dict[str, np.ndarray] = {}
        self.intent_history: List[Intent] = []
        self.match_history: List[ResonanceMatch] = []
        
        # Initialize keyword embeddings (simplified)
        self._initialize_keyword_embeddings()
    
    def _initialize_keyword_embeddings(self):
        """Initialize keyword embeddings for intent matching"""
        # This is a simplified version - in production, you'd use real embeddings
        keywords = [
            "healing", "guidance", "protection", "love", "wisdom", "strength",
            "peace", "joy", "abundance", "transformation", "forgiveness",
            "clarity", "courage", "gratitude", "spiritual", "divine", "angel",
            "meditation", "prayer", "ritual", "ceremony", "blessing", "energy",
            "chakra", "aura", "karma", "dharma", "enlightenment", "awakening"
        ]
        
        # Create random embeddings for demonstration
        for keyword in keywords:
            self.keyword_vectors[keyword] = np.random.randn(50)
    
    def add_intent(self, intent: Intent) -> str:
        """Add a new intent to the system"""
        # Create intent vector
        intent_vector = self._create_intent_vector(intent)
        self.intent_vectors[intent.intent_id] = intent_vector
        
        # Store intent
        self.intent_history.append(intent)
        
        return intent.intent_id
    
    def _create_intent_vector(self, intent: Intent) -> np.ndarray:
        """Create a vector representation of an intent"""
        # Combine keyword vectors and energy vectors
        keyword_vector = np.zeros(50)
        for keyword in intent.keywords:
            if keyword in self.keyword_vectors:
                keyword_vector += self.keyword_vectors[keyword]
        
        # Normalize
        if np.linalg.norm(keyword_vector) > 0:
            keyword_vector = keyword_vector / np.linalg.norm(keyword_vector)
        
        # Add energy type information
        energy_vector = np.zeros(10)  # 10 energy types
        for i, energy_type in enumerate(intent.energy_types):
            energy_vector[i] = intent.intensity
        
        # Combine vectors
        combined_vector = np.concatenate([keyword_vector, energy_vector])
        return combined_vector
    
    def find_matching_intents(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Find intents that match a query"""
        # Create query vector
        query_vector = self._create_query_vector(query)
        
        # Calculate similarities
        similarities = []
        for intent_id, intent_vector in self.intent_vectors.items():
            similarity = np.dot(query_vector, intent_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(intent_vector)
            )
            similarities.append((intent_id, similarity))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def _create_query_vector(self, query: str) -> np.ndarray:
        """Create a vector representation of a query"""
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)
        
        # Create keyword vector
        keyword_vector = np.zeros(50)
        for keyword in query_keywords:
            if keyword in self.keyword_vectors:
                keyword_vector += self.keyword_vectors[keyword]
        
        # Normalize
        if np.linalg.norm(keyword_vector) > 0:
            keyword_vector = keyword_vector / np.linalg.norm(keyword_vector)
        
        # Add energy type information (simplified)
        energy_vector = np.zeros(10)
        
        # Combine vectors
        combined_vector = np.concatenate([keyword_vector, energy_vector])
        return combined_vector
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter for spiritual keywords
        spiritual_keywords = set(self.keyword_vectors.keys())
        return [word for word in words if word in spiritual_keywords]
    
    def get_intent_similarity(self, intent1_id: str, intent2_id: str) -> float:
        """Calculate similarity between two intents"""
        if intent1_id not in self.intent_vectors or intent2_id not in self.intent_vectors:
            return 0.0
        
        vector1 = self.intent_vectors[intent1_id]
        vector2 = self.intent_vectors[intent2_id]
        
        similarity = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
        return similarity


class SynchronicityDetector:
    """Synchronicity detection and analysis system"""
    
    def __init__(self):
        self.synchronicity_events: List[SynchronicityEvent] = []
        self.pattern_database: Dict[str, List[str]] = {}
        self.numerical_patterns: Dict[str, List[int]] = {}
        self.symbolic_patterns: Dict[str, List[str]] = {}
        self.temporal_patterns: Dict[str, List[datetime]] = {}
        
        # Initialize pattern databases
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize synchronicity pattern databases"""
        # Numerical patterns
        self.numerical_patterns = {
            "angel_numbers": [111, 222, 333, 444, 555, 666, 777, 888, 999],
            "sacred_numbers": [3, 7, 12, 21, 108, 144, 432],
            "fibonacci": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
            "master_numbers": [11, 22, 33, 44, 55, 66, 77, 88, 99]
        }
        
        # Symbolic patterns
        self.symbolic_patterns = {
            "animals": ["butterfly", "eagle", "owl", "dolphin", "wolf", "lion", "dragon"],
            "elements": ["fire", "water", "air", "earth", "spirit"],
            "colors": ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "white", "gold"],
            "shapes": ["circle", "triangle", "square", "pentagon", "hexagon", "spiral", "infinity"]
        }
        
        # Temporal patterns
        self.temporal_patterns = {
            "moon_phases": ["new_moon", "waxing_crescent", "first_quarter", "waxing_gibbous", 
                           "full_moon", "waning_gibbous", "last_quarter", "waning_crescent"],
            "seasons": ["spring", "summer", "autumn", "winter"],
            "times": ["dawn", "midday", "dusk", "midnight"]
        }
    
    def detect_synchronicity(self, event_description: str, context: Dict[str, Any]) -> Optional[SynchronicityEvent]:
        """Detect synchronicity in an event"""
        event_id = f"sync_{int(time.time())}_{len(self.synchronicity_events)}"
        
        # Analyze the event for patterns
        patterns = self._analyze_patterns(event_description, context)
        
        if not patterns:
            return None
        
        # Calculate significance
        significance = self._calculate_significance(patterns, context)
        
        if significance < 0.3:  # Threshold for synchronicity
            return None
        
        # Determine event type
        event_type = self._determine_event_type(patterns)
        
        # Create synchronicity event
        synchronicity_event = SynchronicityEvent(
            event_id=event_id,
            event_type=event_type,
            description=event_description,
            significance=significance,
            timestamp=datetime.now(),
            location=context.get("location", "unknown"),
            related_intents=context.get("related_intents", []),
            patterns=patterns,
            user_id=context.get("user_id", "unknown")
        )
        
        # Store the event
        self.synchronicity_events.append(synchronicity_event)
        
        return synchronicity_event
    
    def _analyze_patterns(self, event_description: str, context: Dict[str, Any]) -> List[str]:
        """Analyze event for synchronicity patterns"""
        patterns = []
        event_lower = event_description.lower()
        
        # Check for numerical patterns
        numbers = re.findall(r'\b\d+\b', event_description)
        for number in numbers:
            num = int(number)
            for pattern_name, pattern_numbers in self.numerical_patterns.items():
                if num in pattern_numbers:
                    patterns.append(f"{pattern_name}:{num}")
        
        # Check for symbolic patterns
        for category, symbols in self.symbolic_patterns.items():
            for symbol in symbols:
                if symbol in event_lower:
                    patterns.append(f"{category}:{symbol}")
        
        # Check for temporal patterns
        for category, times in self.temporal_patterns.items():
            for time_pattern in times:
                if time_pattern in event_lower:
                    patterns.append(f"{category}:{time_pattern}")
        
        # Check for repeated words or phrases
        words = re.findall(r'\b\w+\b', event_lower)
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if count > 1:
                patterns.append(f"repetition:{word}:{count}")
        
        return patterns
    
    def _calculate_significance(self, patterns: List[str], context: Dict[str, Any]) -> float:
        """Calculate significance of synchronicity patterns"""
        if not patterns:
            return 0.0
        
        base_significance = 0.1
        
        # Pattern type weights
        pattern_weights = {
            "angel_numbers": 0.3,
            "sacred_numbers": 0.25,
            "master_numbers": 0.2,
            "fibonacci": 0.15,
            "animals": 0.1,
            "elements": 0.1,
            "colors": 0.05,
            "shapes": 0.05,
            "repetition": 0.05
        }
        
        # Calculate weighted significance
        total_weight = 0.0
        for pattern in patterns:
            pattern_type = pattern.split(":")[0]
            weight = pattern_weights.get(pattern_type, 0.05)
            total_weight += weight
        
        # Add context bonuses
        context_bonus = 0.0
        if context.get("recent_intent", False):
            context_bonus += 0.2
        if context.get("emotional_intensity", 0) > 0.7:
            context_bonus += 0.1
        if context.get("timing_significance", False):
            context_bonus += 0.1
        
        final_significance = min(1.0, base_significance + total_weight + context_bonus)
        return final_significance
    
    def _determine_event_type(self, patterns: List[str]) -> SynchronicityType:
        """Determine the type of synchronicity event"""
        if any("angel_numbers" in pattern or "sacred_numbers" in pattern for pattern in patterns):
            return SynchronicityType.NUMERICAL
        elif any("animals" in pattern or "elements" in pattern for pattern in patterns):
            return SynchronicityType.SYMBOLIC
        elif any("moon_phases" in pattern or "seasons" in pattern for pattern in patterns):
            return SynchronicityType.TEMPORAL
        elif any("repetition" in pattern for pattern in patterns):
            return SynchronicityType.RELATIONAL
        else:
            return SynchronicityType.SYMBOLIC
    
    def get_synchronicity_frequency(self, days: int = 30) -> Dict[str, int]:
        """Get frequency of synchronicity types over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_events = [
            event for event in self.synchronicity_events
            if event.timestamp > cutoff_date
        ]
        
        frequency = defaultdict(int)
        for event in recent_events:
            frequency[event.event_type.value] += 1
        
        return dict(frequency)
    
    def get_most_significant_synchronicities(self, limit: int = 10) -> List[SynchronicityEvent]:
        """Get the most significant synchronicities"""
        sorted_events = sorted(
            self.synchronicity_events,
            key=lambda x: x.get_importance_score(),
            reverse=True
        )
        return sorted_events[:limit]
    
    def find_synchronicity_patterns(self, intent_id: str) -> List[SynchronicityEvent]:
        """Find synchronicities related to a specific intent"""
        return [
            event for event in self.synchronicity_events
            if intent_id in event.related_intents
        ]


class ResonanceOracle:
    """
    Main Resonance Oracle that coordinates intent matching and synchronicity detection.
    
    This is the central intelligence system that provides personalized spiritual guidance
    through ML-based intent matching and synchronicity analysis.
    """
    
    def __init__(self, angelic_core: AngelicCore):
        self.angelic_core = angelic_core
        self.intent_matcher = IntentMatcher()
        self.synchronicity_detector = SynchronicityDetector()
        self.guidance_database: Dict[str, Dict] = {}
        self.resonance_history: List[ResonanceMatch] = []
        
        # Initialize guidance database
        self._initialize_guidance_database()
    
    def _initialize_guidance_database(self):
        """Initialize the guidance database with spiritual wisdom"""
        self.guidance_database = {
            "healing": {
                "guidance": [
                    "Healing begins with self-compassion and acceptance.",
                    "Your body has an innate wisdom for healing - trust it.",
                    "Healing is not linear - be patient with the process.",
                    "Forgiveness is a powerful healing tool.",
                    "Surround yourself with healing energy and positive influences."
                ],
                "practices": [
                    "Daily meditation for inner peace",
                    "Energy healing sessions",
                    "Gratitude journaling",
                    "Nature walks for grounding",
                    "Breathing exercises for stress relief"
                ],
                "energy_types": [AngelicEnergyType.HEALING, AngelicEnergyType.LOVE]
            },
            "guidance": {
                "guidance": [
                    "Trust your inner wisdom - it knows the way.",
                    "Guidance comes in many forms - stay open to all possibilities.",
                    "Sometimes the best guidance is to wait and observe.",
                    "Your higher self is always communicating with you.",
                    "Guidance often comes through synchronicities and signs."
                ],
                "practices": [
                    "Morning meditation for clarity",
                    "Journaling to connect with inner wisdom",
                    "Oracle card readings",
                    "Nature meditation",
                    "Dream journaling"
                ],
                "energy_types": [AngelicEnergyType.GUIDANCE, AngelicEnergyType.WISDOM]
            },
            "protection": {
                "guidance": [
                    "You are always protected by divine love.",
                    "Set clear boundaries to protect your energy.",
                    "Visualize a protective light surrounding you.",
                    "Trust your instincts about people and situations.",
                    "Regular cleansing rituals help maintain protection."
                ],
                "practices": [
                    "Protection meditation",
                    "Energy cleansing rituals",
                    "Crystal protection grids",
                    "Prayer for divine protection",
                    "Visualization of protective barriers"
                ],
                "energy_types": [AngelicEnergyType.PROTECTION, AngelicEnergyType.STRENGTH]
            },
            "love": {
                "guidance": [
                    "Love begins with self-love and acceptance.",
                    "Unconditional love is the highest form of love.",
                    "Love is not about possession - it's about freedom.",
                    "Forgiveness opens the heart to deeper love.",
                    "Love yourself first, then others can love you fully."
                ],
                "practices": [
                    "Self-love affirmations",
                    "Heart chakra meditation",
                    "Loving-kindness meditation",
                    "Gratitude for love in your life",
                    "Acts of kindness and compassion"
                ],
                "energy_types": [AngelicEnergyType.LOVE, AngelicEnergyType.JOY]
            },
            "wisdom": {
                "guidance": [
                    "Wisdom comes from experience and reflection.",
                    "True wisdom is knowing that you know nothing.",
                    "Wisdom is the ability to see the bigger picture.",
                    "Ancient wisdom is always available to guide you.",
                    "Wisdom grows through contemplation and meditation."
                ],
                "practices": [
                    "Study of spiritual texts",
                    "Contemplative meditation",
                    "Learning from elders and teachers",
                    "Reflection on life experiences",
                    "Seeking knowledge from multiple sources"
                ],
                "energy_types": [AngelicEnergyType.WISDOM, AngelicEnergyType.GUIDANCE]
            }
        }
    
    def process_intent(self, intent_description: str, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a spiritual intent and provide guidance.
        
        Args:
            intent_description: Description of the spiritual intent
            user_id: ID of the user
            context: Additional context information
            
        Returns:
            Dictionary containing guidance and recommendations
        """
        context = context or {}
        
        # Create intent object
        intent = Intent(
            intent_id=f"intent_{int(time.time())}_{len(self.intent_matcher.intent_history)}",
            intent_type=self._classify_intent(intent_description),
            description=intent_description,
            keywords=self.intent_matcher._extract_keywords(intent_description),
            energy_types=self._determine_energy_types(intent_description),
            intensity=context.get("intensity", 0.7),
            created_at=datetime.now(),
            user_id=user_id,
            context=context
        )
        
        # Add intent to matcher
        self.intent_matcher.add_intent(intent)
        
        # Find matching guidance
        guidance_matches = self._find_guidance_matches(intent)
        
        # Check for synchronicities
        synchronicities = self._check_recent_synchronicities(intent, context)
        
        # Generate personalized guidance
        personalized_guidance = self._generate_personalized_guidance(intent, guidance_matches, synchronicities)
        
        # Create resonance match
        if guidance_matches:
            resonance_match = ResonanceMatch(
                match_id=f"match_{int(time.time())}_{len(self.resonance_history)}",
                intent_id=intent.intent_id,
                guidance_id=guidance_matches[0]["guidance_id"],
                resonance_score=guidance_matches[0]["score"],
                matching_keywords=guidance_matches[0]["matching_keywords"],
                energy_alignment=guidance_matches[0]["energy_alignment"],
                synchronicity_boost=len(synchronicities) * 0.1,
                timestamp=datetime.now()
            )
            self.resonance_history.append(resonance_match)
        
        return {
            "intent_id": intent.intent_id,
            "intent_type": intent.intent_type.value,
            "guidance": personalized_guidance,
            "synchronicities": synchronicities,
            "energy_types": [et.value for et in intent.energy_types],
            "resonance_score": guidance_matches[0]["score"] if guidance_matches else 0.0,
            "recommendations": self._generate_recommendations(intent, guidance_matches),
            "timestamp": datetime.now().isoformat()
        }
    
    def _classify_intent(self, intent_description: str) -> IntentType:
        """Classify the type of spiritual intent"""
        intent_lower = intent_description.lower()
        
        # Simple keyword-based classification
        if any(word in intent_lower for word in ["heal", "healing", "recover", "restore"]):
            return IntentType.HEALING
        elif any(word in intent_lower for word in ["guide", "guidance", "direction", "path"]):
            return IntentType.GUIDANCE
        elif any(word in intent_lower for word in ["protect", "protection", "safety", "shield"]):
            return IntentType.PROTECTION
        elif any(word in intent_lower for word in ["love", "loving", "relationship", "heart"]):
            return IntentType.LOVE
        elif any(word in intent_lower for word in ["wisdom", "knowledge", "understanding", "learn"]):
            return IntentType.WISDOM
        elif any(word in intent_lower for word in ["strength", "strong", "courage", "brave"]):
            return IntentType.STRENGTH
        elif any(word in intent_lower for word in ["peace", "calm", "tranquil", "serene"]):
            return IntentType.PEACE
        elif any(word in intent_lower for word in ["joy", "happy", "celebrate", "joyful"]):
            return IntentType.JOY
        elif any(word in intent_lower for word in ["abundance", "prosperity", "wealth", "success"]):
            return IntentType.ABUNDANCE
        elif any(word in intent_lower for word in ["transform", "change", "evolve", "grow"]):
            return IntentType.TRANSFORMATION
        elif any(word in intent_lower for word in ["forgive", "forgiveness", "release", "let go"]):
            return IntentType.FORGIVENESS
        elif any(word in intent_lower for word in ["clarity", "clear", "understand", "see"]):
            return IntentType.CLARITY
        elif any(word in intent_lower for word in ["courage", "brave", "fearless", "bold"]):
            return IntentType.COURAGE
        elif any(word in intent_lower for word in ["gratitude", "thankful", "appreciate", "blessed"]):
            return IntentType.GRATITUDE
        else:
            return IntentType.GUIDANCE  # Default
    
    def _determine_energy_types(self, intent_description: str) -> List[AngelicEnergyType]:
        """Determine energy types for an intent"""
        intent_lower = intent_description.lower()
        energy_types = []
        
        if any(word in intent_lower for word in ["heal", "healing", "recover"]):
            energy_types.append(AngelicEnergyType.HEALING)
        if any(word in intent_lower for word in ["guide", "guidance", "wisdom"]):
            energy_types.append(AngelicEnergyType.GUIDANCE)
        if any(word in intent_lower for word in ["protect", "protection", "strength"]):
            energy_types.append(AngelicEnergyType.PROTECTION)
        if any(word in intent_lower for word in ["love", "loving", "heart"]):
            energy_types.append(AngelicEnergyType.LOVE)
        if any(word in intent_lower for word in ["wisdom", "knowledge", "understanding"]):
            energy_types.append(AngelicEnergyType.WISDOM)
        if any(word in intent_lower for word in ["strength", "courage", "power"]):
            energy_types.append(AngelicEnergyType.STRENGTH)
        if any(word in intent_lower for word in ["peace", "calm", "tranquil"]):
            energy_types.append(AngelicEnergyType.PEACE)
        if any(word in intent_lower for word in ["joy", "happy", "celebrate"]):
            energy_types.append(AngelicEnergyType.JOY)
        if any(word in intent_lower for word in ["abundance", "prosperity", "wealth"]):
            energy_types.append(AngelicEnergyType.ABUNDANCE)
        if any(word in intent_lower for word in ["transform", "change", "evolve"]):
            energy_types.append(AngelicEnergyType.TRANSFORMATION)
        
        # Default to guidance if no specific energy types found
        if not energy_types:
            energy_types.append(AngelicEnergyType.GUIDANCE)
        
        return energy_types
    
    def _find_guidance_matches(self, intent: Intent) -> List[Dict[str, Any]]:
        """Find guidance that matches the intent"""
        intent_type = intent.intent_type.value
        matches = []
        
        if intent_type in self.guidance_database:
            guidance_data = self.guidance_database[intent_type]
            
            # Calculate matching keywords
            matching_keywords = [
                keyword for keyword in intent.keywords
                if keyword in " ".join(guidance_data["guidance"]).lower()
            ]
            
            # Calculate energy alignment
            energy_alignment = 0.0
            for energy_type in intent.energy_types:
                if energy_type in guidance_data["energy_types"]:
                    energy_alignment += 0.5
            
            # Calculate overall score
            score = (len(matching_keywords) * 0.3 + energy_alignment * 0.4 + 
                    intent.intensity * 0.3)
            
            matches.append({
                "guidance_id": f"guidance_{intent_type}_{int(time.time())}",
                "guidance": guidance_data["guidance"],
                "practices": guidance_data["practices"],
                "score": score,
                "matching_keywords": matching_keywords,
                "energy_alignment": energy_alignment
            })
        
        return matches
    
    def _check_recent_synchronicities(self, intent: Intent, context: Dict[str, Any]) -> List[SynchronicityEvent]:
        """Check for recent synchronicities related to the intent"""
        # This would typically check for synchronicities in the last few days
        # For now, we'll simulate some synchronicities
        synchronicities = []
        
        if random.random() < 0.3:  # 30% chance of synchronicity
            synchronicity = self.synchronicity_detector.detect_synchronicity(
                f"Synchronicity related to {intent.description}",
                {
                    "user_id": intent.user_id,
                    "related_intents": [intent.intent_id],
                    "location": context.get("location", "unknown"),
                    "recent_intent": True,
                    "emotional_intensity": intent.intensity
                }
            )
            if synchronicity:
                synchronicities.append(synchronicity)
        
        return synchronicities
    
    def _generate_personalized_guidance(self, intent: Intent, guidance_matches: List[Dict], synchronicities: List[SynchronicityEvent]) -> List[str]:
        """Generate personalized guidance based on intent and context"""
        if not guidance_matches:
            return ["No specific guidance found for this intent."]
        
        guidance = guidance_matches[0]["guidance"]
        
        # Add synchronicity-based guidance
        if synchronicities:
            synchronicity_guidance = [
                f"Notice the synchronicities around {intent.description} - they are signs from the universe.",
                "The patterns you're seeing are not coincidences - they're guidance.",
                "Pay attention to the signs and symbols that appear in your life."
            ]
            guidance.extend(synchronicity_guidance[:1])  # Add one synchronicity guidance
        
        # Add intensity-based guidance
        if intent.intensity > 0.8:
            intensity_guidance = [
                "Your strong intention is already manifesting - trust the process.",
                "The universe is responding to your clear intention."
            ]
            guidance.extend(intensity_guidance[:1])
        
        return guidance[:5]  # Return up to 5 guidance points
    
    def _generate_recommendations(self, intent: Intent, guidance_matches: List[Dict]) -> List[str]:
        """Generate specific recommendations for the intent"""
        if not guidance_matches:
            return []
        
        practices = guidance_matches[0]["practices"]
        
        # Add intent-specific recommendations
        recommendations = practices.copy()
        
        # Add energy type recommendations
        for energy_type in intent.energy_types:
            if energy_type == AngelicEnergyType.HEALING:
                recommendations.append("Consider energy healing or Reiki sessions")
            elif energy_type == AngelicEnergyType.GUIDANCE:
                recommendations.append("Try meditation or oracle card readings")
            elif energy_type == AngelicEnergyType.PROTECTION:
                recommendations.append("Use protective crystals or visualization")
            elif energy_type == AngelicEnergyType.LOVE:
                recommendations.append("Practice self-love affirmations daily")
        
        return recommendations[:5]  # Return up to 5 recommendations
    
    def get_resonance_analysis(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get resonance analysis for a user"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get recent intents
        recent_intents = [
            intent for intent in self.intent_matcher.intent_history
            if intent.user_id == user_id and intent.created_at > cutoff_date
        ]
        
        # Get recent synchronicities
        recent_synchronicities = [
            event for event in self.synchronicity_detector.synchronicity_events
            if event.user_id == user_id and event.timestamp > cutoff_date
        ]
        
        # Get recent resonance matches
        recent_matches = [
            match for match in self.resonance_history
            if match.timestamp > cutoff_date
        ]
        
        # Calculate statistics
        intent_types = Counter(intent.intent_type.value for intent in recent_intents)
        synchronicity_types = Counter(event.event_type.value for event in recent_synchronicities)
        average_resonance = np.mean([match.get_total_score() for match in recent_matches]) if recent_matches else 0.0
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_intents": len(recent_intents),
            "total_synchronicities": len(recent_synchronicities),
            "total_matches": len(recent_matches),
            "intent_types": dict(intent_types),
            "synchronicity_types": dict(synchronicity_types),
            "average_resonance_score": average_resonance,
            "most_common_intent": intent_types.most_common(1)[0][0] if intent_types else None,
            "most_common_synchronicity": synchronicity_types.most_common(1)[0][0] if synchronicity_types else None
        }
    
    def save_state(self, filepath: str) -> bool:
        """Save oracle state to a file"""
        try:
            state = {
                "intent_history": [
                    {
                        "intent_id": intent.intent_id,
                        "intent_type": intent.intent_type.value,
                        "description": intent.description,
                        "keywords": intent.keywords,
                        "energy_types": [et.value for et in intent.energy_types],
                        "intensity": intent.intensity,
                        "created_at": intent.created_at.isoformat(),
                        "user_id": intent.user_id,
                        "context": intent.context
                    } for intent in self.intent_matcher.intent_history
                ],
                "synchronicity_events": [
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "description": event.description,
                        "significance": event.significance,
                        "timestamp": event.timestamp.isoformat(),
                        "location": event.location,
                        "related_intents": event.related_intents,
                        "patterns": event.patterns,
                        "user_id": event.user_id
                    } for event in self.synchronicity_detector.synchronicity_events
                ],
                "resonance_history": [
                    {
                        "match_id": match.match_id,
                        "intent_id": match.intent_id,
                        "guidance_id": match.guidance_id,
                        "resonance_score": match.resonance_score,
                        "matching_keywords": match.matching_keywords,
                        "energy_alignment": match.energy_alignment,
                        "synchronicity_boost": match.synchronicity_boost,
                        "timestamp": match.timestamp.isoformat()
                    } for match in self.resonance_history
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving oracle state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """Load oracle state from a file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.intent_matcher.intent_history.clear()
            self.synchronicity_detector.synchronicity_events.clear()
            self.resonance_history.clear()
            
            # Load intent history
            for intent_data in state.get("intent_history", []):
                intent = Intent(
                    intent_id=intent_data["intent_id"],
                    intent_type=IntentType(intent_data["intent_type"]),
                    description=intent_data["description"],
                    keywords=intent_data["keywords"],
                    energy_types=[AngelicEnergyType(et) for et in intent_data["energy_types"]],
                    intensity=intent_data["intensity"],
                    created_at=datetime.fromisoformat(intent_data["created_at"]),
                    user_id=intent_data["user_id"],
                    context=intent_data["context"]
                )
                self.intent_matcher.intent_history.append(intent)
            
            # Load synchronicity events
            for event_data in state.get("synchronicity_events", []):
                event = SynchronicityEvent(
                    event_id=event_data["event_id"],
                    event_type=SynchronicityType(event_data["event_type"]),
                    description=event_data["description"],
                    significance=event_data["significance"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    location=event_data["location"],
                    related_intents=event_data["related_intents"],
                    patterns=event_data["patterns"],
                    user_id=event_data["user_id"]
                )
                self.synchronicity_detector.synchronicity_events.append(event)
            
            # Load resonance history
            for match_data in state.get("resonance_history", []):
                match = ResonanceMatch(
                    match_id=match_data["match_id"],
                    intent_id=match_data["intent_id"],
                    guidance_id=match_data["guidance_id"],
                    resonance_score=match_data["resonance_score"],
                    matching_keywords=match_data["matching_keywords"],
                    energy_alignment=match_data["energy_alignment"],
                    synchronicity_boost=match_data["synchronicity_boost"],
                    timestamp=datetime.fromisoformat(match_data["timestamp"])
                )
                self.resonance_history.append(match)
            
            return True
        except Exception as e:
            print(f"Error loading oracle state: {e}")
            return False