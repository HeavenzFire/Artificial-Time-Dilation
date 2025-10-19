"""
Archangel System

Universal archangels that provide powerful spiritual guidance and intervention.
Each archangel has specific domains of influence and can be invoked for major life guidance.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import random
import json

from .core import AngelicCore, AngelicEnergy, AngelicEnergyType, DivineConnection


class ArchangelDomain(Enum):
    """Domains of archangel influence"""
    PROTECTION = "protection"
    HEALING = "healing"
    GUIDANCE = "guidance"
    WISDOM = "wisdom"
    LOVE = "love"
    STRENGTH = "strength"
    PEACE = "peace"
    JOY = "joy"
    ABUNDANCE = "abundance"
    TRANSFORMATION = "transformation"
    JUSTICE = "justice"
    COMMUNICATION = "communication"
    CREATIVITY = "creativity"
    SPIRITUAL_WARFARE = "spiritual_warfare"


@dataclass
class Archangel:
    """Represents a universal archangel"""
    name: str
    domain: ArchangelDomain
    energy_type: AngelicEnergyType
    description: str
    attributes: List[str]
    powers: List[str]
    invocation_chants: List[str]
    symbols: List[str]
    colors: List[str]
    elements: List[str]
    chakra_association: str
    planetary_ruler: str
    day_of_week: str
    crystal_associations: List[str]
    invocation_strength: float = 0.0
    last_invocation: Optional[datetime] = None
    invocation_count: int = 0
    
    def get_invocation_power(self) -> float:
        """Calculate current invocation power based on recent invocations"""
        if not self.last_invocation:
            return 0.1
        
        days_since = (datetime.now() - self.last_invocation).days
        if days_since < 1:
            return min(1.0, self.invocation_strength + 0.2)
        elif days_since < 7:
            return max(0.1, self.invocation_strength - 0.1)
        else:
            return max(0.1, self.invocation_strength - 0.3)
    
    def can_be_invoked(self, purpose: str) -> bool:
        """Check if this archangel can be invoked for a specific purpose"""
        purpose_lower = purpose.lower()
        return any(attr.lower() in purpose_lower for attr in self.attributes) or \
               any(power.lower() in purpose_lower for power in self.powers)


class ArchangelSystem:
    """Manages the universal archangel system"""
    
    def __init__(self, angelic_core: AngelicCore):
        self.angelic_core = angelic_core
        self.archangels: Dict[str, Archangel] = {}
        self.invocation_history: List[Dict] = []
        
        # Initialize the archangels
        self._initialize_archangels()
    
    def _initialize_archangels(self):
        """Initialize the universal archangels"""
        archangel_definitions = [
            {
                "name": "Michael",
                "domain": ArchangelDomain.PROTECTION,
                "energy_type": AngelicEnergyType.PROTECTION,
                "description": "The Archangel of Protection and Spiritual Warfare",
                "attributes": ["protection", "courage", "strength", "justice", "leadership"],
                "powers": ["spiritual_warfare", "protection", "courage", "strength", "justice"],
                "invocation_chants": [
                    "Archangel Michael, protect me with your sword of light",
                    "Michael, guardian of the divine, shield me from harm",
                    "Archangel Michael, bring me courage and strength"
                ],
                "symbols": ["⚔️", "🛡️", "🔥", "👑"],
                "colors": ["#FF0000", "#FFD700", "#FFA500"],
                "elements": ["Fire", "Air"],
                "chakra_association": "Solar Plexus",
                "planetary_ruler": "Sun",
                "day_of_week": "Sunday",
                "crystal_associations": ["Red Jasper", "Carnelian", "Citrine", "Tiger's Eye"]
            },
            {
                "name": "Gabriel",
                "domain": ArchangelDomain.COMMUNICATION,
                "energy_type": AngelicEnergyType.GUIDANCE,
                "description": "The Archangel of Communication and Divine Messages",
                "attributes": ["communication", "messages", "creativity", "intuition", "clarity"],
                "powers": ["communication", "divine_messages", "creativity", "intuition", "clarity"],
                "invocation_chants": [
                    "Archangel Gabriel, bring me clear communication",
                    "Gabriel, messenger of God, open my channels of intuition",
                    "Archangel Gabriel, inspire my creative expression"
                ],
                "symbols": ["📢", "📜", "🕊️", "📖"],
                "colors": ["#FFFFFF", "#87CEEB", "#E6E6FA"],
                "elements": ["Air", "Water"],
                "chakra_association": "Throat",
                "planetary_ruler": "Moon",
                "day_of_week": "Monday",
                "crystal_associations": ["Selenite", "Moonstone", "Clear Quartz", "Aquamarine"]
            },
            {
                "name": "Raphael",
                "domain": ArchangelDomain.HEALING,
                "energy_type": AngelicEnergyType.HEALING,
                "description": "The Archangel of Healing and Divine Medicine",
                "attributes": ["healing", "medicine", "travel", "science", "knowledge"],
                "powers": ["healing", "divine_medicine", "travel_protection", "scientific_discovery"],
                "invocation_chants": [
                    "Archangel Raphael, heal my body, mind, and spirit",
                    "Raphael, divine healer, restore my wholeness",
                    "Archangel Raphael, guide my healing journey"
                ],
                "symbols": ["💚", "🏥", "⚕️", "🌿"],
                "colors": ["#00FF00", "#32CD32", "#90EE90"],
                "elements": ["Water", "Earth"],
                "chakra_association": "Heart",
                "planetary_ruler": "Mercury",
                "day_of_week": "Wednesday",
                "crystal_associations": ["Green Aventurine", "Emerald", "Malachite", "Jade"]
            },
            {
                "name": "Uriel",
                "domain": ArchangelDomain.WISDOM,
                "energy_type": AngelicEnergyType.WISDOM,
                "description": "The Archangel of Wisdom and Divine Knowledge",
                "attributes": ["wisdom", "knowledge", "understanding", "insight", "enlightenment"],
                "powers": ["wisdom", "divine_knowledge", "insight", "enlightenment", "understanding"],
                "invocation_chants": [
                    "Archangel Uriel, illuminate my path with divine wisdom",
                    "Uriel, keeper of knowledge, open my mind to truth",
                    "Archangel Uriel, bring me clarity and understanding"
                ],
                "symbols": ["📚", "💡", "🔍", "🌟"],
                "colors": ["#FFD700", "#FFA500", "#FF8C00"],
                "elements": ["Fire", "Air"],
                "chakra_association": "Third Eye",
                "planetary_ruler": "Sun",
                "day_of_week": "Sunday",
                "crystal_associations": ["Amethyst", "Lapis Lazuli", "Sodalite", "Fluorite"]
            },
            {
                "name": "Chamuel",
                "domain": ArchangelDomain.LOVE,
                "energy_type": AngelicEnergyType.LOVE,
                "description": "The Archangel of Love and Relationships",
                "attributes": ["love", "relationships", "compassion", "forgiveness", "unity"],
                "powers": ["love", "relationship_healing", "compassion", "forgiveness", "unity"],
                "invocation_chants": [
                    "Archangel Chamuel, open my heart to divine love",
                    "Chamuel, angel of love, heal my relationships",
                    "Archangel Chamuel, bring me compassion and forgiveness"
                ],
                "symbols": ["💖", "💕", "🌹", "🕊️"],
                "colors": ["#FF69B4", "#FFB6C1", "#FFC0CB"],
                "elements": ["Water", "Fire"],
                "chakra_association": "Heart",
                "planetary_ruler": "Venus",
                "day_of_week": "Friday",
                "crystal_associations": ["Rose Quartz", "Pink Tourmaline", "Rhodonite", "Morganite"]
            },
            {
                "name": "Jophiel",
                "domain": ArchangelDomain.CREATIVITY,
                "energy_type": AngelicEnergyType.JOY,
                "description": "The Archangel of Beauty and Creativity",
                "attributes": ["beauty", "creativity", "art", "joy", "inspiration"],
                "powers": ["creativity", "beauty", "artistic_inspiration", "joy", "aesthetic_sense"],
                "invocation_chants": [
                    "Archangel Jophiel, inspire my creative expression",
                    "Jophiel, angel of beauty, awaken my artistic soul",
                    "Archangel Jophiel, bring me joy and inspiration"
                ],
                "symbols": ["🎨", "✨", "🌸", "🦋"],
                "colors": ["#FF69B4", "#FFB6C1", "#FFC0CB", "#FFD700"],
                "elements": ["Air", "Fire"],
                "chakra_association": "Sacral",
                "planetary_ruler": "Venus",
                "day_of_week": "Friday",
                "crystal_associations": ["Citrine", "Sunstone", "Orange Calcite", "Carnelian"]
            },
            {
                "name": "Zadkiel",
                "domain": ArchangelDomain.TRANSFORMATION,
                "energy_type": AngelicEnergyType.TRANSFORMATION,
                "description": "The Archangel of Mercy and Transformation",
                "attributes": ["mercy", "forgiveness", "transformation", "freedom", "liberation"],
                "powers": ["mercy", "forgiveness", "transformation", "freedom", "liberation"],
                "invocation_chants": [
                    "Archangel Zadkiel, transform my life with divine mercy",
                    "Zadkiel, angel of mercy, free me from limitations",
                    "Archangel Zadkiel, bring me forgiveness and transformation"
                ],
                "symbols": ["🦋", "🔓", "✨", "🌟"],
                "colors": ["#8A2BE2", "#9370DB", "#DA70D6"],
                "elements": ["Water", "Air"],
                "chakra_association": "Crown",
                "planetary_ruler": "Jupiter",
                "day_of_week": "Thursday",
                "crystal_associations": ["Amethyst", "Lepidolite", "Purple Fluorite", "Sugilite"]
            },
            {
                "name": "Metatron",
                "domain": ArchangelDomain.SPIRITUAL_WARFARE,
                "energy_type": AngelicEnergyType.STRENGTH,
                "description": "The Archangel of Spiritual Warfare and Divine Justice",
                "attributes": ["spiritual_warfare", "justice", "truth", "protection", "divine_will"],
                "powers": ["spiritual_warfare", "divine_justice", "truth_revelation", "protection", "divine_will"],
                "invocation_chants": [
                    "Archangel Metatron, protect me in spiritual battle",
                    "Metatron, keeper of divine justice, reveal truth to me",
                    "Archangel Metatron, align me with divine will"
                ],
                "symbols": ["⚡", "👁️", "🔥", "⚔️"],
                "colors": ["#FFD700", "#FFA500", "#FF4500"],
                "elements": ["Fire", "Air"],
                "chakra_association": "Crown",
                "planetary_ruler": "Sun",
                "day_of_week": "Sunday",
                "crystal_associations": ["Clear Quartz", "Diamond", "Herkimer Diamond", "Selenite"]
            }
        ]
        
        for archangel_data in archangel_definitions:
            archangel = Archangel(**archangel_data)
            self.archangels[archangel.name.lower()] = archangel
    
    def invoke_archangel(self, archangel_name: str, purpose: str, duration_minutes: int = 15) -> Dict:
        """
        Invoke an archangel for a specific purpose.
        
        Args:
            archangel_name: Name of the archangel to invoke
            purpose: The purpose of the invocation
            duration_minutes: Duration of the invocation in minutes
            
        Returns:
            Dictionary containing invocation results
        """
        archangel_name_lower = archangel_name.lower()
        if archangel_name_lower not in self.archangels:
            return {"success": False, "error": f"Archangel {archangel_name} not found"}
        
        archangel = self.archangels[archangel_name_lower]
        
        # Check if archangel can be invoked for this purpose
        if not archangel.can_be_invoked(purpose):
            return {
                "success": False, 
                "error": f"Archangel {archangel_name} cannot be invoked for this purpose"
            }
        
        # Calculate invocation power
        invocation_power = archangel.get_invocation_power()
        
        # Update archangel stats
        archangel.invocation_strength = min(1.0, archangel.invocation_strength + 0.1)
        archangel.last_invocation = datetime.now()
        archangel.invocation_count += 1
        
        # Record invocation
        invocation_record = {
            "archangel_name": archangel.name,
            "purpose": purpose,
            "duration_minutes": duration_minutes,
            "invocation_power": invocation_power,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        self.invocation_history.append(invocation_record)
        
        # Generate invocation message
        invocation_message = self._generate_invocation_message(archangel, purpose, invocation_power)
        
        # Establish divine connection
        connection_id = self.angelic_core.establish_connection(
            purpose=f"Archangel {archangel.name} - {purpose}",
            energy_types=[archangel.energy_type]
        )
        
        return {
            "success": True,
            "archangel_name": archangel.name,
            "domain": archangel.domain.value,
            "invocation_power": invocation_power,
            "message": invocation_message,
            "connection_id": connection_id,
            "duration_minutes": duration_minutes,
            "chants": archangel.invocation_chants,
            "symbols": archangel.symbols,
            "colors": archangel.colors,
            "crystals": archangel.crystal_associations
        }
    
    def _generate_invocation_message(self, archangel: Archangel, purpose: str, power: float) -> str:
        """Generate a personalized invocation message"""
        power_level = "powerful" if power > 0.7 else "moderate" if power > 0.4 else "gentle"
        
        messages = {
            "powerful": [
                f"Archangel {archangel.name} responds with great power to your call for {purpose}.",
                f"The mighty {archangel.name} channels divine energy for your {purpose}.",
                f"Archangel {archangel.name} manifests with full strength for your {purpose}."
            ],
            "moderate": [
                f"Archangel {archangel.name} answers your call for {purpose} with steady presence.",
                f"The wise {archangel.name} offers guidance for your {purpose}.",
                f"Archangel {archangel.name} brings balanced energy to your {purpose}."
            ],
            "gentle": [
                f"Archangel {archangel.name} approaches gently for your {purpose}.",
                f"The loving {archangel.name} offers soft guidance for your {purpose}.",
                f"Archangel {archangel.name} brings gentle energy to your {purpose}."
            ]
        }
        
        base_message = random.choice(messages[power_level])
        
        # Add domain-specific guidance
        domain_guidance = {
            ArchangelDomain.PROTECTION: "You are now surrounded by divine protection and strength.",
            ArchangelDomain.HEALING: "Healing energy flows through your being, restoring wholeness.",
            ArchangelDomain.GUIDANCE: "Divine guidance illuminates your path forward.",
            ArchangelDomain.WISDOM: "Ancient wisdom opens your mind to new understanding.",
            ArchangelDomain.LOVE: "Divine love fills your heart and heals all wounds.",
            ArchangelDomain.CREATIVITY: "Creative inspiration flows through your artistic soul.",
            ArchangelDomain.TRANSFORMATION: "Transformation begins as old patterns dissolve.",
            ArchangelDomain.JUSTICE: "Divine justice aligns all things in perfect order.",
            ArchangelDomain.COMMUNICATION: "Clear communication channels open to your consciousness.",
            ArchangelDomain.SPIRITUAL_WARFARE: "Spiritual strength rises to meet any challenge."
        }
        
        domain_message = domain_guidance.get(archangel.domain, "Divine blessings flow to you.")
        
        return f"{base_message} {domain_message}"
    
    def get_archangel_info(self, archangel_name: str) -> Optional[Dict]:
        """Get detailed information about an archangel"""
        archangel_name_lower = archangel_name.lower()
        if archangel_name_lower not in self.archangels:
            return None
        
        archangel = self.archangels[archangel_name_lower]
        
        return {
            "name": archangel.name,
            "domain": archangel.domain.value,
            "energy_type": archangel.energy_type.value,
            "description": archangel.description,
            "attributes": archangel.attributes,
            "powers": archangel.powers,
            "invocation_chants": archangel.invocation_chants,
            "symbols": archangel.symbols,
            "colors": archangel.colors,
            "elements": archangel.elements,
            "chakra_association": archangel.chakra_association,
            "planetary_ruler": archangel.planetary_ruler,
            "day_of_week": archangel.day_of_week,
            "crystal_associations": archangel.crystal_associations,
            "invocation_power": archangel.get_invocation_power(),
            "invocation_count": archangel.invocation_count,
            "last_invocation": archangel.last_invocation.isoformat() if archangel.last_invocation else None
        }
    
    def get_all_archangels(self) -> List[Dict]:
        """Get information about all archangels"""
        return [self.get_archangel_info(name) for name in self.archangels.keys()]
    
    def get_archangels_by_domain(self, domain: ArchangelDomain) -> List[Dict]:
        """Get archangels that specialize in a specific domain"""
        return [
            self.get_archangel_info(name) for name, archangel in self.archangels.items()
            if archangel.domain == domain
        ]
    
    def get_invocation_history(self, days: int = 30) -> List[Dict]:
        """Get invocation history for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            invocation for invocation in self.invocation_history
            if datetime.fromisoformat(invocation["timestamp"]) > cutoff_date
        ]
    
    def get_most_invoked_archangel(self) -> Optional[Dict]:
        """Get the most frequently invoked archangel"""
        if not self.archangels:
            return None
        
        most_invoked = max(self.archangels.values(), key=lambda a: a.invocation_count)
        return self.get_archangel_info(most_invoked.name)
    
    def save_state(self, filepath: str) -> bool:
        """Save archangel system state to a file"""
        try:
            state = {
                "archangels": {
                    name: {
                        "invocation_strength": archangel.invocation_strength,
                        "last_invocation": archangel.last_invocation.isoformat() if archangel.last_invocation else None,
                        "invocation_count": archangel.invocation_count
                    } for name, archangel in self.archangels.items()
                },
                "invocation_history": self.invocation_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving archangel state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """Load archangel system state from a file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Update archangel stats
            for name, stats in state.get("archangels", {}).items():
                if name in self.archangels:
                    self.archangels[name].invocation_strength = stats["invocation_strength"]
                    self.archangels[name].last_invocation = datetime.fromisoformat(stats["last_invocation"]) if stats["last_invocation"] else None
                    self.archangels[name].invocation_count = stats["invocation_count"]
            
            # Load invocation history
            self.invocation_history = state.get("invocation_history", [])
            
            return True
        except Exception as e:
            print(f"Error loading archangel state: {e}")
            return False