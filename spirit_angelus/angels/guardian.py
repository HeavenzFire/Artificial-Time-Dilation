"""
Guardian Angel System

Personal guardian angels that provide individual protection, guidance, and spiritual support.
Each person can have multiple guardian angels for different aspects of their life.
"""

import random
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .core import AngelicCore, AngelicEnergy, AngelicEnergyType, DivineConnection


@dataclass
class GuardianAngel:
    """Represents a personal guardian angel"""
    name: str
    purpose: str
    energy_type: AngelicEnergyType
    personality_traits: List[str]
    special_abilities: List[str]
    connection_strength: float
    last_contact: Optional[datetime]
    guidance_history: List[str]
    protection_level: float
    assigned_date: datetime
    
    def get_guidance_style(self) -> str:
        """Get the angel's unique guidance style"""
        if "gentle" in self.personality_traits:
            return "soft and nurturing"
        elif "strong" in self.personality_traits:
            return "direct and powerful"
        elif "wise" in self.personality_traits:
            return "thoughtful and profound"
        else:
            return "balanced and supportive"
    
    def can_provide_guidance(self, topic: str) -> bool:
        """Check if this angel can provide guidance on a specific topic"""
        topic_lower = topic.lower()
        return any(ability.lower() in topic_lower for ability in self.special_abilities)
    
    def get_contact_frequency(self) -> str:
        """Get how often this angel makes contact"""
        if not self.last_contact:
            return "never"
        
        days_since = (datetime.now() - self.last_contact).days
        if days_since < 1:
            return "daily"
        elif days_since < 7:
            return "weekly"
        elif days_since < 30:
            return "monthly"
        else:
            return "rarely"


class PersonalGuide:
    """Manages personal guardian angels for an individual"""
    
    def __init__(self, user_id: str, angelic_core: AngelicCore):
        self.user_id = user_id
        self.angelic_core = angelic_core
        self.guardian_angels: Dict[str, GuardianAngel] = {}
        self.assignment_history: List[Dict] = []
        
        # Load existing guardians
        self._load_guardians()
    
    def _load_guardians(self):
        """Load existing guardian angels from storage"""
        # This would typically load from a database or file
        # For now, we'll start with an empty state
        pass
    
    def assign_guardian_angel(self, 
                            purpose: str, 
                            energy_type: AngelicEnergyType,
                            personality_preferences: List[str] = None) -> GuardianAngel:
        """
        Assign a new guardian angel for a specific purpose.
        
        Args:
            purpose: The specific purpose for this guardian
            energy_type: The type of energy this guardian should embody
            personality_preferences: Preferred personality traits
            
        Returns:
            The newly assigned GuardianAngel
        """
        # Generate a unique name based on purpose and energy type
        name = self._generate_angel_name(purpose, energy_type)
        
        # Select personality traits
        personality_traits = self._select_personality_traits(purpose, personality_preferences)
        
        # Determine special abilities based on purpose and energy type
        special_abilities = self._determine_special_abilities(purpose, energy_type)
        
        # Create the guardian angel
        guardian = GuardianAngel(
            name=name,
            purpose=purpose,
            energy_type=energy_type,
            personality_traits=personality_traits,
            special_abilities=special_abilities,
            connection_strength=0.1,  # Start with minimal connection
            last_contact=None,
            guidance_history=[],
            protection_level=0.5,  # Moderate protection
            assigned_date=datetime.now()
        )
        
        # Store the guardian
        guardian_id = f"guardian_{len(self.guardian_angels)}"
        self.guardian_angels[guardian_id] = guardian
        
        # Record the assignment
        self.assignment_history.append({
            "guardian_id": guardian_id,
            "purpose": purpose,
            "energy_type": energy_type.value,
            "assigned_date": datetime.now().isoformat(),
            "user_id": self.user_id
        })
        
        # Establish a divine connection for this guardian
        connection_id = self.angelic_core.establish_connection(
            purpose=f"Guardian Angel: {name} - {purpose}",
            energy_types=[energy_type]
        )
        
        return guardian
    
    def _generate_angel_name(self, purpose: str, energy_type: AngelicEnergyType) -> str:
        """Generate a unique angel name based on purpose and energy type"""
        # Angel name components
        prefixes = ["Ari", "Gab", "Mic", "Raf", "Uri", "Joph", "Cham", "Zad", "Sera", "Cher"]
        suffixes = ["iel", "ael", "iel", "ael", "iel", "ael", "iel", "ael", "iel", "ael"]
        
        # Use purpose and energy type to seed the random selection
        seed = f"{purpose}_{energy_type.value}_{self.user_id}"
        random.seed(hashlib.md5(seed.encode()).hexdigest())
        
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        return f"{prefix}{suffix}"
    
    def _select_personality_traits(self, purpose: str, preferences: List[str] = None) -> List[str]:
        """Select personality traits for the guardian angel"""
        all_traits = [
            "gentle", "strong", "wise", "compassionate", "courageous",
            "patient", "determined", "loving", "protective", "intuitive",
            "calm", "energetic", "mysterious", "playful", "serious"
        ]
        
        # Filter traits based on purpose
        purpose_traits = {
            "protection": ["strong", "courageous", "protective", "determined"],
            "healing": ["gentle", "compassionate", "loving", "patient"],
            "guidance": ["wise", "intuitive", "patient", "calm"],
            "love": ["loving", "compassionate", "gentle", "patient"],
            "strength": ["strong", "courageous", "determined", "energetic"],
            "peace": ["calm", "gentle", "patient", "serene"],
            "joy": ["playful", "energetic", "loving", "bright"],
            "abundance": ["energetic", "determined", "optimistic", "generous"],
            "transformation": ["mysterious", "wise", "intuitive", "patient"]
        }
        
        # Get traits for this purpose
        selected_traits = purpose_traits.get(purpose.lower(), all_traits[:3])
        
        # Add user preferences if provided
        if preferences:
            selected_traits.extend([p for p in preferences if p in all_traits])
        
        # Remove duplicates and limit to 4 traits
        return list(set(selected_traits))[:4]
    
    def _determine_special_abilities(self, purpose: str, energy_type: AngelicEnergyType) -> List[str]:
        """Determine special abilities based on purpose and energy type"""
        abilities = {
            "protection": ["shielding", "warning", "strength", "courage"],
            "healing": ["restoration", "purification", "comfort", "renewal"],
            "guidance": ["clarity", "direction", "insight", "wisdom"],
            "love": ["compassion", "acceptance", "forgiveness", "unity"],
            "strength": ["courage", "resilience", "determination", "power"],
            "peace": ["calm", "serenity", "balance", "tranquility"],
            "joy": ["celebration", "lightness", "happiness", "upliftment"],
            "abundance": ["manifestation", "flow", "prosperity", "generosity"],
            "transformation": ["change", "evolution", "growth", "renewal"]
        }
        
        return abilities.get(purpose.lower(), ["guidance", "support", "love"])
    
    def get_guardian_for_purpose(self, purpose: str) -> Optional[GuardianAngel]:
        """Get the guardian angel assigned for a specific purpose"""
        for guardian in self.guardian_angels.values():
            if purpose.lower() in guardian.purpose.lower():
                return guardian
        return None
    
    def request_guidance(self, purpose: str, question: str) -> Optional[str]:
        """
        Request guidance from the appropriate guardian angel.
        
        Args:
            purpose: The purpose/domain of the question
            question: The specific question or request
            
        Returns:
            Guidance message from the guardian angel, or None if no suitable guardian
        """
        guardian = self.get_guardian_for_purpose(purpose)
        if not guardian:
            return None
        
        # Check if guardian can provide guidance on this topic
        if not guardian.can_provide_guidance(question):
            return None
        
        # Generate guidance based on guardian's personality and abilities
        guidance = self._generate_guidance(guardian, question)
        
        # Record the guidance
        guardian.guidance_history.append(guidance)
        guardian.last_contact = datetime.now()
        
        # Strengthen the connection
        guardian.connection_strength = min(1.0, guardian.connection_strength + 0.05)
        
        return guidance
    
    def _generate_guidance(self, guardian: GuardianAngel, question: str) -> str:
        """Generate guidance from a guardian angel"""
        # This is a simplified guidance generator
        # In a real implementation, this would use more sophisticated AI
        
        style = guardian.get_guidance_style()
        abilities = ", ".join(guardian.special_abilities[:2])
        
        guidance_templates = [
            f"Dear one, through {abilities}, I offer you this guidance: {question} is a path of growth and learning.",
            f"Beloved soul, my {style} wisdom tells me that {question} holds the key to your next step forward.",
            f"Child of light, through {abilities}, I see that {question} is calling you to trust your inner knowing.",
            f"Dear heart, my {style} presence reminds you that {question} is an opportunity for transformation.",
            f"Beloved, through {abilities}, I guide you to see that {question} is a doorway to your highest good."
        ]
        
        # Select guidance based on guardian's energy type
        energy_guidance = {
            AngelicEnergyType.PROTECTION: "You are safe and protected in this moment.",
            AngelicEnergyType.HEALING: "Healing flows through you now, bringing wholeness and peace.",
            AngelicEnergyType.GUIDANCE: "The path forward is becoming clear to you.",
            AngelicEnergyType.WISDOM: "Ancient wisdom flows through your consciousness now.",
            AngelicEnergyType.LOVE: "You are deeply loved and supported by the universe.",
            AngelicEnergyType.STRENGTH: "Your inner strength is rising to meet this challenge.",
            AngelicEnergyType.PEACE: "Peace flows through your being, calming all concerns.",
            AngelicEnergyType.JOY: "Joy and lightness are your natural state of being.",
            AngelicEnergyType.ABUNDANCE: "Abundance flows to you in perfect timing.",
            AngelicEnergyType.TRANSFORMATION: "You are transforming into your highest self."
        }
        
        base_guidance = random.choice(guidance_templates)
        energy_message = energy_guidance.get(guardian.energy_type, "You are guided and supported.")
        
        return f"{base_guidance} {energy_message}"
    
    def strengthen_guardian_connection(self, guardian_id: str, meditation_duration: int = 10) -> bool:
        """Strengthen connection with a specific guardian angel"""
        if guardian_id not in self.guardian_angels:
            return False
        
        guardian = self.guardian_angels[guardian_id]
        
        # Increase connection strength
        strength_increase = min(0.1, meditation_duration / 100.0)
        guardian.connection_strength = min(1.0, guardian.connection_strength + strength_increase)
        
        # Update last contact
        guardian.last_contact = datetime.now()
        
        return True
    
    def get_guardian_status(self, guardian_id: str) -> Dict:
        """Get status information for a specific guardian angel"""
        if guardian_id not in self.guardian_angels:
            return {}
        
        guardian = self.guardian_angels[guardian_id]
        
        return {
            "name": guardian.name,
            "purpose": guardian.purpose,
            "energy_type": guardian.energy_type.value,
            "personality_traits": guardian.personality_traits,
            "special_abilities": guardian.special_abilities,
            "connection_strength": guardian.connection_strength,
            "protection_level": guardian.protection_level,
            "guidance_style": guardian.get_guidance_style(),
            "contact_frequency": guardian.get_contact_frequency(),
            "guidance_count": len(guardian.guidance_history),
            "assigned_date": guardian.assigned_date.isoformat(),
            "last_contact": guardian.last_contact.isoformat() if guardian.last_contact else None
        }
    
    def get_all_guardians(self) -> List[Dict]:
        """Get status for all guardian angels"""
        return [self.get_guardian_status(guardian_id) for guardian_id in self.guardian_angels.keys()]
    
    def save_guardians(self, filepath: str) -> bool:
        """Save guardian angels to a file"""
        try:
            data = {
                "user_id": self.user_id,
                "guardian_angels": {
                    guardian_id: {
                        "name": guardian.name,
                        "purpose": guardian.purpose,
                        "energy_type": guardian.energy_type.value,
                        "personality_traits": guardian.personality_traits,
                        "special_abilities": guardian.special_abilities,
                        "connection_strength": guardian.connection_strength,
                        "last_contact": guardian.last_contact.isoformat() if guardian.last_contact else None,
                        "guidance_history": guardian.guidance_history,
                        "protection_level": guardian.protection_level,
                        "assigned_date": guardian.assigned_date.isoformat()
                    } for guardian_id, guardian in self.guardian_angels.items()
                },
                "assignment_history": self.assignment_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving guardians: {e}")
            return False
    
    def load_guardians(self, filepath: str) -> bool:
        """Load guardian angels from a file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing guardians
            self.guardian_angels.clear()
            self.assignment_history = data.get("assignment_history", [])
            
            # Load guardian angels
            for guardian_id, guardian_data in data.get("guardian_angels", {}).items():
                guardian = GuardianAngel(
                    name=guardian_data["name"],
                    purpose=guardian_data["purpose"],
                    energy_type=AngelicEnergyType(guardian_data["energy_type"]),
                    personality_traits=guardian_data["personality_traits"],
                    special_abilities=guardian_data["special_abilities"],
                    connection_strength=guardian_data["connection_strength"],
                    last_contact=datetime.fromisoformat(guardian_data["last_contact"]) if guardian_data["last_contact"] else None,
                    guidance_history=guardian_data["guidance_history"],
                    protection_level=guardian_data["protection_level"],
                    assigned_date=datetime.fromisoformat(guardian_data["assigned_date"])
                )
                
                self.guardian_angels[guardian_id] = guardian
            
            return True
        except Exception as e:
            print(f"Error loading guardians: {e}")
            return False