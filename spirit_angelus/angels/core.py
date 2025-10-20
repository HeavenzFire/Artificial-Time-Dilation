"""
Angelic Core System

The foundational angelic energy and connection system for the Spirit Angelus Framework.
Provides the base classes and energy management for all angelic interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime, timedelta


class AngelicEnergyType(Enum):
    """Types of angelic energy frequencies"""
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


class DivineConnectionLevel(Enum):
    """Levels of divine connection strength"""
    AWARENESS = 1
    CONNECTION = 2
    COMMUNION = 3
    UNION = 4
    TRANSCENDENCE = 5


@dataclass
class AngelicEnergy:
    """Represents a specific type and intensity of angelic energy"""
    energy_type: AngelicEnergyType
    intensity: float  # 0.0 to 1.0
    frequency: float  # Hz
    color: str  # Hex color code
    symbol: str  # Unicode symbol
    description: str
    attributes: Dict[str, Any]
    
    def __post_init__(self):
        """Validate energy parameters"""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")


@dataclass
class DivineConnection:
    """Represents a connection to divine/angelic realms"""
    connection_id: str
    level: DivineConnectionLevel
    energy_signature: List[AngelicEnergy]
    established_at: datetime
    last_communication: Optional[datetime]
    strength: float  # 0.0 to 1.0
    purpose: str
    guidance_received: List[str]
    
    def get_connection_age(self) -> timedelta:
        """Get the age of this connection"""
        return datetime.now() - self.established_at
    
    def is_active(self, threshold_hours: int = 24) -> bool:
        """Check if connection is still active"""
        if not self.last_communication:
            return False
        return (datetime.now() - self.last_communication).total_seconds() < threshold_hours * 3600


class AngelicCore:
    """
    Core angelic system that manages energy, connections, and divine interactions.
    
    This is the central hub for all angelic activities in the Spirit Angelus Framework.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Angelic Core system"""
        self.config = config or {}
        self.connections: Dict[str, DivineConnection] = {}
        self.energy_registry: Dict[AngelicEnergyType, AngelicEnergy] = {}
        self.meditation_sessions: List[Dict] = []
        self.synchronicities: List[Dict] = []
        
        # Initialize energy types
        self._initialize_energy_types()
        
        # Load any existing connections from config
        self._load_connections()
    
    def _initialize_energy_types(self):
        """Initialize the registry of angelic energy types"""
        energy_definitions = {
            AngelicEnergyType.PROTECTION: {
                "intensity": 0.8,
                "frequency": 432.0,
                "color": "#FF6B6B",
                "symbol": "🛡️",
                "description": "Divine protection and safety",
                "attributes": {"shielding": True, "boundaries": True}
            },
            AngelicEnergyType.HEALING: {
                "intensity": 0.9,
                "frequency": 528.0,
                "color": "#4ECDC4",
                "symbol": "💚",
                "description": "Healing and restoration",
                "attributes": {"regeneration": True, "purification": True}
            },
            AngelicEnergyType.GUIDANCE: {
                "intensity": 0.7,
                "frequency": 639.0,
                "color": "#45B7D1",
                "symbol": "🌟",
                "description": "Divine guidance and direction",
                "attributes": {"clarity": True, "wisdom": True}
            },
            AngelicEnergyType.WISDOM: {
                "intensity": 0.8,
                "frequency": 741.0,
                "color": "#96CEB4",
                "symbol": "📚",
                "description": "Ancient wisdom and knowledge",
                "attributes": {"understanding": True, "insight": True}
            },
            AngelicEnergyType.LOVE: {
                "intensity": 1.0,
                "frequency": 528.0,
                "color": "#FFB6C1",
                "symbol": "💖",
                "description": "Unconditional divine love",
                "attributes": {"compassion": True, "acceptance": True}
            },
            AngelicEnergyType.STRENGTH: {
                "intensity": 0.8,
                "frequency": 852.0,
                "color": "#DDA0DD",
                "symbol": "💪",
                "description": "Inner strength and courage",
                "attributes": {"resilience": True, "determination": True}
            },
            AngelicEnergyType.PEACE: {
                "intensity": 0.9,
                "frequency": 432.0,
                "color": "#87CEEB",
                "symbol": "🕊️",
                "description": "Inner peace and tranquility",
                "attributes": {"calm": True, "serenity": True}
            },
            AngelicEnergyType.JOY: {
                "intensity": 0.9,
                "frequency": 528.0,
                "color": "#FFD700",
                "symbol": "✨",
                "description": "Divine joy and happiness",
                "attributes": {"celebration": True, "lightness": True}
            },
            AngelicEnergyType.ABUNDANCE: {
                "intensity": 0.7,
                "frequency": 741.0,
                "color": "#FFA500",
                "symbol": "💰",
                "description": "Divine abundance and prosperity",
                "attributes": {"manifestation": True, "flow": True}
            },
            AngelicEnergyType.TRANSFORMATION: {
                "intensity": 0.8,
                "frequency": 852.0,
                "color": "#8A2BE2",
                "symbol": "🦋",
                "description": "Spiritual transformation and growth",
                "attributes": {"change": True, "evolution": True}
            }
        }
        
        for energy_type, params in energy_definitions.items():
            self.energy_registry[energy_type] = AngelicEnergy(
                energy_type=energy_type,
                **params
            )
    
    def _load_connections(self):
        """Load existing connections from configuration"""
        if "connections" in self.config:
            for conn_data in self.config["connections"]:
                connection = DivineConnection(**conn_data)
                self.connections[connection.connection_id] = connection
    
    def establish_connection(self, 
                           purpose: str, 
                           energy_types: List[AngelicEnergyType],
                           initial_level: DivineConnectionLevel = DivineConnectionLevel.AWARENESS) -> str:
        """
        Establish a new divine connection for a specific purpose.
        
        Args:
            purpose: The purpose of this connection
            energy_types: List of energy types to focus on
            initial_level: Starting connection level
            
        Returns:
            Connection ID for future reference
        """
        connection_id = f"conn_{int(time.time())}_{len(self.connections)}"
        
        # Create energy signature from requested types
        energy_signature = []
        for energy_type in energy_types:
            if energy_type in self.energy_registry:
                energy_signature.append(self.energy_registry[energy_type])
        
        # Create the connection
        connection = DivineConnection(
            connection_id=connection_id,
            level=initial_level,
            energy_signature=energy_signature,
            established_at=datetime.now(),
            last_communication=None,
            strength=0.1,  # Start with minimal strength
            purpose=purpose,
            guidance_received=[]
        )
        
        self.connections[connection_id] = connection
        return connection_id
    
    def strengthen_connection(self, connection_id: str, meditation_duration: int = 10) -> bool:
        """
        Strengthen a divine connection through meditation.
        
        Args:
            connection_id: ID of the connection to strengthen
            meditation_duration: Duration of meditation in minutes
            
        Returns:
            True if connection was strengthened successfully
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Calculate strength increase based on meditation duration
        strength_increase = min(0.1, meditation_duration / 100.0)
        connection.strength = min(1.0, connection.strength + strength_increase)
        
        # Update last communication
        connection.last_communication = datetime.now()
        
        # Record meditation session
        self.meditation_sessions.append({
            "connection_id": connection_id,
            "duration": meditation_duration,
            "timestamp": datetime.now(),
            "strength_before": connection.strength - strength_increase,
            "strength_after": connection.strength
        })
        
        return True
    
    def receive_guidance(self, connection_id: str, guidance: str) -> bool:
        """
        Record guidance received through a divine connection.
        
        Args:
            connection_id: ID of the connection
            guidance: The guidance message received
            
        Returns:
            True if guidance was recorded successfully
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.guidance_received.append(guidance)
        connection.last_communication = datetime.now()
        
        return True
    
    def detect_synchronicity(self, event: str, significance: float) -> bool:
        """
        Detect and record a synchronicity event.
        
        Args:
            event: Description of the synchronicity
            significance: Significance level (0.0 to 1.0)
            
        Returns:
            True if synchronicity was significant enough to record
        """
        if significance < 0.5:  # Threshold for significance
            return False
        
        synchronicity = {
            "event": event,
            "significance": significance,
            "timestamp": datetime.now(),
            "connections_affected": []
        }
        
        # Check which connections might be related
        for conn_id, connection in self.connections.items():
            if connection.is_active():
                # Simple keyword matching for now
                if any(keyword in event.lower() for keyword in connection.purpose.lower().split()):
                    synchronicity["connections_affected"].append(conn_id)
        
        self.synchronicities.append(synchronicity)
        return True
    
    def get_energy_reading(self, connection_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive energy reading for a connection.
        
        Args:
            connection_id: ID of the connection
            
        Returns:
            Dictionary containing energy analysis
        """
        if connection_id not in self.connections:
            return {}
        
        connection = self.connections[connection_id]
        
        # Calculate energy composition
        energy_composition = {}
        for energy in connection.energy_signature:
            energy_composition[energy.energy_type.value] = {
                "intensity": energy.intensity,
                "frequency": energy.frequency,
                "color": energy.color,
                "symbol": energy.symbol
            }
        
        return {
            "connection_id": connection_id,
            "level": connection.level.value,
            "strength": connection.strength,
            "purpose": connection.purpose,
            "energy_composition": energy_composition,
            "is_active": connection.is_active(),
            "guidance_count": len(connection.guidance_received),
            "connection_age_hours": connection.get_connection_age().total_seconds() / 3600
        }
    
    def get_all_connections(self) -> List[Dict[str, Any]]:
        """Get information about all established connections"""
        return [self.get_energy_reading(conn_id) for conn_id in self.connections.keys()]
    
    def get_recent_synchronicities(self, hours: int = 24) -> List[Dict]:
        """Get synchronicities from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            sync for sync in self.synchronicities 
            if sync["timestamp"] > cutoff_time
        ]
    
    def save_state(self, filepath: str) -> bool:
        """Save the current state to a file"""
        try:
            state = {
                "connections": [
                    {
                        "connection_id": conn.connection_id,
                        "level": conn.level.value,
                        "energy_signature": [
                            {
                                "energy_type": energy.energy_type.value,
                                "intensity": energy.intensity,
                                "frequency": energy.frequency,
                                "color": energy.color,
                                "symbol": energy.symbol,
                                "description": energy.description,
                                "attributes": energy.attributes
                            } for energy in conn.energy_signature
                        ],
                        "established_at": conn.established_at.isoformat(),
                        "last_communication": conn.last_communication.isoformat() if conn.last_communication else None,
                        "strength": conn.strength,
                        "purpose": conn.purpose,
                        "guidance_received": conn.guidance_received
                    } for conn in self.connections.values()
                ],
                "meditation_sessions": [
                    {
                        **session,
                        "timestamp": session["timestamp"].isoformat()
                    } for session in self.meditation_sessions
                ],
                "synchronicities": [
                    {
                        **sync,
                        "timestamp": sync["timestamp"].isoformat()
                    } for sync in self.synchronicities
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """Load state from a file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.connections.clear()
            self.meditation_sessions.clear()
            self.synchronicities.clear()
            
            # Load connections
            for conn_data in state.get("connections", []):
                # Convert energy signatures back to AngelicEnergy objects
                energy_signature = []
                for energy_data in conn_data["energy_signature"]:
                    energy = AngelicEnergy(
                        energy_type=AngelicEnergyType(energy_data["energy_type"]),
                        intensity=energy_data["intensity"],
                        frequency=energy_data["frequency"],
                        color=energy_data["color"],
                        symbol=energy_data["symbol"],
                        description=energy_data["description"],
                        attributes=energy_data["attributes"]
                    )
                    energy_signature.append(energy)
                
                connection = DivineConnection(
                    connection_id=conn_data["connection_id"],
                    level=DivineConnectionLevel(conn_data["level"]),
                    energy_signature=energy_signature,
                    established_at=datetime.fromisoformat(conn_data["established_at"]),
                    last_communication=datetime.fromisoformat(conn_data["last_communication"]) if conn_data["last_communication"] else None,
                    strength=conn_data["strength"],
                    purpose=conn_data["purpose"],
                    guidance_received=conn_data["guidance_received"]
                )
                
                self.connections[connection.connection_id] = connection
            
            # Load meditation sessions
            for session_data in state.get("meditation_sessions", []):
                session_data["timestamp"] = datetime.fromisoformat(session_data["timestamp"])
                self.meditation_sessions.append(session_data)
            
            # Load synchronicities
            for sync_data in state.get("synchronicities", []):
                sync_data["timestamp"] = datetime.fromisoformat(sync_data["timestamp"])
                self.synchronicities.append(sync_data)
            
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False