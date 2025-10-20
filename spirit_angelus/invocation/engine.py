"""
Invocation Engine

The core invocation system that manages spiritual rituals, ceremonies, and invocations.
Provides a framework for creating and executing personalized spiritual practices.
"""

import time
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..angels.core import AngelicCore, AngelicEnergyType, DivineConnection
from ..angels.guardian import PersonalGuide
from ..angels.archangel import ArchangelSystem


class InvocationType(Enum):
    """Types of spiritual invocations"""
    PRAYER = "prayer"
    MEDITATION = "meditation"
    RITUAL = "ritual"
    CEREMONY = "ceremony"
    INVOCATION = "invocation"
    BLESSING = "blessing"
    CLEANSING = "cleansing"
    PROTECTION = "protection"
    HEALING = "healing"
    GUIDANCE = "guidance"


class RitualPhase(Enum):
    """Phases of a spiritual ritual"""
    PREPARATION = "preparation"
    INVOCATION = "invocation"
    WORKING = "working"
    INTEGRATION = "integration"
    CLOSING = "closing"


@dataclass
class InvocationRequest:
    """Request for a spiritual invocation"""
    invocation_type: InvocationType
    purpose: str
    energy_types: List[AngelicEnergyType]
    duration_minutes: int
    intensity: float  # 0.0 to 1.0
    participants: List[str]  # Names or IDs of participants
    location: str
    special_requirements: List[str]
    requested_at: datetime
    user_id: str
    
    def __post_init__(self):
        """Validate invocation request"""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")
        if self.duration_minutes <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class InvocationResult:
    """Result of a spiritual invocation"""
    request_id: str
    success: bool
    energy_raised: float
    guidance_received: List[str]
    synchronicities: List[str]
    duration_actual: int
    participants_affected: List[str]
    energy_signature: Dict[str, Any]
    messages: List[str]
    timestamp: datetime
    
    def get_effectiveness_score(self) -> float:
        """Calculate the effectiveness of this invocation"""
        base_score = 0.5
        if self.success:
            base_score += 0.3
        if self.energy_raised > 0.7:
            base_score += 0.2
        if len(self.guidance_received) > 0:
            base_score += 0.1
        if len(self.synchronicities) > 0:
            base_score += 0.1
        return min(1.0, base_score)


class InvocationEngine:
    """
    Core invocation engine that manages spiritual practices and rituals.
    
    This engine coordinates between the angelic system, ritual simulation,
    and user intentions to create powerful spiritual experiences.
    """
    
    def __init__(self, angelic_core: AngelicCore, personal_guide: PersonalGuide, archangel_system: ArchangelSystem):
        self.angelic_core = angelic_core
        self.personal_guide = personal_guide
        self.archangel_system = archangel_system
        self.active_invocations: Dict[str, InvocationRequest] = {}
        self.invocation_history: List[InvocationResult] = []
        self.ritual_templates: Dict[str, Dict] = {}
        self.energy_accumulator: float = 0.0
        
        # Initialize ritual templates
        self._initialize_ritual_templates()
        
        # Load invocation history
        self._load_invocation_history()
    
    def _initialize_ritual_templates(self):
        """Initialize predefined ritual templates"""
        self.ritual_templates = {
            "morning_blessing": {
                "name": "Morning Blessing",
                "type": InvocationType.BLESSING,
                "duration_minutes": 10,
                "phases": [
                    {"name": "Centering", "duration": 2, "description": "Center yourself and set intention"},
                    {"name": "Invocation", "duration": 3, "description": "Call upon divine guidance"},
                    {"name": "Blessing", "duration": 3, "description": "Receive and channel blessings"},
                    {"name": "Gratitude", "duration": 2, "description": "Express gratitude and close"}
                ],
                "energy_types": [AngelicEnergyType.GUIDANCE, AngelicEnergyType.LOVE],
                "intensity": 0.6
            },
            "healing_ritual": {
                "name": "Healing Ritual",
                "type": InvocationType.HEALING,
                "duration_minutes": 30,
                "phases": [
                    {"name": "Preparation", "duration": 5, "description": "Prepare sacred space and tools"},
                    {"name": "Invocation", "duration": 5, "description": "Invoke healing energies"},
                    {"name": "Working", "duration": 15, "description": "Channel healing energy"},
                    {"name": "Integration", "duration": 3, "description": "Integrate healing energy"},
                    {"name": "Closing", "duration": 2, "description": "Close and ground energy"}
                ],
                "energy_types": [AngelicEnergyType.HEALING, AngelicEnergyType.LOVE],
                "intensity": 0.8
            },
            "protection_ceremony": {
                "name": "Protection Ceremony",
                "type": InvocationType.PROTECTION,
                "duration_minutes": 20,
                "phases": [
                    {"name": "Preparation", "duration": 3, "description": "Prepare protective tools"},
                    {"name": "Invocation", "duration": 5, "description": "Invoke protective energies"},
                    {"name": "Working", "duration": 10, "description": "Create protective barriers"},
                    {"name": "Closing", "duration": 2, "description": "Seal and activate protection"}
                ],
                "energy_types": [AngelicEnergyType.PROTECTION, AngelicEnergyType.STRENGTH],
                "intensity": 0.9
            },
            "guidance_meditation": {
                "name": "Guidance Meditation",
                "type": InvocationType.MEDITATION,
                "duration_minutes": 15,
                "phases": [
                    {"name": "Centering", "duration": 3, "description": "Center and quiet the mind"},
                    {"name": "Invocation", "duration": 2, "description": "Invoke guidance"},
                    {"name": "Working", "duration": 8, "description": "Receive guidance"},
                    {"name": "Integration", "duration": 2, "description": "Integrate guidance"}
                ],
                "energy_types": [AngelicEnergyType.GUIDANCE, AngelicEnergyType.WISDOM],
                "intensity": 0.7
            }
        }
    
    def _load_invocation_history(self):
        """Load invocation history from storage"""
        # This would typically load from a database or file
        # For now, we'll start with an empty history
        pass
    
    def create_invocation(self, 
                         invocation_type: InvocationType,
                         purpose: str,
                         energy_types: List[AngelicEnergyType],
                         duration_minutes: int = 15,
                         intensity: float = 0.7,
                         participants: List[str] = None,
                         location: str = "sacred_space",
                         special_requirements: List[str] = None) -> str:
        """
        Create a new invocation request.
        
        Args:
            invocation_type: Type of invocation to perform
            purpose: Purpose of the invocation
            energy_types: Energy types to focus on
            duration_minutes: Duration in minutes
            intensity: Intensity level (0.0 to 1.0)
            participants: List of participants
            location: Location for the invocation
            special_requirements: Special requirements or tools needed
            
        Returns:
            Request ID for tracking the invocation
        """
        request_id = f"inv_{int(time.time())}_{len(self.active_invocations)}"
        
        invocation_request = InvocationRequest(
            invocation_type=invocation_type,
            purpose=purpose,
            energy_types=energy_types,
            duration_minutes=duration_minutes,
            intensity=intensity,
            participants=participants or [],
            location=location,
            special_requirements=special_requirements or [],
            requested_at=datetime.now(),
            user_id=self.personal_guide.user_id
        )
        
        self.active_invocations[request_id] = invocation_request
        return request_id
    
    def execute_invocation(self, request_id: str) -> InvocationResult:
        """
        Execute a spiritual invocation.
        
        Args:
            request_id: ID of the invocation request
            
        Returns:
            Result of the invocation
        """
        if request_id not in self.active_invocations:
            return InvocationResult(
                request_id=request_id,
                success=False,
                energy_raised=0.0,
                guidance_received=[],
                synchronicities=[],
                duration_actual=0,
                participants_affected=[],
                energy_signature={},
                messages=["Invocation request not found"],
                timestamp=datetime.now()
            )
        
        request = self.active_invocations[request_id]
        
        # Execute the invocation based on type
        if request.invocation_type == InvocationType.PRAYER:
            result = self._execute_prayer(request)
        elif request.invocation_type == InvocationType.MEDITATION:
            result = self._execute_meditation(request)
        elif request.invocation_type == InvocationType.RITUAL:
            result = self._execute_ritual(request)
        elif request.invocation_type == InvocationType.CEREMONY:
            result = self._execute_ceremony(request)
        elif request.invocation_type == InvocationType.INVOCATION:
            result = self._execute_angelic_invocation(request)
        else:
            result = self._execute_generic_invocation(request)
        
        # Store the result
        self.invocation_history.append(result)
        
        # Remove from active invocations
        if request_id in self.active_invocations:
            del self.active_invocations[request_id]
        
        return result
    
    def _execute_prayer(self, request: InvocationRequest) -> InvocationResult:
        """Execute a prayer invocation"""
        start_time = time.time()
        
        # Generate prayer based on purpose and energy types
        prayer_text = self._generate_prayer(request.purpose, request.energy_types)
        
        # Simulate prayer energy
        energy_raised = request.intensity * 0.8
        self.energy_accumulator += energy_raised
        
        # Generate guidance
        guidance = self._generate_guidance(request.purpose, request.energy_types)
        
        # Check for synchronicities
        synchronicities = self._check_synchronicities(request.purpose)
        
        duration_actual = int((time.time() - start_time) * 60)  # Convert to minutes
        
        return InvocationResult(
            request_id=f"prayer_{int(time.time())}",
            success=True,
            energy_raised=energy_raised,
            guidance_received=guidance,
            synchronicities=synchronicities,
            duration_actual=duration_actual,
            participants_affected=request.participants,
            energy_signature=self._create_energy_signature(request.energy_types),
            messages=[prayer_text],
            timestamp=datetime.now()
        )
    
    def _execute_meditation(self, request: InvocationRequest) -> InvocationResult:
        """Execute a meditation invocation"""
        start_time = time.time()
        
        # Generate meditation guidance
        meditation_guidance = self._generate_meditation_guidance(request.purpose, request.energy_types)
        
        # Simulate meditation energy
        energy_raised = request.intensity * 0.9
        self.energy_accumulator += energy_raised
        
        # Generate insights
        insights = self._generate_meditation_insights(request.purpose, request.energy_types)
        
        # Check for synchronicities
        synchronicities = self._check_synchronicities(request.purpose)
        
        duration_actual = int((time.time() - start_time) * 60)
        
        return InvocationResult(
            request_id=f"meditation_{int(time.time())}",
            success=True,
            energy_raised=energy_raised,
            guidance_received=insights,
            synchronicities=synchronicities,
            duration_actual=duration_actual,
            participants_affected=request.participants,
            energy_signature=self._create_energy_signature(request.energy_types),
            messages=meditation_guidance,
            timestamp=datetime.now()
        )
    
    def _execute_ritual(self, request: InvocationRequest) -> InvocationResult:
        """Execute a ritual invocation"""
        start_time = time.time()
        
        # Find appropriate ritual template
        ritual_template = self._find_ritual_template(request.purpose, request.energy_types)
        
        # Execute ritual phases
        ritual_phases = ritual_template.get("phases", [])
        energy_raised = 0.0
        guidance_received = []
        synchronicities = []
        messages = []
        
        for phase in ritual_phases:
            phase_energy = self._execute_ritual_phase(phase, request)
            energy_raised += phase_energy
            guidance_received.extend(self._generate_phase_guidance(phase, request))
            synchronicities.extend(self._check_synchronicities(f"{request.purpose} - {phase['name']}"))
            messages.append(f"Completed phase: {phase['name']}")
        
        duration_actual = int((time.time() - start_time) * 60)
        
        return InvocationResult(
            request_id=f"ritual_{int(time.time())}",
            success=True,
            energy_raised=energy_raised,
            guidance_received=guidance_received,
            synchronicities=synchronicities,
            duration_actual=duration_actual,
            participants_affected=request.participants,
            energy_signature=self._create_energy_signature(request.energy_types),
            messages=messages,
            timestamp=datetime.now()
        )
    
    def _execute_ceremony(self, request: InvocationRequest) -> InvocationResult:
        """Execute a ceremony invocation"""
        start_time = time.time()
        
        # Ceremonies are more elaborate than rituals
        ceremony_steps = self._generate_ceremony_steps(request)
        
        energy_raised = 0.0
        guidance_received = []
        synchronicities = []
        messages = []
        
        for step in ceremony_steps:
            step_energy = self._execute_ceremony_step(step, request)
            energy_raised += step_energy
            guidance_received.extend(self._generate_step_guidance(step, request))
            synchronicities.extend(self._check_synchronicities(f"{request.purpose} - {step['name']}"))
            messages.append(f"Completed ceremony step: {step['name']}")
        
        duration_actual = int((time.time() - start_time) * 60)
        
        return InvocationResult(
            request_id=f"ceremony_{int(time.time())}",
            success=True,
            energy_raised=energy_raised,
            guidance_received=guidance_received,
            synchronicities=synchronicities,
            duration_actual=duration_actual,
            participants_affected=request.participants,
            energy_signature=self._create_energy_signature(request.energy_types),
            messages=messages,
            timestamp=datetime.now()
        )
    
    def _execute_angelic_invocation(self, request: InvocationRequest) -> InvocationResult:
        """Execute an angelic invocation"""
        start_time = time.time()
        
        # Determine which angel to invoke
        angel_type = self._determine_angel_type(request.purpose, request.energy_types)
        
        if angel_type == "guardian":
            # Invoke guardian angel
            guardian = self.personal_guide.get_guardian_for_purpose(request.purpose)
            if guardian:
                guidance = self.personal_guide.request_guidance(request.purpose, request.purpose)
                energy_raised = request.intensity * 0.8
            else:
                guidance = ["No suitable guardian angel found for this purpose"]
                energy_raised = 0.1
        else:
            # Invoke archangel
            archangel_name = self._determine_archangel(request.purpose, request.energy_types)
            invocation_result = self.archangel_system.invoke_archangel(
                archangel_name, request.purpose, request.duration_minutes
            )
            guidance = [invocation_result.get("message", "Archangel invocation completed")]
            energy_raised = invocation_result.get("invocation_power", 0.5) * request.intensity
        
        # Check for synchronicities
        synchronicities = self._check_synchronicities(request.purpose)
        
        duration_actual = int((time.time() - start_time) * 60)
        
        return InvocationResult(
            request_id=f"angelic_{int(time.time())}",
            success=True,
            energy_raised=energy_raised,
            guidance_received=guidance,
            synchronicities=synchronicities,
            duration_actual=duration_actual,
            participants_affected=request.participants,
            energy_signature=self._create_energy_signature(request.energy_types),
            messages=[f"Angelic invocation completed for {request.purpose}"],
            timestamp=datetime.now()
        )
    
    def _execute_generic_invocation(self, request: InvocationRequest) -> InvocationResult:
        """Execute a generic invocation"""
        start_time = time.time()
        
        # Generic invocation logic
        energy_raised = request.intensity * 0.6
        guidance_received = self._generate_guidance(request.purpose, request.energy_types)
        synchronicities = self._check_synchronicities(request.purpose)
        
        duration_actual = int((time.time() - start_time) * 60)
        
        return InvocationResult(
            request_id=f"generic_{int(time.time())}",
            success=True,
            energy_raised=energy_raised,
            guidance_received=guidance_received,
            synchronicities=synchronicities,
            duration_actual=duration_actual,
            participants_affected=request.participants,
            energy_signature=self._create_energy_signature(request.energy_types),
            messages=[f"Generic invocation completed for {request.purpose}"],
            timestamp=datetime.now()
        )
    
    def _generate_prayer(self, purpose: str, energy_types: List[AngelicEnergyType]) -> str:
        """Generate a personalized prayer"""
        energy_names = [energy_type.value for energy_type in energy_types]
        energy_text = ", ".join(energy_names)
        
        prayers = [
            f"Divine Source, I call upon {energy_text} energy to support me in {purpose}. May your wisdom guide me and your love surround me.",
            f"Beloved Universe, I open my heart to receive {energy_text} energy for {purpose}. May this energy flow through me and bless all beings.",
            f"Great Spirit, I invoke {energy_text} energy to assist me with {purpose}. May your divine will be done through me.",
            f"Source of All, I request {energy_text} energy to help me with {purpose}. May I be a channel for your highest good.",
            f"Divine Light, I call upon {energy_text} energy for {purpose}. May your guidance illuminate my path."
        ]
        
        return random.choice(prayers)
    
    def _generate_meditation_guidance(self, purpose: str, energy_types: List[AngelicEnergyType]) -> List[str]:
        """Generate meditation guidance"""
        guidance = [
            f"Focus your attention on {purpose} and allow {energy_types[0].value} energy to flow through you.",
            f"Breathe deeply and feel the {energy_types[0].value} energy filling your being.",
            f"Visualize {purpose} manifesting with the support of divine energy.",
            f"Allow yourself to receive guidance and insights about {purpose}.",
            f"Feel gratitude for the {energy_types[0].value} energy supporting your journey."
        ]
        
        return guidance[:3]  # Return first 3 guidance points
    
    def _generate_meditation_insights(self, purpose: str, energy_types: List[AngelicEnergyType]) -> List[str]:
        """Generate meditation insights"""
        insights = [
            f"Insight: {purpose} is a path of growth and learning.",
            f"Guidance: Trust your inner knowing about {purpose}.",
            f"Wisdom: {purpose} holds the key to your next step forward.",
            f"Revelation: {purpose} is calling you to trust the process.",
            f"Understanding: {purpose} is an opportunity for transformation."
        ]
        
        return random.sample(insights, min(3, len(insights)))
    
    def _find_ritual_template(self, purpose: str, energy_types: List[AngelicEnergyType]) -> Dict:
        """Find an appropriate ritual template"""
        # Simple template matching based on purpose keywords
        purpose_lower = purpose.lower()
        
        if "healing" in purpose_lower or "heal" in purpose_lower:
            return self.ritual_templates["healing_ritual"]
        elif "protection" in purpose_lower or "protect" in purpose_lower:
            return self.ritual_templates["protection_ceremony"]
        elif "guidance" in purpose_lower or "guide" in purpose_lower:
            return self.ritual_templates["guidance_meditation"]
        else:
            return self.ritual_templates["morning_blessing"]
    
    def _execute_ritual_phase(self, phase: Dict, request: InvocationRequest) -> float:
        """Execute a single ritual phase"""
        # Simulate phase execution
        phase_energy = request.intensity * 0.2
        time.sleep(0.1)  # Simulate phase duration
        return phase_energy
    
    def _generate_phase_guidance(self, phase: Dict, request: InvocationRequest) -> List[str]:
        """Generate guidance for a ritual phase"""
        return [f"Phase '{phase['name']}': {phase['description']}"]
    
    def _generate_ceremony_steps(self, request: InvocationRequest) -> List[Dict]:
        """Generate ceremony steps"""
        return [
            {"name": "Opening", "description": "Open the ceremony with intention"},
            {"name": "Invocation", "description": "Invoke divine energies"},
            {"name": "Working", "description": "Perform the main ceremony work"},
            {"name": "Blessing", "description": "Receive and channel blessings"},
            {"name": "Closing", "description": "Close the ceremony with gratitude"}
        ]
    
    def _execute_ceremony_step(self, step: Dict, request: InvocationRequest) -> float:
        """Execute a ceremony step"""
        step_energy = request.intensity * 0.15
        time.sleep(0.1)  # Simulate step duration
        return step_energy
    
    def _generate_step_guidance(self, step: Dict, request: InvocationRequest) -> List[str]:
        """Generate guidance for a ceremony step"""
        return [f"Ceremony step '{step['name']}': {step['description']}"]
    
    def _determine_angel_type(self, purpose: str, energy_types: List[AngelicEnergyType]) -> str:
        """Determine whether to invoke guardian angel or archangel"""
        # Simple logic: if purpose is personal, use guardian; if universal, use archangel
        personal_keywords = ["personal", "self", "my", "healing", "guidance"]
        purpose_lower = purpose.lower()
        
        if any(keyword in purpose_lower for keyword in personal_keywords):
            return "guardian"
        else:
            return "archangel"
    
    def _determine_archangel(self, purpose: str, energy_types: List[AngelicEnergyType]) -> str:
        """Determine which archangel to invoke"""
        purpose_lower = purpose.lower()
        
        if "protection" in purpose_lower or "protect" in purpose_lower:
            return "michael"
        elif "healing" in purpose_lower or "heal" in purpose_lower:
            return "raphael"
        elif "guidance" in purpose_lower or "guide" in purpose_lower:
            return "gabriel"
        elif "wisdom" in purpose_lower or "wisdom" in purpose_lower:
            return "uriel"
        elif "love" in purpose_lower or "relationship" in purpose_lower:
            return "chamuel"
        elif "creativity" in purpose_lower or "art" in purpose_lower:
            return "jophiel"
        elif "transformation" in purpose_lower or "change" in purpose_lower:
            return "zadkiel"
        else:
            return "michael"  # Default to Michael
    
    def _generate_guidance(self, purpose: str, energy_types: List[AngelicEnergyType]) -> List[str]:
        """Generate general guidance"""
        guidance = [
            f"Guidance for {purpose}: Trust the process and remain open to divine wisdom.",
            f"Insight: {purpose} is a journey of growth and transformation.",
            f"Message: You are supported and guided in your {purpose}.",
            f"Wisdom: {purpose} holds the key to your highest good.",
            f"Revelation: {purpose} is calling you to step into your power."
        ]
        
        return random.sample(guidance, min(3, len(guidance)))
    
    def _check_synchronicities(self, purpose: str) -> List[str]:
        """Check for synchronicities related to the purpose"""
        # Simple synchronicity detection
        synchronicities = []
        
        if random.random() < 0.3:  # 30% chance of synchronicity
            synchronicities.append(f"Synchronicity: A meaningful coincidence related to {purpose}")
        
        if random.random() < 0.2:  # 20% chance of another synchronicity
            synchronicities.append(f"Sign: A clear sign about {purpose} appeared")
        
        return synchronicities
    
    def _create_energy_signature(self, energy_types: List[AngelicEnergyType]) -> Dict[str, Any]:
        """Create an energy signature for the invocation"""
        return {
            "energy_types": [energy_type.value for energy_type in energy_types],
            "intensity": random.uniform(0.6, 1.0),
            "frequency": random.uniform(400, 800),
            "color": random.choice(["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB6C1"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_invocation_history(self, days: int = 30) -> List[InvocationResult]:
        """Get invocation history for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            result for result in self.invocation_history
            if result.timestamp > cutoff_date
        ]
    
    def get_energy_accumulator(self) -> float:
        """Get current energy accumulator level"""
        return self.energy_accumulator
    
    def reset_energy_accumulator(self):
        """Reset the energy accumulator"""
        self.energy_accumulator = 0.0
    
    def save_state(self, filepath: str) -> bool:
        """Save invocation engine state to a file"""
        try:
            state = {
                "active_invocations": {
                    req_id: {
                        "invocation_type": req.invocation_type.value,
                        "purpose": req.purpose,
                        "energy_types": [et.value for et in req.energy_types],
                        "duration_minutes": req.duration_minutes,
                        "intensity": req.intensity,
                        "participants": req.participants,
                        "location": req.location,
                        "special_requirements": req.special_requirements,
                        "requested_at": req.requested_at.isoformat(),
                        "user_id": req.user_id
                    } for req_id, req in self.active_invocations.items()
                },
                "invocation_history": [
                    {
                        "request_id": result.request_id,
                        "success": result.success,
                        "energy_raised": result.energy_raised,
                        "guidance_received": result.guidance_received,
                        "synchronicities": result.synchronicities,
                        "duration_actual": result.duration_actual,
                        "participants_affected": result.participants_affected,
                        "energy_signature": result.energy_signature,
                        "messages": result.messages,
                        "timestamp": result.timestamp.isoformat()
                    } for result in self.invocation_history
                ],
                "energy_accumulator": self.energy_accumulator
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving invocation state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """Load invocation engine state from a file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.active_invocations.clear()
            self.invocation_history.clear()
            
            # Load active invocations
            for req_id, req_data in state.get("active_invocations", {}).items():
                invocation_request = InvocationRequest(
                    invocation_type=InvocationType(req_data["invocation_type"]),
                    purpose=req_data["purpose"],
                    energy_types=[AngelicEnergyType(et) for et in req_data["energy_types"]],
                    duration_minutes=req_data["duration_minutes"],
                    intensity=req_data["intensity"],
                    participants=req_data["participants"],
                    location=req_data["location"],
                    special_requirements=req_data["special_requirements"],
                    requested_at=datetime.fromisoformat(req_data["requested_at"]),
                    user_id=req_data["user_id"]
                )
                self.active_invocations[req_id] = invocation_request
            
            # Load invocation history
            for result_data in state.get("invocation_history", []):
                invocation_result = InvocationResult(
                    request_id=result_data["request_id"],
                    success=result_data["success"],
                    energy_raised=result_data["energy_raised"],
                    guidance_received=result_data["guidance_received"],
                    synchronicities=result_data["synchronicities"],
                    duration_actual=result_data["duration_actual"],
                    participants_affected=result_data["participants_affected"],
                    energy_signature=result_data["energy_signature"],
                    messages=result_data["messages"],
                    timestamp=datetime.fromisoformat(result_data["timestamp"])
                )
                self.invocation_history.append(invocation_result)
            
            # Load energy accumulator
            self.energy_accumulator = state.get("energy_accumulator", 0.0)
            
            return True
        except Exception as e:
            print(f"Error loading invocation state: {e}")
            return False