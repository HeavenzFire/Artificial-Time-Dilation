from dataclasses import dataclass
from typing import List, Optional

from ..common.types import Intent, Guidance


@dataclass
class InvocationProfile:
    protection_level: int = 1  # 0..3
    archetype_bias: Optional[str] = None  # e.g., "Gabriel", "Michael"


class InvocationEngine:
    def __init__(self, profile: Optional[InvocationProfile] = None) -> None:
        self.profile = profile or InvocationProfile()

    def normalize(self, intent_text: str) -> str:
        text = intent_text.strip()
        text = " ".join(text.split())
        return text

    def expand(self, intent: Intent) -> List[str]:
        text = self.normalize(intent.text).lower()
        steps: List[str] = []
        if any(k in text for k in ["heal", "trauma", "wound"]):
            steps.extend([
                "Center breath for 60 seconds",
                "Acknowledge the wound without judgment",
                "Invite protective presence to witness",
            ])
        if any(k in text for k in ["clarity", "decision", "path"]):
            steps.extend([
                "List 3 options and consequences",
                "Ask for the most compassionate outcome",
            ])
        if not steps:
            steps = [
                "State intention clearly",
                "Ask for aligned guidance",
                "Commit to one small action today",
            ]
        if self.profile.protection_level > 0:
            steps.insert(0, "Establish protective boundary")
        return steps

    def invoke(self, intent: Intent) -> Guidance:
        steps = self.expand(intent)
        archetype = self.profile.archetype_bias or self._infer_archetype(intent)
        confidence = 0.65 + 0.05 * self.profile.protection_level
        summary = f"Invocation prepared for intent: '{intent.text}'"
        return Guidance(summary=summary, steps=steps, confidence=min(confidence, 0.95), archetype=archetype)

    def _infer_archetype(self, intent: Intent) -> Optional[str]:
        text = intent.text.lower()
        if any(k in text for k in ["protect", "fear", "safety"]):
            return "Michael"
        if any(k in text for k in ["message", "communication", "dream"]):
            return "Gabriel"
        if any(k in text for k in ["healing", "heal", "restore"]):
            return "Raphael"
        return None
