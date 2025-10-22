from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Intent:
    user_id: Optional[str]
    text: str
    tags: List[str]


@dataclass(frozen=True)
class Guidance:
    summary: str
    steps: List[str]
    confidence: float  # 0..1
    archetype: Optional[str] = None


@dataclass(frozen=True)
class SigilSpec:
    seed: str
    color: str = "#4B6FFF"
    background: str = "#0B1020"
    size: int = 256
    stroke: int = 3
