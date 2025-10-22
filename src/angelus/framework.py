from dataclasses import dataclass
from typing import Optional

from .invocation.engine import InvocationEngine, InvocationProfile
from .resonance.oracle import ResonanceOracle
from .lattice.weaver import LatticeWeaver
from .common.types import Intent, Guidance


@dataclass
class AngelusConfig:
    protection_level: int = 1
    archetype_bias: Optional[str] = None


class AngelusFramework:
    def __init__(self, config: Optional[AngelusConfig] = None) -> None:
        config = config or AngelusConfig()
        self.invocation = InvocationEngine(
            InvocationProfile(
                protection_level=config.protection_level,
                archetype_bias=config.archetype_bias,
            )
        )
        self.oracle = ResonanceOracle()
        self.lattice = LatticeWeaver()

    def run(self, intent: Intent) -> Guidance:
        guidance = self.invocation.invoke(intent)
        refined = self.oracle.refine(guidance, intent)
        self.lattice.add_intent(intent)
        return refined
