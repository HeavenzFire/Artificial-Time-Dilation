from typing import List, Tuple
import math
import json
from pathlib import Path

from ..common.types import Intent, Guidance


class ResonanceOracle:
    def __init__(self, archetype_vectors: dict | None = None) -> None:
        # simple keyword vectors as stand-in for embeddings
        self.archetype_vectors = archetype_vectors or {
            "Michael": [1, 0, 0, 1, 0],  # protection, courage
            "Gabriel": [0, 1, 1, 0, 0],  # messages, clarity
            "Raphael": [0, 0, 1, 0, 1],  # healing, restoration
        }
        self._sigil_weights: List[float] | None = self._load_sigil_weights()

    def _load_sigil_weights(self) -> List[float] | None:
        try:
            base = Path.cwd() / "data" / "angelus" / "model.json"
            if not base.exists():
                return None
            js = json.loads(base.read_text(encoding="utf-8"))
            w = js.get("w")
            if isinstance(w, list) and len(w) == 5:
                return [float(v) for v in w]
        except Exception:
            return None
        return None

    def vectorize(self, text: str) -> List[float]:
        text = text.lower()
        # base features
        feats = [
            ("protect" in text) or ("courage" in text),
            ("message" in text) or ("clarity" in text) or ("decision" in text),
            ("heal" in text) or ("restore" in text) or ("trauma" in text),
            ("fear" in text) or ("boundary" in text),
            ("body" in text) or ("health" in text),
        ]
        base = [1.0 if f else 0.0 for f in feats]
        # if trained weights exist, lightly reweight the base features by their magnitude
        if self._sigil_weights:
            # normalize weights to [0,1] scale
            max_abs = max(abs(v) for v in self._sigil_weights) or 1.0
            scale = [abs(v) / max_abs for v in self._sigil_weights]
            return [b * (0.7 + 0.3 * s) for b, s in zip(base, scale)]
        return base

    def cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def match_archetype(self, intent: Intent) -> Tuple[str | None, float]:
        v = self.vectorize(intent.text)
        best = None
        best_score = -1.0
        for name, vec in self.archetype_vectors.items():
            score = self.cosine(v, vec)
            if score > best_score:
                best = name
                best_score = score
        if best_score <= 0:
            return None, 0.0
        return best, best_score

    def refine(self, guidance: Guidance, intent: Intent) -> Guidance:
        name, score = self.match_archetype(intent)
        confidence = max(guidance.confidence, min(0.99, 0.6 + 0.3 * score))
        return Guidance(
            summary=guidance.summary,
            steps=guidance.steps,
            confidence=confidence,
            archetype=guidance.archetype or name,
        )
