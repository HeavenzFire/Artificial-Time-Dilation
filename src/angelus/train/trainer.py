from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..data.registry import load_sigils, ensure_data_dir


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.05


class SimpleSigilModel:
    def __init__(self) -> None:
        # weights for 5 pseudo-features
        self.w = np.zeros(5, dtype=float)

    def features(self, text: str) -> np.ndarray:
        t = text.lower()
        return np.array([
            sum(ch in "aeiou" for ch in t) / max(1, len(t)),
            t.count("❖") / 10.0,
            t.count("☀") / 10.0 + t.count("☾") / 10.0,
            sum(c in "╔╝╚╗" for c in t) / 20.0,
            sum(c in "↻↺" for c in t) / 10.0,
        ], dtype=float)

    def predict(self, x: np.ndarray) -> float:
        return float(np.clip(self.w @ x, 0.0, 1.0))

    def train(self, samples: List[Tuple[str, float]], cfg: TrainConfig) -> None:
        for _ in range(cfg.epochs):
            for text, y in samples:
                x = self.features(text)
                y_hat = self.w @ x
                err = y - y_hat
                self.w += cfg.lr * err * x

    def to_json(self) -> Dict:
        return {"w": self.w.tolist()}


TARGETS: Dict[str, float] = {
    # assign high qualities; can be refined later
    "RadiantEquilibrium": 0.95,
    "UnityPulse": 0.92,
    "EternalRebirth": 0.94,
}


def run_training(cfg: TrainConfig = TrainConfig()) -> Path:
    sigils = load_sigils()
    # build samples
    samples: List[Tuple[str, float]] = []
    for s in sigils:
        y = TARGETS.get(s.name, 0.8)
        samples.append((s.ascii_art, y))

    model = SimpleSigilModel()
    model.train(samples, cfg)

    out_dir = ensure_data_dir()
    out_path = out_dir / "model.json"
    out_path.write_text(json.dumps(model.to_json(), indent=2), encoding="utf-8")
    return out_path
