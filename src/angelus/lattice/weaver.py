from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..common.types import Intent


@dataclass
class Edge:
    target: int
    weight: float


class LatticeWeaver:
    def __init__(self) -> None:
        self.nodes: List[str] = []  # labels
        self.edges: Dict[int, List[Edge]] = {}

    def _get_or_add(self, label: str) -> int:
        if label in self.nodes:
            return self.nodes.index(label)
        self.nodes.append(label)
        idx = len(self.nodes) - 1
        self.edges[idx] = []
        return idx

    def add_intent(self, intent: Intent) -> int:
        key = intent.text.lower()
        return self._get_or_add(key)

    def connect_similarity(self, a: Intent, b: Intent) -> None:
        ia = self.add_intent(a)
        ib = self.add_intent(b)
        weight = self._jaccard(a.text, b.text)
        self.edges[ia].append(Edge(target=ib, weight=weight))
        self.edges[ib].append(Edge(target=ia, weight=weight))

    def neighborhood(self, intent: Intent, min_weight: float = 0.2) -> List[Tuple[str, float]]:
        i = self.add_intent(intent)
        results: List[Tuple[str, float]] = []
        for e in self.edges.get(i, []):
            if e.weight >= min_weight:
                results.append((self.nodes[e.target], e.weight))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _jaccard(self, a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
