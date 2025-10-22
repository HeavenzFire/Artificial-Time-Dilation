import hashlib
from math import sin, cos, pi
from typing import List

from .types import SigilSpec


def _hash_to_floats(seed: str, count: int) -> List[float]:
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    # repeat if needed
    data = (h * ((count * 4 // len(h)) + 1))[: count * 4]
    vals: List[float] = []
    for i in range(0, len(data), 4):
        chunk = int.from_bytes(data[i : i + 4], "big")
        vals.append((chunk % 10_000) / 10_000.0)
    return vals


def generate_sigil_svg(spec: SigilSpec) -> str:
    n_points = 9
    r_outer = spec.size * 0.42
    r_inner = spec.size * 0.18
    cx = cy = spec.size / 2

    rnd = _hash_to_floats(spec.seed, 64)

    def pt(r: float, t: float):
        return (cx + r * cos(t), cy + r * sin(t))

    # base star polygon
    pts = []
    for i in range(n_points):
        theta = 2 * pi * i / n_points + rnd[i] * 0.4
        r = r_outer if i % 2 == 0 else r_inner * (0.9 + 0.2 * rnd[i + 1])
        pts.append(pt(r, theta))

    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)

    # orbiting circles
    circles = []
    for j in range(5):
        theta = 2 * pi * rnd[10 + j]
        rr = r_inner * (0.6 + 0.6 * rnd[20 + j])
        x, y = pt(rr, theta)
        r = 3 + int(5 * rnd[30 + j])
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{spec.color}" opacity="0.8" />')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{spec.size}" height="{spec.size}" viewBox="0 0 {spec.size} {spec.size}">
  <rect width="100%" height="100%" fill="{spec.background}"/>
  <polygon points="{poly}" fill="none" stroke="{spec.color}" stroke-width="{spec.stroke}" />
  {''.join(circles)}
</svg>'''
    return svg
