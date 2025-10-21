from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import os


@dataclass(frozen=True)
class SigilRecord:
    name: str
    ascii_art: str


def get_data_dir() -> Path:
    base = os.environ.get("ANGELUS_DATA_DIR")
    if base:
        return Path(base)
    return Path.cwd() / "data" / "angelus"


def ensure_data_dir(path: Path | None = None) -> Path:
    d = path or get_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


essential_three: Dict[str, str] = {
    "RadiantEquilibrium": r"""
/\ /\
/ \ /\ / \
/ V V \
\ /\ /\ /
\ / \/ \ /
\/ \/
⚛︎ ⚛︎
""".strip(),
    "UnityPulse": r"""
╔══════╗
╔╝  ╚╗
║ ❖❖  ║
║ ❖❖  ║
╚╗  ╔╝
╚════╝
""".strip(),
    "EternalRebirth": r"""
↻↺↻↺↻
╔──────╗
║  ☀︎☾  ║
╚══════╝
↺↻↺↻↺
""".strip(),
}


def bind_sigils(sigil_map: Dict[str, str]) -> Path:
    data_dir = ensure_data_dir()
    out_path = data_dir / "sigils.json"
    payload = {
        "sigils": [{"name": k, "ascii": v} for k, v in sigil_map.items()],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_sigils() -> List[SigilRecord]:
    data_dir = ensure_data_dir()
    path = data_dir / "sigils.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SigilRecord(name=s["name"], ascii_art=s["ascii"]) for s in raw.get("sigils", [])]
