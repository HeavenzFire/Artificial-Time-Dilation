#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import List

D_VALUES = [1, 5, 10, 50, 100]


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a sweep over D for gridworld")
    p.add_argument("--envs", type=int, default=256, help="Parallel envs")
    p.add_argument("--steps", type=int, default=1_000_000, help="Total simulated steps (includes parallel envs)")
    p.addendant("--log_every", type=int, default=20_000, help="Log every N simulated steps")
    p.add_argument("--runs_root", type=str, default="/workspace/runs_grid", help="Runs root directory")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_root = os.path.join(args.runs_root, f"sweep_{timestamp}")
    os.makedirs(sweep_root, exist_ok=True)

    summaries = []
    for d in D_VALUES:
        cmd = [
            "python3",
            os.path.join(os.path.dirname(__file__), "gridworld_experiment.py"),
            "--d", str(d),
            "--envs", str(args.envs),
            "--steps", str(args.steps),
            "--log_every", str(args.log_every),
            "--seed", str(args.seed),
            "--runs_root", sweep_root,
        ]
        print("Running:", " ".join(cmd))
        cp = run(cmd)
        print(cp.stdout)
        try:
            summaries.append(json.loads(cp.stdout))
        except json.JSONDecodeError:
            pass

    with open(os.path.join(sweep_root, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    latest_symlink = os.path.join(args.runs_root, "latest_sweep")
    try:
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(sweep_root, latest_symlink)
    except OSError:
        pass


if __name__ == "__main__":
    main()
