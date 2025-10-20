#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import List


D_VALUES = [1, 10, 100, 1000]


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep over dilation factors D.")
    parser.add_argument("--envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=500_000, help="Total simulated steps (includes parallel envs)")
    parser.add_argument("--log_every", type=int, default=10_000, help="Log every N simulated steps")
    parser.add_argument("--runs_root", type=str, default="/workspace/runs", help="Runs root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_root = os.path.join(args.runs_root, f"sweep_{timestamp}")
    os.makedirs(sweep_root, exist_ok=True)

    summaries = []
    for d in D_VALUES:
        cmd = [
            "python3",
            os.path.join(os.path.dirname(__file__), "dilation_experiment.py"),
            "--d", str(d),
            "--envs", str(args.envs),
            "--steps", str(args.steps),
            "--log_every", str(args.log_every),
            "--seed", str(args.seed),
            "--runs_root", sweep_root,
        ]
        print("Running:", " ".join(cmd))
        completed = run(cmd)
        print(completed.stdout)
        try:
            summaries.append(json.loads(completed.stdout))
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
