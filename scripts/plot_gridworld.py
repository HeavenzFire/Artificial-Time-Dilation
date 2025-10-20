#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_dir: str) -> Dict[str, np.ndarray]:
    xs: List[int] = []
    hours: List[float] = []
    succ: List[float] = []
    dist: List[float] = []
    rewards: List[float] = []
    with open(os.path.join(run_dir, "metrics.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(int(row["simulated_steps"]))
            hours.append(float(row["real_hours"]))
            succ.append(float(row["success_rate_window"]))
            dist.append(float(row["avg_distance_window"]))
            rewards.append(float(row["avg_reward_window"]))
    return {
        "sim_steps": np.array(xs),
        "real_hours": np.array(hours),
        "success": np.array(succ),
        "distance": np.array(dist),
        "reward": np.array(rewards),
    }


def discover_runs(sweep_root: str) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(sweep_root, "D_*")) if os.path.isdir(p)])


def plot_breakthrough_time(sweep_root: str, output_path: str) -> None:
    run_dirs = discover_runs(sweep_root)
    d_values: List[int] = []
    breakthrough_hours: List[float] = []
    for run_dir in run_dirs:
        d_str = os.path.basename(run_dir).split("_")[-1]
        try:
            d_val = int(d_str)
        except ValueError:
            continue
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        d_values.append(d_val)
        breakthrough_hours.append(summary.get("breakthrough_real_hours", 0.0))

    fig, ax = plt.subplots(figsize=(7, 4.25))
    ax.plot(d_values, breakthrough_hours, marker="o", color="#8B5CF6")
    ax.set_xscale("log")
    ax.set_xlabel("Dilation Factor D (log scale)")
    ax.set_ylabel("Breakthrough Time (Real Hours)")
    ax.set_title("Gridworld Breakthrough vs. D")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


def plot_success_curves(sweep_root: str, output_path: str) -> None:
    run_dirs = discover_runs(sweep_root)
    fig, ax = plt.subplots(figsize=(7, 4.25))
    for run_dir in run_dirs:
        d_str = os.path.basename(run_dir).split("_")[-1]
        try:
            d_val = int(d_str)
        except ValueError:
            continue
        data = load_metrics(run_dir)
        ax.plot(data["real_hours"], data["success"], label=f"D={d_val}")

    ax.set_xlabel("Real Hours")
    ax.set_ylabel("Success Rate (window)")
    ax.set_title("Gridworld Success vs. Real Time")
    ax.legend(title="D")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot gridworld acceleration figures")
    p.add_argument("--sweep_root", type=str, default="/workspace/runs_grid/latest_sweep", help="Sweep root")
    p.add_argument("--outdir", type=str, default="/workspace/figures", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_breakthrough_time(args.sweep_root, os.path.join(args.outdir, "grid_breakthrough_vs_D.png"))
    plot_success_curves(args.sweep_root, os.path.join(args.outdir, "grid_success_vs_time.png"))


if __name__ == "__main__":
    main()
