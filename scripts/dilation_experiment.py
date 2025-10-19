#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Deque, Dict, List, Tuple

import numpy as np


@dataclass
class BanditConfig:
    num_arms: int = 10
    arm_stddev: float = 0.5
    random_seed: int = 42


@dataclass
class AgentConfig:
    epsilon: float = 0.1
    step_size: float = 0.05


@dataclass
class DilationConfig:
    dilation_factor: float = 1.0
    num_parallel_envs: int = 256
    baseline_steps_per_second_per_env: float = 5000.0  # conceptual throughput per env


@dataclass
class ExperimentConfig:
    total_simulated_steps: int = 500_000  # counts every env step (includes parallel envs)
    log_every_simulated_steps: int = 10_000
    breakthrough_selection_rate_threshold: float = 0.9  # selects best arm at least this fraction over window
    breakthrough_window_steps: int = 20_000  # window size (sim steps) for measuring selection rate


class VectorizedBanditEnv:
    def __init__(self, config: BanditConfig, batch_size: int):
        self.config = config
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(config.random_seed)
        # Sample stationary arm means once per run
        # Spread means around 0..1 for clearer best arm separation
        base_means = self.random_state.normal(loc=0.0, scale=1.0, size=config.num_arms)
        # Normalize to 0..1 range
        min_m = base_means.min()
        max_m = base_means.max()
        self.arm_means = (base_means - min_m) / (max_m - min_m + 1e-12)
        self.best_arm_index = int(np.argmax(self.arm_means))

    def step(self, actions: np.ndarray) -> np.ndarray:
        # actions shape: (batch_size,)
        means = self.arm_means[actions]
        rewards = means + np.random.normal(0.0, self.config.arm_stddev, size=self.batch_size)
        return rewards


class LinearQBanditAgent:
    def __init__(self, config: AgentConfig, num_arms: int):
        self.config = config
        self.num_arms = num_arms
        # Linear Q over one-hot features equals per-arm values
        self.action_values = np.zeros(num_arms, dtype=np.float64)
        self.action_counts = np.zeros(num_arms, dtype=np.int64)

    def select_actions(self, batch_size: int, rng: np.random.RandomState) -> np.ndarray:
        if rng.rand() < self.config.epsilon:
            return rng.randint(0, self.num_arms, size=batch_size)
        greedy_action = int(np.argmax(self.action_values))
        return np.full(batch_size, greedy_action, dtype=np.int64)

    def update(self, actions: np.ndarray, rewards: np.ndarray) -> None:
        # Vectorized incremental update with constant step-size alpha (linear function approx)
        alpha = self.config.step_size
        # For each action, compute mean reward among batch where it was taken
        for arm in range(self.num_arms):
            mask = actions == arm
            if not np.any(mask):
                continue
            avg_reward_for_arm = rewards[mask].mean()
            self.action_values[arm] += alpha * (avg_reward_for_arm - self.action_values[arm])
            self.action_counts[arm] += int(mask.sum())


@dataclass
class RunSummary:
    run_dir: str
    best_arm_index: int
    best_arm_mean: float
    breakthrough_reached: bool
    breakthrough_real_seconds: float
    breakthrough_real_hours: float
    final_selection_rate_window: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_real_seconds(simulated_steps: int, dilation: DilationConfig) -> float:
    effective_steps_per_second = (
        dilation.baseline_steps_per_second_per_env * dilation.num_parallel_envs * max(dilation.dilation_factor, 1e-9)
    )
    return simulated_steps / effective_steps_per_second


def run_experiment(
    bandit_cfg: BanditConfig,
    agent_cfg: AgentConfig,
    dilation_cfg: DilationConfig,
    exp_cfg: ExperimentConfig,
    run_root: str,
) -> RunSummary:
    ensure_dir(run_root)
    run_dir = os.path.join(run_root, f"D_{int(dilation_cfg.dilation_factor)}")
    ensure_dir(run_dir)

    rng = np.random.RandomState(bandit_cfg.random_seed)

    env = VectorizedBanditEnv(bandit_cfg, batch_size=dilation_cfg.num_parallel_envs)
    agent = LinearQBanditAgent(agent_cfg, num_arms=bandit_cfg.num_arms)

    # Logging setup
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    config_json_path = os.path.join(run_root, "config.json")

    # Persist top-level config once
    if not os.path.exists(config_json_path):
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "bandit": asdict(bandit_cfg),
                    "agent": asdict(agent_cfg),
                    "dilation": asdict(dilation_cfg),
                    "experiment": asdict(exp_cfg),
                    "arm_means": env.arm_means.tolist(),
                },
                f,
                indent=2,
            )

    # Rolling windows for stats
    recent_rewards: Deque[float] = deque(maxlen=exp_cfg.breakthrough_window_steps)
    recent_best_action_selected: Deque[int] = deque(maxlen=exp_cfg.breakthrough_window_steps)

    simulated_steps = 0
    num_updates = max(1, exp_cfg.total_simulated_steps // dilation_cfg.num_parallel_envs)
    log_every_updates = max(1, exp_cfg.log_every_simulated_steps // dilation_cfg.num_parallel_envs)

    breakthrough_reached = False
    breakthrough_real_seconds = 0.0

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "update_idx",
                "simulated_steps",
                "real_seconds",
                "real_hours",
                "avg_reward_window",
                "avg_reward_cumulative",
                "best_action_selection_rate_window",
                "epsilon",
            ]
        )

        cumulative_reward_sum = 0.0
        cumulative_reward_count = 0

        for update_idx in range(1, num_updates + 1):
            actions = agent.select_actions(dilation_cfg.num_parallel_envs, rng)
            rewards = env.step(actions)
            agent.update(actions, rewards)

            # Stats
            simulated_steps += dilation_cfg.num_parallel_envs
            cumulative_reward_sum += float(rewards.sum())
            cumulative_reward_count += rewards.size
            recent_rewards.extend(rewards.tolist())
            recent_best_action_selected.extend((actions == env.best_arm_index).astype(int).tolist())

            if update_idx % log_every_updates == 0 or update_idx == num_updates:
                real_seconds = compute_real_seconds(simulated_steps, dilation_cfg)
                real_hours = real_seconds / 3600.0
                avg_reward_window = (np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0)
                avg_reward_cumulative = (
                    cumulative_reward_sum / max(1, cumulative_reward_count)
                )
                best_action_selection_rate_window = (
                    float(np.mean(recent_best_action_selected)) if len(recent_best_action_selected) > 0 else 0.0
                )

                writer.writerow(
                    [
                        update_idx,
                        simulated_steps,
                        real_seconds,
                        real_hours,
                        avg_reward_window,
                        avg_reward_cumulative,
                        best_action_selection_rate_window,
                        agent.config.epsilon,
                    ]
                )

                # Breakthrough: selects best arm at least threshold over recent window
                if not breakthrough_reached and len(recent_best_action_selected) == recent_best_action_selected.maxlen:
                    if best_action_selection_rate_window >= exp_cfg.breakthrough_selection_rate_threshold:
                        breakthrough_reached = True
                        breakthrough_real_seconds = real_seconds

    summary = RunSummary(
        run_dir=run_dir,
        best_arm_index=env.best_arm_index,
        best_arm_mean=float(env.arm_means[env.best_arm_index]),
        breakthrough_reached=breakthrough_reached,
        breakthrough_real_seconds=breakthrough_real_seconds,
        breakthrough_real_hours=(breakthrough_real_seconds / 3600.0),
        final_selection_rate_window=float(np.mean(recent_best_action_selected)) if len(recent_best_action_selected) > 0 else 0.0,
    )

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f_sum:
        json.dump(asdict(summary), f_sum, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a time dilation bandit experiment.")
    parser.add_argument("--d", "--dilation", dest="dilation", type=float, default=1.0, help="Dilation factor D")
    parser.add_argument("--envs", dest="num_envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--steps", dest="steps", type=int, default=500_000, help="Total simulated steps (includes parallel envs)")
    parser.add_argument("--log_every", dest="log_every", type=int, default=10_000, help="Log every N simulated steps")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs_root", dest="runs_root", type=str, default="/workspace/runs", help="Runs root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.runs_root, f"run_{timestamp}")

    bandit_cfg = BanditConfig(num_arms=10, arm_stddev=0.5, random_seed=args.seed)
    agent_cfg = AgentConfig(epsilon=0.1, step_size=0.05)
    dilation_cfg = DilationConfig(
        dilation_factor=args.dilation,
        num_parallel_envs=args.num_envs,
        baseline_steps_per_second_per_env=5000.0,
    )
    exp_cfg = ExperimentConfig(
        total_simulated_steps=args.steps,
        log_every_simulated_steps=args.log_every,
        breakthrough_selection_rate_threshold=0.9,
        breakthrough_window_steps=20_000,
    )

    os.makedirs(run_root, exist_ok=True)

    summary = run_experiment(bandit_cfg, agent_cfg, dilation_cfg, exp_cfg, run_root)

    latest_symlink = os.path.join(os.path.dirname(run_root), "latest")
    try:
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(run_root, latest_symlink)
    except OSError:
        # Non-fatal on systems that restrict symlinks
        pass

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
