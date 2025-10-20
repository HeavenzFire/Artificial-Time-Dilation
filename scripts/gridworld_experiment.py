#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Deque, Tuple

import numpy as np


@dataclass
class GridConfig:
    width: int = 8
    height: int = 8
    random_seed: int = 123
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    max_steps_per_episode: int = 256


@dataclass
class AgentConfig:
    epsilon: float = 0.1
    alpha: float = 0.2
    gamma: float = 0.95


@dataclass
class DilationConfig:
    dilation_factor: float = 1.0
    num_parallel_envs: int = 256
    baseline_steps_per_second_per_env: float = 2000.0


@dataclass
class ExperimentConfig:
    total_simulated_steps: int = 1_000_000
    log_every_simulated_steps: int = 20_000
    breakthrough_success_rate_threshold: float = 0.05  # per-step success rate over window
    breakthrough_window_steps: int = 50_000


class VectorizedGridworldEnv:
    def __init__(self, cfg: GridConfig, batch_size: int):
        self.cfg = cfg
        self.batch_size = batch_size
        self.rng = np.random.RandomState(cfg.random_seed)
        self.width = cfg.width
        self.height = cfg.height
        self.num_states = self.width * self.height
        self.num_actions = 4  # 0: up, 1: right, 2: down, 3: left
        # Goal at bottom-right
        self.goal_xy: Tuple[int, int] = (self.width - 1, self.height - 1)
        # State
        self.pos_x = np.zeros(batch_size, dtype=np.int32)
        self.pos_y = np.zeros(batch_size, dtype=np.int32)
        self.steps_in_episode = np.zeros(batch_size, dtype=np.int32)
        self.reset_all()

    def reset_all(self) -> None:
        # Random start positions anywhere except the goal to encourage exploration
        self.pos_x = self.rng.randint(0, self.width, size=self.batch_size, dtype=np.int32)
        self.pos_y = self.rng.randint(0, self.height, size=self.batch_size, dtype=np.int32)
        at_goal = (self.pos_x == self.goal_xy[0]) & (self.pos_y == self.goal_xy[1])
        # If at goal, shift left by 1 safely
        self.pos_x[at_goal] = np.maximum(0, self.pos_x[at_goal] - 1)
        self.steps_in_episode.fill(0)

    def state_indices(self) -> np.ndarray:
        return self.pos_y * self.width + self.pos_x

    def manhattan_distance(self) -> np.ndarray:
        dx = np.abs(self.pos_x - self.goal_xy[0])
        dy = np.abs(self.pos_y - self.goal_xy[1])
        return dx + dy

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Apply actions with boundary clipping
        # 0: up, 1: right, 2: down, 3: left
        new_x = self.pos_x.copy()
        new_y = self.pos_y.copy()
        new_y = np.where(actions == 0, np.maximum(0, new_y - 1), new_y)
        new_x = np.where(actions == 1, np.minimum(self.width - 1, new_x + 1), new_x)
        new_y = np.where(actions == 2, np.minimum(self.height - 1, new_y + 1), new_y)
        new_x = np.where(actions == 3, np.maximum(0, new_x - 1), new_x)

        self.pos_x = new_x
        self.pos_y = new_y
        self.steps_in_episode += 1

        # Rewards and termination
        at_goal = (self.pos_x == self.goal_xy[0]) & (self.pos_y == self.goal_xy[1])
        done = at_goal | (self.steps_in_episode >= self.cfg.max_steps_per_episode)
        rewards = np.where(at_goal, self.cfg.goal_reward, self.cfg.step_penalty)

        next_states = self.state_indices()

        # Auto-reset done envs
        if np.any(done):
            # Record resets but return next_states and done before reset so agent learns from terminal
            self.pos_x[done] = self.rng.randint(0, self.width, size=done.sum(), dtype=np.int32)
            self.pos_y[done] = self.rng.randint(0, self.height, size=done.sum(), dtype=np.int32)
            at_goal_reset = (self.pos_x[done] == self.goal_xy[0]) & (self.pos_y[done] == self.goal_xy[1])
            # Avoid resetting directly to goal
            idx_done = np.where(done)[0]
            self.pos_x[idx_done[at_goal_reset]] = np.maximum(0, self.pos_x[idx_done[at_goal_reset]] - 1)
            self.steps_in_episode[done] = 0

        return next_states, rewards, done, at_goal


class TabularQAgent:
    def __init__(self, cfg: AgentConfig, num_states: int, num_actions: int, seed: int):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.Q = np.zeros((num_states, num_actions), dtype=np.float64)

    def select(self, states: np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        if self.rng.rand() < self.cfg.epsilon:
            return self.rng.randint(0, self.Q.shape[1], size=batch_size)
        q_rows = self.Q[states]
        greedy_actions = np.argmax(q_rows, axis=1)
        return greedy_actions.astype(np.int64)

    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, done: np.ndarray) -> None:
        alpha = self.cfg.alpha
        gamma = self.cfg.gamma
        q_sa = self.Q[states, actions]
        max_q_next = np.max(self.Q[next_states], axis=1)
        targets = rewards + gamma * max_q_next * (1.0 - done.astype(np.float64))
        self.Q[states, actions] = q_sa + alpha * (targets - q_sa)


@dataclass
class RunSummary:
    run_dir: str
    breakthrough_reached: bool
    breakthrough_real_seconds: float
    breakthrough_real_hours: float
    final_success_rate_window: float
    final_avg_distance_window: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_real_seconds(simulated_steps: int, dilation: DilationConfig) -> float:
    effective_sps = (
        dilation.baseline_steps_per_second_per_env * dilation.num_parallel_envs * max(dilation.dilation_factor, 1e-9)
    )
    return simulated_steps / effective_sps


def run_experiment(grid_cfg: GridConfig, agent_cfg: AgentConfig, dilation_cfg: DilationConfig, exp_cfg: ExperimentConfig, run_root: str) -> RunSummary:
    ensure_dir(run_root)
    run_dir = os.path.join(run_root, f"D_{int(dilation_cfg.dilation_factor)}")
    ensure_dir(run_dir)

    env = VectorizedGridworldEnv(grid_cfg, batch_size=dilation_cfg.num_parallel_envs)
    agent = TabularQAgent(agent_cfg, num_states=env.num_states, num_actions=env.num_actions, seed=grid_cfg.random_seed)

    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    config_json_path = os.path.join(run_root, "config.json")

    if not os.path.exists(config_json_path):
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "grid": asdict(grid_cfg),
                    "agent": asdict(agent_cfg),
                    "dilation": asdict(dilation_cfg),
                    "experiment": asdict(exp_cfg),
                },
                f,
                indent=2,
            )

    recent_success: Deque[int] = deque(maxlen=exp_cfg.breakthrough_window_steps)
    recent_distance: Deque[float] = deque(maxlen=exp_cfg.breakthrough_window_steps)
    recent_rewards: Deque[float] = deque(maxlen=exp_cfg.breakthrough_window_steps)

    simulated_steps = 0
    num_updates = max(1, exp_cfg.total_simulated_steps // dilation_cfg.num_parallel_envs)
    log_every_updates = max(1, exp_cfg.log_every_simulated_steps // dilation_cfg.num_parallel_envs)

    breakthrough_reached = False
    breakthrough_real_seconds = 0.0

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "update_idx",
            "simulated_steps",
            "real_seconds",
            "real_hours",
            "success_rate_window",
            "avg_distance_window",
            "avg_reward_window",
        ])

        for update_idx in range(1, num_updates + 1):
            states = env.state_indices()
            actions = agent.select(states)
            next_states, rewards, done, at_goal = env.step(actions)
            agent.update(states, actions, rewards, next_states, done)

            simulated_steps += dilation_cfg.num_parallel_envs

            # Update windows
            recent_success.extend(at_goal.astype(int).tolist())
            recent_distance.extend(env.manhattan_distance().astype(float).tolist())
            recent_rewards.extend(rewards.astype(float).tolist())

            if update_idx % log_every_updates == 0 or update_idx == num_updates:
                real_seconds = compute_real_seconds(simulated_steps, dilation_cfg)
                real_hours = real_seconds / 3600.0
                success_rate_window = float(np.mean(recent_success)) if len(recent_success) else 0.0
                avg_distance_window = float(np.mean(recent_distance)) if len(recent_distance) else 0.0
                avg_reward_window = float(np.mean(recent_rewards)) if len(recent_rewards) else 0.0

                writer.writerow([
                    update_idx,
                    simulated_steps,
                    real_seconds,
                    real_hours,
                    success_rate_window,
                    avg_distance_window,
                    avg_reward_window,
                ])

                if (not breakthrough_reached) and len(recent_success) == recent_success.maxlen:
                    if success_rate_window >= exp_cfg.breakthrough_success_rate_threshold:
                        breakthrough_reached = True
                        breakthrough_real_seconds = real_seconds

    summary = RunSummary(
        run_dir=run_dir,
        breakthrough_reached=breakthrough_reached,
        breakthrough_real_seconds=breakthrough_real_seconds,
        breakthrough_real_hours=breakthrough_real_seconds / 3600.0,
        final_success_rate_window=float(np.mean(recent_success)) if len(recent_success) else 0.0,
        final_avg_distance_window=float(np.mean(recent_distance)) if len(recent_distance) else 0.0,
    )

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f_sum:
        json.dump(asdict(summary), f_sum, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gridworld time dilation experiment")
    p.add_argument("--d", type=float, default=1.0, help="Dilation factor D")
    p.add_argument("--envs", type=int, default=256, help="Number of parallel envs")
    p.add_argument("--steps", type=int, default=1_000_000, help="Total simulated steps (includes parallel envs)")
    p.add_argument("--log_every", type=int, default=20_000, help="Log every N simulated steps")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--runs_root", type=str, default="/workspace/runs_grid", help="Runs root directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.runs_root, f"run_{timestamp}")

    grid_cfg = GridConfig(random_seed=args.seed)
    agent_cfg = AgentConfig(epsilon=0.1, alpha=0.2, gamma=0.95)
    dilation_cfg = DilationConfig(dilation_factor=args.d, num_parallel_envs=args.envs, baseline_steps_per_second_per_env=2000.0)
    exp_cfg = ExperimentConfig(total_simulated_steps=args.steps, log_every_simulated_steps=args.log_every, breakthrough_success_rate_threshold=0.05, breakthrough_window_steps=50_000)

    os.makedirs(run_root, exist_ok=True)
    summary = run_experiment(grid_cfg, agent_cfg, dilation_cfg, exp_cfg, run_root)

    latest_symlink = os.path.join(os.path.dirname(run_root), "latest")
    try:
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(run_root, latest_symlink)
    except OSError:
        pass

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
