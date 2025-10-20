#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    output_path = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure: {output_path}")


def generate_d_scaling_chart() -> None:
    hours = np.array([0, 12, 24, 36, 48, 60, 72], dtype=float)
    years_per_hour_at_1x = (60 * 60) / (365.0 * 24.0 * 60.0 * 60.0)  # seconds/hour divided by seconds/year
    # Convert hours to years of simulated time experienced
    years_1x = hours * years_per_hour_at_1x
    years_100x = years_1x * 100.0
    years_1000x = years_1x * 1000.0

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(hours, years_1x, label="1x (Real-Time)", color="#FF0000")
    ax.plot(hours, years_100x, label="100x Speed", color="#2196F3")
    ax.plot(hours, years_1000x, label="1000x Speed", color="#4CAF50")

    ax.fill_between(hours, 0, years_1x, color="#FF0000", alpha=0.15)
    ax.fill_between(hours, 0, years_100x, color="#2196F3", alpha=0.15)
    ax.fill_between(hours, 0, years_1000x, color="#4CAF50", alpha=0.15)

    ax.set_title("(D)-Scaling: Simulated vs. Real-World Time")
    ax.set_xlabel("Real-World Time (Hours)")
    ax.set_ylabel("Simulated Time (Years)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    save_figure(fig, "d_scaling_chart.png")
    plt.close(fig)


def generate_reward_curves() -> None:
    hours = np.array([0, 12, 24, 36, 48], dtype=float)
    rewards = np.array([0, 2000, 5000, 7000, 8500], dtype=float)
    steps_millions = np.array([0, 2.5, 5, 7.5, 10], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

    ln1 = ax1.plot(hours, rewards, label="Cumulative Reward", color="#2196F3")
    ax1.fill_between(hours, 0, rewards, color="#2196F3", alpha=0.15)
    ax1.set_xlabel("Real-World Time (Hours)")
    ax1.set_ylabel("Cumulative Reward")
    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(hours, steps_millions, label="Simulated Steps (M)", color="#FF9800")
    ax2.fill_between(hours, 0, steps_millions, color="#FF9800", alpha=0.15)
    ax2.set_ylabel("Simulated Steps (M)")

    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc="lower right")

    ax1.set_title("RL Agent Performance in Dilated Time")

    save_figure(fig, "reward_curves.png")
    plt.close(fig)


def main() -> None:
    generate_d_scaling_chart()
    generate_reward_curves()


if __name__ == "__main__":
    main()
