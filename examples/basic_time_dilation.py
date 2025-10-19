#!/usr/bin/env python3
"""
Basic example demonstrating artificial time dilation in RL.

This script shows how to use the time dilation simulator with a simple
RL environment to demonstrate the effects of accelerated training.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.time_dilation import TimeDilationSimulator, DilationFactor
from core.rl_environment import RLDilatedEnvironment, EnvironmentConfig
from visualization.charts import DScalingChart, RewardCurveChart


def create_simple_environment():
    """Create a simple RL environment for demonstration."""
    try:
        import gymnasium as gym
        env_name = "CartPole-v1"
    except ImportError:
        try:
            import gym
            env_name = "CartPole-v1"
        except ImportError:
            print("Neither gymnasium nor gym is available. Using mock environment.")
            return None
    
    config = EnvironmentConfig(
        env_name=env_name,
        dilation_factor=1.0,
        max_episode_steps=500
    )
    
    return RLDilatedEnvironment(config)


def simple_agent_act(obs: np.ndarray) -> int:
    """Simple agent that takes random actions."""
    return np.random.randint(0, 2)


def run_time_dilation_experiment(
    dilation_factors: List[float],
    episodes_per_factor: int = 5,
    max_steps_per_episode: int = 200
) -> Dict[str, Any]:
    """
    Run time dilation experiment with different dilation factors.
    
    Args:
        dilation_factors: List of dilation factors to test
        episodes_per_factor: Number of episodes per dilation factor
        max_steps_per_episode: Maximum steps per episode
        
    Returns:
        Dictionary containing experiment results
    """
    results = {
        "dilation_factors": [],
        "episode_rewards": [],
        "episode_lengths": [],
        "training_times": [],
        "simulated_times": []
    }
    
    env = create_simple_environment()
    if env is None:
        print("Could not create environment. Using mock data.")
        return generate_mock_results(dilation_factors, episodes_per_factor)
    
    for dilation_factor in dilation_factors:
        print(f"\nTesting dilation factor: {dilation_factor}x")
        
        # Set dilation factor
        env.set_dilation_factor(dilation_factor)
        
        factor_rewards = []
        factor_lengths = []
        factor_times = []
        factor_simulated_times = []
        
        for episode in range(episodes_per_factor):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                action = simple_agent_act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Get episode metrics
            episode_metrics = env.get_current_episode_metrics()
            
            factor_rewards.append(episode_reward)
            factor_lengths.append(episode_length)
            factor_times.append(episode_metrics["real_time"])
            factor_simulated_times.append(episode_metrics["simulated_time"])
            
            print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, Time={episode_metrics['real_time']:.2f}s")
        
        # Store results for this dilation factor
        results["dilation_factors"].extend([dilation_factor] * episodes_per_factor)
        results["episode_rewards"].extend(factor_rewards)
        results["episode_lengths"].extend(factor_lengths)
        results["training_times"].extend(factor_times)
        results["simulated_times"].extend(factor_simulated_times)
    
    env.close()
    return results


def generate_mock_results(
    dilation_factors: List[float],
    episodes_per_factor: int
) -> Dict[str, Any]:
    """Generate mock results for demonstration when environment is not available."""
    results = {
        "dilation_factors": [],
        "episode_rewards": [],
        "episode_lengths": [],
        "training_times": [],
        "simulated_times": []
    }
    
    for dilation_factor in dilation_factors:
        for episode in range(episodes_per_factor):
            # Generate mock data with some variation
            base_reward = 100 + np.random.normal(0, 20)
            base_length = 200 + np.random.normal(0, 50)
            base_time = 10 + np.random.normal(0, 2)
            
            results["dilation_factors"].append(dilation_factor)
            results["episode_rewards"].append(max(0, base_reward))
            results["episode_lengths"].append(max(1, int(base_length)))
            results["training_times"].append(max(0.1, base_time))
            results["simulated_times"].append(max(0.1, base_time * dilation_factor))
    
    return results


def create_visualizations(results: Dict[str, Any]) -> None:
    """Create visualizations from experiment results."""
    print("\nCreating visualizations...")
    
    # Create D-scaling chart
    d_chart = DScalingChart()
    time_points = np.linspace(0, 72, 100)  # 72 hours
    dilation_factors = [1, 10, 100, 1000]
    
    d_data = d_chart.generate_d_scaling_data(time_points, dilation_factors)
    d_fig = d_chart.plot_matplotlib(time_points, dilation_factors)
    d_fig.savefig("d_scaling_chart.png", dpi=300, bbox_inches='tight')
    print("Saved D-scaling chart as d_scaling_chart.png")
    
    # Create reward curve chart
    reward_chart = RewardCurveChart()
    
    # Group results by dilation factor
    unique_factors = sorted(set(results["dilation_factors"]))
    
    for factor in unique_factors:
        # Filter results for this factor
        factor_mask = [df == factor for df in results["dilation_factors"]]
        factor_rewards = [r for r, m in zip(results["episode_rewards"], factor_mask) if m]
        factor_times = [t for t, m in zip(results["training_times"], factor_mask) if m]
        
        if len(factor_rewards) > 1:
            # Create cumulative rewards
            cumulative_rewards = np.cumsum(factor_rewards)
            time_points = np.cumsum(factor_times) / 3600  # Convert to hours
            steps = np.arange(len(factor_rewards)) * 200  # Approximate steps
            
            reward_fig = reward_chart.plot_matplotlib(
                time_points, cumulative_rewards, steps, factor
            )
            reward_fig.savefig(f"reward_curves_{factor}x.png", dpi=300, bbox_inches='tight')
            print(f"Saved reward curve chart for {factor}x as reward_curves_{factor}x.png")
    
    plt.close('all')


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the experiment results."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    unique_factors = sorted(set(results["dilation_factors"]))
    
    for factor in unique_factors:
        factor_mask = [df == factor for df in results["dilation_factors"]]
        factor_rewards = [r for r, m in zip(results["episode_rewards"], factor_mask) if m]
        factor_times = [t for t, m in zip(results["training_times"], factor_mask) if m]
        factor_simulated_times = [st for st, m in zip(results["simulated_times"], factor_mask) if m]
        
        avg_reward = np.mean(factor_rewards)
        std_reward = np.std(factor_rewards)
        avg_time = np.mean(factor_times)
        avg_simulated_time = np.mean(factor_simulated_times)
        time_compression = avg_simulated_time / avg_time if avg_time > 0 else 0
        
        print(f"\nDilation Factor: {factor}x")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Real Time: {avg_time:.2f} seconds")
        print(f"  Average Simulated Time: {avg_simulated_time:.2f} seconds")
        print(f"  Time Compression: {time_compression:.2f}x")
    
    print("\n" + "="*60)


def main():
    """Main function to run the time dilation example."""
    print("Artificial Time Dilation for RL - Basic Example")
    print("=" * 50)
    
    # Configuration
    dilation_factors = [1, 10, 100, 1000]
    episodes_per_factor = 5
    max_steps_per_episode = 200
    
    print(f"Testing dilation factors: {dilation_factors}")
    print(f"Episodes per factor: {episodes_per_factor}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    
    # Run experiment
    results = run_time_dilation_experiment(
        dilation_factors, episodes_per_factor, max_steps_per_episode
    )
    
    # Print summary
    print_experiment_summary(results)
    
    # Create visualizations
    create_visualizations(results)
    
    print("\nExample completed! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()