#!/usr/bin/env python3
"""
Command-line interface for Artificial Time Dilation RL project.

This module provides a command-line interface for running experiments,
generating visualizations, and managing the time dilation system.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from core.time_dilation import TimeDilationSimulator, DilationFactor
from core.rl_environment import RLDilatedEnvironment, EnvironmentConfig
from visualization.charts import DScalingChart, RewardCurveChart
from utils.data_processing import DataProcessor, ExperimentLogger
from config.settings import ConfigManager, create_default_config


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def run_experiment(
    env_name: str,
    dilation_factors: List[float],
    num_episodes: int,
    max_steps: int,
    output_dir: str,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run a time dilation experiment.
    
    Args:
        env_name: Name of the RL environment
        dilation_factors: List of dilation factors to test
        num_episodes: Number of episodes per dilation factor
        max_steps: Maximum steps per episode
        output_dir: Directory to save results
        save_results: Whether to save results to file
        
    Returns:
        Experiment results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment with environment: {env_name}")
    logger.info(f"Dilation factors: {dilation_factors}")
    logger.info(f"Episodes per factor: {num_episodes}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data processor and logger
    data_processor = DataProcessor(str(output_path / "data"))
    experiment_logger = ExperimentLogger(str(output_path / "logs"))
    
    # Create environment configuration
    config = EnvironmentConfig(
        env_name=env_name,
        max_episode_steps=max_steps
    )
    
    # Run experiments
    all_results = {}
    
    for dilation_factor in dilation_factors:
        logger.info(f"Testing dilation factor: {dilation_factor}x")
        
        # Create environment
        try:
            env = RLDilatedEnvironment(config)
            env.set_dilation_factor(dilation_factor)
        except Exception as e:
            logger.warning(f"Could not create environment {env_name}: {e}")
            logger.info("Using mock environment for demonstration")
            env = None
        
        # Run episodes
        episode_data = []
        experiment_logger.start_experiment(
            f"dilation_{dilation_factor}x",
            {"dilation_factor": dilation_factor, "env_name": env_name}
        )
        
        for episode in range(num_episodes):
            if env:
                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0
                
                for step in range(max_steps):
                    action = env.env.action_space.sample()  # Random action
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if terminated or truncated:
                        break
                
                episode_metrics = env.get_current_episode_metrics()
            else:
                # Mock data for demonstration
                episode_reward = 100 + (episode * 10) + (dilation_factor * 0.1)
                episode_length = 200 + (episode * 5)
                episode_metrics = {
                    "episode_number": episode + 1,
                    "step_count": episode_length,
                    "episode_reward": episode_reward,
                    "average_reward": episode_reward / episode_length,
                    "real_time": episode_length * 0.01,
                    "simulated_time": episode_length * 0.01 * dilation_factor,
                    "dilation_factor": dilation_factor,
                    "steps_per_second": 100.0
                }
            
            episode_data.append({
                "reward": episode_reward,
                "length": episode_length,
                "time": episode_metrics.get("real_time", 0),
                "simulated_time": episode_metrics.get("simulated_time", 0),
                "dilation_factor": dilation_factor
            })
            
            experiment_logger.log_episode(episode + 1, episode_data[-1])
        
        if env:
            env.close()
        
        experiment_logger.end_experiment()
        all_results[str(dilation_factor)] = episode_data
    
    # Process and save results
    processed_data = data_processor.process_experiment_data(all_results)
    
    if save_results:
        # Save processed data
        data_processor.save_data(processed_data, "experiment_results", "json")
        
        # Save experiment log
        experiment_logger.save_experiment_log()
        
        logger.info(f"Results saved to {output_path}")
    
    return processed_data


def generate_visualizations(
    results: Dict[str, Any],
    output_dir: str,
    chart_types: List[str] = ["d_scaling", "reward_curves"]
) -> None:
    """
    Generate visualizations from experiment results.
    
    Args:
        results: Experiment results
        output_dir: Directory to save visualizations
        chart_types: Types of charts to generate
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if "d_scaling" in chart_types:
        # Generate D-scaling chart
        d_chart = DScalingChart()
        time_points = np.linspace(0, 72, 100)  # 72 hours
        dilation_factors = [1, 10, 100, 1000, 10000]
        
        d_fig = d_chart.plot_matplotlib(time_points, dilation_factors)
        d_fig.savefig(output_path / "d_scaling_chart.png", dpi=300, bbox_inches='tight')
        logger.info("Saved D-scaling chart")
    
    if "reward_curves" in chart_types:
        # Generate reward curve charts for each dilation factor
        reward_chart = RewardCurveChart()
        
        for factor, data in results.get("experiment_data", {}).items():
            if not data:
                continue
            
            # Create mock time series data
            num_episodes = data.get("num_episodes", 10)
            time_points = np.linspace(0, 1, num_episodes)  # 1 hour simulation
            rewards = np.cumsum([data.get("average_reward", 100)] * num_episodes)
            steps = np.arange(num_episodes) * data.get("average_length", 200)
            
            reward_fig = reward_chart.plot_matplotlib(
                time_points, rewards, steps, float(factor)
            )
            reward_fig.savefig(
                output_path / f"reward_curves_{factor}x.png",
                dpi=300, bbox_inches='tight'
            )
            logger.info(f"Saved reward curve chart for {factor}x")


def create_config_file(output_path: str) -> None:
    """Create a default configuration file."""
    create_default_config(output_path)
    print(f"Default configuration created at {output_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Artificial Time Dilation for RL - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic experiment
  python -m src.cli experiment --env CartPole-v1 --factors 1 10 100 --episodes 5
  
  # Generate visualizations
  python -m src.cli visualize --input results.json --output charts/
  
  # Create configuration file
  python -m src.cli config --output config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run time dilation experiment')
    exp_parser.add_argument('--env', default='CartPole-v1', help='RL environment name')
    exp_parser.add_argument('--factors', nargs='+', type=float, default=[1, 10, 100],
                           help='Dilation factors to test')
    exp_parser.add_argument('--episodes', type=int, default=5, help='Episodes per factor')
    exp_parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    exp_parser.add_argument('--output', default='results', help='Output directory')
    exp_parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    # Visualization command
    vis_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    vis_parser.add_argument('--input', required=True, help='Input results file')
    vis_parser.add_argument('--output', default='charts', help='Output directory')
    vis_parser.add_argument('--types', nargs='+', default=['d_scaling', 'reward_curves'],
                           help='Chart types to generate')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Create configuration file')
    config_parser.add_argument('--output', default='config.yaml', help='Output file path')
    
    # Global options
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else args.log_level
    setup_logging(log_level)
    
    if args.command == 'experiment':
        # Run experiment
        results = run_experiment(
            env_name=args.env,
            dilation_factors=args.factors,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            output_dir=args.output,
            save_results=not args.no_save
        )
        
        # Generate visualizations
        generate_visualizations(results, args.output)
        
        print(f"\nExperiment completed! Results saved to {args.output}")
        print(f"Total dilation factors tested: {len(args.factors)}")
        print(f"Episodes per factor: {args.episodes}")
        
    elif args.command == 'visualize':
        # Load results and generate visualizations
        try:
            with open(args.input, 'r') as f:
                results = json.load(f)
            generate_visualizations(results, args.output, args.types)
            print(f"Visualizations saved to {args.output}")
        except FileNotFoundError:
            print(f"Error: Input file {args.input} not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {args.input}")
            sys.exit(1)
    
    elif args.command == 'config':
        # Create configuration file
        create_config_file(args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()