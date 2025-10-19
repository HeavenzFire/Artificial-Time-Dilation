"""
RL environment wrapper with time dilation support.

This module provides RL environment integration with time dilation capabilities,
supporting various RL frameworks and environments including MuJoCo.
"""

import gym
import gymnasium
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from .time_dilation import TimeDilationSimulator, DilationFactor

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for RL environment with time dilation."""
    env_name: str
    dilation_factor: float = 1.0
    max_episode_steps: int = 1000
    render_mode: Optional[str] = None
    physics_timestep: float = 0.01
    enable_gravity_modification: bool = False
    gravity_scale: float = 1.0
    custom_reward_scale: float = 1.0


class RLDilatedEnvironment:
    """
    RL environment wrapper with time dilation capabilities.
    
    This class wraps standard RL environments (OpenAI Gym/Gymnasium) and adds
    time dilation functionality, allowing for accelerated training through
    simulation speed scaling.
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        time_dilation_simulator: Optional[TimeDilationSimulator] = None
    ):
        """
        Initialize the dilated RL environment.
        
        Args:
            config: Environment configuration
            time_dilation_simulator: Optional pre-configured time dilation simulator
        """
        self.config = config
        self.env = None
        self.time_dilation = time_dilation_simulator or TimeDilationSimulator(
            base_dilation_factor=config.dilation_factor
        )
        
        # Environment state
        self.episode_count = 0
        self.step_count = 0
        self.episode_rewards = []
        self.total_rewards = 0.0
        
        # Performance tracking
        self.performance_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "training_time": [],
            "dilation_factors": []
        }
        
        self._initialize_environment()
    
    def _initialize_environment(self) -> None:
        """Initialize the underlying RL environment."""
        try:
            # Try Gymnasium first (newer API)
            self.env = gymnasium.make(
                self.config.env_name,
                render_mode=self.config.render_mode,
                max_episode_steps=self.config.max_episode_steps
            )
            self.is_gymnasium = True
        except Exception:
            try:
                # Fallback to OpenAI Gym
                self.env = gym.make(self.config.env_name)
                self.is_gymnasium = False
            except Exception as e:
                raise RuntimeError(f"Failed to create environment {self.config.env_name}: {e}")
        
        logger.info(f"Initialized environment {self.config.env_name} with dilation factor {self.config.dilation_factor}")
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and start a new episode.
        
        Returns:
            Tuple of (observation, info)
        """
        if self.is_gymnasium:
            obs, info = self.env.reset()
        else:
            obs = self.env.reset()
            info = {}
        
        self.episode_count += 1
        self.step_count = 0
        self.episode_rewards = []
        
        # Reset time dilation for new episode
        self.time_dilation.reset()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform one step in the environment with time dilation.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Take step in environment
        if self.is_gymnasium:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Convert gym format to gymnasium format
            if isinstance(truncated, bool):
                truncated = truncated
            else:
                truncated = False
        
        # Apply custom reward scaling
        reward *= self.config.custom_reward_scale
        
        # Update time dilation with reward
        time_metrics = self.time_dilation.step(reward)
        
        # Track performance
        self.step_count += 1
        self.episode_rewards.append(reward)
        self.total_rewards += reward
        
        # Add time dilation info to environment info
        info.update({
            "time_dilation": {
                "real_time": time_metrics.real_time,
                "simulated_time": time_metrics.simulated_time,
                "dilation_factor": time_metrics.dilation_factor,
                "steps_per_second": time_metrics.steps_per_second
            }
        })
        
        # Check if episode is done
        done = terminated or truncated
        
        if done:
            self._finalize_episode()
        
        return obs, reward, terminated, truncated, info
    
    def _finalize_episode(self) -> None:
        """Finalize episode and update performance metrics."""
        episode_reward = sum(self.episode_rewards)
        episode_length = len(self.episode_rewards)
        
        # Update performance metrics
        self.performance_metrics["episode_rewards"].append(episode_reward)
        self.performance_metrics["episode_lengths"].append(episode_length)
        self.performance_metrics["training_time"].append(self.time_dilation.get_time_metrics().real_time)
        self.performance_metrics["dilation_factors"].append(self.time_dilation.get_dilation_factor())
        
        logger.info(
            f"Episode {self.episode_count} completed: "
            f"reward={episode_reward:.2f}, length={episode_length}, "
            f"dilation={self.time_dilation.get_dilation_factor():.1f}x"
        )
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None
    
    def close(self) -> None:
        """Close the environment."""
        if self.env:
            self.env.close()
    
    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """Get performance metrics for all episodes."""
        return self.performance_metrics.copy()
    
    def get_current_episode_metrics(self) -> Dict[str, Any]:
        """Get metrics for the current episode."""
        time_metrics = self.time_dilation.get_time_metrics()
        return {
            "episode_number": self.episode_count,
            "step_count": self.step_count,
            "episode_reward": sum(self.episode_rewards),
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "real_time": time_metrics.real_time,
            "simulated_time": time_metrics.simulated_time,
            "dilation_factor": time_metrics.dilation_factor,
            "steps_per_second": time_metrics.steps_per_second
        }
    
    def set_dilation_factor(self, factor: Union[float, DilationFactor]) -> None:
        """Set the dilation factor."""
        self.time_dilation.set_dilation_factor(factor)
    
    def get_dilation_factor(self) -> float:
        """Get the current dilation factor."""
        return self.time_dilation.get_dilation_factor()
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive simulation summary."""
        time_summary = self.time_dilation.get_simulation_summary()
        performance_summary = self.get_performance_metrics()
        
        return {
            "time_dilation": time_summary,
            "performance": {
                "total_episodes": self.episode_count,
                "total_steps": sum(performance_summary["episode_lengths"]),
                "average_episode_reward": np.mean(performance_summary["episode_rewards"]) if performance_summary["episode_rewards"] else 0.0,
                "average_episode_length": np.mean(performance_summary["episode_lengths"]) if performance_summary["episode_lengths"] else 0.0,
                "best_episode_reward": max(performance_summary["episode_rewards"]) if performance_summary["episode_rewards"] else 0.0,
                "worst_episode_reward": min(performance_summary["episode_rewards"]) if performance_summary["episode_rewards"] else 0.0
            }
        }


class MultiEnvironmentDilatedTrainer:
    """
    Trainer for multiple dilated environments with different dilation factors.
    
    This class enables training across multiple environments with different
    time dilation factors, allowing for comparison and analysis of dilation effects.
    """
    
    def __init__(self, environment_configs: List[EnvironmentConfig]):
        """
        Initialize the multi-environment trainer.
        
        Args:
            environment_configs: List of environment configurations
        """
        self.environments = []
        self.configs = environment_configs
        
        for config in environment_configs:
            env = RLDilatedEnvironment(config)
            self.environments.append(env)
    
    def train_episode(self, env_index: int, agent, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Train one episode in a specific environment.
        
        Args:
            env_index: Index of the environment to use
            agent: RL agent to train
            max_steps: Maximum steps per episode
            
        Returns:
            Episode metrics
        """
        if env_index >= len(self.environments):
            raise ValueError(f"Environment index {env_index} out of range")
        
        env = self.environments[env_index]
        obs, info = env.reset()
        
        episode_reward = 0.0
        step_count = 0
        
        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        return {
            "env_index": env_index,
            "dilation_factor": env.get_dilation_factor(),
            "episode_reward": episode_reward,
            "step_count": step_count,
            "time_metrics": env.get_current_episode_metrics()
        }
    
    def compare_dilation_factors(self, agent, episodes_per_env: int = 10) -> Dict[str, Any]:
        """
        Compare performance across different dilation factors.
        
        Args:
            agent: RL agent to test
            episodes_per_env: Number of episodes per environment
            
        Returns:
            Comparison results
        """
        results = {}
        
        for i, env in enumerate(self.environments):
            env_results = []
            
            for episode in range(episodes_per_env):
                episode_result = self.train_episode(i, agent)
                env_results.append(episode_result)
            
            dilation_factor = env.get_dilation_factor()
            results[f"dilation_{dilation_factor}x"] = {
                "episodes": env_results,
                "average_reward": np.mean([r["episode_reward"] for r in env_results]),
                "std_reward": np.std([r["episode_reward"] for r in env_results]),
                "average_steps": np.mean([r["step_count"] for r in env_results])
            }
        
        return results
    
    def close_all(self) -> None:
        """Close all environments."""
        for env in self.environments:
            env.close()