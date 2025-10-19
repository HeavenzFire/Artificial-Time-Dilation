"""
Configuration settings for the Artificial Time Dilation RL project.

This module provides centralized configuration management for all components
of the time dilation system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import yaml
import json
from pathlib import Path


@dataclass
class TimeDilationConfig:
    """Configuration for time dilation simulation."""
    base_dilation_factor: float = 1.0
    max_dilation_factor: float = 10000.0
    time_step: float = 0.01
    enable_adaptive_scaling: bool = True
    adaptive_threshold: float = 0.1
    performance_window_size: int = 10


@dataclass
class RLEnvironmentConfig:
    """Configuration for RL environments."""
    env_name: str = "CartPole-v1"
    max_episode_steps: int = 1000
    render_mode: Optional[str] = None
    physics_timestep: float = 0.01
    enable_gravity_modification: bool = False
    gravity_scale: float = 1.0
    custom_reward_scale: float = 1.0


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    width: int = 800
    height: int = 600
    title_font_size: int = 16
    label_font_size: int = 12
    legend_font_size: int = 10
    color_palette: str = "viridis"
    style: str = "whitegrid"
    dpi: int = 300
    save_format: str = "png"


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    num_episodes: int = 10
    max_steps_per_episode: int = 200
    dilation_factors: List[float] = field(default_factory=lambda: [1, 10, 100, 1000])
    num_runs: int = 1
    save_results: bool = True
    results_dir: str = "data/results"
    log_level: str = "INFO"


@dataclass
class WebConfig:
    """Configuration for web interface."""
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    theme: str = "light"
    enable_animations: bool = True


@dataclass
class ProjectConfig:
    """Main project configuration."""
    time_dilation: TimeDilationConfig = field(default_factory=TimeDilationConfig)
    rl_environment: RLEnvironmentConfig = field(default_factory=RLEnvironmentConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    web: WebConfig = field(default_factory=WebConfig)
    
    # Project metadata
    project_name: str = "Artificial Time Dilation for RL"
    version: str = "0.1.0"
    author: str = "[Your Name]"
    email: str = "[your-email@domain.com]"
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "data/models"
    results_dir: str = "data/results"
    assets_dir: str = "assets"
    logs_dir: str = "logs"


class ConfigManager:
    """Configuration manager for loading and saving settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config.yaml"
        self.config = ProjectConfig()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self._update_config_from_dict(config_data)
                print(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
                print("Using default configuration")
        else:
            print(f"Configuration file {self.config_path} not found, using defaults")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'time_dilation' in config_data:
            for key, value in config_data['time_dilation'].items():
                if hasattr(self.config.time_dilation, key):
                    setattr(self.config.time_dilation, key, value)
        
        if 'rl_environment' in config_data:
            for key, value in config_data['rl_environment'].items():
                if hasattr(self.config.rl_environment, key):
                    setattr(self.config.rl_environment, key, value)
        
        if 'visualization' in config_data:
            for key, value in config_data['visualization'].items():
                if hasattr(self.config.visualization, key):
                    setattr(self.config.visualization, key, value)
        
        if 'experiment' in config_data:
            for key, value in config_data['experiment'].items():
                if hasattr(self.config.experiment, key):
                    setattr(self.config.experiment, key, value)
        
        if 'web' in config_data:
            for key, value in config_data['web'].items():
                if hasattr(self.config.web, key):
                    setattr(self.config.web, key, value)
        
        # Update project metadata
        for key in ['project_name', 'version', 'author', 'email']:
            if key in config_data:
                setattr(self.config, key, config_data[key])
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        
        config_dict = {
            'time_dilation': {
                'base_dilation_factor': self.config.time_dilation.base_dilation_factor,
                'max_dilation_factor': self.config.time_dilation.max_dilation_factor,
                'time_step': self.config.time_dilation.time_step,
                'enable_adaptive_scaling': self.config.time_dilation.enable_adaptive_scaling,
                'adaptive_threshold': self.config.time_dilation.adaptive_threshold,
                'performance_window_size': self.config.time_dilation.performance_window_size
            },
            'rl_environment': {
                'env_name': self.config.rl_environment.env_name,
                'max_episode_steps': self.config.rl_environment.max_episode_steps,
                'render_mode': self.config.rl_environment.render_mode,
                'physics_timestep': self.config.rl_environment.physics_timestep,
                'enable_gravity_modification': self.config.rl_environment.enable_gravity_modification,
                'gravity_scale': self.config.rl_environment.gravity_scale,
                'custom_reward_scale': self.config.rl_environment.custom_reward_scale
            },
            'visualization': {
                'width': self.config.visualization.width,
                'height': self.config.visualization.height,
                'title_font_size': self.config.visualization.title_font_size,
                'label_font_size': self.config.visualization.label_font_size,
                'legend_font_size': self.config.visualization.legend_font_size,
                'color_palette': self.config.visualization.color_palette,
                'style': self.config.visualization.style,
                'dpi': self.config.visualization.dpi,
                'save_format': self.config.visualization.save_format
            },
            'experiment': {
                'num_episodes': self.config.experiment.num_episodes,
                'max_steps_per_episode': self.config.experiment.max_steps_per_episode,
                'dilation_factors': self.config.experiment.dilation_factors,
                'num_runs': self.config.experiment.num_runs,
                'save_results': self.config.experiment.save_results,
                'results_dir': self.config.experiment.results_dir,
                'log_level': self.config.experiment.log_level
            },
            'web': {
                'host': self.config.web.host,
                'port': self.config.web.port,
                'debug': self.config.web.debug,
                'theme': self.config.web.theme,
                'enable_animations': self.config.web.enable_animations
            },
            'project_name': self.config.project_name,
            'version': self.config.version,
            'author': self.config.author,
            'email': self.config.email
        }
        
        try:
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_config(self) -> ProjectConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Project configuration object
    """
    manager = ConfigManager(config_path)
    return manager.get_config()


def create_default_config(config_path: str = "config.yaml") -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to save the configuration file
    """
    manager = ConfigManager()
    manager.save_config(config_path)
    print(f"Default configuration created at {config_path}")


# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    "CartPole-v1": {
        "max_episode_steps": 500,
        "physics_timestep": 0.02,
        "custom_reward_scale": 1.0
    },
    "Humanoid-v4": {
        "max_episode_steps": 1000,
        "physics_timestep": 0.01,
        "custom_reward_scale": 0.1
    },
    "Atari-Breakout-v4": {
        "max_episode_steps": 10000,
        "physics_timestep": 0.01,
        "custom_reward_scale": 1.0
    }
}

# Default dilation factors for different scenarios
DILATION_FACTORS = {
    "real_time": [1.0],
    "fast": [1, 10, 100],
    "very_fast": [1, 100, 1000],
    "extreme": [1, 1000, 10000],
    "research": [1, 10, 100, 1000, 10000]
}