"""
Core modules for time dilation simulation and RL environment management.
"""

from .time_dilation import TimeDilationSimulator, DilationFactor
from .rl_environment import RLDilatedEnvironment, EnvironmentConfig
from .physics import PhysicsSimulator, GravityModifier

__all__ = [
    "TimeDilationSimulator",
    "DilationFactor", 
    "RLDilatedEnvironment",
    "EnvironmentConfig",
    "PhysicsSimulator",
    "GravityModifier",
]