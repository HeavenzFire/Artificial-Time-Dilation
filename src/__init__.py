"""
Artificial Time Dilation for Reinforcement Learning

A research project exploring the application of artificial time dilation concepts
to accelerate reinforcement learning training through simulation speed scaling.
"""

__version__ = "0.1.0"
__author__ = "[Your Name]"
__email__ = "[your-email@domain.com]"

from .core.time_dilation import TimeDilationSimulator
from .core.rl_environment import RLDilatedEnvironment
from .visualization.charts import DScalingChart, RewardCurveChart

__all__ = [
    "TimeDilationSimulator",
    "RLDilatedEnvironment", 
    "DScalingChart",
    "RewardCurveChart",
]