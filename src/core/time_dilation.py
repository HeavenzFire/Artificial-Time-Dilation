"""
Core time dilation simulation algorithms.

This module implements the fundamental time dilation concepts applied to
reinforcement learning environments, enabling accelerated training through
simulation speed scaling.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DilationFactor(Enum):
    """Predefined dilation factors for common simulation speeds."""
    REAL_TIME = 1.0
    FAST = 10.0
    VERY_FAST = 100.0
    ULTRA_FAST = 1000.0
    EXTREME = 10000.0


@dataclass
class TimeMetrics:
    """Container for time dilation metrics."""
    real_time: float
    simulated_time: float
    dilation_factor: float
    steps_per_second: float
    total_steps: int
    
    @property
    def time_compression_ratio(self) -> float:
        """Ratio of simulated time to real time."""
        return self.simulated_time / self.real_time if self.real_time > 0 else 0.0


class TimeDilationSimulator:
    """
    Core time dilation simulator for RL environments.
    
    Implements velocity-based time dilation concepts where simulation speed
    (analogous to relativistic velocity) scales the effective "experienced time"
    for RL agents, compressing years of training into real-world hours.
    """
    
    def __init__(
        self,
        base_dilation_factor: float = 1.0,
        max_dilation_factor: float = 10000.0,
        time_step: float = 0.01,
        enable_adaptive_scaling: bool = True
    ):
        """
        Initialize the time dilation simulator.
        
        Args:
            base_dilation_factor: Base dilation factor (1.0 = real-time)
            max_dilation_factor: Maximum allowed dilation factor
            time_step: Simulation time step in seconds
            enable_adaptive_scaling: Whether to enable adaptive scaling based on performance
        """
        self.base_dilation_factor = base_dilation_factor
        self.max_dilation_factor = max_dilation_factor
        self.time_step = time_step
        self.enable_adaptive_scaling = enable_adaptive_scaling
        
        self.current_dilation_factor = base_dilation_factor
        self.start_time = None
        self.total_simulated_time = 0.0
        self.total_real_time = 0.0
        self.step_count = 0
        
        # Performance tracking for adaptive scaling
        self.performance_history = []
        self.adaptive_threshold = 0.1  # 10% performance change triggers scaling
        
    def start_simulation(self) -> None:
        """Start the time dilation simulation."""
        self.start_time = time.time()
        self.total_simulated_time = 0.0
        self.total_real_time = 0.0
        self.step_count = 0
        logger.info(f"Started time dilation simulation with factor {self.current_dilation_factor}")
        
    def step(self, reward: Optional[float] = None) -> TimeMetrics:
        """
        Perform one simulation step with time dilation.
        
        Args:
            reward: Optional reward value for adaptive scaling
            
        Returns:
            TimeMetrics object containing current time dilation information
        """
        if self.start_time is None:
            self.start_simulation()
            
        # Calculate real time elapsed
        current_real_time = time.time() - self.start_time
        real_time_delta = current_real_time - self.total_real_time
        
        # Calculate simulated time based on dilation factor
        simulated_time_delta = real_time_delta * self.current_dilation_factor
        self.total_simulated_time += simulated_time_delta
        self.total_real_time = current_real_time
        self.step_count += 1
        
        # Update performance history for adaptive scaling
        if reward is not None:
            self.performance_history.append(reward)
            if self.enable_adaptive_scaling:
                self._update_adaptive_scaling()
        
        # Calculate steps per second
        steps_per_second = self.step_count / self.total_real_time if self.total_real_time > 0 else 0
        
        return TimeMetrics(
            real_time=self.total_real_time,
            simulated_time=self.total_simulated_time,
            dilation_factor=self.current_dilation_factor,
            steps_per_second=steps_per_second,
            total_steps=self.step_count
        )
    
    def set_dilation_factor(self, factor: Union[float, DilationFactor]) -> None:
        """
        Set the dilation factor.
        
        Args:
            factor: Dilation factor (float or DilationFactor enum)
        """
        if isinstance(factor, DilationFactor):
            factor = factor.value
            
        if factor < 1.0:
            raise ValueError("Dilation factor must be >= 1.0")
        if factor > self.max_dilation_factor:
            raise ValueError(f"Dilation factor must be <= {self.max_dilation_factor}")
            
        self.current_dilation_factor = factor
        logger.info(f"Updated dilation factor to {factor}")
    
    def get_dilation_factor(self) -> float:
        """Get the current dilation factor."""
        return self.current_dilation_factor
    
    def get_time_metrics(self) -> TimeMetrics:
        """Get current time dilation metrics."""
        if self.start_time is None:
            return TimeMetrics(0, 0, self.current_dilation_factor, 0, 0)
            
        current_real_time = time.time() - self.start_time
        steps_per_second = self.step_count / current_real_time if current_real_time > 0 else 0
        
        return TimeMetrics(
            real_time=current_real_time,
            simulated_time=self.total_simulated_time,
            dilation_factor=self.current_dilation_factor,
            steps_per_second=steps_per_second,
            total_steps=self.step_count
        )
    
    def _update_adaptive_scaling(self) -> None:
        """Update dilation factor based on performance trends."""
        if len(self.performance_history) < 10:
            return
            
        # Calculate performance trend
        recent_performance = np.mean(self.performance_history[-10:])
        older_performance = np.mean(self.performance_history[-20:-10]) if len(self.performance_history) >= 20 else recent_performance
        
        performance_change = (recent_performance - older_performance) / abs(older_performance) if older_performance != 0 else 0
        
        # Adjust dilation factor based on performance
        if performance_change > self.adaptive_threshold:
            # Performance improving, increase dilation
            new_factor = min(self.current_dilation_factor * 1.1, self.max_dilation_factor)
        elif performance_change < -self.adaptive_threshold:
            # Performance degrading, decrease dilation
            new_factor = max(self.current_dilation_factor * 0.9, 1.0)
        else:
            return
            
        if new_factor != self.current_dilation_factor:
            self.set_dilation_factor(new_factor)
    
    def reset(self) -> None:
        """Reset the simulation state."""
        self.start_time = None
        self.total_simulated_time = 0.0
        self.total_real_time = 0.0
        self.step_count = 0
        self.performance_history = []
        self.current_dilation_factor = self.base_dilation_factor
        logger.info("Reset time dilation simulation")
    
    def get_simulation_summary(self) -> Dict[str, Union[float, int, str]]:
        """Get a summary of the simulation state."""
        metrics = self.get_time_metrics()
        return {
            "real_time_hours": metrics.real_time / 3600,
            "simulated_time_hours": metrics.simulated_time / 3600,
            "dilation_factor": metrics.dilation_factor,
            "time_compression_ratio": metrics.time_compression_ratio,
            "steps_per_second": metrics.steps_per_second,
            "total_steps": metrics.total_steps,
            "simulation_speed": f"{metrics.dilation_factor:.1f}x real-time"
        }


def calculate_lorentz_factor(velocity_fraction: float) -> float:
    """
    Calculate the Lorentz factor for relativistic time dilation.
    
    Args:
        velocity_fraction: Velocity as a fraction of the speed of light (0-1)
        
    Returns:
        Lorentz factor (gamma)
    """
    if velocity_fraction < 0 or velocity_fraction >= 1:
        raise ValueError("Velocity fraction must be in range [0, 1)")
    
    return 1.0 / np.sqrt(1 - velocity_fraction**2)


def calculate_effective_time_dilation(
    simulation_speed: float,
    base_time: float,
    lorentz_factor: Optional[float] = None
) -> float:
    """
    Calculate effective time dilation for simulation.
    
    Args:
        simulation_speed: Simulation speed multiplier
        base_time: Base time in seconds
        lorentz_factor: Optional Lorentz factor for relativistic effects
        
    Returns:
        Effective dilated time
    """
    if lorentz_factor is not None:
        return base_time * simulation_speed * lorentz_factor
    else:
        return base_time * simulation_speed


def generate_dilation_curve(
    time_points: np.ndarray,
    dilation_factors: List[float]
) -> Dict[str, np.ndarray]:
    """
    Generate time dilation curves for visualization.
    
    Args:
        time_points: Array of time points in hours
        dilation_factors: List of dilation factors to plot
        
    Returns:
        Dictionary mapping dilation factor names to time curves
    """
    curves = {}
    
    for factor in dilation_factors:
        simulated_times = time_points * factor
        curves[f"{factor}x"] = simulated_times
    
    return curves