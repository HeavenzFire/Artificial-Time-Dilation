"""
Physics simulation components for time dilation effects.

This module provides physics simulation utilities that can be modified
to demonstrate time dilation effects in RL environments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhysicsState:
    """Container for physics simulation state."""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    time: float
    gravity: float
    mass: float


class PhysicsSimulator:
    """
    Physics simulator with time dilation support.
    
    This class provides a basic physics simulation that can be used
    to demonstrate time dilation effects in RL environments.
    """
    
    def __init__(
        self,
        gravity: float = 9.81,
        time_step: float = 0.01,
        enable_drag: bool = True,
        drag_coefficient: float = 0.1
    ):
        """
        Initialize the physics simulator.
        
        Args:
            gravity: Gravitational acceleration
            time_step: Simulation time step
            enable_drag: Whether to enable air resistance
            drag_coefficient: Air resistance coefficient
        """
        self.gravity = gravity
        self.time_step = time_step
        self.enable_drag = enable_drag
        self.drag_coefficient = drag_coefficient
        
        self.state = None
        self.time = 0.0
        self.history = []
    
    def initialize_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray = None,
        mass: float = 1.0
    ) -> None:
        """
        Initialize the physics state.
        
        Args:
            position: Initial position
            velocity: Initial velocity (defaults to zero)
            mass: Object mass
        """
        if velocity is None:
            velocity = np.zeros_like(position)
        
        self.state = PhysicsState(
            position=position.copy(),
            velocity=velocity.copy(),
            acceleration=np.zeros_like(position),
            time=0.0,
            gravity=self.gravity,
            mass=mass
        )
        
        self.time = 0.0
        self.history = [self.state]
        
        logger.info(f"Initialized physics state at position {position}")
    
    def step(self, force: np.ndarray = None, dilation_factor: float = 1.0) -> PhysicsState:
        """
        Perform one physics simulation step.
        
        Args:
            force: External force applied to the object
            dilation_factor: Time dilation factor
            
        Returns:
            Updated physics state
        """
        if self.state is None:
            raise RuntimeError("Physics state not initialized")
        
        if force is None:
            force = np.zeros_like(self.state.position)
        
        # Calculate acceleration from forces
        gravity_force = np.array([0, 0, -self.gravity * self.state.mass])
        total_force = force + gravity_force
        
        # Add drag force if enabled
        if self.enable_drag:
            drag_force = -self.drag_coefficient * self.state.velocity
            total_force += drag_force
        
        # Calculate acceleration
        acceleration = total_force / self.state.mass
        
        # Apply time dilation to time step
        dilated_time_step = self.time_step * dilation_factor
        
        # Update state using Verlet integration
        new_position = (
            self.state.position + 
            self.state.velocity * dilated_time_step + 
            0.5 * acceleration * dilated_time_step**2
        )
        
        new_velocity = self.state.velocity + acceleration * dilated_time_step
        
        # Update state
        self.state = PhysicsState(
            position=new_position,
            velocity=new_velocity,
            acceleration=acceleration,
            time=self.state.time + dilated_time_step,
            gravity=self.gravity,
            mass=self.state.mass
        )
        
        self.time += dilated_time_step
        self.history.append(self.state)
        
        return self.state
    
    def get_state(self) -> Optional[PhysicsState]:
        """Get the current physics state."""
        return self.state
    
    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the trajectory history.
        
        Returns:
            Tuple of (positions, velocities, times)
        """
        if not self.history:
            return np.array([]), np.array([]), np.array([])
        
        positions = np.array([state.position for state in self.history])
        velocities = np.array([state.velocity for state in self.history])
        times = np.array([state.time for state in self.history])
        
        return positions, velocities, times
    
    def reset(self) -> None:
        """Reset the physics simulation."""
        self.state = None
        self.time = 0.0
        self.history = []
        logger.info("Reset physics simulation")


class GravityModifier:
    """
    Gravity modification utility for demonstrating time dilation effects.
    
    This class allows for dynamic modification of gravitational effects
    to simulate different time dilation scenarios.
    """
    
    def __init__(self, base_gravity: float = 9.81):
        """
        Initialize the gravity modifier.
        
        Args:
            base_gravity: Base gravitational acceleration
        """
        self.base_gravity = base_gravity
        self.current_gravity = base_gravity
        self.gravity_history = []
    
    def set_gravity(self, gravity: float) -> None:
        """
        Set the current gravity value.
        
        Args:
            gravity: New gravity value
        """
        self.current_gravity = gravity
        self.gravity_history.append(gravity)
        logger.info(f"Set gravity to {gravity}")
    
    def scale_gravity(self, scale_factor: float) -> None:
        """
        Scale the gravity by a factor.
        
        Args:
            scale_factor: Factor to scale gravity by
        """
        new_gravity = self.base_gravity * scale_factor
        self.set_gravity(new_gravity)
    
    def apply_time_dilation_gravity(self, dilation_factor: float) -> None:
        """
        Apply time dilation effects to gravity.
        
        In time dilation scenarios, gravitational effects may be perceived
        differently due to the accelerated time frame.
        
        Args:
            dilation_factor: Time dilation factor
        """
        # Scale gravity based on time dilation
        # This is a simplified model - in reality, time dilation effects
        # on gravity are more complex and depend on the specific scenario
        scaled_gravity = self.base_gravity * (1.0 + np.log(dilation_factor) * 0.1)
        self.set_gravity(scaled_gravity)
    
    def get_gravity(self) -> float:
        """Get the current gravity value."""
        return self.current_gravity
    
    def reset(self) -> None:
        """Reset gravity to base value."""
        self.current_gravity = self.base_gravity
        self.gravity_history = []
        logger.info("Reset gravity to base value")


class TimeDilatedPhysics:
    """
    Combined physics simulator with time dilation effects.
    
    This class combines physics simulation with time dilation concepts,
    allowing for realistic simulation of accelerated environments.
    """
    
    def __init__(
        self,
        physics_simulator: PhysicsSimulator,
        gravity_modifier: GravityModifier,
        dilation_factor: float = 1.0
    ):
        """
        Initialize the time-dilated physics system.
        
        Args:
            physics_simulator: Base physics simulator
            gravity_modifier: Gravity modification utility
            dilation_factor: Initial time dilation factor
        """
        self.physics = physics_simulator
        self.gravity_modifier = gravity_modifier
        self.dilation_factor = dilation_factor
        
        self.time_dilation_history = []
        self.performance_metrics = {
            "simulation_speed": [],
            "energy_consumption": [],
            "computational_efficiency": []
        }
    
    def set_dilation_factor(self, factor: float) -> None:
        """
        Set the time dilation factor.
        
        Args:
            factor: New dilation factor
        """
        self.dilation_factor = factor
        self.time_dilation_history.append(factor)
        
        # Apply time dilation effects to gravity
        self.gravity_modifier.apply_time_dilation_gravity(factor)
        
        logger.info(f"Set time dilation factor to {factor}")
    
    def step(self, force: np.ndarray = None) -> PhysicsState:
        """
        Perform one time-dilated physics step.
        
        Args:
            force: External force applied to the object
            
        Returns:
            Updated physics state
        """
        # Update physics simulator with current gravity
        self.physics.gravity = self.gravity_modifier.get_gravity()
        
        # Perform physics step with dilation
        state = self.physics.step(force, self.dilation_factor)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return state
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for the simulation."""
        # Calculate simulation speed (steps per second)
        if len(self.physics.history) > 1:
            time_delta = self.physics.history[-1].time - self.physics.history[-2].time
            if time_delta > 0:
                simulation_speed = 1.0 / time_delta
                self.performance_metrics["simulation_speed"].append(simulation_speed)
        
        # Calculate energy consumption (simplified model)
        if self.physics.state:
            kinetic_energy = 0.5 * self.physics.state.mass * np.sum(self.physics.state.velocity**2)
            potential_energy = self.physics.state.mass * self.gravity_modifier.get_gravity() * self.physics.state.position[2]
            total_energy = kinetic_energy + potential_energy
            self.performance_metrics["energy_consumption"].append(total_energy)
        
        # Calculate computational efficiency
        efficiency = self.dilation_factor / (1.0 + self.dilation_factor * 0.1)  # Simplified model
        self.performance_metrics["computational_efficiency"].append(efficiency)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get a summary of performance metrics."""
        summary = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[f"average_{metric_name}"] = np.mean(values)
                summary[f"max_{metric_name}"] = np.max(values)
                summary[f"min_{metric_name}"] = np.min(values)
        
        summary["dilation_factor"] = self.dilation_factor
        summary["total_steps"] = len(self.physics.history)
        
        return summary
    
    def reset(self) -> None:
        """Reset the time-dilated physics system."""
        self.physics.reset()
        self.gravity_modifier.reset()
        self.dilation_factor = 1.0
        self.time_dilation_history = []
        self.performance_metrics = {
            "simulation_speed": [],
            "energy_consumption": [],
            "computational_efficiency": []
        }
        logger.info("Reset time-dilated physics system")