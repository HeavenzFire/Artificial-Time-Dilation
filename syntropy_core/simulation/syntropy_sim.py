#!/usr/bin/env python3
"""
Syntropy Core Simulation Framework

Simulates the FPGA-based syntropy core behavior for testing and validation.
Implements the same flux computation and learning algorithms as the Verilog implementation.
"""

import numpy as np
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque


@dataclass
class SensorData:
    """Sensor data structure matching FPGA inputs"""
    voltage_drift: float  # 0.0-1.0 range
    packet_loss: float    # 0.0-1.0 range
    temp_variance: float  # 0.0-100.0 range
    phase_jitter: float   # 0.0-1.0 range
    timestamp: float


@dataclass
class FluxResult:
    """Flux computation result"""
    flux_value: float
    stable: bool
    correction_vector: float
    optimized_params: List[float]
    learning_active: bool
    iteration_count: int
    state: str
    error_flag: bool


class SyntropyCore:
    """
    Python simulation of the FPGA Syntropy Core
    
    Implements the same logic as the Verilog module for testing and validation.
    """
    
    def __init__(self, 
                 threshold: float = 0.9,
                 learning_rate: float = 0.1,
                 max_iterations: int = 10,
                 enable_learning: bool = True,
                 enable_mesh: bool = True):
        
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.enable_learning = enable_learning
        self.enable_mesh = enable_mesh
        
        # State variables
        self.stable = False
        self.flux_value = 0.0
        self.correction_vector = 0.0
        self.optimized_params = [0.0, 0.0, 0.0, 0.0]  # [voltage, packet, temp, phase]
        self.learning_active = False
        self.iteration_count = 0
        self.state = "IDLE"
        self.error_flag = False
        
        # Learning parameters
        self.params = [0.0, 0.0, 0.0, 0.0]
        self.gradients = [0.0, 0.0, 0.0, 0.0]
        
        # History for analysis
        self.flux_history = deque(maxlen=1000)
        self.stability_history = deque(maxlen=1000)
        self.correction_history = deque(maxlen=1000)
        self.learning_history = deque(maxlen=100)
        
        # Mesh communication
        self.mesh_flux_in = 0.0
        self.mesh_correction_in = 0.0
        self.mesh_valid_in = False
        self.mesh_flux_out = 0.0
        self.mesh_correction_out = 0.0
        self.mesh_valid_out = False
        self.mesh_broadcast_req = False
    
    def compute_flux(self, metrics: List[float]) -> float:
        """
        Compute flux value: flux = 1.0 - (voltage_drift + packet_loss + temp_variance/100 + phase_jitter)
        
        Args:
            metrics: [voltage_drift, packet_loss, temp_variance, phase_jitter]
            
        Returns:
            Flux value (0.0-1.0+)
        """
        voltage_drift, packet_loss, temp_variance, phase_jitter = metrics
        
        # Apply learned parameters if available
        if self.enable_learning and any(p != 0.0 for p in self.params):
            voltage_drift *= (1.0 + self.params[0])
            packet_loss *= (1.0 + self.params[1])
            temp_variance *= (1.0 + self.params[2])
            phase_jitter *= (1.0 + self.params[3])
        
        flux = 1.0 - (voltage_drift + packet_loss + temp_variance/100.0 + phase_jitter)
        
        # Clamp to prevent overflow (matching FPGA behavior)
        flux = max(0.0, min(2.0, flux))
        
        return flux
    
    def compute_gradients(self, metrics: List[float]) -> List[float]:
        """
        Compute gradients for learning algorithm
        
        Args:
            metrics: Current sensor metrics
            
        Returns:
            Gradients for each parameter
        """
        flux = self.compute_flux(metrics)
        error = self.threshold - flux
        
        gradients = [
            -2 * error,           # voltage_drift gradient
            -2 * error,           # packet_loss gradient
            -2 * error / 100.0,   # temp_variance gradient (scaled)
            -2 * error            # phase_jitter gradient
        ]
        
        return gradients
    
    def update_parameters(self, gradients: List[float]) -> None:
        """
        Update learning parameters using gradient descent
        
        Args:
            gradients: Computed gradients
        """
        for i in range(4):
            self.params[i] -= self.learning_rate * gradients[i]
            # Clamp parameters to prevent overflow
            self.params[i] = max(-1.0, min(1.0, self.params[i]))
    
    def evaluate_flux(self, sensor_data: SensorData) -> FluxResult:
        """
        Main flux evaluation function - matches Verilog state machine
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            FluxResult with all computed values
        """
        metrics = [
            sensor_data.voltage_drift,
            sensor_data.packet_loss,
            sensor_data.temp_variance,
            sensor_data.phase_jitter
        ]
        
        # Check for overflow conditions
        if any(m < 0 or m > 1.0 for m in metrics[:2]) or metrics[2] < 0 or metrics[2] > 100.0 or metrics[3] < 0 or metrics[3] > 1.0:
            self.error_flag = True
            self.state = "ERROR"
            return FluxResult(
                flux_value=0.0,
                stable=False,
                correction_vector=0.0,
                optimized_params=self.optimized_params.copy(),
                learning_active=False,
                iteration_count=self.iteration_count,
                state=self.state,
                error_flag=True
            )
        
        # Compute flux
        self.flux_value = self.compute_flux(metrics)
        self.flux_history.append(self.flux_value)
        
        # Check stability
        if self.flux_value >= self.threshold:
            self.stable = True
            self.correction_vector = 0.0
            self.state = "STABLE"
            self.learning_active = False
        else:
            self.stable = False
            self.correction_vector = self.threshold - self.flux_value
            
            if self.enable_learning:
                self.state = "LEARNING"
                self.learning_active = True
                self.iteration_count = 0
                
                # Learning loop
                for iteration in range(self.max_iterations):
                    gradients = self.compute_gradients(metrics)
                    self.update_parameters(gradients)
                    
                    # Recompute flux with new parameters
                    new_flux = self.compute_flux(metrics)
                    
                    self.iteration_count = iteration + 1
                    self.flux_value = new_flux
                    
                    # Check if we've converged
                    if new_flux >= self.threshold:
                        break
                
                self.learning_active = False
                self.optimized_params = self.params.copy()
                self.learning_history.append({
                    'iteration': self.iteration_count,
                    'final_flux': self.flux_value,
                    'params': self.params.copy()
                })
            
            self.state = "UNSTABLE"
        
        # Update history
        self.stability_history.append(self.stable)
        self.correction_history.append(self.correction_vector)
        
        # Mesh communication
        if self.enable_mesh:
            self.mesh_flux_out = self.flux_value
            self.mesh_correction_out = self.correction_vector
            self.mesh_valid_out = True
            self.mesh_broadcast_req = True
        
        return FluxResult(
            flux_value=self.flux_value,
            stable=self.stable,
            correction_vector=self.correction_vector,
            optimized_params=self.optimized_params.copy(),
            learning_active=self.learning_active,
            iteration_count=self.iteration_count,
            state=self.state,
            error_flag=self.error_flag
        )
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.flux_history:
            return {}
        
        flux_array = np.array(self.flux_history)
        stability_array = np.array(self.stability_history)
        
        return {
            'total_evaluations': len(self.flux_history),
            'stability_rate': np.mean(stability_array) * 100,
            'average_flux': np.mean(flux_array),
            'min_flux': np.min(flux_array),
            'max_flux': np.max(flux_array),
            'flux_std': np.std(flux_array),
            'learning_episodes': len(self.learning_history),
            'average_correction': np.mean(self.correction_history) if self.correction_history else 0.0
        }


class SyntropyMesh:
    """
    Mesh network simulation for multiple Syntropy Core instances
    """
    
    def __init__(self, num_nodes: int = 4):
        self.num_nodes = num_nodes
        self.nodes = [SyntropyCore() for _ in range(num_nodes)]
        self.consensus_flux = 0.0
        self.consensus_correction = 0.0
        self.consensus_valid = False
        self.leader_node = 0
        
        # Mesh statistics
        self.communication_rounds = 0
        self.consensus_history = deque(maxlen=1000)
    
    def update_mesh(self, sensor_data: List[SensorData]) -> List[FluxResult]:
        """
        Update all nodes and perform mesh consensus
        
        Args:
            sensor_data: Sensor data for each node
            
        Returns:
            Flux results for each node
        """
        results = []
        
        # Update each node
        for i, (node, sensors) in enumerate(zip(self.nodes, sensor_data)):
            result = node.evaluate_flux(sensors)
            results.append(result)
        
        # Perform consensus (simple majority voting)
        valid_fluxes = [r.flux_value for r in results if r.flux_value > 0]
        
        if valid_fluxes:
            # Find node with highest flux as leader
            max_flux = max(valid_fluxes)
            leader_idx = next(i for i, r in enumerate(results) if r.flux_value == max_flux)
            
            self.consensus_flux = max_flux
            self.consensus_correction = results[leader_idx].correction_vector
            self.consensus_valid = True
            self.leader_node = leader_idx
            
            # Broadcast consensus to all nodes
            for node in self.nodes:
                node.mesh_flux_in = self.consensus_flux
                node.mesh_correction_in = self.consensus_correction
                node.mesh_valid_in = True
        else:
            self.consensus_valid = False
        
        self.communication_rounds += 1
        self.consensus_history.append({
            'round': self.communication_rounds,
            'consensus_flux': self.consensus_flux,
            'leader_node': self.leader_node,
            'valid': self.consensus_valid
        })
        
        return results
    
    def get_mesh_statistics(self) -> Dict:
        """Get mesh-wide statistics"""
        node_stats = [node.get_statistics() for node in self.nodes]
        
        return {
            'num_nodes': self.num_nodes,
            'communication_rounds': self.communication_rounds,
            'consensus_rate': sum(1 for h in self.consensus_history if h['valid']) / len(self.consensus_history) * 100 if self.consensus_history else 0,
            'average_consensus_flux': np.mean([h['consensus_flux'] for h in self.consensus_history]) if self.consensus_history else 0,
            'node_statistics': node_stats
        }


def generate_sensor_data(duration: float, 
                        base_values: List[float] = None,
                        noise_level: float = 0.1,
                        disturbance_events: List[Tuple[float, float, str]] = None) -> List[SensorData]:
    """
    Generate realistic sensor data for simulation
    
    Args:
        duration: Simulation duration in seconds
        base_values: Base sensor values [voltage_drift, packet_loss, temp_variance, phase_jitter]
        noise_level: Amount of random noise to add
        disturbance_events: List of (time, intensity, type) disturbance events
        
    Returns:
        List of SensorData objects
    """
    if base_values is None:
        base_values = [0.05, 0.02, 10.0, 0.1]  # Stable baseline
    
    if disturbance_events is None:
        disturbance_events = [
            (duration * 0.3, 0.5, "voltage_spike"),
            (duration * 0.6, 0.3, "packet_loss"),
            (duration * 0.8, 0.4, "temp_rise")
        ]
    
    dt = 0.01  # 100Hz sampling rate
    timesteps = int(duration / dt)
    data = []
    
    for t in range(timesteps):
        time_val = t * dt
        
        # Base values with noise
        voltage_drift = base_values[0] + np.random.normal(0, noise_level * 0.1)
        packet_loss = base_values[1] + np.random.normal(0, noise_level * 0.05)
        temp_variance = base_values[2] + np.random.normal(0, noise_level * 2.0)
        phase_jitter = base_values[3] + np.random.normal(0, noise_level * 0.05)
        
        # Apply disturbance events
        for event_time, intensity, event_type in disturbance_events:
            if abs(time_val - event_time) < 0.1:  # 100ms event window
                if event_type == "voltage_spike":
                    voltage_drift += intensity
                elif event_type == "packet_loss":
                    packet_loss += intensity
                elif event_type == "temp_rise":
                    temp_variance += intensity * 20.0
        
        # Clamp values to valid ranges
        voltage_drift = max(0.0, min(1.0, voltage_drift))
        packet_loss = max(0.0, min(1.0, packet_loss))
        temp_variance = max(0.0, min(100.0, temp_variance))
        phase_jitter = max(0.0, min(1.0, phase_jitter))
        
        data.append(SensorData(
            voltage_drift=voltage_drift,
            packet_loss=packet_loss,
            temp_variance=temp_variance,
            phase_jitter=phase_jitter,
            timestamp=time_val
        ))
    
    return data


def run_simulation(num_nodes: int = 4, 
                  duration: float = 60.0,
                  enable_learning: bool = True,
                  enable_mesh: bool = True,
                  save_results: bool = True) -> Dict:
    """
    Run complete syntropy core simulation
    
    Args:
        num_nodes: Number of nodes in mesh
        duration: Simulation duration in seconds
        enable_learning: Enable adaptive learning
        enable_mesh: Enable mesh communication
        save_results: Save results to file
        
    Returns:
        Simulation results dictionary
    """
    print(f"🌟 Starting Syntropy Core Simulation")
    print(f"   Nodes: {num_nodes}")
    print(f"   Duration: {duration}s")
    print(f"   Learning: {'Enabled' if enable_learning else 'Disabled'}")
    print(f"   Mesh: {'Enabled' if enable_mesh else 'Disabled'}")
    print("=" * 50)
    
    # Initialize mesh
    mesh = SyntropyMesh(num_nodes)
    
    # Generate sensor data for each node
    sensor_data_sets = []
    for i in range(num_nodes):
        # Add some variation between nodes
        base_variation = [0.01 * i, 0.005 * i, 2.0 * i, 0.01 * i]
        base_values = [0.05 + base_variation[0], 0.02 + base_variation[1], 
                      10.0 + base_variation[2], 0.1 + base_variation[3]]
        
        sensor_data = generate_sensor_data(duration, base_values)
        sensor_data_sets.append(sensor_data)
    
    # Run simulation
    start_time = time.time()
    all_results = []
    
    for timestep in range(len(sensor_data_sets[0])):
        # Get current sensor data for all nodes
        current_sensors = [sensor_data_sets[i][timestep] for i in range(num_nodes)]
        
        # Update mesh
        results = mesh.update_mesh(current_sensors)
        all_results.append(results)
        
        # Progress indicator
        if timestep % 100 == 0:
            progress = (timestep / len(sensor_data_sets[0])) * 100
            print(f"   Progress: {progress:.1f}%", end='\r')
    
    simulation_time = time.time() - start_time
    print(f"\n✅ Simulation completed in {simulation_time:.2f}s")
    
    # Analyze results
    print("\n📊 Results Analysis:")
    print("-" * 30)
    
    mesh_stats = mesh.get_mesh_statistics()
    print(f"Communication Rounds: {mesh_stats['communication_rounds']}")
    print(f"Consensus Rate: {mesh_stats['consensus_rate']:.1f}%")
    print(f"Average Consensus Flux: {mesh_stats['average_consensus_flux']:.3f}")
    
    for i, node_stats in enumerate(mesh_stats['node_statistics']):
        print(f"\nNode {i}:")
        print(f"  Stability Rate: {node_stats['stability_rate']:.1f}%")
        print(f"  Average Flux: {node_stats['average_flux']:.3f}")
        print(f"  Learning Episodes: {node_stats['learning_episodes']}")
    
    # Save results
    if save_results:
        results_data = {
            'simulation_config': {
                'num_nodes': num_nodes,
                'duration': duration,
                'enable_learning': enable_learning,
                'enable_mesh': enable_mesh,
                'simulation_time': simulation_time
            },
            'mesh_statistics': mesh_stats,
            'node_results': all_results
        }
        
        filename = f"syntropy_sim_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
    
    return {
        'mesh_statistics': mesh_stats,
        'simulation_time': simulation_time,
        'all_results': all_results
    }


def main():
    """Main simulation function"""
    parser = argparse.ArgumentParser(description='Syntropy Core Simulation')
    parser.add_argument('--nodes', type=int, default=4, help='Number of nodes')
    parser.add_argument('--duration', type=float, default=60.0, help='Duration in seconds')
    parser.add_argument('--no-learning', action='store_true', help='Disable learning')
    parser.add_argument('--no-mesh', action='store_true', help='Disable mesh communication')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results')
    
    args = parser.parse_args()
    
    # Run simulation
    results = run_simulation(
        num_nodes=args.nodes,
        duration=args.duration,
        enable_learning=not args.no_learning,
        enable_mesh=not args.no_mesh,
        save_results=not args.no_save
    )
    
    print(f"\n🎯 Simulation Summary:")
    print(f"   Total Time: {results['simulation_time']:.2f}s")
    print(f"   Mesh Consensus Rate: {results['mesh_statistics']['consensus_rate']:.1f}%")
    print(f"   Average Stability: {np.mean([node['stability_rate'] for node in results['mesh_statistics']['node_statistics']]):.1f}%")


if __name__ == "__main__":
    main()