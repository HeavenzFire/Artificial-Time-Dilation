"""
Data processing utilities for time dilation experiments.

This module provides tools for processing, storing, and analyzing
experimental data from time dilation simulations.
"""

import numpy as np
import pandas as pd
import json
import h5py
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime
import csv

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processor for time dilation experiment data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory to store processed data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_data = {}
        self.metadata = {}
    
    def process_episode_data(
        self,
        episode_data: List[Dict[str, Any]],
        dilation_factor: float
    ) -> Dict[str, Any]:
        """
        Process episode data and extract key metrics.
        
        Args:
            episode_data: List of episode dictionaries
            dilation_factor: Dilation factor used
            
        Returns:
            Processed episode metrics
        """
        if not episode_data:
            return {}
        
        # Extract metrics
        rewards = [ep.get('reward', 0) for ep in episode_data]
        lengths = [ep.get('length', 0) for ep in episode_data]
        times = [ep.get('time', 0) for ep in episode_data]
        simulated_times = [ep.get('simulated_time', 0) for ep in episode_data]
        
        # Calculate statistics
        metrics = {
            'dilation_factor': dilation_factor,
            'num_episodes': len(episode_data),
            'total_reward': sum(rewards),
            'average_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'average_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'total_time': sum(times),
            'average_time': np.mean(times),
            'total_simulated_time': sum(simulated_times),
            'average_simulated_time': np.mean(simulated_times),
            'time_compression_ratio': sum(simulated_times) / sum(times) if sum(times) > 0 else 0,
            'efficiency': sum(rewards) / sum(times) if sum(times) > 0 else 0
        }
        
        return metrics
    
    def process_experiment_data(
        self,
        experiment_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Process complete experiment data.
        
        Args:
            experiment_data: Dictionary mapping dilation factors to episode data
            
        Returns:
            Processed experiment metrics
        """
        processed_data = {}
        
        for dilation_factor, episode_data in experiment_data.items():
            factor = float(dilation_factor)
            processed_data[dilation_factor] = self.process_episode_data(episode_data, factor)
        
        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(processed_data)
        
        return {
            'experiment_data': processed_data,
            'comparison_metrics': comparison_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_comparison_metrics(
        self,
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics for comparing different dilation factors."""
        if not processed_data:
            return {}
        
        # Extract data for comparison
        dilation_factors = []
        average_rewards = []
        time_compression_ratios = []
        efficiencies = []
        
        for factor, data in processed_data.items():
            dilation_factors.append(float(factor))
            average_rewards.append(data['average_reward'])
            time_compression_ratios.append(data['time_compression_ratio'])
            efficiencies.append(data['efficiency'])
        
        # Calculate comparison statistics
        best_reward_idx = np.argmax(average_rewards)
        best_efficiency_idx = np.argmax(efficiencies)
        
        return {
            'dilation_factors': dilation_factors,
            'average_rewards': average_rewards,
            'time_compression_ratios': time_compression_ratios,
            'efficiencies': efficiencies,
            'best_reward_factor': dilation_factors[best_reward_idx],
            'best_efficiency_factor': dilation_factors[best_efficiency_idx],
            'reward_improvement': (max(average_rewards) - min(average_rewards)) / min(average_rewards) * 100,
            'efficiency_improvement': (max(efficiencies) - min(efficiencies)) / min(efficiencies) * 100
        }
    
    def save_data(
        self,
        data: Dict[str, Any],
        filename: str,
        format: str = "json"
    ) -> str:
        """
        Save processed data to file.
        
        Args:
            data: Data to save
            filename: Filename (without extension)
            format: File format ("json", "pickle", "hdf5", "csv")
            
        Returns:
            Path to saved file
        """
        file_path = self.data_dir / f"{filename}.{format}"
        
        try:
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            elif format == "hdf5":
                with h5py.File(file_path, 'w') as f:
                    self._save_to_hdf5(f, data)
            elif format == "csv":
                self._save_to_csv(file_path, data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data saved to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def _save_to_hdf5(self, h5file: h5py.File, data: Dict[str, Any]) -> None:
        """Save data to HDF5 format."""
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                h5file.create_dataset(key, data=value)
            elif isinstance(value, dict):
                group = h5file.create_group(key)
                self._save_to_hdf5(group, value)
            else:
                h5file.attrs[key] = value
    
    def _save_to_csv(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data to CSV format."""
        # Flatten nested data for CSV
        flattened_data = self._flatten_dict(data)
        
        # Convert to DataFrame
        df = pd.DataFrame([flattened_data])
        df.to_csv(file_path, index=False)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def load_data(self, filename: str, format: str = "json") -> Dict[str, Any]:
        """
        Load data from file.
        
        Args:
            filename: Filename (without extension)
            format: File format ("json", "pickle", "hdf5")
            
        Returns:
            Loaded data
        """
        file_path = self.data_dir / f"{filename}.{format}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if format == "json":
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif format == "pickle":
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif format == "hdf5":
                with h5py.File(file_path, 'r') as f:
                    return self._load_from_hdf5(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_from_hdf5(self, h5file: h5py.File) -> Dict[str, Any]:
        """Load data from HDF5 format."""
        data = {}
        
        for key in h5file.keys():
            if isinstance(h5file[key], h5py.Group):
                data[key] = self._load_from_hdf5(h5file[key])
            else:
                data[key] = h5file[key][:]
        
        # Add attributes
        for key, value in h5file.attrs.items():
            data[key] = value
        
        return data


class ExperimentLogger:
    """Logger for time dilation experiments."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the experiment logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.log_data = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
    
    def start_experiment(
        self,
        experiment_name: str,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Start a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            parameters: Experiment parameters
        """
        self.current_experiment = {
            'name': experiment_name,
            'start_time': datetime.now(),
            'parameters': parameters,
            'episodes': []
        }
        
        logger.info(f"Started experiment: {experiment_name}")
        logger.info(f"Parameters: {parameters}")
    
    def log_episode(
        self,
        episode_number: int,
        episode_data: Dict[str, Any]
    ) -> None:
        """
        Log episode data.
        
        Args:
            episode_number: Episode number
            episode_data: Episode data dictionary
        """
        if self.current_experiment is None:
            logger.warning("No active experiment. Call start_experiment first.")
            return
        
        episode_log = {
            'episode_number': episode_number,
            'timestamp': datetime.now(),
            'data': episode_data
        }
        
        self.current_experiment['episodes'].append(episode_log)
        self.log_data.append(episode_log)
        
        logger.info(f"Episode {episode_number}: Reward={episode_data.get('reward', 0):.2f}, "
                   f"Length={episode_data.get('length', 0)}, "
                   f"Dilation={episode_data.get('dilation_factor', 1):.1f}x")
    
    def end_experiment(self) -> Dict[str, Any]:
        """
        End the current experiment.
        
        Returns:
            Complete experiment data
        """
        if self.current_experiment is None:
            logger.warning("No active experiment to end.")
            return {}
        
        self.current_experiment['end_time'] = datetime.now()
        self.current_experiment['duration'] = (
            self.current_experiment['end_time'] - self.current_experiment['start_time']
        ).total_seconds()
        
        # Calculate summary statistics
        episodes = self.current_experiment['episodes']
        if episodes:
            rewards = [ep['data'].get('reward', 0) for ep in episodes]
            lengths = [ep['data'].get('length', 0) for ep in episodes]
            
            self.current_experiment['summary'] = {
                'total_episodes': len(episodes),
                'total_reward': sum(rewards),
                'average_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'average_length': np.mean(lengths),
                'std_length': np.std(lengths)
            }
        
        logger.info(f"Ended experiment: {self.current_experiment['name']}")
        logger.info(f"Duration: {self.current_experiment['duration']:.2f} seconds")
        
        # Save experiment data
        experiment_data = self.current_experiment.copy()
        self.current_experiment = None
        
        return experiment_data
    
    def save_experiment_log(self, filename: Optional[str] = None) -> str:
        """
        Save experiment log to file.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{timestamp}.json"
        
        file_path = self.log_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(self.log_data, f, indent=2, default=str)
        
        logger.info(f"Experiment log saved to {file_path}")
        return str(file_path)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all logged experiments."""
        if not self.log_data:
            return {}
        
        # Group by experiment
        experiments = {}
        for log_entry in self.log_data:
            # This is a simplified grouping - in practice, you'd need
            # to track experiment boundaries more carefully
            pass
        
        return experiments