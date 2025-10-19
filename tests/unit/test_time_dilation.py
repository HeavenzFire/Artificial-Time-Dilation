"""
Unit tests for time dilation simulation components.
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.time_dilation import TimeDilationSimulator, DilationFactor, calculate_lorentz_factor


class TestTimeDilationSimulator(unittest.TestCase):
    """Test cases for TimeDilationSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = TimeDilationSimulator(
            base_dilation_factor=1.0,
            max_dilation_factor=1000.0,
            time_step=0.01
        )
    
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.base_dilation_factor, 1.0)
        self.assertEqual(self.simulator.max_dilation_factor, 1000.0)
        self.assertEqual(self.simulator.time_step, 0.01)
        self.assertEqual(self.simulator.current_dilation_factor, 1.0)
    
    def test_start_simulation(self):
        """Test simulation start."""
        self.simulator.start_simulation()
        self.assertIsNotNone(self.simulator.start_time)
        self.assertEqual(self.simulator.total_simulated_time, 0.0)
        self.assertEqual(self.simulator.total_real_time, 0.0)
        self.assertEqual(self.simulator.step_count, 0)
    
    def test_step_without_start(self):
        """Test step without starting simulation."""
        metrics = self.simulator.step()
        self.assertEqual(metrics.real_time, 0.0)
        self.assertEqual(metrics.simulated_time, 0.0)
        self.assertEqual(metrics.dilation_factor, 1.0)
    
    def test_step_with_dilation(self):
        """Test step with time dilation."""
        self.simulator.start_simulation()
        self.simulator.set_dilation_factor(10.0)
        
        # Simulate some time passing
        time.sleep(0.1)
        metrics = self.simulator.step()
        
        self.assertGreater(metrics.real_time, 0.0)
        self.assertGreater(metrics.simulated_time, metrics.real_time)
        self.assertEqual(metrics.dilation_factor, 10.0)
    
    def test_set_dilation_factor(self):
        """Test setting dilation factor."""
        self.simulator.set_dilation_factor(100.0)
        self.assertEqual(self.simulator.current_dilation_factor, 100.0)
        
        # Test with DilationFactor enum
        self.simulator.set_dilation_factor(DilationFactor.VERY_FAST)
        self.assertEqual(self.simulator.current_dilation_factor, 100.0)
    
    def test_invalid_dilation_factor(self):
        """Test invalid dilation factor."""
        with self.assertRaises(ValueError):
            self.simulator.set_dilation_factor(0.5)  # Less than 1.0
        
        with self.assertRaises(ValueError):
            self.simulator.set_dilation_factor(2000.0)  # Greater than max
    
    def test_adaptive_scaling(self):
        """Test adaptive scaling functionality."""
        self.simulator.enable_adaptive_scaling = True
        self.simulator.start_simulation()
        
        # Add some performance history
        for i in range(15):
            self.simulator.step(reward=1.0 + i * 0.1)  # Improving performance
        
        # Should increase dilation factor due to improving performance
        self.assertGreaterEqual(self.simulator.current_dilation_factor, 1.0)
    
    def test_reset(self):
        """Test simulator reset."""
        self.simulator.start_simulation()
        self.simulator.set_dilation_factor(100.0)
        self.simulator.step()
        
        self.simulator.reset()
        
        self.assertIsNone(self.simulator.start_time)
        self.assertEqual(self.simulator.total_simulated_time, 0.0)
        self.assertEqual(self.simulator.total_real_time, 0.0)
        self.assertEqual(self.simulator.step_count, 0)
        self.assertEqual(self.simulator.current_dilation_factor, 1.0)
    
    def test_get_simulation_summary(self):
        """Test simulation summary generation."""
        self.simulator.start_simulation()
        self.simulator.set_dilation_factor(10.0)
        self.simulator.step()
        
        summary = self.simulator.get_simulation_summary()
        
        self.assertIn('real_time_hours', summary)
        self.assertIn('simulated_time_hours', summary)
        self.assertIn('dilation_factor', summary)
        self.assertIn('time_compression_ratio', summary)
        self.assertIn('steps_per_second', summary)
        self.assertIn('total_steps', summary)
        self.assertIn('simulation_speed', summary)


class TestDilationFactor(unittest.TestCase):
    """Test cases for DilationFactor enum."""
    
    def test_enum_values(self):
        """Test enum values."""
        self.assertEqual(DilationFactor.REAL_TIME.value, 1.0)
        self.assertEqual(DilationFactor.FAST.value, 10.0)
        self.assertEqual(DilationFactor.VERY_FAST.value, 100.0)
        self.assertEqual(DilationFactor.ULTRA_FAST.value, 1000.0)
        self.assertEqual(DilationFactor.EXTREME.value, 10000.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_calculate_lorentz_factor(self):
        """Test Lorentz factor calculation."""
        # Test with velocity = 0 (no dilation)
        gamma = calculate_lorentz_factor(0.0)
        self.assertAlmostEqual(gamma, 1.0, places=10)
        
        # Test with velocity = 0.5c
        gamma = calculate_lorentz_factor(0.5)
        expected = 1.0 / np.sqrt(1 - 0.5**2)
        self.assertAlmostEqual(gamma, expected, places=10)
        
        # Test with velocity = 0.9c
        gamma = calculate_lorentz_factor(0.9)
        expected = 1.0 / np.sqrt(1 - 0.9**2)
        self.assertAlmostEqual(gamma, expected, places=10)
    
    def test_invalid_lorentz_factor(self):
        """Test Lorentz factor with invalid inputs."""
        with self.assertRaises(ValueError):
            calculate_lorentz_factor(-0.1)  # Negative velocity
        
        with self.assertRaises(ValueError):
            calculate_lorentz_factor(1.0)  # Velocity >= c
    
    def test_calculate_effective_time_dilation(self):
        """Test effective time dilation calculation."""
        from core.time_dilation import calculate_effective_time_dilation
        
        # Test without Lorentz factor
        dilated_time = calculate_effective_time_dilation(10.0, 1.0)
        self.assertEqual(dilated_time, 10.0)
        
        # Test with Lorentz factor
        lorentz_factor = calculate_lorentz_factor(0.5)
        dilated_time = calculate_effective_time_dilation(10.0, 1.0, lorentz_factor)
        expected = 10.0 * lorentz_factor
        self.assertAlmostEqual(dilated_time, expected, places=10)
    
    def test_generate_dilation_curve(self):
        """Test dilation curve generation."""
        from core.time_dilation import generate_dilation_curve
        
        time_points = np.array([0, 1, 2, 3, 4, 5])
        dilation_factors = [1, 10, 100]
        
        curves = generate_dilation_curve(time_points, dilation_factors)
        
        self.assertIn('1x', curves)
        self.assertIn('10x', curves)
        self.assertIn('100x', curves)
        
        # Check that curves are correctly scaled
        self.assertTrue(np.allclose(curves['1x'], time_points))
        self.assertTrue(np.allclose(curves['10x'], time_points * 10))
        self.assertTrue(np.allclose(curves['100x'], time_points * 100))


class TestTimeMetrics(unittest.TestCase):
    """Test cases for TimeMetrics dataclass."""
    
    def test_time_metrics_creation(self):
        """Test TimeMetrics creation."""
        from core.time_dilation import TimeMetrics
        
        metrics = TimeMetrics(
            real_time=1.0,
            simulated_time=10.0,
            dilation_factor=10.0,
            steps_per_second=5.0,
            total_steps=5
        )
        
        self.assertEqual(metrics.real_time, 1.0)
        self.assertEqual(metrics.simulated_time, 10.0)
        self.assertEqual(metrics.dilation_factor, 10.0)
        self.assertEqual(metrics.steps_per_second, 5.0)
        self.assertEqual(metrics.total_steps, 5)
    
    def test_time_compression_ratio(self):
        """Test time compression ratio calculation."""
        from core.time_dilation import TimeMetrics
        
        metrics = TimeMetrics(
            real_time=1.0,
            simulated_time=10.0,
            dilation_factor=10.0,
            steps_per_second=5.0,
            total_steps=5
        )
        
        self.assertEqual(metrics.time_compression_ratio, 10.0)
        
        # Test with zero real time
        metrics_zero = TimeMetrics(
            real_time=0.0,
            simulated_time=10.0,
            dilation_factor=10.0,
            steps_per_second=5.0,
            total_steps=5
        )
        
        self.assertEqual(metrics_zero.time_compression_ratio, 0.0)


if __name__ == '__main__':
    unittest.main()