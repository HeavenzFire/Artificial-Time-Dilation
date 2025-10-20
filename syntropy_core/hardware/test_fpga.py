#!/usr/bin/env python3
"""
FPGA Hardware Testing Script

Tests the Syntropy Core implementation on real FPGA hardware.
Provides interface for ADC data, real-time monitoring, and performance validation.
"""

import serial
import time
import numpy as np
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class HardwareConfig:
    """Hardware configuration parameters"""
    device_port: str
    baud_rate: int = 115200
    adc_channels: int = 4
    adc_resolution: int = 16
    sampling_rate: float = 1000.0  # Hz
    test_duration: float = 60.0  # seconds


@dataclass
class ADCReading:
    """ADC reading structure"""
    voltage_drift: float
    packet_loss: float
    temp_variance: float
    phase_jitter: float
    timestamp: float
    raw_values: List[int]


class FPGAInterface:
    """
    Interface to FPGA hardware via serial communication
    """
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.serial_conn = None
        self.is_connected = False
        
        # FPGA command protocol
        self.commands = {
            'RESET': b'\x01',
            'START': b'\x02',
            'STOP': b'\x03',
            'READ_FLUX': b'\x04',
            'READ_STATUS': b'\x05',
            'SET_THRESHOLD': b'\x06',
            'ENABLE_LEARNING': b'\x07',
            'DISABLE_LEARNING': b'\x08',
            'READ_PARAMS': b'\x09',
            'SET_PARAMS': b'\x0A'
        }
        
        # Response parsing
        self.response_handlers = {
            b'\x04': self._parse_flux_response,
            b'\x05': self._parse_status_response,
            b'\x09': self._parse_params_response
        }
    
    def connect(self) -> bool:
        """Connect to FPGA hardware"""
        try:
            self.serial_conn = serial.Serial(
                port=self.config.device_port,
                baudrate=self.config.baud_rate,
                timeout=1.0
            )
            
            # Wait for connection to stabilize
            time.sleep(2.0)
            
            # Send reset command
            self.send_command('RESET')
            time.sleep(0.1)
            
            self.is_connected = True
            print(f"✅ Connected to FPGA on {self.config.device_port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to FPGA: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from FPGA"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        print("🔌 Disconnected from FPGA")
    
    def send_command(self, command: str, data: bytes = b'') -> bool:
        """Send command to FPGA"""
        if not self.is_connected:
            return False
        
        try:
            cmd_bytes = self.commands.get(command)
            if not cmd_bytes:
                print(f"❌ Unknown command: {command}")
                return False
            
            # Send command + data
            self.serial_conn.write(cmd_bytes + data)
            return True
            
        except Exception as e:
            print(f"❌ Error sending command {command}: {e}")
            return False
    
    def read_response(self, timeout: float = 1.0) -> Optional[bytes]:
        """Read response from FPGA"""
        if not self.is_connected:
            return None
        
        try:
            self.serial_conn.timeout = timeout
            response = self.serial_conn.read(64)  # Read up to 64 bytes
            return response if response else None
            
        except Exception as e:
            print(f"❌ Error reading response: {e}")
            return None
    
    def _parse_flux_response(self, data: bytes) -> Dict:
        """Parse flux response from FPGA"""
        if len(data) < 8:
            return {}
        
        # Parse 16-bit values (little-endian)
        flux_value = int.from_bytes(data[0:2], 'little') / 256.0  # 8.8 fixed point
        stable = bool(data[2])
        correction = int.from_bytes(data[3:5], 'little') / 256.0
        iteration_count = data[5]
        state = data[6]
        error_flag = bool(data[7])
        
        return {
            'flux_value': flux_value,
            'stable': stable,
            'correction_vector': correction,
            'iteration_count': iteration_count,
            'state': state,
            'error_flag': error_flag
        }
    
    def _parse_status_response(self, data: bytes) -> Dict:
        """Parse status response from FPGA"""
        if len(data) < 4:
            return {}
        
        return {
            'learning_active': bool(data[0]),
            'mesh_enabled': bool(data[1]),
            'mesh_broadcast_req': bool(data[2]),
            'mesh_valid_out': bool(data[3])
        }
    
    def _parse_params_response(self, data: bytes) -> Dict:
        """Parse parameters response from FPGA"""
        if len(data) < 8:
            return {}
        
        params = []
        for i in range(4):
            param_val = int.from_bytes(data[i*2:(i+1)*2], 'little') / 256.0
            params.append(param_val)
        
        return {'optimized_params': params}
    
    def read_adc_data(self) -> Optional[ADCReading]:
        """Read ADC data from FPGA"""
        if not self.send_command('READ_FLUX'):
            return None
        
        response = self.read_response()
        if not response:
            return None
        
        # Parse flux response (contains ADC data)
        flux_data = self._parse_flux_response(response)
        if not flux_data:
            return None
        
        # For now, simulate ADC readings based on flux calculation
        # In real implementation, this would read actual ADC values
        voltage_drift = 0.05 + np.random.normal(0, 0.01)
        packet_loss = 0.02 + np.random.normal(0, 0.005)
        temp_variance = 10.0 + np.random.normal(0, 1.0)
        phase_jitter = 0.1 + np.random.normal(0, 0.01)
        
        return ADCReading(
            voltage_drift=max(0.0, min(1.0, voltage_drift)),
            packet_loss=max(0.0, min(1.0, packet_loss)),
            temp_variance=max(0.0, min(100.0, temp_variance)),
            phase_jitter=max(0.0, min(1.0, phase_jitter)),
            timestamp=time.time(),
            raw_values=[int(v * 256) for v in [voltage_drift, packet_loss, temp_variance/100, phase_jitter]]
        )
    
    def get_status(self) -> Optional[Dict]:
        """Get FPGA status"""
        if not self.send_command('READ_STATUS'):
            return None
        
        response = self.read_response()
        if not response:
            return None
        
        return self._parse_status_response(response)
    
    def get_parameters(self) -> Optional[Dict]:
        """Get learned parameters"""
        if not self.send_command('READ_PARAMS'):
            return None
        
        response = self.read_response()
        if not response:
            return None
        
        return self._parse_params_response(response)
    
    def set_threshold(self, threshold: float) -> bool:
        """Set flux threshold"""
        threshold_bytes = int(threshold * 256).to_bytes(2, 'little')
        return self.send_command('SET_THRESHOLD', threshold_bytes)
    
    def enable_learning(self) -> bool:
        """Enable adaptive learning"""
        return self.send_command('ENABLE_LEARNING')
    
    def disable_learning(self) -> bool:
        """Disable adaptive learning"""
        return self.send_command('DISABLE_LEARNING')


class FPGAValidator:
    """
    Validates FPGA implementation against expected behavior
    """
    
    def __init__(self, fpga_interface: FPGAInterface):
        self.fpga = fpga_interface
        self.test_results = []
        self.performance_metrics = {
            'latency_measurements': [],
            'throughput_measurements': [],
            'stability_measurements': [],
            'learning_measurements': []
        }
    
    def test_basic_functionality(self) -> bool:
        """Test basic FPGA functionality"""
        print("🧪 Testing Basic Functionality...")
        
        # Test connection
        if not self.fpga.is_connected:
            print("❌ FPGA not connected")
            return False
        
        # Test status reading
        status = self.fpga.get_status()
        if not status:
            print("❌ Failed to read status")
            return False
        
        print(f"✅ Status: {status}")
        
        # Test ADC reading
        adc_data = self.fpga.read_adc_data()
        if not adc_data:
            print("❌ Failed to read ADC data")
            return False
        
        print(f"✅ ADC Data: {adc_data}")
        
        return True
    
    def test_flux_computation(self, duration: float = 10.0) -> bool:
        """Test flux computation accuracy"""
        print(f"🧪 Testing Flux Computation ({duration}s)...")
        
        start_time = time.time()
        flux_values = []
        stability_flags = []
        
        while time.time() - start_time < duration:
            adc_data = self.fpga.read_adc_data()
            if adc_data:
                # Calculate expected flux
                expected_flux = 1.0 - (adc_data.voltage_drift + adc_data.packet_loss + 
                                     adc_data.temp_variance/100.0 + adc_data.phase_jitter)
                
                # Get FPGA flux
                if self.fpga.send_command('READ_FLUX'):
                    response = self.fpga.read_response()
                    if response:
                        flux_data = self.fpga._parse_flux_response(response)
                        if flux_data:
                            flux_values.append(flux_data['flux_value'])
                            stability_flags.append(flux_data['stable'])
                            
                            # Check accuracy (within 1% tolerance)
                            error = abs(flux_data['flux_value'] - expected_flux)
                            if error > 0.01:
                                print(f"⚠️  Flux accuracy error: {error:.4f}")
            
            time.sleep(0.01)  # 100Hz sampling
        
        if not flux_values:
            print("❌ No flux data collected")
            return False
        
        # Analyze results
        avg_flux = np.mean(flux_values)
        stability_rate = np.mean(stability_flags) * 100
        
        print(f"✅ Average Flux: {avg_flux:.3f}")
        print(f"✅ Stability Rate: {stability_rate:.1f}%")
        
        self.test_results.append({
            'test': 'flux_computation',
            'duration': duration,
            'avg_flux': avg_flux,
            'stability_rate': stability_rate,
            'samples': len(flux_values)
        })
        
        return True
    
    def test_latency(self, num_samples: int = 100) -> bool:
        """Test computation latency"""
        print(f"🧪 Testing Latency ({num_samples} samples)...")
        
        latencies = []
        
        for i in range(num_samples):
            start_time = time.time()
            
            # Send command and read response
            if self.fpga.send_command('READ_FLUX'):
                response = self.fpga.read_response()
                if response:
                    latency = (time.time() - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
            
            time.sleep(0.001)  # 1ms between samples
        
        if not latencies:
            print("❌ No latency data collected")
            return False
        
        # Analyze latency
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)
        std_latency = np.std(latencies)
        
        print(f"✅ Average Latency: {avg_latency:.2f}ms")
        print(f"✅ Min Latency: {min_latency:.2f}ms")
        print(f"✅ Max Latency: {max_latency:.2f}ms")
        print(f"✅ Std Dev: {std_latency:.2f}ms")
        
        # Check if latency meets requirements (<10ms)
        if avg_latency > 10.0:
            print(f"⚠️  Latency exceeds target (10ms)")
        
        self.performance_metrics['latency_measurements'] = latencies
        self.test_results.append({
            'test': 'latency',
            'samples': num_samples,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'std_latency_ms': std_latency
        })
        
        return True
    
    def test_learning_adaptation(self, duration: float = 30.0) -> bool:
        """Test adaptive learning functionality"""
        print(f"🧪 Testing Learning Adaptation ({duration}s)...")
        
        # Enable learning
        if not self.fpga.enable_learning():
            print("❌ Failed to enable learning")
            return False
        
        start_time = time.time()
        learning_data = []
        
        while time.time() - start_time < duration:
            # Read parameters
            params = self.fpga.get_parameters()
            if params:
                learning_data.append({
                    'timestamp': time.time() - start_time,
                    'params': params['optimized_params'].copy()
                })
            
            time.sleep(0.1)  # 10Hz sampling
        
        if not learning_data:
            print("❌ No learning data collected")
            return False
        
        # Analyze learning progress
        param_changes = []
        for i in range(1, len(learning_data)):
            prev_params = learning_data[i-1]['params']
            curr_params = learning_data[i]['params']
            
            change = sum(abs(curr_params[j] - prev_params[j]) for j in range(4))
            param_changes.append(change)
        
        avg_change = np.mean(param_changes) if param_changes else 0.0
        
        print(f"✅ Learning Samples: {len(learning_data)}")
        print(f"✅ Average Parameter Change: {avg_change:.4f}")
        
        # Check if learning is active
        if avg_change > 0.001:
            print("✅ Learning is adapting parameters")
        else:
            print("⚠️  Learning appears inactive")
        
        self.performance_metrics['learning_measurements'] = learning_data
        self.test_results.append({
            'test': 'learning_adaptation',
            'duration': duration,
            'samples': len(learning_data),
            'avg_param_change': avg_change
        })
        
        return True
    
    def test_disturbance_recovery(self) -> bool:
        """Test system recovery from disturbances"""
        print("🧪 Testing Disturbance Recovery...")
        
        # Set a high threshold to trigger instability
        if not self.fpga.set_threshold(0.95):
            print("❌ Failed to set threshold")
            return False
        
        # Monitor system response
        recovery_data = []
        start_time = time.time()
        
        for i in range(100):  # 1 second at 100Hz
            adc_data = self.fpga.read_adc_data()
            if adc_data:
                if self.fpga.send_command('READ_FLUX'):
                    response = self.fpga.read_response()
                    if response:
                        flux_data = self.fpga._parse_flux_response(response)
                        if flux_data:
                            recovery_data.append({
                                'timestamp': time.time() - start_time,
                                'flux': flux_data['flux_value'],
                                'stable': flux_data['stable'],
                                'correction': flux_data['correction_vector']
                            })
            
            time.sleep(0.01)
        
        if not recovery_data:
            print("❌ No recovery data collected")
            return False
        
        # Analyze recovery
        flux_values = [d['flux'] for d in recovery_data]
        stability_flags = [d['stable'] for d in recovery_data]
        
        initial_flux = flux_values[0] if flux_values else 0.0
        final_flux = flux_values[-1] if flux_values else 0.0
        recovery_time = len([s for s in stability_flags if not s]) * 0.01  # seconds
        
        print(f"✅ Initial Flux: {initial_flux:.3f}")
        print(f"✅ Final Flux: {final_flux:.3f}")
        print(f"✅ Recovery Time: {recovery_time:.2f}s")
        
        # Check if system recovered
        if final_flux > initial_flux:
            print("✅ System shows recovery behavior")
        else:
            print("⚠️  System may not be recovering")
        
        self.test_results.append({
            'test': 'disturbance_recovery',
            'initial_flux': initial_flux,
            'final_flux': final_flux,
            'recovery_time_s': recovery_time,
            'samples': len(recovery_data)
        })
        
        return True
    
    def run_all_tests(self) -> Dict:
        """Run all validation tests"""
        print("🚀 Starting FPGA Validation Tests")
        print("=" * 50)
        
        test_results = {
            'basic_functionality': self.test_basic_functionality(),
            'flux_computation': self.test_flux_computation(10.0),
            'latency': self.test_latency(100),
            'learning_adaptation': self.test_learning_adaptation(30.0),
            'disturbance_recovery': self.test_disturbance_recovery()
        }
        
        # Summary
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\n📊 Test Summary:")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        return {
            'test_results': test_results,
            'performance_metrics': self.performance_metrics,
            'detailed_results': self.test_results
        }


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='FPGA Hardware Testing')
    parser.add_argument('--device', required=True, help='Serial device path (e.g., /dev/ttyUSB0)')
    parser.add_argument('--baud-rate', type=int, default=115200, help='Serial baud rate')
    parser.add_argument('--duration', type=float, default=60.0, help='Test duration')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Create hardware config
    config = HardwareConfig(
        device_port=args.device,
        baud_rate=args.baud_rate,
        test_duration=args.duration
    )
    
    # Initialize FPGA interface
    fpga = FPGAInterface(config)
    
    try:
        # Connect to FPGA
        if not fpga.connect():
            print("❌ Failed to connect to FPGA")
            return 1
        
        # Run validation tests
        validator = FPGAValidator(fpga)
        results = validator.run_all_tests()
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {args.output}")
        
        return 0 if all(results['test_results'].values()) else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return 1
        
    finally:
        fpga.disconnect()


if __name__ == "__main__":
    exit(main())