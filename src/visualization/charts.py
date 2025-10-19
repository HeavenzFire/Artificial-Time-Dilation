"""
Chart generation and visualization components for time dilation analysis.

This module provides various chart types for visualizing time dilation effects,
RL performance, and simulation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    width: int = 800
    height: int = 600
    title_font_size: int = 16
    label_font_size: int = 12
    legend_font_size: int = 10
    color_palette: str = "viridis"
    style: str = "whitegrid"
    dpi: int = 300


class DScalingChart:
    """
    Chart for visualizing time dilation scaling effects.
    
    This class generates charts showing how simulation speed (dilation factor)
    scales with effective training time, demonstrating the time compression
    achieved through artificial time dilation.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize the D-scaling chart generator.
        
        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self.setup_style()
    
    def setup_style(self) -> None:
        """Setup matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        sns.set_palette(self.config.color_palette)
        sns.set_style(self.config.style)
    
    def generate_d_scaling_data(
        self,
        time_points: np.ndarray,
        dilation_factors: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        Generate data for D-scaling chart.
        
        Args:
            time_points: Real-world time points in hours
            dilation_factors: List of dilation factors to plot
            
        Returns:
            Dictionary mapping dilation factor names to simulated time arrays
        """
        data = {}
        
        for factor in dilation_factors:
            simulated_times = time_points * factor
            data[f"{factor}x"] = simulated_times
        
        return data
    
    def plot_matplotlib(
        self,
        time_points: np.ndarray,
        dilation_factors: List[float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate D-scaling chart using matplotlib.
        
        Args:
            time_points: Real-world time points in hours
            dilation_factors: List of dilation factors to plot
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(dilation_factors)))
        
        for i, factor in enumerate(dilation_factors):
            simulated_times = time_points * factor
            ax.plot(
                time_points, 
                simulated_times, 
                label=f"{factor}x Speed",
                color=colors[i],
                linewidth=2,
                marker='o',
                markersize=4
            )
        
        ax.set_xlabel("Real-World Time (Hours)", fontsize=self.config.label_font_size)
        ax.set_ylabel("Simulated Time (Years)", fontsize=self.config.label_font_size)
        ax.set_title("(D)-Scaling: Simulated vs. Real-World Time", fontsize=self.config.title_font_size)
        ax.legend(fontsize=self.config.legend_font_size)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for time compression
        max_time = max(time_points)
        max_simulated = max(time_points) * max(dilation_factors)
        ax.annotate(
            f"Time Compression: {max_simulated/max_time:.1f}x",
            xy=(max_time*0.7, max_simulated*0.3),
            fontsize=self.config.label_font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_plotly(
        self,
        time_points: np.ndarray,
        dilation_factors: List[float],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Generate D-scaling chart using Plotly.
        
        Args:
            time_points: Real-world time points in hours
            dilation_factors: List of dilation factors to plot
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, factor in enumerate(dilation_factors):
            simulated_times = time_points * factor
            fig.add_trace(go.Scatter(
                x=time_points,
                y=simulated_times,
                mode='lines+markers',
                name=f"{factor}x Speed",
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="(D)-Scaling: Simulated vs. Real-World Time",
            xaxis_title="Real-World Time (Hours)",
            yaxis_title="Simulated Time (Years)",
            width=self.config.width,
            height=self.config.height,
            font=dict(size=self.config.label_font_size),
            legend=dict(font=dict(size=self.config.legend_font_size)),
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_chart_js_config(
        self,
        time_points: np.ndarray,
        dilation_factors: List[float]
    ) -> Dict[str, Any]:
        """
        Generate Chart.js configuration for web display.
        
        Args:
            time_points: Real-world time points in hours
            dilation_factors: List of dilation factors to plot
            
        Returns:
            Chart.js configuration dictionary
        """
        datasets = []
        colors = [
            "#FF0000", "#2196F3", "#4CAF50", "#FF9800", 
            "#9C27B0", "#F44336", "#00BCD4", "#8BC34A"
        ]
        
        for i, factor in enumerate(dilation_factors):
            simulated_times = time_points * factor
            datasets.append({
                "label": f"{factor}x Speed",
                "data": simulated_times.tolist(),
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)] + "20",
                "fill": False,
                "tension": 0.1
            })
        
        config = {
            "type": "line",
            "data": {
                "labels": time_points.tolist(),
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Simulated Time (Years)"}
                    },
                    "x": {
                        "title": {"display": True, "text": "Real-World Time (Hours)"}
                    }
                },
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "(D)-Scaling: Simulated vs. Real-World Time"}
                }
            }
        }
        
        return config


class RewardCurveChart:
    """
    Chart for visualizing RL agent performance with time dilation.
    
    This class generates charts showing how RL agent performance (cumulative reward)
    improves over time, comparing dilated simulation time to real-world time.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize the reward curve chart generator.
        
        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self.setup_style()
    
    def setup_style(self) -> None:
        """Setup matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        sns.set_palette(self.config.color_palette)
        sns.set_style(self.config.style)
    
    def plot_reward_curves(
        self,
        time_points: np.ndarray,
        rewards: np.ndarray,
        steps: np.ndarray,
        dilation_factor: float = 1.0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot reward curves with dual y-axes.
        
        Args:
            time_points: Time points in hours
            rewards: Cumulative rewards
            steps: Simulation steps
            dilation_factor: Time dilation factor
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax1 = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # Plot rewards
        color1 = 'tab:blue'
        ax1.set_xlabel("Real-World Time (Hours)", fontsize=self.config.label_font_size)
        ax1.set_ylabel("Cumulative Reward", color=color1, fontsize=self.config.label_font_size)
        line1 = ax1.plot(time_points, rewards, color=color1, linewidth=2, label="Cumulative Reward")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for steps
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel("Simulated Steps (Millions)", color=color2, fontsize=self.config.label_font_size)
        line2 = ax2.plot(time_points, steps/1e6, color=color2, linewidth=2, linestyle='--', label="Simulated Steps")
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add title and legend
        plt.title(f"RL Agent Performance in Dilated Time ({dilation_factor}x)", fontsize=self.config.title_font_size)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=self.config.legend_font_size)
        
        # Add performance metrics annotation
        final_reward = rewards[-1] if len(rewards) > 0 else 0
        final_steps = steps[-1] if len(steps) > 0 else 0
        ax1.annotate(
            f"Final Reward: {final_reward:.0f}\nFinal Steps: {final_steps/1e6:.1f}M\nDilation: {dilation_factor}x",
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            fontsize=self.config.label_font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            verticalalignment='top'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_plotly(
        self,
        time_points: np.ndarray,
        rewards: np.ndarray,
        steps: np.ndarray,
        dilation_factor: float = 1.0,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Generate reward curve chart using Plotly.
        
        Args:
            time_points: Time points in hours
            rewards: Cumulative rewards
            steps: Simulation steps
            dilation_factor: Time dilation factor
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add reward curve
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=rewards,
                mode='lines',
                name="Cumulative Reward",
                line=dict(color='blue', width=3)
            ),
            secondary_y=False
        )
        
        # Add steps curve
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=steps/1e6,
                mode='lines',
                name="Simulated Steps (M)",
                line=dict(color='orange', width=3, dash='dash')
            ),
            secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Real-World Time (Hours)")
        fig.update_yaxes(title_text="Cumulative Reward", secondary_y=False)
        fig.update_yaxes(title_text="Simulated Steps (Millions)", secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title=f"RL Agent Performance in Dilated Time ({dilation_factor}x)",
            width=self.config.width,
            height=self.config.height,
            font=dict(size=self.config.label_font_size),
            legend=dict(font=dict(size=self.config.legend_font_size))
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_chart_js_config(
        self,
        time_points: np.ndarray,
        rewards: np.ndarray,
        steps: np.ndarray,
        dilation_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate Chart.js configuration for web display.
        
        Args:
            time_points: Time points in hours
            rewards: Cumulative rewards
            steps: Simulation steps
            dilation_factor: Time dilation factor
            
        Returns:
            Chart.js configuration dictionary
        """
        config = {
            "type": "line",
            "data": {
                "labels": time_points.tolist(),
                "datasets": [
                    {
                        "label": "Cumulative Reward",
                        "data": rewards.tolist(),
                        "borderColor": "#2196F3",
                        "backgroundColor": "rgba(33, 150, 243, 0.2)",
                        "fill": True,
                        "yAxisID": "y"
                    },
                    {
                        "label": "Simulated Steps (Millions)",
                        "data": (steps/1e6).tolist(),
                        "borderColor": "#FF9800",
                        "backgroundColor": "rgba(255, 152, 0, 0.2)",
                        "fill": False,
                        "yAxisID": "y2"
                    }
                ]
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Cumulative Reward"},
                        "position": "left"
                    },
                    "y2": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Simulated Steps (M)"},
                        "position": "right"
                    },
                    "x": {
                        "title": {"display": True, "text": "Real-World Time (Hours)"}
                    }
                },
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": f"RL Agent Performance in Dilated Time ({dilation_factor}x)"}
                }
            }
        }
        
        return config


class PerformanceDashboard:
    """
    Comprehensive performance dashboard for time dilation analysis.
    
    This class generates multi-panel dashboards showing various aspects
    of time dilation performance and RL training metrics.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize the performance dashboard.
        
        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self.setup_style()
    
    def setup_style(self) -> None:
        """Setup matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        sns.set_palette(self.config.color_palette)
        sns.set_style(self.config.style)
    
    def create_dashboard(
        self,
        performance_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            performance_data: Dictionary containing performance metrics
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure with subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Time Dilation Performance Dashboard", fontsize=20, fontweight='bold')
        
        # Plot 1: Dilation factor over time
        if 'dilation_factors' in performance_data:
            ax1 = axes[0, 0]
            ax1.plot(performance_data['dilation_factors'], linewidth=2)
            ax1.set_title("Dilation Factor Over Time")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Dilation Factor")
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode rewards
        if 'episode_rewards' in performance_data:
            ax2 = axes[0, 1]
            ax2.plot(performance_data['episode_rewards'], linewidth=2, color='green')
            ax2.set_title("Episode Rewards")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Reward")
            ax2.grid(True, alpha=0.3)
            
            # Add moving average
            if len(performance_data['episode_rewards']) > 10:
                window_size = min(50, len(performance_data['episode_rewards']) // 5)
                moving_avg = pd.Series(performance_data['episode_rewards']).rolling(window=window_size).mean()
                ax2.plot(moving_avg, linewidth=2, color='red', alpha=0.7, label=f'Moving Avg ({window_size})')
                ax2.legend()
        
        # Plot 3: Training time vs episodes
        if 'training_time' in performance_data:
            ax3 = axes[1, 0]
            ax3.plot(performance_data['training_time'], linewidth=2, color='purple')
            ax3.set_title("Training Time per Episode")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Time (seconds)")
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison
        if 'episode_lengths' in performance_data:
            ax4 = axes[1, 1]
            ax4.hist(performance_data['episode_lengths'], bins=20, alpha=0.7, color='orange')
            ax4.set_title("Episode Length Distribution")
            ax4.set_xlabel("Episode Length")
            ax4.set_ylabel("Frequency")
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def create_plotly_dashboard(
        self,
        performance_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a Plotly dashboard.
        
        Args:
            performance_data: Dictionary containing performance metrics
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Dilation Factor Over Time", "Episode Rewards", 
                          "Training Time per Episode", "Episode Length Distribution"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Dilation factor over time
        if 'dilation_factors' in performance_data:
            fig.add_trace(
                go.Scatter(
                    y=performance_data['dilation_factors'],
                    mode='lines',
                    name="Dilation Factor",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Plot 2: Episode rewards
        if 'episode_rewards' in performance_data:
            fig.add_trace(
                go.Scatter(
                    y=performance_data['episode_rewards'],
                    mode='lines',
                    name="Episode Rewards",
                    line=dict(width=2, color='green')
                ),
                row=1, col=2
            )
        
        # Plot 3: Training time
        if 'training_time' in performance_data:
            fig.add_trace(
                go.Scatter(
                    y=performance_data['training_time'],
                    mode='lines',
                    name="Training Time",
                    line=dict(width=2, color='purple')
                ),
                row=2, col=1
            )
        
        # Plot 4: Episode length distribution
        if 'episode_lengths' in performance_data:
            fig.add_trace(
                go.Histogram(
                    x=performance_data['episode_lengths'],
                    name="Episode Lengths",
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Time Dilation Performance Dashboard",
            width=self.config.width * 1.5,
            height=self.config.height * 1.5,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig