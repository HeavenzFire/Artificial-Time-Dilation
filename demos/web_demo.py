#!/usr/bin/env python3
"""
Web demo for Artificial Time Dilation in RL.

This script creates an interactive web interface using Streamlit to demonstrate
time dilation effects in reinforcement learning environments.
"""

import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.time_dilation import TimeDilationSimulator, DilationFactor
from core.rl_environment import RLDilatedEnvironment, EnvironmentConfig
from visualization.charts import DScalingChart, RewardCurveChart, PerformanceDashboard


def create_mock_environment():
    """Create a mock environment for demonstration."""
    class MockEnvironment:
        def __init__(self, dilation_factor: float = 1.0):
            self.dilation_factor = dilation_factor
            self.step_count = 0
            self.episode_count = 0
            self.episode_rewards = []
            self.total_rewards = 0.0
        
        def reset(self):
            self.episode_count += 1
            self.step_count = 0
            self.episode_rewards = []
            return np.random.random(4), {}
        
        def step(self, action):
            self.step_count += 1
            reward = np.random.normal(1.0, 0.5) * self.dilation_factor
            self.episode_rewards.append(reward)
            self.total_rewards += reward
            
            # Simulate episode termination
            terminated = self.step_count >= 200 or np.random.random() < 0.01
            truncated = self.step_count >= 200
            
            return (np.random.random(4), reward, terminated, truncated, 
                   {"dilation_factor": self.dilation_factor})
        
        def set_dilation_factor(self, factor):
            self.dilation_factor = factor
        
        def get_dilation_factor(self):
            return self.dilation_factor
        
        def get_current_episode_metrics(self):
            return {
                "episode_number": self.episode_count,
                "step_count": self.step_count,
                "episode_reward": sum(self.episode_rewards),
                "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            }
    
    return MockEnvironment()


def simple_agent(obs: np.ndarray) -> int:
    """Simple agent that takes random actions."""
    return np.random.randint(0, 2)


def run_simulation(
    dilation_factor: float,
    num_episodes: int,
    max_steps: int
) -> Dict[str, Any]:
    """Run a simulation with the given parameters."""
    env = create_mock_environment()
    env.set_dilation_factor(dilation_factor)
    
    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "cumulative_rewards": [],
        "time_points": [],
        "dilation_factor": dilation_factor
    }
    
    cumulative_reward = 0
    total_time = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = simple_agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        cumulative_reward += episode_reward
        total_time += episode_length * 0.01  # Simulate time per step
        
        results["episode_rewards"].append(episode_reward)
        results["episode_lengths"].append(episode_length)
        results["cumulative_rewards"].append(cumulative_reward)
        results["time_points"].append(total_time)
    
    return results


def create_d_scaling_chart() -> go.Figure:
    """Create the D-scaling chart."""
    time_points = np.linspace(0, 72, 100)
    dilation_factors = [1, 10, 100, 1000, 10000]
    
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
        width=800,
        height=500,
        font=dict(size=12),
        legend=dict(font=dict(size=10)),
        hovermode='x unified'
    )
    
    return fig


def create_reward_curve_chart(results: Dict[str, Any]) -> go.Figure:
    """Create a reward curve chart from simulation results."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add reward curve
    fig.add_trace(
        go.Scatter(
            x=results["time_points"],
            y=results["cumulative_rewards"],
            mode='lines',
            name="Cumulative Reward",
            line=dict(color='blue', width=3)
        ),
        secondary_y=False
    )
    
    # Add episode rewards
    fig.add_trace(
        go.Scatter(
            x=results["time_points"],
            y=results["episode_rewards"],
            mode='markers',
            name="Episode Rewards",
            marker=dict(color='red', size=6)
        ),
        secondary_y=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Simulation Time (Hours)")
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    
    # Update layout
    fig.update_layout(
        title=f"RL Agent Performance with {results['dilation_factor']}x Time Dilation",
        width=800,
        height=500,
        font=dict(size=12),
        legend=dict(font=dict(size=10))
    )
    
    return fig


def create_performance_comparison(results_list: List[Dict[str, Any]]) -> go.Figure:
    """Create a performance comparison chart."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for i, results in enumerate(results_list):
        fig.add_trace(go.Scatter(
            x=results["time_points"],
            y=results["cumulative_rewards"],
            mode='lines',
            name=f"{results['dilation_factor']}x Dilation",
            line=dict(color=colors[i % len(colors)], width=3)
        ))
    
    fig.update_layout(
        title="Performance Comparison Across Dilation Factors",
        xaxis_title="Simulation Time (Hours)",
        yaxis_title="Cumulative Reward",
        width=800,
        height=500,
        font=dict(size=12),
        legend=dict(font=dict(size=10))
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Artificial Time Dilation for RL",
        page_icon="⏰",
        layout="wide"
    )
    
    st.title("⏰ Artificial Time Dilation for Reinforcement Learning")
    st.markdown("""
    This demo showcases how artificial time dilation can accelerate RL training
    by compressing years of simulation into real-world hours.
    """)
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    dilation_factors = st.sidebar.multiselect(
        "Dilation Factors",
        options=[1, 10, 100, 1000, 10000],
        default=[1, 100, 1000],
        help="Select dilation factors to compare"
    )
    
    num_episodes = st.sidebar.slider(
        "Number of Episodes",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of episodes to run per simulation"
    )
    
    max_steps = st.sidebar.slider(
        "Max Steps per Episode",
        min_value=50,
        max_value=500,
        value=200,
        help="Maximum steps per episode"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Theory", "D-Scaling Chart", "Simulation", "Comparison"])
    
    with tab1:
        st.header("Time Dilation Theory")
        st.markdown("""
        ### What is Artificial Time Dilation?
        
        Artificial time dilation applies relativistic concepts to digital environments,
        allowing RL agents to experience accelerated time through simulation speed scaling.
        
        **Key Concepts:**
        - **Dilation Factor (D)**: Ratio of simulated time to real-world time
        - **Time Compression**: Years of training compressed into hours
        - **Velocity Scaling**: Simulation speed analogous to relativistic velocity
        
        **Benefits:**
        - Accelerated RL training
        - Rapid algorithm exploration
        - Resource optimization
        - Breakthrough discovery potential
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Real-Time (1x)")
            st.metric("Simulation Speed", "1x", "Real-time")
            st.metric("Time Compression", "1:1", "No compression")
        
        with col2:
            st.subheader("Dilated (1000x)")
            st.metric("Simulation Speed", "1000x", "+99900%")
            st.metric("Time Compression", "1000:1", "1000x faster")
    
    with tab2:
        st.header("D-Scaling Visualization")
        st.markdown("""
        This chart shows how different dilation factors scale simulated time
        compared to real-world time.
        """)
        
        d_chart = create_d_scaling_chart()
        st.plotly_chart(d_chart, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - Higher dilation factors enable more simulated time in less real time
        - 1000x dilation allows 1000 years of simulation in 1 real year
        - This enables rapid exploration of RL algorithms and environments
        """)
    
    with tab3:
        st.header("Interactive Simulation")
        st.markdown("""
        Run simulations with different dilation factors to see the effects
        on RL agent performance.
        """)
        
        if st.button("Run Simulation", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            
            for i, factor in enumerate(dilation_factors):
                status_text.text(f"Running simulation with {factor}x dilation...")
                progress_bar.progress((i + 1) / len(dilation_factors))
                
                results = run_simulation(factor, num_episodes, max_steps)
                all_results.append(results)
                
                time.sleep(0.5)  # Simulate processing time
            
            status_text.text("Simulation complete!")
            progress_bar.empty()
            
            # Display results
            for i, (factor, results) in enumerate(zip(dilation_factors, all_results)):
                st.subheader(f"{factor}x Dilation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Episodes", len(results["episode_rewards"]))
                with col2:
                    st.metric("Avg Reward", f"{np.mean(results['episode_rewards']):.2f}")
                with col3:
                    st.metric("Total Reward", f"{results['cumulative_rewards'][-1]:.2f}")
                with col4:
                    st.metric("Dilation Factor", f"{factor}x")
                
                # Create reward curve
                reward_chart = create_reward_curve_chart(results)
                st.plotly_chart(reward_chart, use_container_width=True)
    
    with tab4:
        st.header("Performance Comparison")
        st.markdown("""
        Compare performance across different dilation factors to see
        the benefits of time dilation.
        """)
        
        if st.button("Run Comparison", type="primary"):
            with st.spinner("Running comparison simulations..."):
                all_results = []
                
                for factor in dilation_factors:
                    results = run_simulation(factor, num_episodes, max_steps)
                    all_results.append(results)
            
            # Create comparison chart
            comparison_chart = create_performance_comparison(all_results)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Create summary table
            st.subheader("Performance Summary")
            
            summary_data = []
            for results in all_results:
                summary_data.append({
                    "Dilation Factor": f"{results['dilation_factor']}x",
                    "Episodes": len(results["episode_rewards"]),
                    "Avg Reward": f"{np.mean(results['episode_rewards']):.2f}",
                    "Total Reward": f"{results['cumulative_rewards'][-1]:.2f}",
                    "Final Time": f"{results['time_points'][-1]:.2f}h"
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Artificial Time Dilation for RL** - A research project exploring accelerated
    reinforcement learning through simulation speed scaling.
    """)


if __name__ == "__main__":
    main()