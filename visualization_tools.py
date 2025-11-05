"""
Visualization Tools for AdaptiveGenieNetwork
===========================================

This module provides comprehensive visualization capabilities for understanding
and analyzing the dialectical optimization process, including real-time monitoring,
system dynamics analysis, and interactive exploration tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dataclasses import dataclass
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.signal import spectrogram
import warnings
warnings.filterwarnings('ignore')

from adaptive_genie_network import AdaptiveGenieNetwork
from mathematical_models import NegationDensityCalculator, ResonanceCalculator
from example_applications import OptimizationResult


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    style: str = 'seaborn-v0_8'
    color_palette: str = 'viridis'
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    animation_interval: int = 100
    save_format: str = 'png'


class SystemDynamicsVisualizer:
    """
    Comprehensive visualization of AdaptiveGenieNetwork system dynamics
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        plt.style.use('default')  # Use default style since seaborn-v0_8 might not be available
        self.color_palette = plt.cm.get_cmap(self.config.color_palette)
        
    def create_comprehensive_dashboard(self, network: AdaptiveGenieNetwork, 
                                     save_path: str = None) -> go.Figure:
        """Create an interactive dashboard showing all system dynamics"""
        if not network.negotiation_history:
            print("No negotiation history available for visualization")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Population Breathing', 'Natural Rhythm Evolution', 'Solution Crystallization',
                'System Energy Dynamics', 'Dialectical Tension', 'Collective Consciousness',
                'Agent Consciousness Evolution', 'Negotiation Frequency', 'System Phase Space'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "scatter"}]
            ]
        )
        
        # Extract data from negotiation history
        timestamps = [entry['timestamp'] for entry in network.negotiation_history]
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]
        
        # Population dynamics
        pop_sizes = [entry['proposals']['population']['population_size'] 
                    for entry in network.negotiation_history]
        fig.add_trace(
            go.Scatter(x=relative_times, y=pop_sizes, mode='lines+markers',
                      name='Population Size', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Natural rhythm
        frequencies = [entry['proposals']['rhythm']['natural_frequency'] 
                      for entry in network.negotiation_history]
        fig.add_trace(
            go.Scatter(x=relative_times, y=frequencies, mode='lines+markers',
                      name='Natural Frequency', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # Crystallization
        crystallization = [entry['proposals']['resonance']['crystallization_level'] 
                          for entry in network.negotiation_history]
        fig.add_trace(
            go.Scatter(x=relative_times, y=crystallization, mode='lines+markers',
                      name='Crystallization', line=dict(color='red', width=2)),
            row=1, col=3
        )
        
        # System energy
        energies = [entry['synthesis']['system_energy'] 
                   for entry in network.negotiation_history]
        fig.add_trace(
            go.Scatter(x=relative_times, y=energies, mode='lines+markers',
                      name='System Energy', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # Dialectical tension
        tensions = [entry['complexity'].dialectical_tension 
                   for entry in network.negotiation_history]
        fig.add_trace(
            go.Scatter(x=relative_times, y=tensions, mode='lines+markers',
                      name='Dialectical Tension', line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        # Collective consciousness evolution
        consciousness_history = self._calculate_consciousness_evolution(network)
        fig.add_trace(
            go.Scatter(x=relative_times, y=consciousness_history, mode='lines+markers',
                      name='Collective Consciousness', line=dict(color='magenta', width=2)),
            row=2, col=3
        )
        
        # Agent consciousness evolution
        pop_consciousness = [network.population.consciousness_level] * len(relative_times)
        rhythm_consciousness = [network.iterator.consciousness_level] * len(relative_times)
        resonance_consciousness = [network.convergence.consciousness_level] * len(relative_times)
        
        fig.add_trace(
            go.Scatter(x=relative_times, y=pop_consciousness, mode='lines',
                      name='Population Agent', line=dict(color='lightblue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=relative_times, y=rhythm_consciousness, mode='lines',
                      name='Rhythm Agent', line=dict(color='lightgreen')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=relative_times, y=resonance_consciousness, mode='lines',
                      name='Resonance Agent', line=dict(color='lightcoral')),
            row=3, col=1
        )
        
        # Negotiation frequency (how often parameters change significantly)
        negotiation_changes = self._calculate_negotiation_frequency(network)
        fig.add_trace(
            go.Scatter(x=relative_times[1:], y=negotiation_changes, mode='lines+markers',
                      name='Change Frequency', line=dict(color='brown', width=2)),
            row=3, col=2
        )
        
        # System phase space (Energy vs Crystallization)
        fig.add_trace(
            go.Scatter(x=energies, y=crystallization, mode='markers+lines',
                      name='Phase Trajectory', 
                      marker=dict(color=relative_times, colorscale='Viridis', size=8),
                      line=dict(color='gray', width=1)),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title="AdaptiveGenieNetwork System Dynamics Dashboard",
            height=900,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        for i in range(1, 4):
            for j in range(1, 4):
                if i < 3 or j < 3:
                    fig.update_xaxes(title_text="Time (seconds)", row=i, col=j)
        
        fig.update_xaxes(title_text="System Energy", row=3, col=3)
        fig.update_yaxes(title_text="Crystallization Level", row=3, col=3)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _calculate_consciousness_evolution(self, network: AdaptiveGenieNetwork) -> List[float]:
        """Calculate the evolution of collective consciousness"""
        consciousness_history = []
        temp_consciousness = 0.5
        
        for entry in network.negotiation_history:
            harmony = entry['proposals']['resonance']['harmony_level']
            energy = entry['synthesis']['system_energy']
            consciousness_delta = 0.01 * (harmony + energy - 1.0)
            temp_consciousness = np.clip(temp_consciousness + consciousness_delta, 0.0, 1.0)
            consciousness_history.append(temp_consciousness)
        
        return consciousness_history
    
    def _calculate_negotiation_frequency(self, network: AdaptiveGenieNetwork) -> List[float]:
        """Calculate how frequently significant parameter changes occur"""
        if len(network.negotiation_history) < 2:
            return []
        
        changes = []
        for i in range(1, len(network.negotiation_history)):
            prev_params = network.negotiation_history[i-1]['synthesis']
            curr_params = network.negotiation_history[i]['synthesis']
            
            # Calculate parameter change magnitude
            change_magnitude = 0
            for key in ['population_size', 'system_energy', 'crystallization_level']:
                if key in prev_params and key in curr_params:
                    change_magnitude += abs(curr_params[key] - prev_params[key])
            
            changes.append(change_magnitude)
        
        return changes
    
    def create_agent_interaction_network(self, network: AdaptiveGenieNetwork,
                                       save_path: str = None) -> go.Figure:
        """Visualize agent interactions as a network graph"""
        # Create network graph showing agent relationships
        fig = go.Figure()
        
        # Agent positions (arranged in a circle)
        agents = ['Population', 'Rhythm', 'Resonance', 'GradientField']
        n_agents = len(agents)
        angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
        
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Add agent nodes
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(size=50, color=['blue', 'green', 'red', 'purple']),
            text=agents,
            textposition="middle center",
            textfont=dict(color="white", size=12),
            name="Agents"
        ))
        
        # Add interaction edges (all agents interact with all others)
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                fig.add_trace(go.Scatter(
                    x=[x_pos[i], x_pos[j]], y=[y_pos[i], y_pos[j]],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash'),
                    showlegend=False
                ))
        
        # Add central consciousness node
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=30, color='gold'),
            text=['Collective<br>Consciousness'],
            textposition="middle center",
            textfont=dict(color="black", size=10),
            name="Collective Consciousness"
        ))
        
        # Connect all agents to central consciousness
        for i in range(n_agents):
            fig.add_trace(go.Scatter(
                x=[x_pos[i], 0], y=[y_pos[i], 0],
                mode='lines',
                line=dict(color='gold', width=1),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Agent Interaction Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=True,
            template="plotly_white",
            width=600, height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dialectical_synthesis_animation(self, synthesis_history: List[Dict],
                                             save_path: str = None) -> animation.FuncAnimation:
        """Create animation showing dialectical synthesis process"""
        if not synthesis_history:
            print("No synthesis history available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Setup for 2D visualization (using first 2 dimensions)
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_title('Dialectical Synthesis Process')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.grid(True, alpha=0.3)
        
        # Setup for tension evolution
        ax2.set_xlim(0, len(synthesis_history))
        ax2.set_ylim(0, 1)
        ax2.set_title('Dialectical Tension Evolution')
        ax2.set_xlabel('Synthesis Step')
        ax2.set_ylabel('Tension Level')
        ax2.grid(True, alpha=0.3)
        
        # Initialize plot elements
        thesis_point, = ax1.plot([], [], 'ro', markersize=10, label='Thesis')
        antithesis_point, = ax1.plot([], [], 'bo', markersize=10, label='Antithesis')
        synthesis_point, = ax1.plot([], [], 'go', markersize=12, label='Synthesis')
        synthesis_trail, = ax1.plot([], [], 'g-', alpha=0.5, linewidth=2)
        
        tension_line, = ax2.plot([], [], 'r-', linewidth=2)
        tension_points, = ax2.plot([], [], 'ro', markersize=6)
        
        ax1.legend()
        
        def animate(frame):
            if frame >= len(synthesis_history):
                return thesis_point, antithesis_point, synthesis_point, synthesis_trail, tension_line, tension_points
            
            entry = synthesis_history[frame]
            
            # Extract 2D coordinates (use first 2 dimensions)
            thesis = entry.get('thesis', np.array([0, 0]))[:2]
            antithesis = entry.get('antithesis', np.array([0, 0]))[:2]
            synthesis = entry.get('synthesis', np.array([0, 0]))[:2]
            tension = entry.get('tension', 0)
            
            # Update synthesis process plot
            thesis_point.set_data([thesis[0]], [thesis[1]])
            antithesis_point.set_data([antithesis[0]], [antithesis[1]])
            synthesis_point.set_data([synthesis[0]], [synthesis[1]])
            
            # Update synthesis trail
            if frame > 0:
                trail_x = [synthesis_history[i].get('synthesis', np.array([0, 0]))[0] 
                          for i in range(min(frame+1, len(synthesis_history)))]
                trail_y = [synthesis_history[i].get('synthesis', np.array([0, 0]))[1] 
                          for i in range(min(frame+1, len(synthesis_history)))]
                synthesis_trail.set_data(trail_x, trail_y)
            
            # Update tension evolution
            tensions = [synthesis_history[i].get('tension', 0) for i in range(frame+1)]
            tension_line.set_data(range(frame+1), tensions)
            tension_points.set_data([frame], [tension])
            
            return thesis_point, antithesis_point, synthesis_point, synthesis_trail, tension_line, tension_points
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(synthesis_history),
            interval=self.config.animation_interval, blit=True, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim


class OptimizationLandscapeVisualizer:
    """
    Visualization tools for optimization landscapes and problem analysis
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
    def visualize_negation_density(self, objective_function: Callable,
                                  bounds: List[Tuple[float, float]],
                                  save_path: str = None) -> go.Figure:
        """Visualize negation density across the optimization landscape"""
        if len(bounds) != 2:
            print("Negation density visualization currently supports 2D problems only")
            return None
        
        # Create mesh grid
        x_bounds, y_bounds = bounds
        x = np.linspace(x_bounds[0], x_bounds[1], 50)
        y = np.linspace(y_bounds[0], y_bounds[1], 50)
        X, Y = np.meshgrid(x, y)
        
        # Calculate negation density
        negation_calc = NegationDensityCalculator()
        negation_metrics = negation_calc.calculate_negation_density(
            objective_function, bounds, sample_points=500
        )
        
        # Calculate fitness landscape
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = objective_function([X[i, j], Y[i, j]])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Fitness Landscape', 'Gradient Field',
                'Negation Density Heatmap', 'Dialectical Tension'
            ],
            specs=[[{"type": "surface"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Fitness landscape
        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', name='Fitness'),
            row=1, col=1
        )
        
        # Gradient field (simplified)
        skip = 5
        U = np.gradient(Z)[1][::skip, ::skip]
        V = np.gradient(Z)[0][::skip, ::skip]
        X_skip = X[::skip, ::skip]
        Y_skip = Y[::skip, ::skip]
        
        fig.add_trace(
            go.Scatter(x=X_skip.flatten(), y=Y_skip.flatten(),
                      mode='markers',
                      marker=dict(size=5, color='blue'),
                      name='Gradient Field'),
            row=1, col=2
        )
        
        # Add gradient arrows (simplified representation)
        for i in range(0, X_skip.shape[0], 2):
            for j in range(0, X_skip.shape[1], 2):
                fig.add_trace(
                    go.Scatter(x=[X_skip[i,j], X_skip[i,j] + U[i,j]*0.1],
                              y=[Y_skip[i,j], Y_skip[i,j] + V[i,j]*0.1],
                              mode='lines',
                              line=dict(color='red', width=1),
                              showlegend=False),
                    row=1, col=2
                )
        
        # Negation density heatmap (simulated)
        negation_density = negation_metrics['negation_density']
        density_map = np.random.random(X.shape) * negation_density
        
        fig.add_trace(
            go.Heatmap(z=density_map, x=x, y=y, colorscale='Reds',
                      name='Negation Density'),
            row=2, col=1
        )
        
        # Dialectical tension visualization
        tension_level = negation_metrics['dialectical_tension']
        fig.add_trace(
            go.Scatter(x=[0], y=[tension_level],
                      mode='markers',
                      marker=dict(size=20, color='orange'),
                      name=f'Tension: {tension_level:.3f}'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Optimization Landscape Analysis",
            height=800,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_crystallization_spectrogram(self, crystallization_history: List[float],
                                         save_path: str = None) -> go.Figure:
        """Create spectrogram of crystallization process"""
        if len(crystallization_history) < 10:
            print("Insufficient data for spectrogram")
            return None
        
        # Calculate spectrogram
        signal = np.array(crystallization_history)
        frequencies, times, Sxx = spectrogram(signal, fs=1.0, nperseg=min(len(signal)//4, 32))
        
        fig = go.Figure(data=go.Heatmap(
            z=10 * np.log10(Sxx + 1e-10),  # Convert to dB
            x=times,
            y=frequencies,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)")
        ))
        
        fig.update_layout(
            title="Crystallization Process Spectrogram",
            xaxis_title="Time",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class RealTimeMonitor:
    """
    Real-time monitoring dashboard for optimization process
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.data_buffer = {
            'timestamps': [],
            'fitness': [],
            'population_size': [],
            'crystallization': [],
            'energy': [],
            'consciousness': []
        }
        
    def update_data(self, timestamp: float, fitness: float, population_size: int,
                   crystallization: float, energy: float, consciousness: float):
        """Update monitoring data"""
        self.data_buffer['timestamps'].append(timestamp)
        self.data_buffer['fitness'].append(fitness)
        self.data_buffer['population_size'].append(population_size)
        self.data_buffer['crystallization'].append(crystallization)
        self.data_buffer['energy'].append(energy)
        self.data_buffer['consciousness'].append(consciousness)
        
        # Keep only recent data (last 1000 points)
        for key in self.data_buffer:
            if len(self.data_buffer[key]) > 1000:
                self.data_buffer[key] = self.data_buffer[key][-1000:]
    
    def create_live_dashboard(self) -> go.Figure:
        """Create live monitoring dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Fitness Evolution', 'Population Dynamics',
                'Crystallization Level', 'System Energy',
                'Collective Consciousness', 'System Status'
            ]
        )
        
        timestamps = self.data_buffer['timestamps']
        
        # Fitness evolution
        fig.add_trace(
            go.Scatter(x=timestamps, y=self.data_buffer['fitness'],
                      mode='lines', name='Fitness', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Population dynamics
        fig.add_trace(
            go.Scatter(x=timestamps, y=self.data_buffer['population_size'],
                      mode='lines', name='Population', line=dict(color='green')),
            row=1, col=2
        )
        
        # Crystallization
        fig.add_trace(
            go.Scatter(x=timestamps, y=self.data_buffer['crystallization'],
                      mode='lines', name='Crystallization', line=dict(color='red')),
            row=2, col=1
        )
        
        # System energy
        fig.add_trace(
            go.Scatter(x=timestamps, y=self.data_buffer['energy'],
                      mode='lines', name='Energy', line=dict(color='purple')),
            row=2, col=2
        )
        
        # Collective consciousness
        fig.add_trace(
            go.Scatter(x=timestamps, y=self.data_buffer['consciousness'],
                      mode='lines', name='Consciousness', line=dict(color='orange')),
            row=3, col=1
        )
        
        # System status (current values as gauge)
        if timestamps:
            current_fitness = self.data_buffer['fitness'][-1]
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=current_fitness,
                    title={'text': "Current Fitness"},
                    gauge={'axis': {'range': [None, max(self.data_buffer['fitness']) if self.data_buffer['fitness'] else 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, max(self.data_buffer['fitness']) if self.data_buffer['fitness'] else 1], 'color': "lightgray"}]}
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title="Real-Time Optimization Monitor",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig


def create_comprehensive_report(network: AdaptiveGenieNetwork, 
                              optimization_result: OptimizationResult,
                              save_directory: str = "visualization_output") -> Dict[str, str]:
    """
    Create a comprehensive visualization report
    """
    import os
    os.makedirs(save_directory, exist_ok=True)
    
    config = VisualizationConfig()
    
    # Initialize visualizers
    system_viz = SystemDynamicsVisualizer(config)
    landscape_viz = OptimizationLandscapeVisualizer(config)
    
    report_files = {}
    
    # 1. System dynamics dashboard
    dashboard_fig = system_viz.create_comprehensive_dashboard(network)
    if dashboard_fig:
        dashboard_path = os.path.join(save_directory, "system_dynamics_dashboard.html")
        dashboard_fig.write_html(dashboard_path)
        report_files['dashboard'] = dashboard_path
    
    # 2. Agent interaction network
    network_fig = system_viz.create_agent_interaction_network(network)
    if network_fig:
        network_path = os.path.join(save_directory, "agent_interaction_network.html")
        network_fig.write_html(network_path)
        report_files['network'] = network_path
    
    # 3. Crystallization spectrogram
    if optimization_result.crystallization_history:
        spectrogram_fig = landscape_viz.create_crystallization_spectrogram(
            optimization_result.crystallization_history
        )
        if spectrogram_fig:
            spectrogram_path = os.path.join(save_directory, "crystallization_spectrogram.html")
            spectrogram_fig.write_html(spectrogram_path)
            report_files['spectrogram'] = spectrogram_path
    
    # 4. Static summary plots
    summary_fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convergence history
    axes[0, 0].semilogy(optimization_result.convergence_history)
    axes[0, 0].set_title('Convergence History')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best Fitness (log scale)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Population evolution
    axes[0, 1].plot(optimization_result.population_history, 'g-', linewidth=2)
    axes[0, 1].set_title('Population Evolution')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Population Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Crystallization evolution
    if optimization_result.crystallization_history:
        axes[1, 0].plot(optimization_result.crystallization_history, 'r-', linewidth=2)
        axes[1, 0].set_title('Crystallization Evolution')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Crystallization Level')
        axes[1, 0].grid(True, alpha=0.3)
    
    # System energy (if available)
    if optimization_result.dialectical_states:
        energies = [state.get('system_energy', 0.5) for state in optimization_result.dialectical_states]
        axes[1, 1].plot(energies, 'm-', linewidth=2)
        axes[1, 1].set_title('System Energy Evolution')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Energy Level')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(save_directory, "optimization_summary.png")
    plt.savefig(summary_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    report_files['summary'] = summary_path
    
    # 5. Generate HTML report index
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AdaptiveGenieNetwork Optimization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .link {{ display: block; margin: 10px 0; padding: 10px; background: #e7f3ff; border-radius: 5px; text-decoration: none; color: #0066cc; }}
            .link:hover {{ background: #d0e7ff; }}
        </style>
    </head>
    <body>
        <h1>AdaptiveGenieNetwork Optimization Report</h1>
        
        <h2>Optimization Results</h2>
        <div class="metric"><strong>Best Fitness:</strong> {optimization_result.best_fitness:.6f}</div>
        <div class="metric"><strong>Total Evaluations:</strong> {optimization_result.total_evaluations}</div>
        <div class="metric"><strong>Execution Time:</strong> {optimization_result.execution_time:.2f} seconds</div>
        <div class="metric"><strong>Final Population Size:</strong> {optimization_result.population_history[-1] if optimization_result.population_history else 'N/A'}</div>
        
        <h2>System State</h2>
        <div class="metric"><strong>Collective Consciousness:</strong> {network.collective_consciousness:.3f}</div>
        <div class="metric"><strong>System Energy:</strong> {network.system_energy:.3f}</div>
        <div class="metric"><strong>Total Negotiations:</strong> {len(network.negotiation_history)}</div>
        
        <h2>Visualizations</h2>
    """
    
    for name, path in report_files.items():
        if path.endswith('.html'):
            html_content += f'<a href="{os.path.basename(path)}" class="link">📊 {name.title().replace("_", " ")}</a>\n'
        else:
            html_content += f'<a href="{os.path.basename(path)}" class="link">📈 {name.title().replace("_", " ")}</a>\n'
    
    html_content += """
        </body>
    </html>
    """
    
    index_path = os.path.join(save_directory, "index.html")
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    report_files['index'] = index_path
    
    print(f"Comprehensive report generated in: {save_directory}")
    print(f"Open {index_path} to view the complete report")
    
    return report_files


if __name__ == "__main__":
    # Example usage
    print("AdaptiveGenieNetwork Visualization Tools")
    print("=" * 50)
    
    # Create example network with some history
    from example_applications import DialecticalParticleSwarm, rastrigin
    
    # Run optimization to generate data
    dpso = DialecticalParticleSwarm(rastrigin, [(-5.12, 5.12)] * 2)
    result = dpso.optimize(max_iterations=30)
    
    # Create comprehensive report
    report_files = create_comprehensive_report(
        dpso.genie_network, result, "example_visualization_output"
    )
    
    print("\nGenerated visualization files:")
    for name, path in report_files.items():
        print(f"  {name}: {path}")