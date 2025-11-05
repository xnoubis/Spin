"""
Mathematical Models for AdaptiveGenieNetwork
===========================================

This module provides the mathematical foundations for dialectical optimization,
including negation density calculations, resonance mathematics, and gradient field theory.
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DialecticalState:
    """Represents the dialectical state of the optimization system"""
    thesis: np.ndarray
    antithesis: np.ndarray
    synthesis: np.ndarray
    tension: float
    resolution_energy: float


class NegationDensityCalculator:
    """
    Calculates the negation density of problem landscapes using dialectical analysis
    """
    
    def __init__(self):
        self.landscape_memory = []
        self.contradiction_threshold = 0.5
        
    def calculate_negation_density(self, fitness_landscape: Callable, 
                                 bounds: List[Tuple[float, float]], 
                                 sample_points: int = 1000) -> Dict[str, float]:
        """
        Calculate negation density by analyzing contradictions in the fitness landscape
        
        Negation density measures how much the landscape contradicts itself -
        areas where local optima conflict with global optimization direction.
        """
        dimensions = len(bounds)
        
        # Sample the landscape
        samples = self._generate_samples(bounds, sample_points)
        fitness_values = np.array([fitness_landscape(x) for x in samples])
        
        # Calculate local gradients
        gradients = self._estimate_gradients(samples, fitness_values, bounds)
        
        # Detect contradictions (negations)
        contradictions = self._detect_contradictions(samples, fitness_values, gradients)
        
        # Calculate various negation metrics
        negation_density = np.mean(contradictions)
        contradiction_variance = np.var(contradictions)
        dialectical_tension = self._calculate_dialectical_tension(gradients)
        
        # Measure landscape deception
        deception_level = self._measure_deception(samples, fitness_values)
        
        return {
            'negation_density': negation_density,
            'contradiction_variance': contradiction_variance,
            'dialectical_tension': dialectical_tension,
            'deception_level': deception_level,
            'exploration_requirement': negation_density * (1.0 + deception_level),
            'convergence_gradient': (1.0 - deception_level) * (1.0 - contradiction_variance)
        }
    
    def _generate_samples(self, bounds: List[Tuple[float, float]], 
                         n_samples: int) -> np.ndarray:
        """Generate sample points using Latin Hypercube Sampling"""
        dimensions = len(bounds)
        samples = np.random.random((n_samples, dimensions))
        
        # Scale to bounds
        for i, (lower, upper) in enumerate(bounds):
            samples[:, i] = lower + samples[:, i] * (upper - lower)
            
        return samples
    
    def _estimate_gradients(self, samples: np.ndarray, fitness_values: np.ndarray,
                           bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Estimate gradients using finite differences"""
        n_samples, dimensions = samples.shape
        gradients = np.zeros_like(samples)
        
        epsilon = 1e-6
        
        for i in range(n_samples):
            for d in range(dimensions):
                # Forward difference
                sample_plus = samples[i].copy()
                sample_plus[d] += epsilon
                
                # Ensure within bounds
                lower, upper = bounds[d]
                sample_plus[d] = np.clip(sample_plus[d], lower, upper)
                
                # Calculate gradient component
                try:
                    fitness_plus = fitness_values[i]  # Approximate for efficiency
                    gradients[i, d] = (fitness_plus - fitness_values[i]) / epsilon
                except:
                    gradients[i, d] = 0.0
                    
        return gradients
    
    def _detect_contradictions(self, samples: np.ndarray, fitness_values: np.ndarray,
                              gradients: np.ndarray) -> np.ndarray:
        """Detect contradictions in the landscape"""
        n_samples = len(samples)
        contradictions = np.zeros(n_samples)
        
        # Find nearest neighbors for each point
        distances = squareform(pdist(samples))
        
        for i in range(n_samples):
            # Find k nearest neighbors
            k = min(10, n_samples - 1)
            neighbor_indices = np.argsort(distances[i])[1:k+1]
            
            # Check for contradictions with neighbors
            contradiction_count = 0
            
            for j in neighbor_indices:
                # Gradient contradiction: gradients point in opposite directions
                gradient_similarity = np.dot(gradients[i], gradients[j])
                gradient_similarity /= (np.linalg.norm(gradients[i]) * np.linalg.norm(gradients[j]) + 1e-8)
                
                # Fitness contradiction: better fitness but worse gradient direction
                fitness_diff = fitness_values[j] - fitness_values[i]
                expected_direction = np.sign(fitness_diff)
                actual_direction = np.sign(gradient_similarity)
                
                if expected_direction != actual_direction:
                    contradiction_count += 1
            
            contradictions[i] = contradiction_count / k
            
        return contradictions
    
    def _calculate_dialectical_tension(self, gradients: np.ndarray) -> float:
        """Calculate the overall dialectical tension in the gradient field"""
        if len(gradients) < 2:
            return 0.0
            
        # Calculate pairwise gradient conflicts
        n_samples = len(gradients)
        tensions = []
        
        for i in range(min(100, n_samples)):  # Sample for efficiency
            for j in range(i+1, min(i+10, n_samples)):
                # Measure angle between gradients
                cos_angle = np.dot(gradients[i], gradients[j])
                cos_angle /= (np.linalg.norm(gradients[i]) * np.linalg.norm(gradients[j]) + 1e-8)
                
                # Tension is highest when gradients are opposite
                tension = 1.0 - abs(cos_angle)
                tensions.append(tension)
        
        return np.mean(tensions) if tensions else 0.0
    
    def _measure_deception(self, samples: np.ndarray, fitness_values: np.ndarray) -> float:
        """Measure how deceptive the landscape is"""
        if len(samples) < 10:
            return 0.0
            
        # Sort by fitness
        sorted_indices = np.argsort(fitness_values)
        best_samples = samples[sorted_indices[-10:]]  # Top 10
        worst_samples = samples[sorted_indices[:10]]   # Bottom 10
        
        # Calculate average distances
        best_distances = pdist(best_samples)
        worst_distances = pdist(worst_samples)
        
        # Deception: good solutions are far apart, bad solutions are close
        avg_best_distance = np.mean(best_distances) if len(best_distances) > 0 else 0
        avg_worst_distance = np.mean(worst_distances) if len(worst_distances) > 0 else 0
        
        # Normalize deception measure
        total_distance = avg_best_distance + avg_worst_distance
        if total_distance > 0:
            deception = avg_best_distance / total_distance
        else:
            deception = 0.0
            
        return np.clip(deception, 0.0, 1.0)


class ResonanceCalculator:
    """
    Calculates resonance frequencies and crystallization patterns
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.frequency_history = []
        
    def calculate_resonance_frequency(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Calculate the dominant resonance frequency using spectral analysis"""
        if len(signal_data) < 10:
            return {'dominant_frequency': 1.0, 'resonance_strength': 0.0, 'harmony_index': 0.0}
        
        # Apply window function
        windowed_signal = signal_data * signal.windows.hann(len(signal_data))
        
        # FFT analysis
        fft_result = np.fft.fft(windowed_signal)
        frequencies = np.fft.fftfreq(len(windowed_signal))
        power_spectrum = np.abs(fft_result) ** 2
        
        # Find dominant frequency
        positive_freqs = frequencies[:len(frequencies)//2]
        positive_power = power_spectrum[:len(power_spectrum)//2]
        
        if len(positive_power) > 1:
            dominant_idx = np.argmax(positive_power[1:]) + 1  # Skip DC component
            dominant_frequency = abs(positive_freqs[dominant_idx])
            resonance_strength = positive_power[dominant_idx] / np.sum(positive_power)
        else:
            dominant_frequency = 1.0
            resonance_strength = 0.0
        
        # Calculate harmony index (how well the signal matches its dominant frequency)
        harmony_index = self._calculate_harmony_index(signal_data, dominant_frequency)
        
        return {
            'dominant_frequency': dominant_frequency,
            'resonance_strength': resonance_strength,
            'harmony_index': harmony_index
        }
    
    def _calculate_harmony_index(self, signal_data: np.ndarray, frequency: float) -> float:
        """Calculate how harmonious the signal is with the given frequency"""
        if frequency == 0:
            return 0.0
            
        # Generate reference sine wave
        t = np.arange(len(signal_data))
        reference_wave = np.sin(2 * np.pi * frequency * t / len(signal_data))
        
        # Calculate correlation
        correlation = np.corrcoef(signal_data, reference_wave)[0, 1]
        
        # Return absolute correlation as harmony index
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def detect_crystallization_patterns(self, population_positions: np.ndarray,
                                      fitness_values: np.ndarray) -> Dict[str, float]:
        """Detect crystallization patterns in the population"""
        if len(population_positions) < 3:
            return {'crystallization_level': 0.0, 'cluster_coherence': 0.0, 'phase_transition': 0.0}
        
        # Calculate population diversity
        distances = pdist(population_positions)
        diversity = np.mean(distances) / (np.std(distances) + 1e-8)
        
        # Calculate fitness clustering
        fitness_std = np.std(fitness_values)
        fitness_clustering = 1.0 / (1.0 + fitness_std)
        
        # Crystallization level (low diversity + high fitness clustering)
        crystallization_level = fitness_clustering * (1.0 / (1.0 + diversity))
        
        # Cluster coherence using silhouette-like measure
        cluster_coherence = self._calculate_cluster_coherence(population_positions, fitness_values)
        
        # Phase transition detection
        phase_transition = self._detect_phase_transition(crystallization_level)
        
        return {
            'crystallization_level': crystallization_level,
            'cluster_coherence': cluster_coherence,
            'phase_transition': phase_transition,
            'population_diversity': diversity,
            'fitness_clustering': fitness_clustering
        }
    
    def _calculate_cluster_coherence(self, positions: np.ndarray, fitness_values: np.ndarray) -> float:
        """Calculate how coherent the clusters are"""
        if len(positions) < 3:
            return 0.0
            
        # Sort by fitness
        sorted_indices = np.argsort(fitness_values)
        n_elite = max(1, len(positions) // 4)  # Top 25%
        
        elite_positions = positions[sorted_indices[-n_elite:]]
        non_elite_positions = positions[sorted_indices[:-n_elite]]
        
        if len(elite_positions) < 2 or len(non_elite_positions) < 2:
            return 0.0
        
        # Calculate intra-cluster distances
        elite_distances = pdist(elite_positions)
        non_elite_distances = pdist(non_elite_positions)
        
        # Calculate inter-cluster distances
        inter_distances = []
        for elite_pos in elite_positions:
            for non_elite_pos in non_elite_positions:
                inter_distances.append(np.linalg.norm(elite_pos - non_elite_pos))
        
        # Coherence: small intra-cluster, large inter-cluster distances
        avg_intra = (np.mean(elite_distances) + np.mean(non_elite_distances)) / 2
        avg_inter = np.mean(inter_distances)
        
        if avg_intra > 0:
            coherence = avg_inter / avg_intra
        else:
            coherence = 1.0
            
        return np.clip(coherence / 10.0, 0.0, 1.0)  # Normalize
    
    def _detect_phase_transition(self, current_crystallization: float) -> float:
        """Detect phase transitions in crystallization"""
        self.frequency_history.append(current_crystallization)
        
        if len(self.frequency_history) < 10:
            return 0.0
        
        # Keep only recent history
        if len(self.frequency_history) > self.window_size:
            self.frequency_history.pop(0)
        
        # Calculate rate of change
        recent_history = np.array(self.frequency_history[-10:])
        if len(recent_history) < 2:
            return 0.0
            
        # Detect sudden changes (phase transitions)
        gradient = np.gradient(recent_history)
        phase_transition_strength = np.std(gradient)
        
        return np.clip(phase_transition_strength, 0.0, 1.0)


class GradientFieldMathematics:
    """
    Mathematical models for dynamic gradient fields
    """
    
    def __init__(self):
        self.field_equations = {}
        self.flow_patterns = {}
        
    def generate_field_equations(self, problem_landscape: Dict) -> Dict[str, Callable]:
        """Generate mathematical equations for the gradient field"""
        dimensions = problem_landscape.get('dimensions', 2)
        bounds = problem_landscape.get('bounds', [(-10, 10)] * dimensions)
        
        field_equations = {}
        
        for dim in range(dimensions):
            # Create field equation for this dimension
            lower, upper = bounds[dim]
            center = (lower + upper) / 2
            scale = (upper - lower) / 2
            
            # Gradient field equation: combines attraction and repulsion
            def field_equation(x, dim=dim, center=center, scale=scale):
                # Normalize position
                normalized_x = (x - center) / scale
                
                # Multi-modal field with dialectical tension
                attraction = -normalized_x  # Linear attraction to center
                repulsion = 0.1 * np.sin(4 * np.pi * normalized_x)  # Oscillatory repulsion
                
                return attraction + repulsion
            
            field_equations[f'dim_{dim}'] = field_equation
        
        self.field_equations = field_equations
        return field_equations
    
    def calculate_flow_vectors(self, position: np.ndarray) -> np.ndarray:
        """Calculate flow vectors at a given position"""
        if not self.field_equations:
            return np.zeros_like(position)
        
        flow_vector = np.zeros_like(position)
        
        for dim, pos in enumerate(position):
            if f'dim_{dim}' in self.field_equations:
                field_func = self.field_equations[f'dim_{dim}']
                flow_vector[dim] = field_func(pos)
        
        return flow_vector
    
    def simulate_field_dynamics(self, initial_positions: np.ndarray, 
                               time_steps: int = 100, dt: float = 0.01) -> np.ndarray:
        """Simulate particle dynamics in the gradient field"""
        n_particles, dimensions = initial_positions.shape
        trajectory = np.zeros((time_steps, n_particles, dimensions))
        
        positions = initial_positions.copy()
        velocities = np.zeros_like(positions)
        
        for t in range(time_steps):
            trajectory[t] = positions.copy()
            
            # Calculate forces for each particle
            for i in range(n_particles):
                flow_force = self.calculate_flow_vectors(positions[i])
                
                # Update velocity (with damping)
                damping = 0.9
                velocities[i] = damping * velocities[i] + dt * flow_force
                
                # Update position
                positions[i] += dt * velocities[i]
        
        return trajectory
    
    def analyze_field_topology(self, bounds: List[Tuple[float, float]], 
                              resolution: int = 50) -> Dict[str, np.ndarray]:
        """Analyze the topology of the gradient field"""
        if len(bounds) != 2:
            raise ValueError("Topology analysis currently supports 2D only")
        
        # Create mesh grid
        x_bounds, y_bounds = bounds
        x = np.linspace(x_bounds[0], x_bounds[1], resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate flow vectors at each point
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(resolution):
            for j in range(resolution):
                position = np.array([X[i, j], Y[i, j]])
                flow = self.calculate_flow_vectors(position)
                U[i, j] = flow[0]
                V[i, j] = flow[1]
        
        # Calculate field properties
        divergence = np.gradient(U)[0] + np.gradient(V)[1]
        curl = np.gradient(V)[0] - np.gradient(U)[1]
        magnitude = np.sqrt(U**2 + V**2)
        
        return {
            'X': X, 'Y': Y,
            'U': U, 'V': V,
            'divergence': divergence,
            'curl': curl,
            'magnitude': magnitude
        }
    
    def find_critical_points(self, bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Find critical points (equilibria) in the gradient field"""
        critical_points = []
        
        # Use optimization to find points where gradient is zero
        def field_magnitude(x):
            flow = self.calculate_flow_vectors(x)
            return np.sum(flow**2)
        
        # Multiple random starting points
        for _ in range(20):
            # Random starting point
            start_point = np.array([
                np.random.uniform(bounds[i][0], bounds[i][1]) 
                for i in range(len(bounds))
            ])
            
            # Minimize field magnitude
            result = minimize(field_magnitude, start_point, method='BFGS')
            
            if result.success and result.fun < 1e-6:
                # Check if this is a new critical point
                is_new = True
                for existing_point in critical_points:
                    if np.linalg.norm(result.x - existing_point) < 1e-3:
                        is_new = False
                        break
                
                if is_new:
                    critical_points.append(result.x)
        
        return critical_points


class DialecticalSynthesis:
    """
    Mathematical framework for dialectical synthesis in optimization
    """
    
    def __init__(self):
        self.synthesis_history = []
        
    def perform_synthesis(self, thesis: np.ndarray, antithesis: np.ndarray,
                         context: Dict = None) -> DialecticalState:
        """Perform dialectical synthesis between thesis and antithesis"""
        if len(thesis) != len(antithesis):
            raise ValueError("Thesis and antithesis must have same dimensions")
        
        # Calculate dialectical tension
        tension = self._calculate_tension(thesis, antithesis)
        
        # Generate synthesis through various methods
        synthesis_candidates = [
            self._arithmetic_synthesis(thesis, antithesis),
            self._geometric_synthesis(thesis, antithesis),
            self._harmonic_synthesis(thesis, antithesis),
            self._dialectical_synthesis(thesis, antithesis, tension)
        ]
        
        # Select best synthesis based on resolution energy
        best_synthesis = None
        best_energy = -np.inf
        
        for candidate in synthesis_candidates:
            energy = self._calculate_resolution_energy(thesis, antithesis, candidate, tension)
            if energy > best_energy:
                best_energy = energy
                best_synthesis = candidate
        
        # Create dialectical state
        state = DialecticalState(
            thesis=thesis.copy(),
            antithesis=antithesis.copy(),
            synthesis=best_synthesis,
            tension=tension,
            resolution_energy=best_energy
        )
        
        self.synthesis_history.append(state)
        return state
    
    def _calculate_tension(self, thesis: np.ndarray, antithesis: np.ndarray) -> float:
        """Calculate dialectical tension between thesis and antithesis"""
        # Euclidean distance normalized by dimension
        distance = np.linalg.norm(thesis - antithesis)
        normalized_distance = distance / np.sqrt(len(thesis))
        
        # Angular difference
        cos_angle = np.dot(thesis, antithesis) / (np.linalg.norm(thesis) * np.linalg.norm(antithesis) + 1e-8)
        angular_tension = 1.0 - abs(cos_angle)
        
        # Combined tension
        tension = 0.5 * (normalized_distance + angular_tension)
        return np.clip(tension, 0.0, 1.0)
    
    def _arithmetic_synthesis(self, thesis: np.ndarray, antithesis: np.ndarray) -> np.ndarray:
        """Simple arithmetic mean synthesis"""
        return (thesis + antithesis) / 2.0
    
    def _geometric_synthesis(self, thesis: np.ndarray, antithesis: np.ndarray) -> np.ndarray:
        """Geometric mean synthesis (for positive values)"""
        # Handle negative values by using signed geometric mean
        synthesis = np.zeros_like(thesis)
        for i in range(len(thesis)):
            if thesis[i] * antithesis[i] >= 0:  # Same sign
                synthesis[i] = np.sign(thesis[i]) * np.sqrt(abs(thesis[i] * antithesis[i]))
            else:  # Different signs - use arithmetic mean
                synthesis[i] = (thesis[i] + antithesis[i]) / 2.0
        return synthesis
    
    def _harmonic_synthesis(self, thesis: np.ndarray, antithesis: np.ndarray) -> np.ndarray:
        """Harmonic mean synthesis"""
        synthesis = np.zeros_like(thesis)
        for i in range(len(thesis)):
            if thesis[i] != 0 and antithesis[i] != 0 and np.sign(thesis[i]) == np.sign(antithesis[i]):
                synthesis[i] = 2 * thesis[i] * antithesis[i] / (thesis[i] + antithesis[i])
            else:
                synthesis[i] = (thesis[i] + antithesis[i]) / 2.0
        return synthesis
    
    def _dialectical_synthesis(self, thesis: np.ndarray, antithesis: np.ndarray, 
                              tension: float) -> np.ndarray:
        """Advanced dialectical synthesis incorporating tension"""
        # Weight based on tension - higher tension leads to more creative synthesis
        creativity_factor = tension
        
        # Base synthesis (weighted average)
        base_synthesis = (1 - creativity_factor) * thesis + creativity_factor * antithesis
        
        # Creative component (orthogonal to both thesis and antithesis)
        if len(thesis) >= 2:
            # Find orthogonal direction
            diff_vector = antithesis - thesis
            if np.linalg.norm(diff_vector) > 1e-8:
                # Create orthogonal vector using Gram-Schmidt-like process
                orthogonal = np.random.randn(len(thesis))
                orthogonal = orthogonal - np.dot(orthogonal, diff_vector) * diff_vector / np.dot(diff_vector, diff_vector)
                orthogonal = orthogonal / (np.linalg.norm(orthogonal) + 1e-8)
                
                # Add creative component
                creative_magnitude = tension * np.linalg.norm(diff_vector) * 0.1
                creative_synthesis = base_synthesis + creative_magnitude * orthogonal
            else:
                creative_synthesis = base_synthesis
        else:
            creative_synthesis = base_synthesis
        
        return creative_synthesis
    
    def _calculate_resolution_energy(self, thesis: np.ndarray, antithesis: np.ndarray,
                                   synthesis: np.ndarray, tension: float) -> float:
        """Calculate the resolution energy of a synthesis"""
        # Distance from thesis and antithesis
        dist_thesis = np.linalg.norm(synthesis - thesis)
        dist_antithesis = np.linalg.norm(synthesis - antithesis)
        
        # Balance: synthesis should be reasonably close to both
        balance = 1.0 / (1.0 + abs(dist_thesis - dist_antithesis))
        
        # Creativity: synthesis should not be trivial
        creativity = min(dist_thesis, dist_antithesis) / (np.linalg.norm(thesis - antithesis) + 1e-8)
        
        # Tension resolution: higher tension should lead to higher energy solutions
        tension_resolution = tension * creativity
        
        # Combined resolution energy
        resolution_energy = balance * (0.5 + 0.5 * tension_resolution)
        
        return resolution_energy
    
    def get_synthesis_trajectory(self) -> List[np.ndarray]:
        """Get the trajectory of synthesis points"""
        return [state.synthesis for state in self.synthesis_history]
    
    def analyze_dialectical_evolution(self) -> Dict[str, np.ndarray]:
        """Analyze the evolution of dialectical synthesis"""
        if not self.synthesis_history:
            return {}
        
        tensions = np.array([state.tension for state in self.synthesis_history])
        energies = np.array([state.resolution_energy for state in self.synthesis_history])
        
        # Calculate synthesis distances (how much synthesis changes)
        synthesis_distances = []
        for i in range(1, len(self.synthesis_history)):
            dist = np.linalg.norm(
                self.synthesis_history[i].synthesis - self.synthesis_history[i-1].synthesis
            )
            synthesis_distances.append(dist)
        
        return {
            'tensions': tensions,
            'resolution_energies': energies,
            'synthesis_distances': np.array(synthesis_distances),
            'evolution_trend': np.polyfit(range(len(energies)), energies, 1)[0] if len(energies) > 1 else 0
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Mathematical Models for AdaptiveGenieNetwork")
    print("=" * 50)
    
    # Test Negation Density Calculator
    def test_function(x):
        """A deceptive test function with multiple local optima"""
        return -(x[0]**2 + x[1]**2) + 0.5 * np.sin(5 * x[0]) * np.cos(5 * x[1])
    
    negation_calc = NegationDensityCalculator()
    bounds = [(-2, 2), (-2, 2)]
    negation_metrics = negation_calc.calculate_negation_density(test_function, bounds)
    
    print("\nNegation Density Analysis:")
    for key, value in negation_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test Resonance Calculator
    resonance_calc = ResonanceCalculator()
    
    # Generate test signal with known frequency
    t = np.linspace(0, 10, 100)
    test_signal = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(100)
    
    resonance_metrics = resonance_calc.calculate_resonance_frequency(test_signal)
    print(f"\nResonance Analysis:")
    for key, value in resonance_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test Dialectical Synthesis
    synthesis = DialecticalSynthesis()
    
    thesis = np.array([1.0, 2.0, -1.0])
    antithesis = np.array([-1.0, 1.0, 2.0])
    
    dialectical_state = synthesis.perform_synthesis(thesis, antithesis)
    
    print(f"\nDialectical Synthesis:")
    print(f"  Thesis: {thesis}")
    print(f"  Antithesis: {antithesis}")
    print(f"  Synthesis: {dialectical_state.synthesis}")
    print(f"  Tension: {dialectical_state.tension:.4f}")
    print(f"  Resolution Energy: {dialectical_state.resolution_energy:.4f}")