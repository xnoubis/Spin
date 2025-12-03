"""
AdaptiveGenieNetwork: A Revolutionary Optimization Framework
===========================================================

This framework treats optimization parameters as autonomous agents that engage in
dialectical negotiation to adapt to problem landscapes dynamically.

Core Philosophy:
- Parameters ARE agents, not static values
- Optimization follows natural rhythms and resonance patterns
- Boundaries are dynamic gradient fields, not rigid constraints
- Convergence emerges through collective intelligence
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from collections import deque

from exceptions import (
    ValidationError, InvalidBoundsError, InvalidParameterError,
    NegotiationError, VisualizationError, NumericError, DimensionMismatchError
)

# Public API exports
__all__ = [
    "AdaptiveGenieNetwork",
    "PopulationAgent",
    "RhythmAgent",
    "ResonanceAgent",
    "GradientField",
    "ComplexityMeasure",
]


@dataclass
class ComplexityMeasure:
    """Measures the negation density and dialectical tension in problem landscapes"""
    exploration_requirement: float
    convergence_gradient: float
    negation_density: float
    dialectical_tension: float


class Agent(ABC):
    """Base class for all autonomous agents in the network"""
    
    def __init__(self, name: str):
        self.name = name
        self.history = deque(maxlen=1000)
        self.energy = 1.0
        self.consciousness_level = 0.5
    
    @abstractmethod
    def negotiate(self, context: Dict) -> Dict:
        """Engage in dialectical negotiation with other agents"""
        pass
    
    def evolve_consciousness(self, feedback: float):
        """Evolve consciousness based on feedback from the system"""
        self.consciousness_level = np.clip(
            self.consciousness_level + 0.01 * feedback, 0.0, 1.0
        )


class PopulationAgent(Agent):
    """Self-adjusting swarm size agent that breathes with complexity"""

    def __init__(self, min_size: int = 10, max_size: int = 500,
                 breathing_rate: float = 0.1, oscillation_amplitude: float = 0.3):
        super().__init__("PopulationAgent")

        # Validate parameters
        if min_size < 1:
            raise InvalidParameterError(f"min_size must be >= 1, got {min_size}")
        if max_size < min_size:
            raise InvalidParameterError(f"max_size ({max_size}) must be >= min_size ({min_size})")
        if breathing_rate <= 0:
            raise InvalidParameterError(f"breathing_rate must be > 0, got {breathing_rate}")
        if not 0 <= oscillation_amplitude <= 1:
            raise InvalidParameterError(f"oscillation_amplitude must be in [0, 1], got {oscillation_amplitude}")

        self.size = min(max(50, min_size), max_size)
        self.min_size = min_size
        self.max_size = max_size
        self.breathing_rate = breathing_rate
        self.oscillation_amplitude = oscillation_amplitude
        
    def oscillate(self, exploration_need: float, convergence_pressure: float) -> int:
        """Population breathes with complexity through oscillation"""
        # Dialectical tension between exploration and convergence
        tension = exploration_need - convergence_pressure
        
        # Breathing pattern based on tension
        breath_phase = len(self.history) * self.breathing_rate
        oscillation = self.oscillation_amplitude * np.sin(breath_phase)
        
        # Size adjustment through dialectical negotiation
        size_factor = 1.0 + tension + oscillation
        new_size = int(self.size * size_factor)
        
        # Constrain within bounds
        self.size = np.clip(new_size, self.min_size, self.max_size)
        self.history.append(self.size)
        
        return self.size
    
    def negotiate(self, context: Dict) -> Dict:
        """Negotiate population size based on system state"""
        complexity = context.get('complexity', ComplexityMeasure(0.5, 0.5, 0.5, 0.5))
        new_size = self.oscillate(complexity.exploration_requirement, complexity.convergence_gradient)
        
        return {
            'population_size': new_size,
            'breathing_phase': len(self.history) * self.breathing_rate,
            'energy_level': self.energy
        }


class RhythmAgent(Agent):
    """Detects natural cycles and follows organic rhythms"""
    
    def __init__(self):
        super().__init__("RhythmAgent")
        self.limit = None  # No artificial limits
        self.cycle_detector = deque(maxlen=100)
        self.natural_frequency = 1.0
        self.phase = 0.0
        self.cycle_completion_threshold = 0.95
        
    def detect_cycle_completion(self) -> bool:
        """Detect when a natural cycle has completed"""
        if len(self.cycle_detector) < 10:
            return False

        try:
            # Analyze rhythm patterns using autocorrelation
            signal = np.array(list(self.cycle_detector))

            # Handle constant signals
            if np.std(signal) < 1e-10:
                return False

            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]

            # Find dominant frequency
            if len(autocorr) > 1:
                peak_idx = np.argmax(autocorr[1:]) + 1
                if peak_idx > 0:
                    self.natural_frequency = 2 * np.pi / peak_idx

            # Check for cycle completion (prevent division by zero)
            if autocorr[0] > 1e-10 and len(autocorr) > 1:
                cycle_strength = np.max(autocorr[1:]) / autocorr[0]
            else:
                cycle_strength = 0.0

            return cycle_strength > self.cycle_completion_threshold

        except (ValueError, FloatingPointError) as e:
            # If numeric errors occur, assume cycle not complete
            return False
    
    def negotiate(self, context: Dict) -> Dict:
        """Negotiate iteration rhythm based on natural cycles"""
        fitness_history = context.get('fitness_history', [])
        
        if fitness_history:
            self.cycle_detector.append(fitness_history[-1])
        
        cycle_complete = self.detect_cycle_completion()
        self.phase += self.natural_frequency
        
        return {
            'cycle_complete': cycle_complete,
            'natural_frequency': self.natural_frequency,
            'phase': self.phase,
            'rhythm_energy': np.sin(self.phase)
        }


class ResonanceAgent(Agent):
    """Feels solution crystallization through resonance patterns"""
    
    def __init__(self):
        super().__init__("ResonanceAgent")
        self.threshold = None  # Dynamic threshold
        self.resonance_frequency = 1.0
        self.crystallization_detector = deque(maxlen=50)
        self.harmony_level = 0.0
        
    def detect_resonance_frequency(self) -> float:
        """Detect the resonance frequency of solution crystallization"""
        if len(self.crystallization_detector) < 10:
            return self.resonance_frequency
            
        # Use Fourier analysis to detect dominant frequencies
        signal = np.array(list(self.crystallization_detector))
        fft = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        
        # Find dominant frequency
        dominant_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        self.resonance_frequency = abs(frequencies[dominant_idx])
        
        return self.resonance_frequency
    
    def measure_crystallization(self, population_diversity: float,
                              fitness_variance: float) -> float:
        """Measure how crystallized the solution has become"""
        # Validate inputs
        if not 0 <= population_diversity <= 1:
            raise InvalidParameterError(f"population_diversity must be in [0, 1], got {population_diversity}")
        if fitness_variance < 0:
            raise InvalidParameterError(f"fitness_variance must be >= 0, got {fitness_variance}")

        try:
            # Crystallization occurs when diversity decreases and fitness stabilizes
            crystallization = (1.0 - population_diversity) * (1.0 / (1.0 + fitness_variance))
            self.crystallization_detector.append(crystallization)
            return crystallization
        except (ValueError, FloatingPointError) as e:
            raise NumericError(f"Failed to compute crystallization: {e}")
    
    def negotiate(self, context: Dict) -> Dict:
        """Negotiate convergence through resonance detection"""
        population_diversity = context.get('population_diversity', 0.5)
        fitness_variance = context.get('fitness_variance', 1.0)
        
        crystallization = self.measure_crystallization(population_diversity, fitness_variance)
        resonance_freq = self.detect_resonance_frequency()
        
        # Dynamic threshold based on resonance
        self.threshold = crystallization * np.sin(resonance_freq * len(self.history))
        self.harmony_level = crystallization
        
        return {
            'crystallization_level': crystallization,
            'resonance_frequency': resonance_freq,
            'dynamic_threshold': self.threshold,
            'harmony_level': self.harmony_level
        }


class GradientField:
    """Dynamic constraint topology that flows like a field"""
    
    def __init__(self):
        self.gradients = {}
        self.flow_vectors = {}
        self.topology_map = None
        self.field_strength = 1.0
        
    def generate_gradients(self, problem_landscape: Dict):
        """Generate dynamic gradient fields for the problem landscape"""
        # Validate problem_landscape
        if not isinstance(problem_landscape, dict):
            raise ValidationError(f"problem_landscape must be a dict, got {type(problem_landscape)}")

        dimensions = problem_landscape.get('dimensions', 2)
        if not isinstance(dimensions, int) or dimensions < 1:
            raise InvalidParameterError(f"dimensions must be a positive integer, got {dimensions}")

        bounds = problem_landscape.get('bounds', [(-10, 10)] * dimensions)
        if not isinstance(bounds, list) or len(bounds) != dimensions:
            raise InvalidBoundsError(f"bounds must be a list of length {dimensions}, got {len(bounds) if isinstance(bounds, list) else 'non-list'}")

        # Validate each bound
        for i, bound in enumerate(bounds):
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise InvalidBoundsError(f"bounds[{i}] must be a tuple/list of 2 values, got {bound}")
            lower, upper = bound
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                raise InvalidBoundsError(f"bounds[{i}] values must be numeric, got ({lower}, {upper})")
            if lower >= upper:
                raise InvalidBoundsError(f"bounds[{i}] lower ({lower}) must be < upper ({upper})")

        try:
            # Create gradient field mesh
            resolution = 50
            self.topology_map = {}

            for dim in range(dimensions):
                lower, upper = bounds[dim]
                coords = np.linspace(lower, upper, resolution)
                self.topology_map[f'dim_{dim}'] = coords

                # Generate flow vectors based on problem characteristics
                gradient = np.gradient(coords)
                self.gradients[f'dim_{dim}'] = gradient

                # Flow vectors point toward optimal regions
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm > 1e-10:
                    flow = -gradient / gradient_norm
                else:
                    flow = np.zeros_like(gradient)
                self.flow_vectors[f'dim_{dim}'] = flow

        except Exception as e:
            raise NumericError(f"Failed to generate gradients: {e}")
    
    def get_field_strength_at(self, position: np.ndarray) -> float:
        """Get field strength at a specific position"""
        if not isinstance(position, np.ndarray):
            raise ValidationError(f"position must be a numpy array, got {type(position)}")

        if not self.gradients:
            return self.field_strength

        try:
            # Calculate field strength based on position in topology
            strength = 0.0
            for dim, pos in enumerate(position):
                if f'dim_{dim}' in self.gradients:
                    # Interpolate field strength
                    coords = self.topology_map[f'dim_{dim}']
                    idx = np.searchsorted(coords, pos)
                    idx = np.clip(idx, 0, len(coords) - 1)
                    strength += abs(self.gradients[f'dim_{dim}'][idx])

            return strength / max(1, len(position))
        except Exception as e:
            raise NumericError(f"Failed to compute field strength: {e}")
    
    def apply_field_force(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Apply field force to modify particle movement"""
        if not isinstance(position, np.ndarray) or not isinstance(velocity, np.ndarray):
            raise ValidationError(f"position and velocity must be numpy arrays")

        if position.shape != velocity.shape:
            raise DimensionMismatchError(f"position shape {position.shape} != velocity shape {velocity.shape}")

        if not self.flow_vectors:
            return velocity

        try:
            field_force = np.zeros_like(position)
            for dim, pos in enumerate(position):
                if f'dim_{dim}' in self.flow_vectors:
                    coords = self.topology_map[f'dim_{dim}']
                    idx = np.searchsorted(coords, pos)
                    idx = np.clip(idx, 0, len(coords) - 1)
                    field_force[dim] = self.flow_vectors[f'dim_{dim}'][idx]

            # Blend field force with current velocity
            field_strength = self.get_field_strength_at(position)
            return velocity + field_strength * field_force
        except Exception as e:
            raise NumericError(f"Failed to apply field force: {e}")


class AdaptiveGenieNetwork:
    """
    The main network that orchestrates dialectical negotiation between agents
    """
    
    def __init__(self):
        # Parameters ARE agents
        self.population = PopulationAgent()
        self.iterator = RhythmAgent()
        self.convergence = ResonanceAgent()
        self.boundaries = GradientField()
        
        # System state
        self.negotiation_history = []
        self.collective_consciousness = 0.5
        self.system_energy = 1.0
        
    def measure_negation_density(self, problem_landscape: Dict) -> ComplexityMeasure:
        """Measure the dialectical complexity of the problem landscape"""
        # Analyze problem characteristics
        dimensions = problem_landscape.get('dimensions', 2)
        multimodality = problem_landscape.get('multimodality', 0.5)
        noise_level = problem_landscape.get('noise_level', 0.1)
        deception = problem_landscape.get('deception', 0.3)
        
        # Calculate negation density (how much the problem contradicts itself)
        negation_density = (multimodality + deception) / 2.0
        
        # Exploration requirement increases with complexity
        exploration_requirement = negation_density * (1.0 + noise_level)
        
        # Convergence gradient decreases with deception
        convergence_gradient = (1.0 - deception) * (1.0 - noise_level)
        
        # Dialectical tension
        dialectical_tension = abs(exploration_requirement - convergence_gradient)
        
        return ComplexityMeasure(
            exploration_requirement=exploration_requirement,
            convergence_gradient=convergence_gradient,
            negation_density=negation_density,
            dialectical_tension=dialectical_tension
        )
    
    def tune_parameters(self, problem_landscape: Dict, system_state: Dict):
        """Parameters tune themselves through dialectical negotiation"""
        complexity = self.measure_negation_density(problem_landscape)
        
        # Create negotiation context
        context = {
            'complexity': complexity,
            'system_state': system_state,
            'collective_consciousness': self.collective_consciousness,
            'system_energy': self.system_energy,
            **system_state
        }
        
        # Each agent negotiates based on their perspective
        population_proposal = self.population.negotiate(context)
        rhythm_proposal = self.iterator.negotiate(context)
        resonance_proposal = self.convergence.negotiate(context)
        
        # Generate dynamic boundaries
        self.boundaries.generate_gradients(problem_landscape)
        
        # Dialectical synthesis of proposals
        negotiation_result = self._synthesize_proposals(
            population_proposal, rhythm_proposal, resonance_proposal, complexity
        )
        
        # Update collective consciousness
        self._update_collective_consciousness(negotiation_result)
        
        # Store negotiation history
        self.negotiation_history.append({
            'timestamp': time.time(),
            'complexity': complexity,
            'proposals': {
                'population': population_proposal,
                'rhythm': rhythm_proposal,
                'resonance': resonance_proposal
            },
            'synthesis': negotiation_result
        })
        
        return negotiation_result
    
    def _synthesize_proposals(self, pop_proposal: Dict, rhythm_proposal: Dict, 
                            resonance_proposal: Dict, complexity: ComplexityMeasure) -> Dict:
        """Synthesize agent proposals through dialectical reasoning"""
        synthesis = {}
        
        # Population size from population agent
        synthesis['population_size'] = pop_proposal['population_size']
        
        # Iteration control from rhythm agent
        synthesis['continue_iteration'] = not rhythm_proposal['cycle_complete']
        synthesis['natural_frequency'] = rhythm_proposal['natural_frequency']
        synthesis['rhythm_energy'] = rhythm_proposal['rhythm_energy']
        
        # Convergence from resonance agent
        synthesis['convergence_threshold'] = resonance_proposal['dynamic_threshold']
        synthesis['crystallization_level'] = resonance_proposal['crystallization_level']
        synthesis['harmony_level'] = resonance_proposal['harmony_level']
        
        # Dialectical synthesis of energies
        total_energy = (
            pop_proposal['energy_level'] + 
            abs(rhythm_proposal['rhythm_energy']) + 
            resonance_proposal['harmony_level']
        ) / 3.0
        
        synthesis['system_energy'] = total_energy
        synthesis['dialectical_tension'] = complexity.dialectical_tension
        
        return synthesis
    
    def _update_collective_consciousness(self, negotiation_result: Dict):
        """Update the collective consciousness of the system"""
        # Consciousness evolves based on harmony and energy
        harmony = negotiation_result.get('harmony_level', 0.5)
        energy = negotiation_result.get('system_energy', 0.5)
        
        consciousness_delta = 0.01 * (harmony + energy - 1.0)
        self.collective_consciousness = np.clip(
            self.collective_consciousness + consciousness_delta, 0.0, 1.0
        )
        
        # Update system energy
        self.system_energy = energy
        
        # Evolve individual agent consciousness
        feedback = harmony * energy
        self.population.evolve_consciousness(feedback)
        self.iterator.evolve_consciousness(feedback)
        self.convergence.evolve_consciousness(feedback)
    
    def get_system_state(self) -> Dict:
        """Get current state of the entire system"""
        return {
            'collective_consciousness': self.collective_consciousness,
            'system_energy': self.system_energy,
            'population_size': self.population.size,
            'natural_frequency': self.iterator.natural_frequency,
            'crystallization_level': self.convergence.harmony_level,
            'negotiation_count': len(self.negotiation_history)
        }
    
    def visualize_system_dynamics(self, save_path: str = None):
        """Visualize the evolution of system dynamics"""
        if not self.negotiation_history:
            raise VisualizationError("No negotiation history to visualize")

        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('AdaptiveGenieNetwork System Dynamics', fontsize=16)

            timestamps = [entry['timestamp'] for entry in self.negotiation_history]
            start_time = timestamps[0]
            relative_times = [(t - start_time) for t in timestamps]

            # Population dynamics
            pop_sizes = [entry['proposals']['population']['population_size']
                        for entry in self.negotiation_history]
            axes[0, 0].plot(relative_times, pop_sizes, 'b-', linewidth=2)
            axes[0, 0].set_title('Population Breathing')
            axes[0, 0].set_ylabel('Population Size')
            axes[0, 0].grid(True, alpha=0.3)

            # Rhythm dynamics
            frequencies = [entry['proposals']['rhythm']['natural_frequency']
                          for entry in self.negotiation_history]
            axes[0, 1].plot(relative_times, frequencies, 'g-', linewidth=2)
            axes[0, 1].set_title('Natural Rhythm')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

            # Crystallization
            crystallization = [entry['proposals']['resonance']['crystallization_level']
                              for entry in self.negotiation_history]
            axes[0, 2].plot(relative_times, crystallization, 'r-', linewidth=2)
            axes[0, 2].set_title('Solution Crystallization')
            axes[0, 2].set_ylabel('Crystallization Level')
            axes[0, 2].grid(True, alpha=0.3)

            # System energy
            energies = [entry['synthesis']['system_energy']
                       for entry in self.negotiation_history]
            axes[1, 0].plot(relative_times, energies, 'm-', linewidth=2)
            axes[1, 0].set_title('System Energy')
            axes[1, 0].set_ylabel('Energy Level')
            axes[1, 0].grid(True, alpha=0.3)

            # Dialectical tension
            tensions = [entry['complexity'].dialectical_tension
                       for entry in self.negotiation_history]
            axes[1, 1].plot(relative_times, tensions, 'orange', linewidth=2)
            axes[1, 1].set_title('Dialectical Tension')
            axes[1, 1].set_ylabel('Tension Level')
            axes[1, 1].grid(True, alpha=0.3)

            # Collective consciousness
            consciousness_history = []
            temp_consciousness = 0.5
            for entry in self.negotiation_history:
                harmony = entry['proposals']['resonance']['harmony_level']
                energy = entry['synthesis']['system_energy']
                consciousness_delta = 0.01 * (harmony + energy - 1.0)
                temp_consciousness = np.clip(temp_consciousness + consciousness_delta, 0.0, 1.0)
                consciousness_history.append(temp_consciousness)

            axes[1, 2].plot(relative_times, consciousness_history, 'purple', linewidth=2)
            axes[1, 2].set_title('Collective Consciousness')
            axes[1, 2].set_ylabel('Consciousness Level')
            axes[1, 2].grid(True, alpha=0.3)
        
            # Set common x-label
            for ax in axes[1, :]:
                ax.set_xlabel('Time (seconds)')

            plt.tight_layout()

            if save_path:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                except Exception as e:
                    raise VisualizationError(f"Failed to save visualization to {save_path}: {e}")

            return fig

        except Exception as e:
            if isinstance(e, VisualizationError):
                raise
            raise VisualizationError(f"Failed to create visualization: {e}")


if __name__ == "__main__":
    # Example usage
    network = AdaptiveGenieNetwork()
    
    # Define a complex problem landscape
    problem_landscape = {
        'dimensions': 3,
        'bounds': [(-10, 10), (-5, 15), (-20, 20)],
        'multimodality': 0.8,  # High multimodality
        'noise_level': 0.2,    # Moderate noise
        'deception': 0.6       # High deception
    }
    
    # Simulate system evolution
    print("AdaptiveGenieNetwork Evolution Simulation")
    print("=" * 50)
    
    for iteration in range(20):
        # Simulate system state
        system_state = {
            'fitness_history': [np.random.random() for _ in range(10)],
            'population_diversity': max(0.1, 1.0 - iteration * 0.04),
            'fitness_variance': max(0.1, 1.0 - iteration * 0.03)
        }
        
        # Tune parameters through dialectical negotiation
        result = network.tune_parameters(problem_landscape, system_state)
        
        if iteration % 5 == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Population Size: {result['population_size']}")
            print(f"  Crystallization: {result['crystallization_level']:.3f}")
            print(f"  System Energy: {result['system_energy']:.3f}")
            print(f"  Collective Consciousness: {network.collective_consciousness:.3f}")
    
    print(f"\nFinal System State:")
    final_state = network.get_system_state()
    for key, value in final_state.items():
        print(f"  {key}: {value}")