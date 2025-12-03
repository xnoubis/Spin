"""
Example Applications for AdaptiveGenieNetwork
============================================

This module demonstrates the AdaptiveGenieNetwork framework applied to various
optimization problems, showcasing its dialectical approach to parameter adaptation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import time
from dataclasses import dataclass

from adaptive_genie_network import AdaptiveGenieNetwork, ComplexityMeasure
from mathematical_models import (
    NegationDensityCalculator,
    ResonanceCalculator,
    DialecticalSynthesis,
    GradientFieldMathematics
)

# Public API exports
__all__ = [
    "OptimizationResult",
    "DialecticalParticleSwarm",
    "DialecticalGeneticAlgorithm",
    "rastrigin",
    "rosenbrock",
    "ackley",
    "schwefel",
    "visualize_optimization_dynamics",
    "run_comparative_study",
]


@dataclass
class OptimizationResult:
    """Results from an optimization run"""
    best_solution: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    population_history: List[int]
    crystallization_history: List[float]
    dialectical_states: List[Dict]
    total_evaluations: int
    execution_time: float


class DialecticalParticleSwarm:
    """
    Particle Swarm Optimization enhanced with dialectical parameter adaptation
    """
    
    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]]):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimensions = len(bounds)
        
        # Initialize AdaptiveGenieNetwork
        self.genie_network = AdaptiveGenieNetwork()
        
        # PSO parameters (will be adapted dialectically)
        self.inertia = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        
        # Population
        self.particles = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')
        
        # History tracking
        self.fitness_history = []
        self.population_size_history = []
        self.crystallization_history = []
        self.dialectical_history = []
        
    def optimize(self, max_iterations: int = 100) -> OptimizationResult:
        """Run dialectical particle swarm optimization"""
        start_time = time.time()
        total_evaluations = 0
        
        # Define problem landscape for the genie network
        problem_landscape = {
            'dimensions': self.dimensions,
            'bounds': self.bounds,
            'multimodality': 0.7,  # Assume moderate multimodality
            'noise_level': 0.1,
            'deception': 0.5
        }
        
        # Initialize with adaptive population size
        initial_system_state = {
            'fitness_history': [],
            'population_diversity': 1.0,
            'fitness_variance': 1.0
        }
        
        # Get initial parameters from genie network
        initial_params = self.genie_network.tune_parameters(problem_landscape, initial_system_state)
        population_size = initial_params['population_size']
        
        # Initialize population
        self._initialize_population(population_size)
        total_evaluations += population_size
        
        print(f"Starting Dialectical PSO with {population_size} particles")
        
        for iteration in range(max_iterations):
            # Update system state for genie network
            system_state = {
                'fitness_history': self.fitness_history[-50:],  # Recent history
                'population_diversity': self._calculate_diversity(),
                'fitness_variance': np.var(self.fitness_history[-20:]) if len(self.fitness_history) >= 20 else 1.0
            }
            
            # Get adaptive parameters from genie network
            adaptive_params = self.genie_network.tune_parameters(problem_landscape, system_state)
            
            # Adapt population size if needed
            new_population_size = adaptive_params['population_size']
            if new_population_size != len(self.particles):
                self._adapt_population_size(new_population_size)
                total_evaluations += abs(new_population_size - len(self.particles))
            
            # Adapt PSO parameters based on dialectical synthesis
            self._adapt_pso_parameters(adaptive_params)
            
            # Update particles
            for i in range(len(self.particles)):
                # Update velocity with dialectical coefficients
                self.velocities[i] = (
                    self.inertia * self.velocities[i] +
                    self.cognitive_coeff * np.random.random() * (self.personal_best[i] - self.particles[i]) +
                    self.social_coeff * np.random.random() * (self.global_best - self.particles[i])
                )
                
                # Apply gradient field influence
                field_force = self.genie_network.boundaries.apply_field_force(
                    self.particles[i], self.velocities[i]
                )
                self.velocities[i] = field_force
                
                # Update position
                self.particles[i] += self.velocities[i]
                
                # Apply bounds
                self.particles[i] = np.clip(
                    self.particles[i], 
                    [bound[0] for bound in self.bounds],
                    [bound[1] for bound in self.bounds]
                )
                
                # Evaluate fitness
                fitness = self.objective_function(self.particles[i])
                total_evaluations += 1
                
                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best[i] = self.particles[i].copy()
                    self.personal_best_fitness[i] = fitness
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best = self.particles[i].copy()
                    self.global_best_fitness = fitness
            
            # Record history
            self.fitness_history.append(self.global_best_fitness)
            self.population_size_history.append(len(self.particles))
            self.crystallization_history.append(adaptive_params['crystallization_level'])
            self.dialectical_history.append(adaptive_params)
            
            # Check for natural cycle completion
            if not adaptive_params['continue_iteration'] and iteration > 20:
                print(f"Natural cycle completed at iteration {iteration}")
                break
            
            # Progress report
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.6f}, "
                      f"Population = {len(self.particles)}, "
                      f"Crystallization = {adaptive_params['crystallization_level']:.3f}")
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            best_solution=self.global_best,
            best_fitness=self.global_best_fitness,
            convergence_history=self.fitness_history,
            population_history=self.population_size_history,
            crystallization_history=self.crystallization_history,
            dialectical_states=self.dialectical_history,
            total_evaluations=total_evaluations,
            execution_time=execution_time
        )
    
    def _initialize_population(self, population_size: int):
        """Initialize particle population"""
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_fitness = []
        
        for _ in range(population_size):
            # Random position within bounds
            particle = np.array([
                np.random.uniform(bound[0], bound[1]) for bound in self.bounds
            ])
            
            # Random velocity
            velocity = np.array([
                np.random.uniform(-1, 1) * (bound[1] - bound[0]) * 0.1 
                for bound in self.bounds
            ])
            
            # Evaluate initial fitness
            fitness = self.objective_function(particle)
            
            self.particles.append(particle)
            self.velocities.append(velocity)
            self.personal_best.append(particle.copy())
            self.personal_best_fitness.append(fitness)
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best = particle.copy()
                self.global_best_fitness = fitness
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.particles) < 2:
            return 1.0
        
        distances = []
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                dist = np.linalg.norm(self.particles[i] - self.particles[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 1.0
    
    def _adapt_population_size(self, new_size: int):
        """Adapt population size dynamically"""
        current_size = len(self.particles)
        
        if new_size > current_size:
            # Add new particles
            for _ in range(new_size - current_size):
                # Create new particle near best solutions
                if np.random.random() < 0.7:  # 70% near best
                    base_particle = self.global_best + np.random.normal(0, 0.1, self.dimensions)
                else:  # 30% random
                    base_particle = np.array([
                        np.random.uniform(bound[0], bound[1]) for bound in self.bounds
                    ])
                
                # Apply bounds
                base_particle = np.clip(
                    base_particle,
                    [bound[0] for bound in self.bounds],
                    [bound[1] for bound in self.bounds]
                )
                
                velocity = np.random.uniform(-0.1, 0.1, self.dimensions)
                fitness = self.objective_function(base_particle)
                
                self.particles.append(base_particle)
                self.velocities.append(velocity)
                self.personal_best.append(base_particle.copy())
                self.personal_best_fitness.append(fitness)
                
                if fitness < self.global_best_fitness:
                    self.global_best = base_particle.copy()
                    self.global_best_fitness = fitness
        
        elif new_size < current_size:
            # Remove worst particles
            indices_to_remove = np.argsort(self.personal_best_fitness)[new_size:]
            
            for idx in sorted(indices_to_remove, reverse=True):
                del self.particles[idx]
                del self.velocities[idx]
                del self.personal_best[idx]
                del self.personal_best_fitness[idx]
    
    def _adapt_pso_parameters(self, adaptive_params: Dict):
        """Adapt PSO parameters based on dialectical synthesis"""
        # Adapt inertia based on crystallization
        crystallization = adaptive_params['crystallization_level']
        self.inertia = 0.9 * (1.0 - crystallization) + 0.4 * crystallization
        
        # Adapt coefficients based on system energy and rhythm
        system_energy = adaptive_params['system_energy']
        rhythm_energy = adaptive_params['rhythm_energy']
        
        # Higher energy = more exploration (higher coefficients)
        base_coeff = 1.0 + system_energy
        
        # Rhythm affects balance between cognitive and social
        if rhythm_energy > 0:
            self.cognitive_coeff = base_coeff * (1.0 + 0.5 * rhythm_energy)
            self.social_coeff = base_coeff * (1.0 - 0.3 * rhythm_energy)
        else:
            self.cognitive_coeff = base_coeff * (1.0 - 0.3 * abs(rhythm_energy))
            self.social_coeff = base_coeff * (1.0 + 0.5 * abs(rhythm_energy))


class DialecticalGeneticAlgorithm:
    """
    Genetic Algorithm with dialectical parameter adaptation
    """
    
    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]]):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimensions = len(bounds)
        
        # Initialize AdaptiveGenieNetwork
        self.genie_network = AdaptiveGenieNetwork()
        
        # GA parameters (will be adapted)
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.selection_pressure = 2.0
        
        # Population
        self.population = None
        self.fitness_values = None
        self.best_individual = None
        self.best_fitness = float('inf')
        
        # History
        self.fitness_history = []
        self.diversity_history = []
        self.parameter_history = []
    
    def optimize(self, max_generations: int = 100) -> OptimizationResult:
        """Run dialectical genetic algorithm"""
        start_time = time.time()
        total_evaluations = 0
        
        # Problem landscape
        problem_landscape = {
            'dimensions': self.dimensions,
            'bounds': self.bounds,
            'multimodality': 0.8,
            'noise_level': 0.15,
            'deception': 0.6
        }
        
        # Initialize population
        initial_params = self.genie_network.tune_parameters(problem_landscape, {
            'fitness_history': [],
            'population_diversity': 1.0,
            'fitness_variance': 1.0
        })
        
        population_size = initial_params['population_size']
        self._initialize_population(population_size)
        total_evaluations += population_size
        
        print(f"Starting Dialectical GA with {population_size} individuals")
        
        for generation in range(max_generations):
            # Calculate system state
            diversity = self._calculate_diversity()
            system_state = {
                'fitness_history': self.fitness_history[-50:],
                'population_diversity': diversity,
                'fitness_variance': np.var(self.fitness_values) if len(self.fitness_values) > 1 else 1.0
            }
            
            # Get adaptive parameters
            adaptive_params = self.genie_network.tune_parameters(problem_landscape, system_state)
            
            # Adapt GA parameters
            self._adapt_ga_parameters(adaptive_params)
            
            # Adapt population size
            new_size = adaptive_params['population_size']
            if new_size != len(self.population):
                self._adapt_population_size(new_size)
                total_evaluations += abs(new_size - len(self.population))
            
            # Selection
            parents = self._selection()
            
            # Crossover and Mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    offspring.extend([child1, child2])
                else:
                    child = self._mutation(parents[i].copy())
                    offspring.append(child)
            
            # Evaluate offspring
            offspring_fitness = []
            for individual in offspring:
                fitness = self.objective_function(individual)
                offspring_fitness.append(fitness)
                total_evaluations += 1
                
                if fitness < self.best_fitness:
                    self.best_individual = individual.copy()
                    self.best_fitness = fitness
            
            # Replacement (elitist)
            combined_population = list(self.population) + offspring
            combined_fitness = list(self.fitness_values) + offspring_fitness
            
            # Sort by fitness and keep best
            sorted_indices = np.argsort(combined_fitness)
            self.population = [combined_population[i] for i in sorted_indices[:len(self.population)]]
            self.fitness_values = [combined_fitness[i] for i in sorted_indices[:len(self.population)]]
            
            # Record history
            self.fitness_history.append(self.best_fitness)
            self.diversity_history.append(diversity)
            self.parameter_history.append(adaptive_params)
            
            # Check for natural stopping
            if not adaptive_params['continue_iteration'] and generation > 20:
                print(f"Natural cycle completed at generation {generation}")
                break
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.6f}, "
                      f"Population = {len(self.population)}, "
                      f"Diversity = {diversity:.3f}")
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            best_solution=self.best_individual,
            best_fitness=self.best_fitness,
            convergence_history=self.fitness_history,
            population_history=[len(self.population)] * len(self.fitness_history),
            crystallization_history=[p['crystallization_level'] for p in self.parameter_history],
            dialectical_states=self.parameter_history,
            total_evaluations=total_evaluations,
            execution_time=execution_time
        )
    
    def _initialize_population(self, population_size: int):
        """Initialize GA population"""
        self.population = []
        self.fitness_values = []
        
        for _ in range(population_size):
            individual = np.array([
                np.random.uniform(bound[0], bound[1]) for bound in self.bounds
            ])
            fitness = self.objective_function(individual)
            
            self.population.append(individual)
            self.fitness_values.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_individual = individual.copy()
                self.best_fitness = fitness
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 1.0
        
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.linalg.norm(self.population[i] - self.population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 1.0
    
    def _adapt_population_size(self, new_size: int):
        """Adapt population size"""
        current_size = len(self.population)
        
        if new_size > current_size:
            # Add new individuals
            for _ in range(new_size - current_size):
                if np.random.random() < 0.5:  # 50% near best
                    base = self.best_individual + np.random.normal(0, 0.2, self.dimensions)
                else:  # 50% random
                    base = np.array([
                        np.random.uniform(bound[0], bound[1]) for bound in self.bounds
                    ])
                
                base = np.clip(base, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
                fitness = self.objective_function(base)
                
                self.population.append(base)
                self.fitness_values.append(fitness)
                
                if fitness < self.best_fitness:
                    self.best_individual = base.copy()
                    self.best_fitness = fitness
        
        elif new_size < current_size:
            # Remove worst individuals
            sorted_indices = np.argsort(self.fitness_values)
            keep_indices = sorted_indices[:new_size]
            
            self.population = [self.population[i] for i in keep_indices]
            self.fitness_values = [self.fitness_values[i] for i in keep_indices]
    
    def _adapt_ga_parameters(self, adaptive_params: Dict):
        """Adapt GA parameters"""
        crystallization = adaptive_params['crystallization_level']
        system_energy = adaptive_params['system_energy']
        
        # Higher crystallization = lower mutation rate
        self.mutation_rate = 0.2 * (1.0 - crystallization) + 0.01 * crystallization
        
        # System energy affects crossover rate
        self.crossover_rate = 0.6 + 0.3 * system_energy
        
        # Selection pressure adapts to harmony
        harmony = adaptive_params.get('harmony_level', 0.5)
        self.selection_pressure = 1.5 + 1.0 * harmony
    
    def _selection(self) -> List[np.ndarray]:
        """Tournament selection"""
        parents = []
        tournament_size = max(2, int(self.selection_pressure))
        
        for _ in range(len(self.population)):
            # Tournament
            tournament_indices = np.random.choice(
                len(self.population), tournament_size, replace=False
            )
            tournament_fitness = [self.fitness_values[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(self.population[winner_idx].copy())
        
        return parents
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        eta = 2.0  # Distribution index
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    
                    # Calculate beta
                    rand = np.random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
                    
                    # Create children
                    child1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Apply bounds
                    child1[i] = np.clip(child1[i], self.bounds[i][0], self.bounds[i][1])
                    child2[i] = np.clip(child2[i], self.bounds[i][0], self.bounds[i][1])
        
        return child1, child2
    
    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
        eta = 20.0  # Distribution index
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() <= self.mutation_rate:
                y = individual[i]
                yl, yu = self.bounds[i]
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (yu - yl)
                mutated[i] = np.clip(y, yl, yu)
        
        return mutated


# Test Functions
def rastrigin(x):
    """Rastrigin function - highly multimodal"""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def rosenbrock(x):
    """Rosenbrock function - narrow valley"""
    return sum([100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)])

def ackley(x):
    """Ackley function - many local minima"""
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([np.cos(c * xi) for xi in x])
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

def schwefel(x):
    """Schwefel function - deceptive global optimum"""
    return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


def run_comparative_study():
    """Run comparative study between dialectical and standard algorithms"""
    print("Comparative Study: Dialectical vs Standard Optimization")
    print("=" * 60)
    
    # Test functions
    test_functions = {
        'Rastrigin': (rastrigin, [(-5.12, 5.12)] * 10),
        'Rosenbrock': (rosenbrock, [(-2.048, 2.048)] * 10),
        'Ackley': (ackley, [(-32.768, 32.768)] * 10),
        'Schwefel': (schwefel, [(-500, 500)] * 10)
    }
    
    results = {}
    
    for func_name, (func, bounds) in test_functions.items():
        print(f"\nTesting {func_name} function...")
        
        # Dialectical PSO
        dpso = DialecticalParticleSwarm(func, bounds)
        dpso_result = dpso.optimize(max_iterations=100)
        
        # Dialectical GA
        dga = DialecticalGeneticAlgorithm(func, bounds)
        dga_result = dga.optimize(max_generations=100)
        
        results[func_name] = {
            'DPSO': dpso_result,
            'DGA': dga_result
        }
        
        print(f"  DPSO: Best = {dpso_result.best_fitness:.6f}, "
              f"Evaluations = {dpso_result.total_evaluations}, "
              f"Time = {dpso_result.execution_time:.2f}s")
        print(f"  DGA:  Best = {dga_result.best_fitness:.6f}, "
              f"Evaluations = {dga_result.total_evaluations}, "
              f"Time = {dga_result.execution_time:.2f}s")
    
    return results


def visualize_optimization_dynamics(result: OptimizationResult, title: str):
    """Visualize the dynamics of dialectical optimization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Dialectical Optimization Dynamics: {title}', fontsize=14)
    
    # Convergence history
    axes[0, 0].semilogy(result.convergence_history)
    axes[0, 0].set_title('Convergence History')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best Fitness (log scale)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Population dynamics
    axes[0, 1].plot(result.population_history, 'g-', linewidth=2)
    axes[0, 1].set_title('Population Breathing')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Population Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Crystallization
    axes[1, 0].plot(result.crystallization_history, 'r-', linewidth=2)
    axes[1, 0].set_title('Solution Crystallization')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Crystallization Level')
    axes[1, 0].grid(True, alpha=0.3)
    
    # System energy evolution
    if result.dialectical_states:
        energies = [state.get('system_energy', 0.5) for state in result.dialectical_states]
        axes[1, 1].plot(energies, 'm-', linewidth=2)
        axes[1, 1].set_title('System Energy')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Energy Level')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Run example applications
    print("AdaptiveGenieNetwork Example Applications")
    print("=" * 50)
    
    # Single function test
    print("\nTesting Rastrigin function with Dialectical PSO...")
    dpso = DialecticalParticleSwarm(rastrigin, [(-5.12, 5.12)] * 5)
    result = dpso.optimize(max_iterations=50)
    
    print(f"Best solution: {result.best_solution}")
    print(f"Best fitness: {result.best_fitness:.6f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    
    # Visualize dynamics
    fig = visualize_optimization_dynamics(result, "Rastrigin Function")
    plt.savefig("dialectical_pso_dynamics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Run comparative study
    # comparative_results = run_comparative_study()