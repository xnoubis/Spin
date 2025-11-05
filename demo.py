"""
AdaptiveGenieNetwork Demonstration
=================================

This script provides a comprehensive demonstration of the AdaptiveGenieNetwork
framework, showcasing its dialectical optimization capabilities across various
problem types and scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List

# Import our framework components
from adaptive_genie_network import AdaptiveGenieNetwork
from mathematical_models import (
    NegationDensityCalculator, 
    ResonanceCalculator, 
    DialecticalSynthesis
)
from example_applications import (
    DialecticalParticleSwarm, 
    DialecticalGeneticAlgorithm,
    rastrigin, rosenbrock, ackley, schwefel
)
from visualization_tools import create_comprehensive_report


def demonstrate_basic_framework():
    """Demonstrate the basic AdaptiveGenieNetwork framework"""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Basic AdaptiveGenieNetwork Framework")
    print("="*60)
    
    # Initialize the network
    network = AdaptiveGenieNetwork()
    
    print("🧠 Initializing AdaptiveGenieNetwork...")
    print(f"   Initial collective consciousness: {network.collective_consciousness:.3f}")
    print(f"   Initial system energy: {network.system_energy:.3f}")
    
    # Define various problem landscapes to test adaptability
    problem_landscapes = [
        {
            'name': 'Simple Landscape',
            'dimensions': 2,
            'bounds': [(-5, 5), (-5, 5)],
            'multimodality': 0.3,
            'noise_level': 0.1,
            'deception': 0.2
        },
        {
            'name': 'Complex Multimodal',
            'dimensions': 5,
            'bounds': [(-10, 10)] * 5,
            'multimodality': 0.8,
            'noise_level': 0.2,
            'deception': 0.6
        },
        {
            'name': 'Highly Deceptive',
            'dimensions': 3,
            'bounds': [(-20, 20)] * 3,
            'multimodality': 0.6,
            'noise_level': 0.3,
            'deception': 0.9
        }
    ]
    
    print("\n🔄 Testing adaptability across different problem landscapes...")
    
    for i, landscape in enumerate(problem_landscapes):
        print(f"\n--- Testing {landscape['name']} ---")
        
        # Simulate optimization progress
        for iteration in range(15):
            # Create evolving system state
            diversity = max(0.1, 1.0 - iteration * 0.05)
            variance = max(0.05, 1.0 - iteration * 0.04)
            fitness_history = [1.0 - i * 0.1 + np.random.random() * 0.1 for i in range(min(iteration + 1, 10))]
            
            system_state = {
                'fitness_history': fitness_history,
                'population_diversity': diversity,
                'fitness_variance': variance
            }
            
            # Get adaptive parameters through dialectical negotiation
            result = network.tune_parameters(landscape, system_state)
            
            if iteration % 5 == 0:
                print(f"   Iteration {iteration:2d}: Pop={result['population_size']:3d}, "
                      f"Crystal={result['crystallization_level']:.3f}, "
                      f"Energy={result['system_energy']:.3f}")
        
        print(f"   Final consciousness: {network.collective_consciousness:.3f}")
    
    print(f"\n✨ Framework demonstration complete!")
    print(f"   Total negotiations: {len(network.negotiation_history)}")
    print(f"   Final collective consciousness: {network.collective_consciousness:.3f}")
    
    return network


def demonstrate_mathematical_models():
    """Demonstrate the mathematical foundations"""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Mathematical Models")
    print("="*60)
    
    # Test Negation Density Calculator
    print("\n🧮 Testing Negation Density Calculator...")
    
    def test_function(x):
        """A complex test function with multiple characteristics"""
        return -(x[0]**2 + x[1]**2) + 0.5 * np.sin(5 * x[0]) * np.cos(5 * x[1]) + 0.1 * np.random.randn()
    
    negation_calc = NegationDensityCalculator()
    bounds = [(-3, 3), (-3, 3)]
    
    print("   Analyzing test function landscape...")
    negation_metrics = negation_calc.calculate_negation_density(test_function, bounds, sample_points=200)
    
    print("   Results:")
    for key, value in negation_metrics.items():
        print(f"     {key}: {value:.4f}")
    
    # Test Resonance Calculator
    print("\n🎵 Testing Resonance Calculator...")
    
    resonance_calc = ResonanceCalculator()
    
    # Generate test signal with known patterns
    t = np.linspace(0, 10, 100)
    test_signal = (np.sin(2 * np.pi * 0.5 * t) + 
                  0.3 * np.sin(2 * np.pi * 1.5 * t) + 
                  0.1 * np.random.randn(100))
    
    resonance_metrics = resonance_calc.calculate_resonance_frequency(test_signal)
    
    print("   Signal analysis results:")
    for key, value in resonance_metrics.items():
        print(f"     {key}: {value:.4f}")
    
    # Test Dialectical Synthesis
    print("\n⚖️  Testing Dialectical Synthesis...")
    
    synthesis = DialecticalSynthesis()
    
    # Create thesis and antithesis
    thesis = np.array([2.0, -1.0, 3.0])
    antithesis = np.array([-1.0, 2.0, -2.0])
    
    print(f"   Thesis: {thesis}")
    print(f"   Antithesis: {antithesis}")
    
    dialectical_state = synthesis.perform_synthesis(thesis, antithesis)
    
    print(f"   Synthesis: {dialectical_state.synthesis}")
    print(f"   Tension: {dialectical_state.tension:.4f}")
    print(f"   Resolution Energy: {dialectical_state.resolution_energy:.4f}")
    
    # Perform multiple syntheses to show evolution
    print("\n   Performing synthesis evolution...")
    for i in range(5):
        new_antithesis = dialectical_state.synthesis + np.random.normal(0, 0.5, 3)
        dialectical_state = synthesis.perform_synthesis(dialectical_state.synthesis, new_antithesis)
        print(f"   Step {i+1}: Synthesis={dialectical_state.synthesis}, Tension={dialectical_state.tension:.3f}")
    
    return negation_metrics, resonance_metrics, synthesis


def demonstrate_optimization_algorithms():
    """Demonstrate dialectical optimization algorithms"""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Dialectical Optimization Algorithms")
    print("="*60)
    
    # Test functions with different characteristics
    test_functions = {
        'Rastrigin (Multimodal)': (rastrigin, [(-5.12, 5.12)] * 3),
        'Rosenbrock (Valley)': (rosenbrock, [(-2.048, 2.048)] * 3),
        'Ackley (Many Minima)': (ackley, [(-32.768, 32.768)] * 3)
    }
    
    results = {}
    
    for func_name, (func, bounds) in test_functions.items():
        print(f"\n🎯 Testing {func_name}...")
        
        # Test Dialectical Particle Swarm
        print("   Running Dialectical PSO...")
        dpso = DialecticalParticleSwarm(func, bounds)
        dpso_result = dpso.optimize(max_iterations=50)
        
        print(f"     Best fitness: {dpso_result.best_fitness:.6f}")
        print(f"     Evaluations: {dpso_result.total_evaluations}")
        print(f"     Time: {dpso_result.execution_time:.2f}s")
        print(f"     Final population: {dpso_result.population_history[-1]}")
        
        # Test Dialectical Genetic Algorithm
        print("   Running Dialectical GA...")
        dga = DialecticalGeneticAlgorithm(func, bounds)
        dga_result = dga.optimize(max_generations=50)
        
        print(f"     Best fitness: {dga_result.best_fitness:.6f}")
        print(f"     Evaluations: {dga_result.total_evaluations}")
        print(f"     Time: {dga_result.execution_time:.2f}s")
        
        results[func_name] = {
            'DPSO': dpso_result,
            'DGA': dga_result
        }
        
        # Show adaptation behavior
        if dpso_result.dialectical_states:
            print("   Adaptation behavior (DPSO):")
            initial_state = dpso_result.dialectical_states[0]
            final_state = dpso_result.dialectical_states[-1]
            
            print(f"     Initial: Pop={initial_state.get('population_size', 'N/A')}, "
                  f"Crystal={initial_state.get('crystallization_level', 0):.3f}")
            print(f"     Final:   Pop={final_state.get('population_size', 'N/A')}, "
                  f"Crystal={final_state.get('crystallization_level', 0):.3f}")
    
    return results


def demonstrate_real_time_adaptation():
    """Demonstrate real-time parameter adaptation"""
    print("\n" + "="*60)
    print("DEMONSTRATION 4: Real-Time Parameter Adaptation")
    print("="*60)
    
    print("🔄 Simulating dynamic optimization environment...")
    
    # Create a time-varying optimization problem
    def dynamic_rastrigin(x, t=0):
        """Rastrigin function that changes over time"""
        A = 10 + 5 * np.sin(t * 0.1)  # Varying amplitude
        shift = np.sin(t * 0.05) * 2   # Shifting optimum
        return A * len(x) + sum([(xi - shift)**2 - A * np.cos(2 * np.pi * (xi - shift)) for xi in x])
    
    # Initialize dialectical PSO
    bounds = [(-5.12, 5.12)] * 2
    dpso = DialecticalParticleSwarm(lambda x: dynamic_rastrigin(x, 0), bounds)
    
    print("   Tracking adaptation over time...")
    
    adaptation_history = []
    time_steps = 20
    
    for t in range(time_steps):
        # Update objective function
        current_time = t * 2.0
        dpso.objective_function = lambda x: dynamic_rastrigin(x, current_time)
        
        # Run a few iterations
        if t == 0:
            # Initial optimization
            result = dpso.optimize(max_iterations=10)
        else:
            # Continue optimization with adapted parameters
            for _ in range(5):  # Mini-optimization steps
                # Simulate system state
                system_state = {
                    'fitness_history': dpso.fitness_history[-10:],
                    'population_diversity': dpso._calculate_diversity(),
                    'fitness_variance': np.var(dpso.fitness_history[-10:]) if len(dpso.fitness_history) >= 10 else 1.0
                }
                
                # Get adaptive parameters
                problem_landscape = {
                    'dimensions': 2,
                    'bounds': bounds,
                    'multimodality': 0.7 + 0.2 * np.sin(current_time * 0.1),
                    'noise_level': 0.1,
                    'deception': 0.5
                }
                
                adaptive_params = dpso.genie_network.tune_parameters(problem_landscape, system_state)
                
                # Record adaptation
                adaptation_history.append({
                    'time': current_time,
                    'population_size': adaptive_params['population_size'],
                    'crystallization': adaptive_params['crystallization_level'],
                    'system_energy': adaptive_params['system_energy'],
                    'best_fitness': dpso.global_best_fitness
                })
        
        if t % 5 == 0:
            current_params = adaptation_history[-1] if adaptation_history else {}
            print(f"   Time {current_time:4.1f}: "
                  f"Pop={current_params.get('population_size', 'N/A')}, "
                  f"Fitness={current_params.get('best_fitness', 'N/A'):.4f}, "
                  f"Energy={current_params.get('system_energy', 'N/A'):.3f}")
    
    print(f"\n   Adaptation complete! Tracked {len(adaptation_history)} adaptation steps.")
    
    # Show adaptation statistics
    if adaptation_history:
        pop_sizes = [entry['population_size'] for entry in adaptation_history]
        energies = [entry['system_energy'] for entry in adaptation_history]
        
        print(f"   Population size range: {min(pop_sizes)} - {max(pop_sizes)}")
        print(f"   System energy range: {min(energies):.3f} - {max(energies):.3f}")
        print(f"   Final collective consciousness: {dpso.genie_network.collective_consciousness:.3f}")
    
    return adaptation_history


def create_demonstration_visualizations():
    """Create visualizations for the demonstration"""
    print("\n" + "="*60)
    print("DEMONSTRATION 5: Visualization Generation")
    print("="*60)
    
    print("📊 Generating comprehensive visualizations...")
    
    # Run a complete optimization for visualization
    dpso = DialecticalParticleSwarm(rastrigin, [(-5.12, 5.12)] * 2)
    result = dpso.optimize(max_iterations=40)
    
    print(f"   Optimization completed: Best fitness = {result.best_fitness:.6f}")
    
    # Create output directory
    output_dir = "demo_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive report
    try:
        report_files = create_comprehensive_report(
            dpso.genie_network, result, output_dir
        )
        
        print(f"   ✅ Visualizations generated in: {output_dir}")
        print("   Generated files:")
        for name, path in report_files.items():
            print(f"     📄 {name}: {os.path.basename(path)}")
        
        return output_dir, report_files
    
    except Exception as e:
        print(f"   ⚠️  Visualization generation encountered an issue: {e}")
        print("   Continuing with demonstration...")
        return None, {}


def run_performance_comparison():
    """Compare dialectical vs standard optimization"""
    print("\n" + "="*60)
    print("DEMONSTRATION 6: Performance Comparison")
    print("="*60)
    
    print("⚡ Comparing dialectical optimization with standard approaches...")
    
    # Simple standard PSO for comparison
    class StandardPSO:
        def __init__(self, objective_function, bounds):
            self.objective_function = objective_function
            self.bounds = bounds
            self.dimensions = len(bounds)
            
        def optimize(self, max_iterations=50, population_size=30):
            start_time = time.time()
            
            # Initialize population
            particles = []
            velocities = []
            personal_best = []
            personal_best_fitness = []
            
            for _ in range(population_size):
                particle = np.array([
                    np.random.uniform(bound[0], bound[1]) for bound in self.bounds
                ])
                velocity = np.zeros(self.dimensions)
                fitness = self.objective_function(particle)
                
                particles.append(particle)
                velocities.append(velocity)
                personal_best.append(particle.copy())
                personal_best_fitness.append(fitness)
            
            global_best = personal_best[np.argmin(personal_best_fitness)].copy()
            global_best_fitness = min(personal_best_fitness)
            
            fitness_history = [global_best_fitness]
            
            # Fixed parameters
            inertia = 0.7
            cognitive = 1.5
            social = 1.5
            
            for iteration in range(max_iterations):
                for i in range(population_size):
                    # Update velocity
                    velocities[i] = (inertia * velocities[i] +
                                   cognitive * np.random.random() * (personal_best[i] - particles[i]) +
                                   social * np.random.random() * (global_best - particles[i]))
                    
                    # Update position
                    particles[i] += velocities[i]
                    
                    # Apply bounds
                    particles[i] = np.clip(particles[i], 
                                         [bound[0] for bound in self.bounds],
                                         [bound[1] for bound in self.bounds])
                    
                    # Evaluate fitness
                    fitness = self.objective_function(particles[i])
                    
                    # Update personal best
                    if fitness < personal_best_fitness[i]:
                        personal_best[i] = particles[i].copy()
                        personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
                
                fitness_history.append(global_best_fitness)
            
            execution_time = time.time() - start_time
            total_evaluations = population_size * (max_iterations + 1)
            
            return {
                'best_fitness': global_best_fitness,
                'total_evaluations': total_evaluations,
                'execution_time': execution_time,
                'convergence_history': fitness_history
            }
    
    # Test on Rastrigin function
    bounds = [(-5.12, 5.12)] * 3
    
    print("   Testing on Rastrigin function (3D)...")
    
    # Dialectical PSO
    print("     Running Dialectical PSO...")
    dpso = DialecticalParticleSwarm(rastrigin, bounds)
    dialectical_result = dpso.optimize(max_iterations=50)
    
    # Standard PSO
    print("     Running Standard PSO...")
    spso = StandardPSO(rastrigin, bounds)
    standard_result = spso.optimize(max_iterations=50, population_size=50)
    
    # Compare results
    print("\n   📈 Comparison Results:")
    print(f"     Dialectical PSO:")
    print(f"       Best fitness: {dialectical_result.best_fitness:.6f}")
    print(f"       Evaluations: {dialectical_result.total_evaluations}")
    print(f"       Time: {dialectical_result.execution_time:.2f}s")
    print(f"       Efficiency: {dialectical_result.best_fitness/dialectical_result.total_evaluations:.8f}")
    
    print(f"     Standard PSO:")
    print(f"       Best fitness: {standard_result['best_fitness']:.6f}")
    print(f"       Evaluations: {standard_result['total_evaluations']}")
    print(f"       Time: {standard_result['execution_time']:.2f}s")
    print(f"       Efficiency: {standard_result['best_fitness']/standard_result['total_evaluations']:.8f}")
    
    # Performance ratio
    fitness_ratio = dialectical_result.best_fitness / standard_result['best_fitness']
    efficiency_ratio = (dialectical_result.best_fitness/dialectical_result.total_evaluations) / (standard_result['best_fitness']/standard_result['total_evaluations'])
    
    print(f"\n   🎯 Performance Analysis:")
    print(f"     Fitness ratio (Dialectical/Standard): {fitness_ratio:.3f}")
    print(f"     Efficiency ratio: {efficiency_ratio:.3f}")
    
    if fitness_ratio < 1.0:
        print("     ✅ Dialectical approach found better solution!")
    if efficiency_ratio < 1.0:
        print("     ✅ Dialectical approach is more efficient!")
    
    return dialectical_result, standard_result


def main():
    """Main demonstration function"""
    print("🌟 AdaptiveGenieNetwork Comprehensive Demonstration")
    print("=" * 60)
    print("This demonstration showcases the revolutionary dialectical optimization framework")
    print("where parameters become autonomous agents engaged in dialectical negotiation.")
    print()
    
    start_time = time.time()
    
    try:
        # Run all demonstrations
        demo_results = {}
        
        # 1. Basic framework
        demo_results['network'] = demonstrate_basic_framework()
        
        # 2. Mathematical models
        demo_results['math_models'] = demonstrate_mathematical_models()
        
        # 3. Optimization algorithms
        demo_results['optimization'] = demonstrate_optimization_algorithms()
        
        # 4. Real-time adaptation
        demo_results['adaptation'] = demonstrate_real_time_adaptation()
        
        # 5. Visualizations
        demo_results['visualizations'] = create_demonstration_visualizations()
        
        # 6. Performance comparison
        demo_results['comparison'] = run_performance_comparison()
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("="*60)
        print(f"Total demonstration time: {total_time:.2f} seconds")
        print()
        print("Key Insights:")
        print("• Parameters as autonomous agents enable dynamic adaptation")
        print("• Dialectical negotiation leads to emergent optimization behavior")
        print("• Natural rhythm detection eliminates arbitrary stopping criteria")
        print("• Collective consciousness evolves throughout optimization")
        print("• System adapts to problem landscape characteristics automatically")
        print()
        
        if demo_results['visualizations'][0]:
            print(f"📊 Comprehensive visualizations available in: {demo_results['visualizations'][0]}")
            print("   Open index.html to explore the interactive dashboard")
        
        print()
        print("🚀 The AdaptiveGenieNetwork represents a paradigm shift in optimization:")
        print("   From static parameters to conscious agents")
        print("   From rigid algorithms to dialectical processes")
        print("   From artificial limits to natural rhythms")
        print("   From individual optimization to collective intelligence")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration encountered an error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Thank you for exploring the AdaptiveGenieNetwork!")
    print("=" * 60)


if __name__ == "__main__":
    main()