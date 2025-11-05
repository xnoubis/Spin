#!/usr/bin/env python3
"""
Quick Demo Runner for AdaptiveGenieNetwork
==========================================

This script provides a simplified way to run the AdaptiveGenieNetwork demonstration
with error handling and dependency checking.
"""

import sys
import subprocess
import importlib
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'plotly', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or install all requirements with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def run_simple_demo():
    """Run a simplified demonstration"""
    print("🌟 AdaptiveGenieNetwork Simple Demo")
    print("=" * 40)
    
    try:
        # Import core components
        from adaptive_genie_network import AdaptiveGenieNetwork
        from example_applications import DialecticalParticleSwarm, rastrigin
        
        print("✅ Successfully imported AdaptiveGenieNetwork components")
        
        # Create and test the network
        print("\n🧠 Testing basic network functionality...")
        network = AdaptiveGenieNetwork()
        
        # Simple test
        problem_landscape = {
            'dimensions': 2,
            'bounds': [(-5, 5), (-5, 5)],
            'multimodality': 0.5,
            'noise_level': 0.1,
            'deception': 0.3
        }
        
        system_state = {
            'fitness_history': [1.0, 0.8, 0.6, 0.4],
            'population_diversity': 0.7,
            'fitness_variance': 0.2
        }
        
        result = network.tune_parameters(problem_landscape, system_state)
        
        print(f"   Population size: {result['population_size']}")
        print(f"   Crystallization level: {result['crystallization_level']:.3f}")
        print(f"   System energy: {result['system_energy']:.3f}")
        print("✅ Basic network test successful!")
        
        # Test optimization
        print("\n🎯 Testing dialectical optimization...")
        dpso = DialecticalParticleSwarm(rastrigin, [(-5.12, 5.12)] * 2)
        opt_result = dpso.optimize(max_iterations=20)
        
        print(f"   Best fitness found: {opt_result.best_fitness:.6f}")
        print(f"   Total evaluations: {opt_result.total_evaluations}")
        print(f"   Execution time: {opt_result.execution_time:.2f} seconds")
        print("✅ Optimization test successful!")
        
        print("\n🎉 Simple demo completed successfully!")
        print("\n💡 For the full demonstration, run: python demo.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Make sure all dependencies are installed")
        print("   2. Check that all Python files are in the same directory")
        print("   3. Try running: pip install -r requirements.txt")
        return False

def main():
    """Main function"""
    print("AdaptiveGenieNetwork Demo Runner")
    print("=" * 35)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    print("✅ All required dependencies found")
    
    # Ask user for demo type
    print("\n🚀 Choose demo type:")
    print("   1. Simple demo (quick test)")
    print("   2. Full demo (comprehensive)")
    print("   3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            run_simple_demo()
        elif choice == "2":
            print("\n🌟 Running full demonstration...")
            try:
                from demo import main as demo_main
                demo_main()
            except Exception as e:
                print(f"❌ Full demo failed: {e}")
                print("💡 Try the simple demo instead")
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()