"""
Tests for AdaptiveGenieNetwork and its agents.
"""
import pytest
import numpy as np
from adaptive_genie_network import (
    AdaptiveGenieNetwork, PopulationAgent, RhythmAgent,
    ResonanceAgent, GradientField, ComplexityMeasure
)
from exceptions import (
    InvalidParameterError, InvalidBoundsError, ValidationError,
    DimensionMismatchError, NumericError, VisualizationError
)


class TestPopulationAgent:
    """Test PopulationAgent"""

    def test_create_with_default_parameters(self):
        """Test creating agent with default parameters"""
        agent = PopulationAgent()
        assert agent.min_size == 10
        assert agent.max_size == 500
        assert agent.breathing_rate == 0.1
        assert agent.oscillation_amplitude == 0.3

    def test_create_with_custom_parameters(self):
        """Test creating agent with custom parameters"""
        agent = PopulationAgent(min_size=20, max_size=200, breathing_rate=0.2, oscillation_amplitude=0.5)
        assert agent.min_size == 20
        assert agent.max_size == 200
        assert agent.breathing_rate == 0.2
        assert agent.oscillation_amplitude == 0.5

    def test_create_with_invalid_min_size(self):
        """Test creating agent with invalid min_size"""
        with pytest.raises(InvalidParameterError, match="min_size must be >= 1"):
            PopulationAgent(min_size=0)

    def test_create_with_invalid_max_size(self):
        """Test creating agent with max_size < min_size"""
        with pytest.raises(InvalidParameterError, match="max_size .* must be >= min_size"):
            PopulationAgent(min_size=100, max_size=50)

    def test_create_with_invalid_breathing_rate(self):
        """Test creating agent with invalid breathing_rate"""
        with pytest.raises(InvalidParameterError, match="breathing_rate must be > 0"):
            PopulationAgent(breathing_rate=0)

    def test_create_with_invalid_oscillation_amplitude(self):
        """Test creating agent with invalid oscillation_amplitude"""
        with pytest.raises(InvalidParameterError, match="oscillation_amplitude must be in"):
            PopulationAgent(oscillation_amplitude=1.5)

    def test_oscillate_returns_valid_size(self):
        """Test oscillate returns size within bounds"""
        agent = PopulationAgent()
        size = agent.oscillate(exploration_need=0.5, convergence_pressure=0.3)
        assert agent.min_size <= size <= agent.max_size

    def test_negotiate_returns_dict(self):
        """Test negotiate returns proper dict"""
        agent = PopulationAgent()
        context = {
            'complexity': ComplexityMeasure(0.5, 0.5, 0.5, 0.5)
        }
        result = agent.negotiate(context)
        assert 'population_size' in result
        assert 'breathing_phase' in result
        assert 'energy_level' in result


class TestRhythmAgent:
    """Test RhythmAgent"""

    def test_detect_cycle_completion_insufficient_data(self):
        """Test cycle detection with insufficient data"""
        agent = RhythmAgent()
        assert agent.detect_cycle_completion() is False

    def test_detect_cycle_completion_with_data(self):
        """Test cycle detection with sufficient data"""
        agent = RhythmAgent()
        # Fill cycle detector with data
        for i in range(20):
            agent.cycle_detector.append(0.5 + 0.1 * np.sin(i * 0.5))
        result = agent.detect_cycle_completion()
        assert isinstance(result, (bool, np.bool_))

    def test_detect_cycle_completion_constant_signal(self):
        """Test cycle detection with constant signal"""
        agent = RhythmAgent()
        # Fill with constant values
        for i in range(20):
            agent.cycle_detector.append(0.5)
        # Should handle constant signals gracefully
        result = agent.detect_cycle_completion()
        assert result is False

    def test_negotiate_returns_dict(self):
        """Test negotiate returns proper dict"""
        agent = RhythmAgent()
        context = {'fitness_history': [0.1, 0.2, 0.3, 0.4, 0.5]}
        result = agent.negotiate(context)
        assert 'cycle_complete' in result
        assert 'natural_frequency' in result
        assert 'phase' in result
        assert 'rhythm_energy' in result


class TestResonanceAgent:
    """Test ResonanceAgent"""

    def test_measure_crystallization_valid_inputs(self):
        """Test measuring crystallization with valid inputs"""
        agent = ResonanceAgent()
        crystallization = agent.measure_crystallization(
            population_diversity=0.5,
            fitness_variance=0.2
        )
        assert 0 <= crystallization <= 1

    def test_measure_crystallization_invalid_diversity(self):
        """Test measuring crystallization with invalid diversity"""
        agent = ResonanceAgent()
        with pytest.raises(InvalidParameterError, match="population_diversity must be in"):
            agent.measure_crystallization(population_diversity=1.5, fitness_variance=0.2)

    def test_measure_crystallization_invalid_variance(self):
        """Test measuring crystallization with invalid variance"""
        agent = ResonanceAgent()
        with pytest.raises(InvalidParameterError, match="fitness_variance must be >= 0"):
            agent.measure_crystallization(population_diversity=0.5, fitness_variance=-0.1)

    def test_negotiate_returns_dict(self):
        """Test negotiate returns proper dict"""
        agent = ResonanceAgent()
        context = {
            'population_diversity': 0.5,
            'fitness_variance': 0.2
        }
        result = agent.negotiate(context)
        assert 'crystallization_level' in result
        assert 'resonance_frequency' in result
        assert 'dynamic_threshold' in result
        assert 'harmony_level' in result


class TestGradientField:
    """Test GradientField"""

    def test_generate_gradients_valid_landscape(self, problem_landscape):
        """Test generating gradients with valid landscape"""
        field = GradientField()
        field.generate_gradients(problem_landscape)
        assert len(field.gradients) == 2
        assert len(field.flow_vectors) == 2

    def test_generate_gradients_invalid_type(self):
        """Test generating gradients with invalid type"""
        field = GradientField()
        with pytest.raises(ValidationError, match="problem_landscape must be a dict"):
            field.generate_gradients("not_a_dict")

    def test_generate_gradients_invalid_dimensions(self):
        """Test generating gradients with invalid dimensions"""
        field = GradientField()
        with pytest.raises(InvalidParameterError, match="dimensions must be a positive integer"):
            field.generate_gradients({'dimensions': -1})

    def test_generate_gradients_invalid_bounds(self):
        """Test generating gradients with invalid bounds"""
        field = GradientField()
        landscape = {
            'dimensions': 2,
            'bounds': [(-10, 10)]  # Wrong length
        }
        with pytest.raises(InvalidBoundsError, match="bounds must be a list of length"):
            field.generate_gradients(landscape)

    def test_generate_gradients_invalid_bound_format(self):
        """Test generating gradients with invalid bound format"""
        field = GradientField()
        landscape = {
            'dimensions': 2,
            'bounds': [(-10, 10), "invalid"]
        }
        with pytest.raises(InvalidBoundsError, match="bounds\\[1\\] must be a tuple/list"):
            field.generate_gradients(landscape)

    def test_generate_gradients_inverted_bounds(self):
        """Test generating gradients with inverted bounds"""
        field = GradientField()
        landscape = {
            'dimensions': 2,
            'bounds': [(10, -10), (-5, 15)]  # lower > upper
        }
        with pytest.raises(InvalidBoundsError, match="bounds\\[0\\] lower .* must be < upper"):
            field.generate_gradients(landscape)

    def test_get_field_strength_at_valid_position(self, problem_landscape):
        """Test getting field strength at valid position"""
        field = GradientField()
        field.generate_gradients(problem_landscape)
        position = np.array([0.0, 0.0])
        strength = field.get_field_strength_at(position)
        assert isinstance(strength, float)
        assert strength >= 0

    def test_get_field_strength_at_invalid_position_type(self):
        """Test getting field strength with invalid position type"""
        field = GradientField()
        with pytest.raises(ValidationError, match="position must be a numpy array"):
            field.get_field_strength_at([0.0, 0.0])

    def test_apply_field_force_valid_inputs(self, problem_landscape):
        """Test applying field force with valid inputs"""
        field = GradientField()
        field.generate_gradients(problem_landscape)
        position = np.array([0.0, 0.0])
        velocity = np.array([1.0, 1.0])
        new_velocity = field.apply_field_force(position, velocity)
        assert new_velocity.shape == velocity.shape

    def test_apply_field_force_dimension_mismatch(self):
        """Test applying field force with dimension mismatch"""
        field = GradientField()
        position = np.array([0.0, 0.0])
        velocity = np.array([1.0])  # Wrong shape
        with pytest.raises(DimensionMismatchError, match="position shape .* != velocity shape"):
            field.apply_field_force(position, velocity)


class TestAdaptiveGenieNetwork:
    """Test AdaptiveGenieNetwork"""

    def test_create_network(self, adaptive_genie_network):
        """Test creating a network"""
        assert isinstance(adaptive_genie_network.population, PopulationAgent)
        assert isinstance(adaptive_genie_network.iterator, RhythmAgent)
        assert isinstance(adaptive_genie_network.convergence, ResonanceAgent)
        assert isinstance(adaptive_genie_network.boundaries, GradientField)

    def test_measure_negation_density(self, adaptive_genie_network, problem_landscape):
        """Test measuring negation density"""
        complexity = adaptive_genie_network.measure_negation_density(problem_landscape)
        assert isinstance(complexity, ComplexityMeasure)
        assert 0 <= complexity.exploration_requirement <= 2
        assert 0 <= complexity.convergence_gradient <= 1
        assert 0 <= complexity.negation_density <= 1

    def test_tune_parameters(self, adaptive_genie_network, problem_landscape, system_state):
        """Test tuning parameters"""
        result = adaptive_genie_network.tune_parameters(problem_landscape, system_state)
        assert 'population_size' in result
        assert 'continue_iteration' in result
        assert 'system_energy' in result
        assert 'dialectical_tension' in result

    def test_get_system_state(self, adaptive_genie_network):
        """Test getting system state"""
        state = adaptive_genie_network.get_system_state()
        assert 'collective_consciousness' in state
        assert 'system_energy' in state
        assert 'population_size' in state

    def test_visualize_system_dynamics_no_history(self, adaptive_genie_network):
        """Test visualization with no history raises error"""
        with pytest.raises(VisualizationError, match="No negotiation history"):
            adaptive_genie_network.visualize_system_dynamics()

    def test_visualize_system_dynamics_with_history(self, adaptive_genie_network, problem_landscape, system_state):
        """Test visualization with history"""
        # Generate some history
        for i in range(5):
            adaptive_genie_network.tune_parameters(problem_landscape, system_state)

        fig = adaptive_genie_network.visualize_system_dynamics()
        assert fig is not None
