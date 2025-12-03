"""
Tests for Dialectical Genie: Aufhebung Implementation
"""
import pytest
import numpy as np
from dialectical_genie import Aufhebung, DialecticalResult

class TestAufhebung:
    """Test Aufhebung Implementation"""

    def test_initialization(self):
        """Test initialization with default and custom parameters"""
        aufhebung = Aufhebung()
        assert aufhebung.tension_threshold == 0.3

        aufhebung_custom = Aufhebung(tension_threshold=0.5)
        assert aufhebung_custom.tension_threshold == 0.5

    def test_low_tension_synthesis(self):
        """Test that low tension leads to quantitative synthesis (weighted average)"""
        aufhebung = Aufhebung(tension_threshold=0.5)

        # Thesis and Antithesis are close (low tension)
        thesis = np.array([1.0, 0.0, 0.0])
        antithesis = np.array([1.1, 0.0, 0.0])

        result = aufhebung.synthesize(thesis, antithesis)

        assert isinstance(result, DialecticalResult)
        assert result.metadata["mode"] == "quantitative"
        assert result.qualitative_shift is False

        # Check if it matches weighted average
        weighted_avg = 0.5 * thesis + 0.5 * antithesis
        np.testing.assert_allclose(result.synthesis, weighted_avg)

    def test_high_tension_synthesis(self):
        """Test that high tension leads to qualitative synthesis (Aufhebung)"""
        aufhebung = Aufhebung(tension_threshold=0.3)

        # Thesis and Antithesis are opposites (high tension)
        thesis = np.array([1.0, 0.0, 0.0])
        antithesis = np.array([-1.0, 0.0, 0.0])

        result = aufhebung.synthesize(thesis, antithesis)

        assert isinstance(result, DialecticalResult)
        assert result.metadata["mode"] == "qualitative_aufhebung"
        assert result.qualitative_shift is True
        assert result.tension > 0.3

        # Check for orthogonal elevation (qualitative shift)
        # Original vectors are on X-axis, result should have non-zero Y or Z
        orthogonal_magnitude = np.linalg.norm(result.synthesis[1:])
        assert orthogonal_magnitude > 1e-6, "Aufhebung should produce orthogonal elevation"

        # Check that it is NOT the weighted average
        weighted_avg = 0.5 * thesis + 0.5 * antithesis # [0, 0, 0]
        assert not np.allclose(result.synthesis, weighted_avg), "Result should not be just weighted average"

    def test_synthesis_preserves_dimensions(self):
        """Test that synthesis output has same dimensions as input"""
        aufhebung = Aufhebung()
        thesis = np.random.rand(5)
        antithesis = np.random.rand(5)

        result = aufhebung.synthesize(thesis, antithesis)
        assert result.synthesis.shape == thesis.shape

    def test_zero_vector_handling(self):
        """Test handling of zero vectors"""
        aufhebung = Aufhebung()
        thesis = np.zeros(3)
        antithesis = np.array([1.0, 0.0, 0.0])

        # Should not crash
        result = aufhebung.synthesize(thesis, antithesis)
        assert result.synthesis.shape == (3,)

    def test_history_tracking(self):
        """Test that history is tracked"""
        aufhebung = Aufhebung()
        thesis = np.array([1.0, 0.0])
        antithesis = np.array([-1.0, 0.0])

        aufhebung.synthesize(thesis, antithesis)
        aufhebung.synthesize(thesis, antithesis)

        assert len(aufhebung.history) == 2
        assert isinstance(aufhebung.history[0], DialecticalResult)
