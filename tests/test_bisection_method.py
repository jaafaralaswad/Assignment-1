import numpy as np
import pytest
from bisectionmethod import bisection, cantilever, mass_center

def test_midpoint():
    # Test for evaluating middle point
    a = 10.0
    b = 20.0
    found = evaluate_middle_point(a, b)
    known = 15.0
    assert np.isclose(known, found)


def test_bisection_finds_root():
    # Test for a simple quadratic function x^2 - 4 = 0 (roots at x = -2, 2)
    result = bisection(lambda x: x**2 - 4, 0, 3, 1e-6, 1e-6, 100)
    assert abs(result['root'] - 2) < 1e-6
    assert result['converged'] is True
    assert result['iterations'] <= 100

def test_bisection_no_root_in_interval():
    # Test for no root in interval [3, 5] for x^2 - 4 = 0
    with pytest.raises(ValueError, match=r"A root in interval.*is not guaranteed."):
        bisection(lambda x: x**2 - 4, 3, 5, 1e-6, 1e-6, 100)

def test_validate_b_greater_a():
    # Test that validate_b_greater_a raises an error when a >= b
    with pytest.raises(ValueError, match="Invalid input: 2.0 is equal to 2.0."):
        validate_b_greater_a(2.0, 2.0)
    with pytest.raises(ValueError, match="Invalid input: 3.0 is greater than 2.0."):
        validate_b_greater_a(3.0, 2.0)

def test_bisection_with_negative_root():
    # Test for finding negative root in [-3, 0]
    result = bisection(lambda x: x**2 - 4, -3, 0, 1e-6, 1e-6, 100)
    assert abs(result['root'] + 2) < 1e-6
    assert result['converged'] is True