import numpy as np
import pytest

# Import functions from both methods
from bisectionmethod import bisection, cantilever, mass_center
from newtonmethod import newton, newton_raphson
from elastoplasticity.elastoplasticity import LinearInterpolator, ElastoplasticMaterial, ElastoplasticSolver

# --- Bisection Method Tests ---
def test_midpoint():
    # Test for evaluating middle point
    a = 10.0
    b = 20.0
    found = bisection.evaluate_middle_point(a, b)
    known = 15.0
    assert np.isclose(known, found)

def test_bisection_finds_root():
    # Test for a simple quadratic function x^2 - 4 = 0 (roots at x = -2, 2)
    result = bisection.bisection(lambda x: x**2 - 4, 0, 3, 1e-6, 1e-6, 100)
    assert abs(result['root'] - 2) < 1e-6
    assert result['converged'] is True
    assert result['iterations'] <= 100

def test_bisection_no_root_in_interval():
    # Test for no root in interval [3, 5] for x^2 - 4 = 0
    with pytest.raises(ValueError, match=r"A root in interval.*is not guaranteed."):
        bisection.bisection(lambda x: x**2 - 4, 3, 5, 1e-6, 1e-6, 100)

def test_validate_b_greater_a():
    # Test that validate_b_greater_a raises an error when a >= b
    with pytest.raises(ValueError, match="Invalid input: 2.0 is equal to 2.0."):
        bisection.validate_b_greater_a(2.0, 2.0)
    with pytest.raises(ValueError, match="Invalid input: 3.0 is greater than 2.0."):
        bisection.validate_b_greater_a(3.0, 2.0)

def test_bisection_with_negative_root():
    # Test for finding negative root in [-3, 0]
    result = bisection.bisection(lambda x: x**2 - 4, -3, 0, 1e-6, 1e-6, 100)
    assert abs(result['root'] + 2) < 1e-6
    assert result['converged'] is True


# --- cantilever Tests ---
# Test cantilever function
@pytest.fixture
def cantilever_params():
    return {
        "L": 10.0,
        "w": 5.0,
        "P": 50.0,  # Increased P to ensure a valid root
        "tol_input": 1e-6,
        "tol_output": 1e-6,
        "max_iterations": 100
    }

def test_cantilever(cantilever_params):
    result = cantilever.cantilever(**cantilever_params)
    assert "root" in result
    assert "iterations" in result
    assert "function_value" in result
    assert "interval" in result
    assert "converged" in result
    assert result["converged"] is True

def test_evaluate_middle_point():
    assert cantilever.evaluate_middle_point(0, 10) == 5.0

def test_validate_b_greater_a():
    with pytest.raises(ValueError):
        cantilever.validate_b_greater_a(10, 10)
    with pytest.raises(ValueError):
        cantilever.validate_b_greater_a(10, 5)
    assert cantilever.validate_b_greater_a(5, 10) is True

def test_validate_interval():
    with pytest.raises(ValueError):
        cantilever.validate_interval(0, 10, 1, 1)
    assert cantilever.validate_interval(0, 10, -1, 1) is None

def test_update_interval_a_b():
    a, b, fnc_a, fnc_b = cantilever.update_interval_a_b(0, 10, 5, -1, 1, 0)
    assert a == b == 5

def test_find_root():
    assert cantilever.find_root(0, 10, 0, 1e-6, 1e-6) is True
    assert cantilever.find_root(0, 10, 1, 1e-6, 1e-6) is False

def test_terminate_max_iter():
    with pytest.raises(ValueError):
        cantilever.terminate_max_iter(101, 100)





# --- mass_canter Tests ---






# --- Newton Method Tests ---
# Define test functions
def f1(x):
    return x**2 - 4

def df1(x):
    return 2*x

def f2(x, y):
    return x**2 + y**2 - 4

def df2x(x, y):
    return 2*x

def df2y(x, y):
    return 2*y

def g2(x, y):
    return x - y

def dg2x(x, y):
    return 1

def dg2y(x, y):
    return -1

# Test evaluate_functions for Newton
def test_evaluate_functions_newton():
    x = 2
    fx, dfx = newton.evaluate_functions(f1, df1, x)
    assert np.isclose(fx, 0)
    assert np.isclose(dfx, 4)

# Test check_convergence_function
@pytest.mark.parametrize("fx, epsilon_2, expected", [
    (0.001, 0.01, True),  
    (0.1, 0.01, False),
])
def test_check_convergence_function(fx, epsilon_2, expected):
    assert newton.check_convergence_function(fx, epsilon_2) == expected

# Test check_derivative_nonzero
def test_check_derivative_nonzero():
    assert newton.check_derivative_nonzero(2.0) is None
    with pytest.raises(ValueError):
        newton.check_derivative_nonzero(0.0)

# Test update_x for Newton method
def test_update_x():
    updated_x = newton.update_x(2.0, 4.0, 2.0, 1e-6)
    assert np.isclose(updated_x, 0.0).all()

# Test Newton method for single variable
def test_newton_success():
    x0 = 2.0
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 50
    x = newton.newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)
    assert np.isclose(x, [2.0, -2.0]).any()

def test_newton_nonconvergence():
    x0 = 0.0  # Derivative is zero, should raise error
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 10
    with pytest.raises(ValueError):
        newton.newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)

def test_newton_forces_convergence_on_step_size(capsys):
    """Forces Newton's method to enter the `if converged:` block due to step size convergence."""
    def f(x):
        return x - 1  # Root is exactly at x = 1

    def df(x):
        return 1  # Constant derivative (nonzero)

    x0 = 0.9999  # Start very close to root
    epsilon_1 = 1e-2 
    epsilon_2 = 1e-6
    max_iter = 50

    x, converged = newton.newton(f, df, x0, epsilon_1, epsilon_2, max_iter)

    assert converged is True 
    assert np.isclose(x, 1.0)  

    captured = capsys.readouterr()
    assert "Root found at x = 1.00000000 after" in captured.out 


def test_newton_max_iterations():
    """Forces Newton's method to reach max iterations without converging."""
    def f(x):
        return np.exp(x) - 1000  # Exponential growth, takes many iterations to reach 0

    def df(x):
        return np.exp(x)  # Always positive, no zero crossings

    x0 = 0.0
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 1  # Too few iterations to reach the root

    with pytest.raises(RuntimeError, match="Maximum iterations reached without convergence."):
        newton.newton(f, df, x0, epsilon_1, epsilon_2, max_iter)


# Test evaluate_functions for Newton-Raphson
def test_evaluate_functions_newton_raphson():
    x, y = 1, 1
    F, J = newton_raphson.evaluate_functions(f2, df2x, df2y, g2, dg2x, dg2y, x, y) 
    assert np.isclose(F[0], -2)
    assert np.isclose(F[1], 0)
    assert np.isclose(J[0, 0], 2)
    assert np.isclose(J[1, 1], -1)

# Test check_convergence for Newton-Raphson
@pytest.mark.parametrize("F, epsilon, expected", [
    (np.array([0.0001, 0.0001]), 1e-3, True),
    (np.array([0.1, 0.1]), 1e-3, False),
])
def test_check_convergence(F, epsilon, expected):
    assert newton_raphson.check_convergence(F, epsilon) == expected 

# Test check_jacobian_singular
def test_check_jacobian_singular():
    J = np.array([[2.0, 2.0], [-2.0, -2.0]])
    with pytest.raises(ValueError):
        newton_raphson.check_jacobian_singular(J)

def test_update_variables():
    J = np.array([[2.0, 1.0], [1.0, 2.0]])
    F = np.array([-2.0, -2.0]).reshape(2, 1)
    x, y = newton_raphson.update_variables(1.0, 1.0, J, F)

    print("\nDEBUG: Expected x = 1.66667, Actual x =", x)
    print("DEBUG: Expected y = 1.66667, Actual y =", y)

    assert np.isclose(x, 1.66667)
    assert np.isclose(y, 1.66667)

# Test Newton-Raphson method for system of equations
def test_newton_raphson_success():
    x0, y0 = 1.0, 1.0
    epsilon = 1e-6
    max_iter = 50
    x, y = newton_raphson.newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)
    assert np.isclose(x**2 + y**2, 4)
    assert np.isclose(x, y)

def test_newton_raphson_nonconvergence():
    x0, y0 = 0.0, 0.0  # Singular point
    epsilon = 1e-6
    max_iter = 10
    with pytest.raises(ValueError):
        newton_raphson.newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)


def test_newton_raphson_max_iterations():
    """Forces Newton-Raphson to reach max iterations without converging."""
    x0, y0 = 1.0, 1.0
    epsilon = 1e-12  # Very strict tolerance
    max_iter = 2  # Too few iterations to allow convergence

    with pytest.raises(RuntimeError, match="Maximum iterations reached without convergence."):
        newton_raphson.newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)





# --- Elastoplasticity Tests ---

# Test LinearInterpolator
@pytest.fixture
def linear_interpolator():
    t_data = np.array([0, 1, 2])
    eps_data = np.array([0, 0.1, 0.2])
    N = 10
    return LinearInterpolator(t_data, eps_data, N)

def test_interpolator(linear_interpolator):
    t, eps = linear_interpolator.interpolate()
    assert len(t) == len(eps)
    assert np.allclose(eps, np.interp(t, linear_interpolator.t_data, linear_interpolator.eps_data))

# Test ElastoplasticMaterial
@pytest.fixture
def elastoplastic_material():
    return ElastoplasticMaterial(E_1=200e9, E_2=10e9, sigma_y0=250e6, H=5e9)

def test_compute_stress(elastoplastic_material):
    assert elastoplastic_material.compute_stress(0.002, 0.001) == 200e9 * (0.002 - 0.001)

def test_compute_back_stress(elastoplastic_material):
    assert elastoplastic_material.compute_back_stress(0.001) == 10e9 * 0.001

def test_yield_function(elastoplastic_material):
    sigma, X, xi = 300e6, 50e6, 0.001
    assert elastoplastic_material.yield_function(sigma, X, xi) == abs(300e6 - 50e6) - (250e6 + 5e9 * 0.001)

def test_plastic_correction(elastoplastic_material):
    deps, eps_p, xi = 0.001, 0.0, 0.0
    delta_eps_p, delta_xi = elastoplastic_material.plastic_correction(deps, eps_p, xi)
    expected_factor = 200e9 / (200e9 + 10e9 + 5e9)
    assert delta_eps_p == expected_factor * deps
    assert delta_xi == expected_factor * abs(deps)

# Test ElastoplasticSolver
@pytest.fixture
def elastoplastic_solver(linear_interpolator, elastoplastic_material):
    return ElastoplasticSolver(elastoplastic_material, linear_interpolator)

def test_solver(elastoplastic_solver):
    t, eps, sigma = elastoplastic_solver.solve()
    assert len(t) == len(eps) == len(sigma)
    assert sigma[0] == 0  # Initial stress should be zero
