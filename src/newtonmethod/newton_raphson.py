import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Union

def newton_raphson(f, dfx, dfy, g, dgx, dgy, x0, y0, epsilon, max_iter):
    """
    This function solves a system of two nonlinear equations using the Newton-Raphson method.
    
    Input:
    f, g: Two functions.
    dfx, dfy: Partial derivatives of f with respect to x and y.
    dgx, dgy: Partial derivatives of g with respect to x and y.
    x0, y0: Initial guesses.
    epsilon: Convergence criterion based on the norm of the residual.
    max_iter: Maximum number of iterations.
    
    Output:
    (x, y): Solution tuple if convergence criteria is satisfied. Otherwise errors.
    """

    # Assign the initial guesses to the first iteration
    x, y = x0, y0
    
    # Print and update convergence history
    print("Iteration |        x        |        y        |      ||F||      ")
    print("---------------------------------------------------------------")
    
    # Main loop
    for i in range(1, max_iter + 1):

        # Evaluate the residual and jacobian using f(x,y), g(x,y) and their partial derivatives
        F, J = evaluate_functions(f, dfx, dfy, g, dgx, dgy, x, y)

        # Evaluate the norm of the residual
        norm_F = np.linalg.norm(F)

        # Print and update convergence history
        print(f"{i:^9} | {x:^15.8f} | {y:^15.8f} | {norm_F:^15.8f}")
        
        # Termination criteria
        if check_convergence(F, epsilon):
            print(f"Root found at (x, y) = ({x:.8f}, {y:.8f}) after {i} iterations.")
            return x, y
        
        # Make sure jacobian matrix is not singular
        check_jacobian_singular(J)

        # Update x and y for nect iteration
        x, y = update_variables(x, y, J, F)
    
    # Throw an error if max_iterations reached without convergence
    raise RuntimeError("Maximum iterations reached without convergence.")

def evaluate_functions(f, dfx, dfy, g, dgx, dgy, x, y):
    """Evaluates the residual and the jacobian."""
    fx, fy = f(x, y), g(x, y)
    J = np.array([[dfx(x, y), dfy(x, y)], [dgx(x, y), dgy(x, y)]])
    F = np.array([fx, fy])
    return F, J

def check_convergence(F, epsilon):
    """Checks if the norm of the residual is smaller than epsilon."""
    return np.linalg.norm(F) < epsilon

def check_jacobian_singular(J):
    """Checks if the Jacobian is singular."""
    if np.linalg.det(J) == 0:
        raise ValueError("Jacobian is singular. Newton-Raphson fails!")

def update_variables(x, y, J, F):
    """Updates x and y"""
    delta = np.linalg.solve(J, -F)
    return x + delta[0], y + delta[1]
