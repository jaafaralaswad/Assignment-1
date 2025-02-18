import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Union

def newton(f, df, x0, epsilon_1, epsilon_2, max_iter):
    """
    This function solves a nonlinear equations using the Newton method.
    
    Input:
    f:  Functions.
    dfx: Derivative of f with respect to x.
    x0: Initial guess.
    epsilon_1 and epsilon_2: Convergence criteria for input and output, respectively.
    max_iter: Maximum number of iterations.
    
    Output:
    x: Solution if convergence criteria is satisfied. Otherwise errors.
    """

    # Assign the initial guess to the first iteration
    x = x0

    # Print and update convergence history
    print("Iteration |        x        |       f(x)      ")
    print("------------------------------------------------")
    
    # Main loop
    for i in range(1, max_iter + 1):

        # Evaluate f(x) and f'(x)
        fx, dfx = evaluate_functions(f, df, x)

        # Print their values
        print(f"{i:^9} | {x:^15.8f} | {fx:^15.8f}")
        
        # Termination criteria on output
        if check_convergence_function(fx, epsilon_2):
            print(f"Root found at x = {x:.8f} after {i} iterations.")
            return x, True
        
        # Make sure derivative is not zero
        check_derivative_nonzero(dfx)
        
        # Update new x value
        x_new, converged = update_x(x, fx, dfx, epsilon_1)
        if converged:
            print(f"Root found at x = {x_new:.8f} after {i} iterations.")
            return x_new, True  # Converged based on termination criteria for input
        
        x = x_new  # Update for next iteration

    # Throw an error if max_iterations reached without convergence
    raise RuntimeError("Maximum iterations reached without convergence.")

def evaluate_functions(f, df, x):
    """Evaluates f(x) and df(x)."""
    return f(x), df(x)

def check_convergence_function(fx, epsilon_2):
    """Checks if |f(x)| < epsilon_2."""
    return abs(fx) < epsilon_2

def check_derivative_nonzero(dfx):
    """Checks if the derivative is nonzero."""
    if dfx == 0:
        raise ValueError("Derivative is zero. Newton's method fails!")

def update_x(x, fx, dfx, epsilon_1):
    """Updates x and checks for convergence based on step size."""
    x_new = x - fx / dfx
    return x_new, abs(x_new - x) < epsilon_1
