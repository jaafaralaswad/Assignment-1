# Bisection method solver used for numerical example 4 - ME700 Assignment 1 "warm-up"
# Author: Jaafar Alaswad (jalaswad@bu.edu)

import numpy as np
from pathlib import Path
from typing import Callable, Union

def cantilever(L: float, w: float, P: float, tol_input: float, tol_output: float, max_iterations: int) -> dict:
    """
    Implements the bisection method to find the location a balance force is applied in a cantilever beam.
    The method iteratively halves the interval [0,L] until a root is found within specified tolerances.
    """

    a = 0
    b = L
    validate_b_greater_a(a, b)  # Ensure b is strictly greater than a

    fnc = lambda x: x - w*L**2/(2*P) # Define the function

    fnc_a = fnc(a)  # Evaluate the function at a
    fnc_b = fnc(b)  # Evaluate the function at b
    validate_interval(a, b, fnc_a, fnc_b)  # Ensure a root is guaranteed in [a, b]
    iteration = 1  # Initialize iteration counter

    print(f"\n{'Iteration':<10} {'a':<15} {'b':<15} {'c':<15} {'|f(c)|':<15}")

    while iteration <= max_iterations:
        c = evaluate_middle_point(a, b)
        fnc_c = fnc(c)

        print(f"{iteration:<10} {a:<15.6f} {b:<15.6f} {c:<15.6f} {abs(fnc_c):<15.6e}")

        if find_root(a, b, fnc_c, tol_input, tol_output):
            break

        a, b, fnc_a, fnc_b = update_interval_a_b(a, b, c, fnc_a, fnc_b, fnc_c)
        iteration += 1

    # Ensure the loop terminates correctly if max_iterations is reached without convergence
    if iteration >= max_iterations and not find_root(a, b, fnc_c, tol_input, tol_output):
        terminate_max_iter(iteration, max_iterations)

    result = {
        'root': c,
        'iterations': iteration,
        'function_value': fnc_c,
        'interval': (a, b),
        'converged': find_root(a, b, fnc_c, tol_input, tol_output)
    }
    
    print(f"\nBalance force applied at x = {c:.6f} after {iteration} iterations.")

    return result


def evaluate_middle_point(a: float, b:float) -> float:
    """
    This function evaluates the middle point based on given bounds a and b.
    """
    c = (a+b)/2
    return c


def validate_b_greater_a(a: float, b: float):
    """
    This function makes sure b is strictly greater than a.
    """
    if a == b:
        raise ValueError(f"Invalid input: {a} is equal to {b}.")
    elif a > b:
        raise ValueError(f"Invalid input: {a} is greater than {b}.")
    return True


def terminate_max_iter(iteration: int, max_iterations: int):
    """
    Given the number of current iterations and maximum number of iterations.
    Terminate procedure and raise an error if #iteration > maximum number of iterations.
    """
    raise ValueError(f"Procedure terminated: specified maximum number of iterations ({max_iterations}) reached without convergence.")


def validate_interval(a: float, b: float, fnc_a: float, fnc_b: float):
    """
    Check the function has different signs at a and b
    Will return true if fnc_a*fnc_b<0
    Will return false otherwise.
    """
    if fnc_a*fnc_b < 0:
        return
    else:
        raise ValueError("Static balance with the entered parameters is impossible.")                  
    return


def update_interval_a_b(a: float, b: float, c: float, fnc_a: float, fnc_b: float, fnc_c: float) -> Union[float, float, float, float]:
    """
    Given endpoints a and b and midpoint c and their function evaluations.
    Will update a and b according to c to to make sure the interval always contains a root.
    If fnc_a, fnc_b, or fnc_c happens to be exactly zero, a and b will both be assigned to the corresponding point.
    """
    if fnc_c == 0: 
        return c, c, fnc_c, fnc_c
    if fnc_a == 0:
        return a, a, fnc_a, fnc_a
    if fnc_b == 0:
        return b, b, fnc_b, fnc_b
    if fnc_a * fnc_c < 0: # Opposite signs indicate root in [a, c]
        return a, c, fnc_a, fnc_c
    else: # Otherwise, root is in [b, c]
        return b, c, fnc_b, fnc_c


def find_root(a: float, b: float, fnc_c: float, tol_input: float, tol_output: float) -> bool:
    """
    Given end points a and b, function evaluation at c, and perscribed tolerances.
    Will return "True" if a root is accordingly found.
    Will return "False" otherwise.
    """
    val_input = abs(a-b)
    val_output = abs(fnc_c)
    if val_input < tol_input or val_output < tol_output:
        return True
    else:
        return False


def update_step(fnc: Callable, a: float, b: float, fnc_a: float, fnc_b: float) -> Union[float, float, float, float]:
    """
    Given a continuous function, bounds a and b, and function values at a and b.
    Will compute the midpoint c and perform the update to a and b according to the bisection method.
    Will return new values of a, b, fcn_a and fcn_b.
    """
    c = evaluate_middle_point(a, b)
    fnc_c = fnc(c)
    a, b, fnc_a, fnc_b = update_interval_a_b(a, b, c, fnc_a, fnc_b, fnc_c)
    return a, b, fnc_a, fnc_b