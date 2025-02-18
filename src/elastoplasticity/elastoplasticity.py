import numpy as np
import matplotlib.pyplot as plt

class LinearInterpolator:
    """Handles linear interpolation of strain over time."""
    def __init__(self, t_data, eps_data, N):
        self.t_data = t_data
        self.eps_data = eps_data
        self.N = N
    
    def interpolate(self):
        t = np.linspace(self.t_data[0], self.t_data[-1], self.N * (len(self.t_data) - 1) + 1)
        eps = np.interp(t, self.t_data, self.eps_data)
        return t, eps

class ElastoplasticMaterial:
    """Represents the elastoplastic material model."""
    def __init__(self, E_1, E_2, sigma_y0, H):
        self.E_1 = E_1  # Elastic modulus
        self.E_2 = E_2  # Kinematic hardening modulus
        self.sigma_y0 = sigma_y0  # Initial yield stress
        self.H = H  # Isotropic hardening modulus
    
    def compute_stress(self, eps, eps_p):
        return self.E_1 * (eps - eps_p)
    
    def compute_back_stress(self, eps_p):
        return self.E_2 * eps_p
    
    def yield_function(self, sigma, X, xi):
        return abs(sigma - X) - (self.sigma_y0 + self.H * xi)
    
    def plastic_correction(self, deps, eps_p, xi):
        delta_eps_p = (self.E_1 / (self.E_1 + self.E_2 + self.H)) * deps
        delta_xi = (self.E_1 / (self.E_1 + self.E_2 + self.H)) * abs(deps)
        return delta_eps_p, delta_xi

class ElastoplasticSolver:
    """Solves the elastoplastic problem."""
    def __init__(self, material, interpolator):
        self.material = material
        self.interpolator = interpolator
    
    def solve(self):
        # Interpolate strain
        t, eps = self.interpolator.interpolate()
        
        # Initialize state variables
        eps_p = np.zeros_like(t)
        xi = np.zeros_like(t)
        sigma = np.zeros_like(t)
        X = np.zeros_like(t)
        
        # Time stepping loop
        for i in range(len(t) - 1):
            deps = eps[i + 1] - eps[i]
            
            # Elastic prediction
            sigma[i + 1] = self.material.compute_stress(eps[i + 1], eps_p[i])
            X[i + 1] = self.material.compute_back_stress(eps_p[i])
            Phi = self.material.yield_function(sigma[i + 1], X[i + 1], xi[i])
            
            # Plastic correction
            if Phi >= 0:
                delta_eps_p, delta_xi = self.material.plastic_correction(deps, eps_p[i], xi[i])
                eps_p[i + 1] = eps_p[i] + delta_eps_p
                xi[i + 1] = xi[i] + delta_xi
                sigma[i + 1] = self.material.compute_stress(eps[i + 1], eps_p[i + 1])
                X[i + 1] = self.material.compute_back_stress(eps_p[i + 1])
            else:
                eps_p[i + 1] = eps_p[i]
                xi[i + 1] = xi[i]
        
        return t, eps, sigma