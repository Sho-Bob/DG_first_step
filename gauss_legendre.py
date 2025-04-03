import numpy as np
from scipy.special import roots_legendre

def gauss_legendre_points(n):
    """
    Compute the Gauss-Legendre quadrature points and weights for a given degree n.

    Parameters:
    n (int): Degree of the polynomial for Gauss-Legendre quadrature points.

    Returns:
    tuple: A tuple containing two numpy arrays:
           - points: The Gauss-Legendre quadrature points.
           - weights: The corresponding weights for the quadrature.
    """
    # Compute the points (roots of the Legendre polynomial) and weights
    points, weights = roots_legendre(n)

    return points, weights

if __name__ == "__main__":
# Example: Compute Gauss-Legendre points and weights for degree 5
    n = 1
    points, weights = gauss_legendre_points(n)
    print("Gauss-Legendre quadrature points for n =", n, ":", points)
    print("Corresponding weights for n =", n, ":", weights)