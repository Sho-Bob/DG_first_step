import numpy as np
from scipy.special import legendre

def gauss_lobatto_points(n):
    """
    Computes the Gauss-Lobatto quadrature points and weights.

    Parameters:
    n (int): The number of quadrature points (including endpoints -1 and 1).

    Returns:
    tuple: A tuple containing two numpy arrays:
           - points: The Gauss-Lobatto quadrature points.
           - weights: The corresponding weights for the quadrature.
    """
    # Check that n is at least 2
    if n < 2:
        raise ValueError("n must be at least 2 to have meaningful Gauss-Lobatto points.")

    # The endpoints are always included
    points = np.zeros(n)
    weights = np.zeros(n)

    # The endpoints -1 and 1
    points[0] = -1
    points[-1] = 1

    if n > 2:
        # Compute the roots of the derivative of the Legendre polynomial P_{n-1}'(x)
        P = legendre(n - 1)  # Legendre polynomial of degree n-1
        dP = np.polyder(P)   # Derivative of P_{n-1}
        interior_points = np.roots(dP)  # Roots of P_{n-1}'(x)

        # Sort roots to maintain order
        interior_points.sort()

        # Set the interior points
        points[1:-1] = interior_points

    # Compute the weights
    weights[0] = 2 / (n * (n - 1))
    weights[-1] = weights[0]

    for i in range(1, n - 1):
        weights[i] = 2 / (n * (n - 1) * (legendre(n - 1)(points[i]) ** 2))

    return points, weights

# Example usage
if __name__ == "__main__":
    n = 5  # Number of quadrature points
    points, weights = gauss_lobatto_points(n)

    print(f"Gauss-Lobatto quadrature points for n = {n}: {points}")
    print(f"Corresponding weights for n = {n}: {weights}")