import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from gauss_lobatto import gauss_lobatto_points
from gauss_legendre import gauss_legendre_points

# Compute dx/dr 
def transform_mat(x):
    num_elem = len(x)-1
    trans = np.zeros(num_elem)
    for i in range(num_elem):
        trans[i] = 0.5*(x[i+1]-x[i])
    return trans

# Initialize lagrange poly
def init_lag_poly_all(xq_points,point):
    """
    Calculates the Lagrange basis polynomial values at given points.
    Returns:
    A 2D array with the dimensions of [points_number,Nq]
    """
    order = len(xq_points)
    order2 = len(point)
    basis_val = np.zeros((order,order2))
    
    for i in range(order):
        for j in range(order2):
            basis_val[i][j] = 1.0
            for k in range(order):
                if k !=i:
                    basis_val[i][j] *= (point[j]-xq_points[k])/(xq_points[i]-xq_points[k])
                # print(basis_val[i])
    return basis_val

def init_lag_poly_indivi(xq_points,point):
    """
    Calculates the Lagrange basis polynomial values at given points.
    Returns:
    A 1D array with the dimensions of [Nq]
    """
    order = len(xq_points)
    basis_val = np.zeros(order)
    
    for i in range(order):
        basis_val[i] = 1.0
        for k in range(order):
            if k !=i:
                basis_val[i] *= (point-xq_points[k])/(xq_points[i]-xq_points[k])
                # print(basis_val[i])
    return basis_val

def init_lag_poly_grad_all(xq_points, points):
    """
    Calculates the derivative of Lagrange basis polynomials at given points.
    Returns:
    A 2D array with the dimensions of [points_number, Nq (basis_number)]
    """
    order = len(xq_points)  # Number of interpolation points
    order2 = len(points)    # Number of evaluation points
    basis_grad_val = np.zeros((order, order2))  # Initialize derivatives array

    # Loop over each Lagrange basis polynomial
    for i in range(order):
        for j in range(order2):
            # Calculate the derivative at points[j] for the i-th basis polynomial
            for l in range(order):
                if l != i:
                    # Compute the product term for the derivative formula
                    lag_temp = 1.0
                    for k in range(order):
                        if k != i and k != l:
                            lag_temp *= (points[j] - xq_points[k]) / (xq_points[i] - xq_points[k])
                    basis_grad_val[i][j] += lag_temp / (xq_points[i] - xq_points[l])
    
    return basis_grad_val

def init_lag_poly_grad_indivi(xq_points, point):
    """
    Calculates the derivative of Lagrange basis polynomials at given points.
    Returns:
    A 1D array with the dimensions of [Nq (basis_number)]
    """
    order = len(xq_points)  # Number of interpolation points
    basis_grad_val = np.zeros(order)  # Initialize derivatives array

    # Loop over each Lagrange basis polynomial
    for i in range(order):
        # Calculate the derivative at points[j] for the i-th basis polynomial
        for l in range(order):
            if l != i:
                # Compute the product term for the derivative formula
                lag_temp = 1.0
                for k in range(order):
                    if k != i and k != l:
                        lag_temp *= (point - xq_points[k]) / (xq_points[i] - xq_points[k])
                basis_grad_val[i] += lag_temp / (xq_points[i] - xq_points[l])
    
    return basis_grad_val

def compute_mass_matrix(xq_points,xq_weights):
    '''Output: M_ij [Np,Np]'''
    order = len(xq_points)
    M = np.zeros((order,order))
    basis_val_node = init_lag_poly_all(xq_points,xq_points)
    for i in range(order):
        for j in range(order):
            for k in range(len(xq_points)):
                M[i][j] += basis_val_node[k,i]*basis_val_node[k,j]*xq_weights[k]
    return M

def compute_stiff_matrix(xq_points,xq_weights):
    '''Output: L_ij [Np,Np]'''

    order = len(xq_points)
    L = np.zeros((order,order))
    basis_value = init_lag_poly_all(xq_points,xq_points)
    basis_grad_value = init_lag_poly_grad_all(xq_points,xq_points)
    for i in range(order):
        for j in range(order):
            for k in range(len(xq_points)):
                L[i][j] = basis_value[k,i]*basis_grad_value[k,j]*xq_weights[k]
    return L

def init_variable(u,x,xq):
    u_shape = u.shape
    x_elem = np.zeros_like(u)
    for i in range(u_shape[0]): # element loop
        for j in range(u_shape[1]): # nq loop
            x_elem[i,j] = 0.5*(x[i+1]+x[i]) + 0.5*(x[i+1]-x[i])*xq[j]
            u[i,j] = np.sin(2.0*x_elem[i,j]*np.pi)
    return u,x_elem

if __name__ == "__main__":
    jmax = 11
    num_element = jmax-1
    approx_order = 2
    time = 0.0
    dt = 0.01
    Np = approx_order+1
    u = np.zeros((num_element,Np))
    du = np.zeros_like(u)
    a = 1.0

    gamma = 1.4

    x_min, x_max = 0.0,1.0
    x = np.linspace(x_min,x_max,jmax)
    element_trans = transform_mat(x)

    xq_points, xq_weights = gauss_legendre_points(Np)

    u,x_element = init_variable(u,x,xq_points)
    # print(x_element)
    flux_points = [-1.0,1.0]
    Mass = compute_mass_matrix(xq_points,xq_weights)
    Stiff = compute_stiff_matrix(xq_points,xq_weights)
    inverse_M = np.linalg.inv(Mass)
    
    res1 = np.zeros(Np)
    for k in range(num_element):
        for i in range(Np): # node loop
            for j in range(Np): #basis loop
                res1[i] += -a*u[k,j]*Stiff[i,j]
            du[k,i] = 1/(element_trans[k])*inverse_M[i,j]*res1[i]



    
    # res1 = 



    x_coord = x_element.flatten()
    u_coord = du.flatten()

    plt.figure(figsize=(8, 6))
    plt.plot(x_coord, u_coord, marker='o', linestyle='-', color='b', label='Data points')
    plt.xlabel('x_coord')
    plt.ylabel('u_coord')
    plt.title('Plot of x_coord vs u_coord')
    plt.legend()
    plt.grid(True)
    plt.show()



# def initial_baisis_setup(x,jmax):

