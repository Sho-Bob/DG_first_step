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
    A 2D array with the dimensions of [Np, number of points]
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
    A 2D array with the dimensions of [Np (basis_number), points_number]
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
                M[i,j] += basis_val_node[i,k]*basis_val_node[j,k]*xq_weights[k]
    return M

def compute_stiff_matrix(xq_points,xq_weights):
    '''Output: L_ij [Np,Np]'''

    order = len(xq_points)
    L = np.zeros((order,order))
    basis_value = init_lag_poly_all(xq_points,xq_points) # li(r_k)
    basis_grad_value = init_lag_poly_grad_all(xq_points,xq_points) # dlj/dx(r_k)
    for i in range(order): # loop for li(rk)
        for j in range(order): # loop for dlj/dx(rk)
            for k in range(len(xq_points)):
                L[i][j] += basis_value[i,k]*basis_grad_value[j,k]*xq_weights[k]
    return L

def init_variable(u,x,xq):
    u_shape = u.shape
    x_elem = np.zeros_like(u)
    for i in range(u_shape[0]): # element loop
        for j in range(u_shape[1]): # np loop
            x_elem[i,j] = 0.5*(x[i+1]+x[i]) + 0.5*(x[i+1]-x[i])*xq[j]
            u[i,j] = np.sin(2.0*x_elem[i,j]*np.pi)
    return u,x_elem

def flux_pint_coor(x,flux_points):
    Ne = x.shape[0]-1
    nf = len(flux_points)
    x_flux_coord = np.zeros((Ne,nf))
    for i in range(Ne):
        for j in range(nf):
            x_flux_coord[i,j] = 0.5*(x[i+1]+x[i]) + 0.5*(x[i+1]-x[i])*flux_points[j]
    return x_flux_coord

def compute_flux(u_flux,a,jmax):
    '''Just an upwind flux'''

    nf = jmax
    flux = np.zeros(jmax)
    for i in range(1,nf-1):
        flux[i] = 0.5*(a*u_flux[i-1,1]+a*u_flux[i,0]) - 0.5*np.abs(a)*(u_flux[i,0]-u_flux[i-1,1])
    #periodic boundary 
    flux[0]  = 0.5*(a*u_flux[0,0]+a*u_flux[nf-2,1]) - 0.5*np.abs(a)*(u_flux[0,0]-u_flux[nf-2,1])
    flux[nf-1] = 0.5*(a*u_flux[0,0]+a*u_flux[nf-2,1])- 0.5*np.abs(a)*(u_flux[0,0]-u_flux[nf-2,1])
    return flux

def compute_stiff_matrix2(Mass_matrix,grad_basis_val):
    stiff_matrix = np.zeros_like(Mass_matrix)
    mass_shape = Mass_matrix.shape
    for i in range(mass_shape[0]):
        for j in range(mass_shape[1]):
            for k in range(grad_basis_val.shape[0]):
                stiff_matrix[i,j] += Mass_matrix[i,k]*grad_basis_val[j,k]
    return stiff_matrix

if __name__ == "__main__":
    jmax = 5
    num_element = jmax-1
    approx_order = 3
    flux_number = 2
    time = 0.0
    Np = approx_order+1
    u = np.zeros((num_element,Np))
    u_flux = np.zeros((num_element,flux_number))
    du = np.zeros_like(u)
    a = 1.0

    gamma = 1.4

    x_min, x_max = 0.0,1.0
    x = np.linspace(x_min,x_max,jmax)
    dx = np.zeros(jmax-1)
    for i in range(jmax-1):
        dx[i] = x[i+1]-x[i]
    dt = 0.1*np.min(dx)/a
    element_trans = transform_mat(x)

    xq_points, xq_weights = gauss_legendre_points(Np)

    u,x_element = init_variable(u,x,xq_points)
    u_ini = u.copy()
    # print(x_element)
    flux_points = [-1.0,1.0]
    Mass = compute_mass_matrix(xq_points,xq_weights)
    Stiff = compute_stiff_matrix(xq_points,xq_weights)
    inverse_M = np.linalg.inv(Mass)
    basis_val_flux_points = init_lag_poly_all(xq_points,flux_points)
    grad_basis_val_at_nodes = init_lag_poly_grad_all(xq_points,xq_points)
    Stiff2 = compute_stiff_matrix2(Mass,grad_basis_val_at_nodes)
    # print(basis_val_flux_points)

    while (time < 1.0):
        coefs = [0.5,1.0]
        u_old = u.copy()
        for coef in coefs:
            # Initialize u_flux at each time step
            u_flux = np.zeros((num_element, flux_number))
            
            # Compute u_flux
            for k in range(num_element):
                for i in range(flux_number):
                    for j in range(Np):
                        u_flux[k, i] += u[k, j] * basis_val_flux_points[j, i]

            # Compute flux coordinates
            x_flux_coord = flux_pint_coor(x, flux_points)
            
            # Compute flux values
            flux = compute_flux(u_flux, a,jmax)
            
            # Loop over elements to update residuals and solution
            du = np.zeros_like(u)  # Initialize du to zero at each time step
            res1 = np.zeros_like(u)
            res2 = np.zeros_like(u)

            # print(Mass)
            for k in range(num_element):
                # Initialize residuals for each element
                
                # Compute residuals
                for i in range(Np):  # basis loop
                    for j in range(Np):  # node loop
                        res1[k,i] += -a * u[k, j] * Stiff[j, i]
                    res2[k,i] = (flux[k + 1] * basis_val_flux_points[i, 1] - flux[k] * basis_val_flux_points[i, 0])

                # Update du and u
                
                for i in range(Np):
                    for j in range(Np):
                        du[k, i] += -coef*dt / element_trans[k] * inverse_M[i, j] * (res1[k,j]+res2[k,j])
                    u[k, i] = u_old[k,i]+ du[k, i]
        time += dt
        
        print("time is", time)
    

    x_coord = x_element.flatten()
    u_coord = u.flatten()


    u_ini_coord = u_ini.flatten()
    

    plt.figure(figsize=(8, 6))
    plt.plot(x_coord, u_coord, marker='o', linestyle='-', color='b', label='Data points')
    plt.plot(x_coord, u_ini_coord, marker='o', linestyle='-', color='r', label='initial')
    plt.xlabel('x_coord')
    plt.ylabel('u_coord')
    plt.title('Plot of x_coord vs u_coord')
    plt.legend()
    plt.grid(True)
    plt.show()



# def initial_baisis_setup(x,jmax):

