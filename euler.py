import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from gauss_lobatto import gauss_lobatto_points
from gauss_legendre import gauss_legendre_points
import interpolation
import RK2nd as RK2
import limiters as lim
import flx as flx
import plotter 
import init

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

def flux_pint_coor(x,flux_points):
    Ne = x.shape[0]-1
    nf = len(flux_points)
    x_flux_coord = np.zeros((Ne,nf))
    for i in range(Ne):
        for j in range(nf):
            x_flux_coord[i,j] = 0.5*(x[i+1]+x[i]) + 0.5*(x[i+1]-x[i])*flux_points[j]
    return x_flux_coord


def compute_stiff_matrix2(Mass_matrix,grad_basis_val):
    stiff_matrix = np.zeros_like(Mass_matrix)
    mass_shape = Mass_matrix.shape
    for i in range(mass_shape[0]):
        for j in range(mass_shape[1]):
            for k in range(grad_basis_val.shape[0]):
                stiff_matrix[i,j] += Mass_matrix[i,k]*grad_basis_val[j,k]
    return stiff_matrix

def compute_dt(dx,primitive,CFL):
    shape = primitive.shape
    gamma = 1.4
    dt_candidate = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(primitive[i,j,2]<0):
                print("ERROR in", i, j, primitive[i,j,2])
            acoustic = np.sqrt(gamma*primitive[i,j,2]/primitive[i,j,0])
            dt_candidate = CFL*dx[i]/(acoustic+np.abs(primitive[i,j,1]))
    dt_candidate = dt_candidate.flatten()
    dt = np.min(dt_candidate)
    return dt

def compute_primitive(u):
    u_shape = u.shape
    primitive_variable = np.zeros_like(u)
    gamma =1.4
    for i in range(u_shape[0]):
        for j in range(u_shape[1]):
            primitive_variable[i,j,0] = u[i,j,0]
            primitive_variable[i,j,1] = u[i,j,1]/u[i,j,0]
            primitive_variable[i,j,2] = (gamma-1.0)*(u[i,j,2]-0.5*u[i,j,1]**2/u[i,j,0])
    return primitive_variable

def compute_cell_average(u,primitive_variable,xq_weights,trans_matrix,dx):
    u_shape = u.shape
    cons_v_cell_average = np.zeros((u_shape[0],3))
    prim_v_cell_average = np.zeros((u_shape[0],3))

    for i in range(u_shape[0]):
        for j in range(u_shape[1]):
            cons_v_cell_average[i,:] += trans_matrix[i]*u[i,j,:]*xq_weights[j]/dx[i]
            prim_v_cell_average[i,:] += trans_matrix[i]*primitive_variable[i,j,:]*xq_weights[j]/dx[i]

    return cons_v_cell_average, prim_v_cell_average 

if __name__ == "__main__":
    jmax = 201
    time_step = 0
    num_element = jmax-1
    approx_order = 2
    flux_number = 2
    time = 0.0
    flux_type = 'Rusanov' # 'Rusanov' or 'HLLC'
    limiter_type = 'PP' # 'PP' or 'minmod'
    initial = 'sod' # 'sod' or 'contact'
    # number of nodes in each element: At quad points
    Np = approx_order+1
    u = np.zeros((num_element,Np,3))
    cons_v_cell_average = np.zeros((num_element))
    prim_v_cell_average = np.zeros((num_element))
    u_flux = np.zeros((num_element,flux_number,3))
    du = np.zeros_like(u)
    a = 1.0
    CFL = 0.03
    gamma = 1.4
    flag_p = False
    x_min, x_max = 0.0,1.0
    x = np.linspace(x_min,x_max,jmax)
    dx = np.zeros(num_element)
    for i in range(jmax-1):
        dx[i] = x[i+1]-x[i]
    # dt = 0.02*np.min(dx)/a
    element_trans = transform_mat(x)
    xq_points, xq_weights = gauss_legendre_points(Np)
    if(initial == 'sod'):
        u,x_element = init.init_variable_sod(u,x,xq_points)
    elif(initial == 'contact'):
        u,x_element = init.init_variable_contact_d(u,x,xq_points)
    primitive_variable = np.zeros_like(u) # [rho, u, p]
    primitive_variable = compute_primitive(u)
    cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
    u_ini = u.copy()

    flux_points = [-1.0,1.0]
    Mass = compute_mass_matrix(xq_points,xq_weights)
    Stiff = compute_stiff_matrix(xq_points,xq_weights)
    inverse_M = np.linalg.inv(Mass)
    basis_val_flux_points = init_lag_poly_all(xq_points,flux_points)
    grad_basis_val_at_nodes = init_lag_poly_grad_all(xq_points,xq_points)
    Stiff2 = compute_stiff_matrix2(Mass,grad_basis_val_at_nodes)
    dt = compute_dt(dx,primitive_variable,CFL)
    print("Precompute martices and weights are done")
    # Initialize u_flux at each time step
    primitive_variable = compute_primitive(u)
    u_flux = np.zeros((num_element, flux_number,3))
    p_flux = np.zeros((num_element, flux_number,3))
    p_from_u_flux = np.zeros((num_element, flux_number,3))
    # Compute u_flux
    u_flux, p_flux = interpolation.compute_flux_point_values(u, primitive_variable, basis_val_flux_points, num_element, flux_number, Np)
    p_from_u_flux = compute_primitive(u_flux)
    x_flux_coord = flux_pint_coor(x,flux_points)
    # plotter.plot(u_flux,x_flux_coord,p_from_u_flux)

    while (time < 0.12):

        un = u.copy()
        dt = compute_dt(dx,primitive_variable,CFL)
        cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
        u_old = u.copy()
        
        
        # Compute flux values
        if(flux_type == 'HLLC'):
            flux = flx.HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)
        elif(flux_type == 'Rusanov'):
            flux = flx.Rusanov(u_flux,p_flux,p_from_u_flux,jmax)

        '''Time integration RK2nd 1st step'''
        u = RK2.RK2nd_1st(u,un,primitive_variable,Stiff,flux,basis_val_flux_points,dt,inverse_M,element_trans,num_element,Np)
        
        cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
        '''Can add some limniters here'''
        u_flux,p_flux = interpolation.compute_flux_point_values(u, primitive_variable, basis_val_flux_points, num_element, flux_number, Np)
        p_from_u_flux = compute_primitive(u_flux)
        
        if(limiter_type == 'PP'):
            u_flux,u = lim.PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
        elif(limiter_type == 'minmod'):
            u_flux,u = lim.minmod(u,u_flux,cons_v_cell_average,x,x_element)
        primitive_variable = compute_primitive(u)
        p_from_u_flux = compute_primitive(u_flux)
        # print("Before limiter")
        # plotter.plot(u,x_element,primitive_variable)
        u_old = u.copy()

        # print("After limiter")
        # plotter.plot(u,x_element,primitive_variable)

        # Compute flux values
        if(flux_type == 'Rusanov'):
            flux = flx.Rusanov(u_flux,p_flux,p_from_u_flux,jmax)
        elif(flux_type == 'HLLC'):
            flux = flx.HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)
        
        '''Time integration RK2nd 2nd step'''
        u = RK2.RK2nd_2nd_step(u,u_old,un,primitive_variable,Stiff,flux,basis_val_flux_points,dt,inverse_M,element_trans)
        primitive_variable_2nd = compute_primitive(u)
        cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable_2nd,xq_weights,element_trans,dx)
        if(limiter_type == 'PP'):
            # Check for negative density or pressure and restart with smaller dt if needed
            if (cons_v_cell_average[:,0] < 0).any() or (prim_v_cell_average[:,2] < 0).any():
                # Reset time step and variables
                dt = dt/2
                u = un.copy() # Reset to start of time step
                primitive_variable = compute_primitive(u)
                continue # Restart the time step
            else:
                u_flux,p_flux = interpolation.compute_flux_point_values(u, primitive_variable, basis_val_flux_points, num_element, flux_number, Np)
                p_from_u_flux = compute_primitive(u_flux)   
                u_flux, u = lim.PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
                primitive_variable = compute_primitive(u)
                p_from_u_flux = compute_primitive(u_flux)
                cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
                time += dt
                time_step += 1
        elif(limiter_type == 'minmod'):
            u_flux,p_flux = interpolation.compute_flux_point_values(u, primitive_variable, basis_val_flux_points, num_element, flux_number, Np)
            p_from_u_flux = compute_primitive(u_flux)   
            # u_flux, u = lim.PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
            u_flux, u = lim.minmod(u,u_flux,cons_v_cell_average,x,x_element)
            primitive_variable = compute_primitive(u)
            p_from_u_flux = compute_primitive(u_flux)
            cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
            time += dt
            time_step += 1
            # if(time_step % 2 == 0):
            #     plotter.plot(u,x_element,primitive_variable)
            #     plotter.plot_cell_average(cons_v_cell_average,prim_v_cell_average)

        

    x_coord = x_element.flatten()
    u_coord = u[:,:,0].flatten()
    p_coord = primitive_variable[:,:,2].flatten()
    velo_coord = primitive_variable[:,:,1].flatten()

    # x_coord = x_flux_coord.flatten()
    # u_coord = p_from_u_flux[:,:,2].flatten()

    x_cell_average = np.linspace(0,1,num_element)

    u_ini_coord = u_ini[:,:,0].flatten()
    

    plt.figure(figsize=(8, 6))
    plt.plot(x_coord, u_coord, linestyle='-', color='k', label='Density')
    plt.plot(x_coord, p_coord, linestyle='-', color='r', label='Pressure')
    plt.plot(x_coord, velo_coord, linestyle='-', color='g', label='Velocity')
    # plt.plot(x_cell_average, prim_v_cell_average[:,0],linestyle='-', color='b', label='Cell average density')
    # plt.plot(x_cell_average, prim_v_cell_average[:,1],linestyle='-', color='g', label='Cell average velocity')
    # plt.plot(x_cell_average, prim_v_cell_average[:,2],linestyle='-', color='r', label='Cell average pressure')
    # plt.plot(x_coord, u_ini_coord, marker='o', linestyle='-', color='r', label='initial')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('cells = '+str(jmax)+', order = '+str(approx_order)+', flux = '+flux_type+', limiter = '+limiter_type + ', CFL = '+str(CFL))
    plt.legend()
    plt.grid(True)
    plt.show()



# def initial_baisis_setup(x,jmax):

