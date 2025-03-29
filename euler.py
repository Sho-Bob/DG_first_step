import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from gauss_lobatto import gauss_lobatto_points
from gauss_legendre import gauss_legendre_points
import interpolation
import RK2nd as RK2

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

def init_variable_sod(u,x,xq):
    u_shape = u.shape
    x_elem = np.zeros((u.shape[0],u.shape[1]))
    gamma = 1.4
    rhoL = 1.0
    pL = 1.0
    uL = 0.0
    rhoR= 0.125
    pR=0.1
    uR=0.0
    for i in range(u_shape[0]): # element loop
        for j in range(u_shape[1]): # np loop
            x_elem[i,j] = 0.5*(x[i+1]+x[i]) + 0.5*(x[i+1]-x[i])*xq[j]
            if(x_elem[i,j]<0.5):
                u[i,j,0] = rhoL
                u[i,j,1] = rhoL*uL
                u[i,j,2] = pL/(gamma-1)+0.5*rhoL*uL**2
            if(x_elem[i,j]>=0.5):
                u[i,j,0] = rhoR
                u[i,j,1] = rhoR*uR
                u[i,j,2] = pR/(gamma-1)+0.5*rhoR*uR**2
    return u,x_elem

def flux_pint_coor(x,flux_points):
    Ne = x.shape[0]-1
    nf = len(flux_points)
    x_flux_coord = np.zeros((Ne,nf))
    for i in range(Ne):
        for j in range(nf):
            x_flux_coord[i,j] = 0.5*(x[i+1]+x[i]) + 0.5*(x[i+1]-x[i])*flux_points[j]
    return x_flux_coord

def HLLC(u_flux,p_flux,p_from_u, jmax,flag_p):
    '''upwind flux'''
    nf = jmax
    flux = np.zeros((jmax,3))
    gamma = 1.4
    sound = np.zeros((nf-1,2))
    # Compute flux for internal points
    for i in range(nf-1):
        if(p_from_u[i,0,2]<0):
            print("ERROR negative pressure",i,p_from_u[i,1,2]  )
        sound[i,0] = np.sqrt(gamma*p_from_u[i,0,2]/p_from_u[i,0,0])
        if(p_from_u[i,1,2]<0):
            print("ERROR negative pressure",i,p_from_u[i,1,2]  )
        sound[i,1] = np.sqrt(gamma*p_from_u[i,1,2]/p_from_u[i,1,0])
    for i in range(1, nf-1):
        '''Extract L and R'''
        aL = sound[i-1,1]
        aR = sound[i,0]
        eL = u_flux[i-1,1,2]
        eR = u_flux[i  ,0,2]
        rhoL = u_flux[i-1,1,0]
        rhoR = u_flux[i,0,0]
        uL = p_from_u[i-1,1,1]
        uR = p_from_u[i,  0,1]
        pL = p_from_u[i-1,1,2]
        pR = p_from_u[i,  0,2]

        '''mid state calclation'''
        a_bar = 0.5*(aL+aR)
        rho_bar = 0.5*(rhoL+rhoR)
        p_star  = 0.5*(pL+pR)-0.5*(uR-uL)*rho_bar*a_bar
        u_star  = 0.5*(uL+uR)-0.5*(pR-pL)/rho_bar/a_bar
        rhoL_star = rhoL + (uL-u_star)*rho_bar/a_bar
        rhoR_star = rhoR + (u_star-uR)*rho_bar/a_bar
        aL_star = np.sqrt(gamma*p_star/rhoL_star)
        aR_star = np.sqrt(gamma*p_star/rhoR_star)

        '''hybrid wave speed estimates/ linearised wave speed estimates'''
        hL = p_star/pL
        hR = p_star/pR
        # if(hL<=1.0):
        #     sL = uL-aL
        # else:
        #     sL = uL-aL*(np.sqrt(1.0+0.5*(gamma+1.0)/gamma*(hL-1.0)))
        # if(hR<=1.0):
        #     sR = uR-aR
        # else:
        #     sR = uR-aR*(np.sqrt(1.0+0.5*(gamma+1.0)/gamma*(hR-1.0)))
        sL = min(uL-aL,u_star-aL_star)
        sR = max(uR+aR,u_star+aR_star)
        sM = u_star

        '''Flux computation of L, R, L_star, and R_star'''
        fL = [rhoL*uL,rhoL*uL**2+pL, (eL+pL)*uL]
        fR = [rhoR*uR,rhoR*uR**2+pR, (eR+pR)*uR]
        uvL = [rhoL,rhoL*uL,eL]
        uvR = [rhoR,rhoR*uR,eR]
        uv_starL = [rhoL_star,rhoL_star*u_star,p_star/(gamma-1.0)+rhoL_star*u_star**2]
        uv_starR = [rhoR_star,rhoR_star*u_star,p_star/(gamma-1.0)+rhoR_star*u_star**2]
        fL = np.array(fL)
        fR = np.array(fR)
        uvL = np.array(uvL)
        uvR = np.array(uvR)
        uv_starL = np.array(uv_starL)
        uv_starR = np.array(uv_starR)
        '''Upwinding'''
        if sL>0:
            flux[i,:] = fL[:]
        elif sL<=0 and sM>=0:
            flux[i,:] = fL[:] + sL*(uv_starL[:]-uvL[:])
        elif sM<0 and sR>=0:
            flux[i,:] = fR[:] + sR*(uv_starR[:]-uvR[:])
        elif sR<0:
            flux[i,:] = fR[:]

    flux[0,:] = [u_flux[0,0,1],(u_flux[0,0,1]**2/u_flux[0,0,0]+p_from_u[0,0,2]),(u_flux[0,0,2]+p_from_u[0,0,2])*p_from_u[0,0,1]]
    flux[nf-1,:] = [u_flux[nf-2,1,1],(u_flux[nf-2,1,1]**2/u_flux[nf-2,1,0]+p_from_u[nf-2,1,2]),(u_flux[nf-2,1,2]+p_from_u[nf-2,1,2])*p_from_u[nf-2,1,1]]
    return flux

def Lax(u_flux,p_flux,p_from_u, jmax,flag_p,dt,dx):
    '''upwind flux'''
    nf = jmax
    flux = np.zeros((jmax,3))
    gamma = 1.4
    sound = np.zeros((nf-1,2))
    # Compute flux for internal points
    for i in range(nf-1):
        sound[i,0] = np.sqrt(gamma*p_from_u[i,0,2]/p_from_u[i,0,0])
        sound[i,1] = np.sqrt(gamma*p_from_u[i,1,2]/p_from_u[i,1,0])
    for i in range(1, nf-1):
        '''Extract L and R'''
        aL = sound[i-1,1]
        aR = sound[i,0]
        eL = u_flux[i-1,1,2]
        eR = u_flux[i  ,0,2]
        rhoL = u_flux[i-1,1,0]
        rhoR = u_flux[i,0,0]
        uL = p_from_u[i-1,1,1]
        uR = p_from_u[i,  0,1]
        pL = p_from_u[i-1,1,2]
        pR = p_from_u[i,  0,2]
        fL = [rhoL*uL,rhoL*uL**2+pL, (eL+pL)*uL]
        fR = [rhoR*uR,rhoR*uR**2+pR, (eR+pR)*uR]
        fL = np.array(fL)
        fR = np.array(fR)
        # flux[i,:] = 0.5*(fL[:]+fR[:] - dt/dx[i]*(u_flux[i,0,:]-u_flux[i-1,1,:]))
        flux[i,:] = 0.5*(fL[:]+fR[:] - max(aL+uL,aR+uR)*(u_flux[i,0,:]-u_flux[i-1,1,:]))
    flux[0,:] = [u_flux[0,0,1],(u_flux[0,0,1]**2/u_flux[0,0,0]+p_from_u[0,0,2]),(u_flux[0,0,2]+p_from_u[0,0,2])*p_from_u[0,0,1]]
    flux[nf-1,:] = [u_flux[nf-2,1,1],(u_flux[nf-2,1,1]**2/u_flux[nf-2,1,0]+p_from_u[nf-2,1,2]),(u_flux[nf-2,1,2]+p_from_u[nf-2,1,2])*p_from_u[nf-2,1,1]]
    return flux


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

def trunc(a, decimals=8):
        '''
    This function truncates a float to a specified decimal place.
    Adapted from:
    https://stackoverflow.com/questions/42021972/
    truncating-decimal-digits-numpy-array-of-floats

    Inputs:
    -------
        a: value(s) to truncate
        decimals: truncated decimal place

    Outputs:
    --------
        truncated float
    '''
        return np.trunc(a*10**decimals)/(10**decimals)

def PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_av,prim_v_cell_av):
    flux_shape = u_flux.shape
    u_shape = u.shape
    RES_TOL = 1.e-10
    unfiltered = u.copy()
    unfiltered_f = u_flux.copy()
    min_density1 = 0.0
    min_density2 = 0.0
    min_density = 0.0
    min_pressure = 0.0
    min_pressure1 = 0.0
    min_pressure2 = 0.0
    alpha = 0.3
    '''Fiirst,limit with the denstiy'''
    for i in range(u_shape[0]):
        rho_bar = prim_v_cell_av[i,0]
        theta1 = np.abs((rho_bar-RES_TOL)/(rho_bar-primitive_variable[i,:,0]+RES_TOL))
        theta2 = np.abs((rho_bar-RES_TOL)/(rho_bar-p_from_u_flux[i,:,0]+RES_TOL))
        theta3 = trunc(np.minimum(1.0, np.min(theta1)))
        theta4 = trunc(np.minimum(1.0, np.min(theta2)))
        theta = min(theta3,theta4)
        # min_density1 = np.min(primitive_variable[i,:,0])
        # min_density2 = np.min(p_from_u_flux[i,:,0])
        # min_density = min(min_density1,min_density2)
        # theta = np.abs((prim_v_cell_av[i,0]-alpha*prim_v_cell_av[i,0])/(prim_v_cell_av[i,0]-min_density+RES_TOL))
        # theta = trunc(theta)
        # theta = min(1.0,theta)
        # print(theta)
        # theta = 0.0
        for j in range(u_shape[1]):
            u[i,j,0] = theta*u[i,j,0] + (1.0-theta)*cons_v_cell_av[i,0]
        for j in range(flux_shape[1]):
            u_flux[i,j,0] = theta*u_flux[i,j,0] + (1.0-theta)*cons_v_cell_av[i,0]
    primitive_variable = compute_primitive(u)
    p_from_u_flux = compute_primitive(u_flux)

    unfiltered = u.copy()
    unfiltered_f = u_flux.copy()
    '''Second, limit with the pressure'''
    for i in range(u_shape[0]):
        p_bar = prim_v_cell_av[i,2]
        theta1 = np.abs((p_bar-alpha*p_bar)/(p_bar-primitive_variable[i,:,2]+RES_TOL))
        theta2 = np.abs((p_bar-alpha*p_bar)/(p_bar-p_from_u_flux[i,:,2]+RES_TOL))
        theta3 = trunc(np.minimum(1.0, np.min(theta1)))
        theta4 = trunc(np.minimum(1.0, np.min(theta2)))
        theta = min(theta3,theta4)
        # min_pressure1 = np.min(primitive_variable[i,:,2])
        # min_pressure2 = np.min(p_from_u_flux[i,:,2])
        # min_pressure = min(min_pressure1,min_pressure2)
        # theta = np.abs((prim_v_cell_av[i,2]-alpha*prim_v_cell_av[i,2])/(prim_v_cell_av[i,2]-min_pressure+RES_TOL))
        # theta = trunc(theta)
        # theta = min(1.0,theta)
        # print(theta)
        # theta = 0.0
        for j in range(u_shape[1]):
            u[i,j,:] = theta*unfiltered[i,j,:] + (1.0-theta)*cons_v_cell_av[i,:]
        for j in range(flux_shape[1]):
            u_flux[i,j,:] = theta*unfiltered_f[i,j,:] + (1.0-theta)*cons_v_cell_av[i,:]
    # primitive_variable = compute_primitive(u)
    # p_from_u_flux = compute_primitive(u_flux)

    return u_flux, u

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
    num_element = jmax-1
    approx_order = 2
    flux_number = 2
    time = 0.0
    flux_type = 'HLLC'
    # number of nodes in each element: At quad points
    Np = approx_order+1
    u = np.zeros((num_element,Np,3))
    cons_v_cell_average = np.zeros((num_element))
    prim_v_cell_average = np.zeros((num_element))
    u_flux = np.zeros((num_element,flux_number,3))
    du = np.zeros_like(u)
    a = 1.0
    CFL = 0.1
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
    u,x_element = init_variable_sod(u,x,xq_points)
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


    while (time < 0.12):
        coefs = [0.5,1.0]
        un = u.copy()
        dt = compute_dt(dx,primitive_variable,CFL)
        cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
        u_old = u.copy()
        # Initialize u_flux at each time step
        primitive_variable = compute_primitive(u)
        u_flux = np.zeros((num_element, flux_number,3))
        p_flux = np.zeros((num_element, flux_number,3))
        p_from_u_flux = np.zeros((num_element, flux_number,3))
        
        # Compute u_flux
        u_flux, p_flux = interpolation.compute_flux_point_values(u, primitive_variable, basis_val_flux_points, num_element, flux_number, Np)
        p_from_u_flux = compute_primitive(u_flux)

        '''Can add some limniters here'''
        u_flux, u = PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
        primitive_variable = compute_primitive(u)
        p_from_u_flux = compute_primitive(u_flux)
        # Compute flux coordinates
        x_flux_coord = flux_pint_coor(x, flux_points)
        
        # Compute flux values
        if(flux_type == 'Rusanov'):
            flux = Lax(u_flux,p_flux,p_from_u_flux,jmax,flag_p,dt,dx)
        elif(flux_type == 'HLLC'):
            flux = HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)

        '''Time integration RK2nd 1st step'''
        u = RK2.RK2nd_1st(u,un,primitive_variable,Stiff,flux,basis_val_flux_points,dt,inverse_M,element_trans,num_element,Np)

        u_old = u.copy()
        primitive_variable = compute_primitive(u)
        u_flux = np.zeros((num_element, flux_number,3))
        p_flux = np.zeros((num_element, flux_number,3))
        p_from_u_flux = np.zeros((num_element, flux_number,3))
        
        # Compute u_flux
        u_flux, p_flux = interpolation.compute_flux_point_values(u, primitive_variable, basis_val_flux_points, num_element, flux_number, Np)
        p_from_u_flux = compute_primitive(u_flux)

        '''Can add some limniters here'''
        u_flux, u = PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
        primitive_variable = compute_primitive(u)
        p_from_u_flux = compute_primitive(u_flux)
        # Compute flux coordinates
        x_flux_coord = flux_pint_coor(x, flux_points)
        
        # Compute flux values
        if(flux_type == 'Rusanov'):
            flux = Lax(u_flux,p_flux,p_from_u_flux,jmax,flag_p,dt,dx)
        elif(flux_type == 'HLLC'):
            flux = HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)
        
        '''Time integration RK2nd 2nd step'''
        u = RK2.RK2nd_2nd_step(u,u_old,un,primitive_variable,Stiff,flux,basis_val_flux_points,dt,inverse_M,element_trans)
        time += dt
    

    # print(x_element.shape)
    # print(u.shape)
    x_coord = x_element.flatten()
    u_coord = u[:,:,0].flatten()
    p_coord = primitive_variable[:,:,2].flatten()

    # x_coord = x_flux_coord.flatten()
    # u_coord = p_from_u_flux[:,:,2].flatten()

    u_ini_coord = u_ini[:,:,0].flatten()
    

    plt.figure(figsize=(8, 6))
    plt.plot(x_coord, u_coord, linestyle='-', color='k', label='Density')
    plt.plot(x_coord, p_coord, linestyle='-', color='r', label='Pressure')
    # plt.plot(x_coord, u_ini_coord, marker='o', linestyle='-', color='r', label='initial')
    plt.xlabel('x')
    plt.ylabel('Density')
    # plt.title('Plot of x_coord vs u_coord')
    plt.legend()
    plt.grid(True)
    plt.show()



# def initial_baisis_setup(x,jmax):

