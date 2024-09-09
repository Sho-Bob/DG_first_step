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

        '''gybrid wave speed estimates/ linearised wave speed estimates'''
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
        flux[i,:] = 0.5*(fL[:]+fR[:] - dt/dx[i]*(u_flux[i,0,:]-u_flux[i-1,1,:]))
    flux[0,:] = [u_flux[0,0,1],(u_flux[0,0,1]**2/u_flux[0,0,0]+p_from_u[0,0,2]),(u_flux[0,0,2]+p_from_u[0,0,2])*p_from_u[0,0,1]]
    flux[nf-1,:] = [u_flux[nf-2,1,1],(u_flux[nf-2,1,1]**2/u_flux[nf-2,1,0]+p_from_u[nf-2,1,2]),(u_flux[nf-2,1,2]+p_from_u[nf-2,1,2])*p_from_u[nf-2,1,1]]
    return flux

def Rusanov(u_flux,p_flux,p_from_u, jmax):
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
        flux[i,:] = 0.5*(fL[:]+fR[:] - 0.5*max((np.abs(u_flux[i-1,1,1]/u_flux[i-1,1,0])+sound[i,0]),(np.abs(u_flux[i,0,1]/u_flux[i,0,0])+sound[i,1]))*(u_flux[i,0,:]-u_flux[i-1,1,:]))
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

def PP_limiter(u_flux1,u1,p_from_u_flux,primitive_variable,cons_v_cell_av,prim_v_cell_av):
    flux_shape = u_flux1.shape
    u_shape = u1.shape
    RES_TOL = 1.e-13
    unfiltered = u1.copy()
    unfiltered_f = u_flux1.copy()
    theta = np.zeros(u_shape[0])
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
        theta[i] = min(theta3,theta4)
        for j in range(u_shape[1]):
            u1[i,j,0] = theta[i]*u1[i,j,0] + (1.0-theta[i])*cons_v_cell_av[i,0]
        for j in range(flux_shape[1]):
            u_flux1[i,j,0] = theta[i]*u_flux1[i,j,0] + (1.0-theta[i])*cons_v_cell_av[i,0]
    primitive_variable = compute_primitive(u1)
    p_from_u_flux = compute_primitive(u_flux1)

    unfiltered = u1
    unfiltered_f = u_flux1
    '''Second, limit with the pressure'''
    for i in range(u_shape[0]):
        p_bar = prim_v_cell_av[i,2]
        theta1 = np.abs((p_bar-RES_TOL)/(p_bar-primitive_variable[i,:,2]+RES_TOL))
        theta2 = np.abs((p_bar-RES_TOL)/(p_bar-p_from_u_flux[i,:,2]+RES_TOL))
        theta3 = trunc(np.minimum(1.0, np.min(theta1)))
        theta4 = trunc(np.minimum(1.0, np.min(theta2)))
        theta[i] = min(theta3,theta4)
        for j in range(u_shape[1]):
            u1[i,j,:] = theta[i]*unfiltered[i,j,:] + (1.0-theta[i])*cons_v_cell_av[i,:]
        for j in range(flux_shape[1]):
            u_flux1[i,j,:] = theta[i]*unfiltered_f[i,j,:] + (1.0-theta[i])*cons_v_cell_av[i,:]
    # primitive_variable = compute_primitive(u)
    # p_from_u_flux = compute_primitive(u_flux)

    return u_flux1, u1, theta

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
    jmax = 101
    num_element = jmax-1
    approx_order = 2
    flux_number = 2
    time = 0.0
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
    print(xq_points)

    u,x_element = init_variable_sod(u,x,xq_points)
    primitive_variable = np.zeros_like(u) # [rho, u, p]
    primitive_variable = compute_primitive(u)
    cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
    u_ini = u.copy()
    # print(x_element)
    flux_points = [-1.0,1.0]
    Mass = compute_mass_matrix(xq_points,xq_weights)
    Stiff = compute_stiff_matrix(xq_points,xq_weights)
    inverse_M = np.linalg.inv(Mass)
    basis_val_flux_points = init_lag_poly_all(xq_points,flux_points)
    grad_basis_val_at_nodes = init_lag_poly_grad_all(xq_points,xq_points)
    Stiff2 = compute_stiff_matrix2(Mass,grad_basis_val_at_nodes)
    dt = compute_dt(dx,primitive_variable,CFL)
    u_flux = np.zeros((num_element, flux_number,3))
    p_flux = np.zeros((num_element, flux_number,3))
    p_from_u_flux = np.zeros((num_element, flux_number,3))
    
    
    print("Initial done")
    # print(basis_val_flux_points)

    while (time < 0.2):
        coefs = [0.5,1.0]
        un = u.copy()
        dt = compute_dt(dx,primitive_variable,CFL)
        cons_v_cell_average = 0.0
        prim_v_cell_average = 0.0
        cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
        # u_old = u.copy()
        # Initialize u_flux at each time step
        primitive_variable = compute_primitive(u)
        u_flux = np.zeros((num_element, flux_number,3))
        p_flux = np.zeros((num_element, flux_number,3))
        p_from_u_flux = np.zeros((num_element, flux_number,3))
        # Compute u_flux
        for k in range(num_element):
            for i in range(flux_number):
                for j in range(Np):
                    u_flux[k, i,:] += u[k, j,:] * basis_val_flux_points[j, i]
                    p_flux[k, i,:] += primitive_variable[k,j,:]*basis_val_flux_points[j, i]
        p_from_u_flux = compute_primitive(u_flux)

        '''Can add some limniters here'''
        u_flux, u, theta = PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
        primitive_variable = compute_primitive(u)
        p_from_u_flux = compute_primitive(u_flux)
        # Compute flux coordinates
        x_flux_coord = flux_pint_coor(x, flux_points)
        
        # Compute flux values
        # flux = HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)
        flux = Rusanov(u_flux,p_flux,p_from_u_flux, jmax)
        # flux = Lax(u_flux,p_flux,p_from_u_flux,jmax,flag_p,dt,dx)
        
        # Loop over elements to update residuals and solution
        du = np.zeros_like(u)  # Initialize du to zero at each time step
        res1 = np.zeros_like(u)
        res2 = np.zeros_like(u)

        '''Time integration SSPRK3rd'''
        # print(Mass)
        for k in range(num_element):
            # Initialize residuals for each element
            
            # Compute residuals
            for i in range(Np):  # basis loop
                for j in range(Np):  # node loop
                    res1[k,i,0] += - u[k,j,1]* Stiff[j, i]
                    res1[k,i,1] += - (u[k,j,1]**2/u[k,j,0]+ primitive_variable[k,j,2])*Stiff[j,i]
                    res1[k,i,2] += - (u[k,j,2]+primitive_variable[k,j,2])*primitive_variable[k,j,1]*Stiff[j,i]
                res2[k,i,:] = (flux[k + 1,:] * basis_val_flux_points[i, 1] - flux[k,:] * basis_val_flux_points[i, 0])

            # Update du and u
            
            for i in range(Np):
                for j in range(Np):
                    du[k, i,:] += -dt / element_trans[k] * inverse_M[i, j] * (res1[k,j,:]+res2[k,j,:])
                u[k, i,:] = un[k,i,:]+ du[k, i,:]

        u_old = u
        # primitive_variable = np.zeros_like(u) # [rho, u, p]
        primitive_variable = compute_primitive(u)
        u_flux = np.zeros((num_element, flux_number,3))
        p_flux = np.zeros((num_element, flux_number,3))
        p_from_u_flux = np.zeros((num_element, flux_number,3))
        
        # Compute u_flux
        for k in range(num_element):
            for i in range(flux_number):
                for j in range(Np):
                    u_flux[k, i,:] += u[k, j,:] * basis_val_flux_points[j, i]
                    p_flux[k, i,:] += primitive_variable[k,j,:]*basis_val_flux_points[j, i]
        p_from_u_flux = compute_primitive(u_flux)

        '''Can add some limniters here'''
        u_flux, u, theta = PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
        primitive_variable = compute_primitive(u)
        p_from_u_flux = compute_primitive(u_flux)
        # Compute flux coordinates
        x_flux_coord = flux_pint_coor(x, flux_points)
        
        # Compute flux values
        # flux = HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)
        flux = Rusanov(u_flux,p_flux,p_from_u_flux, jmax)
        # flux = Lax(u_flux,p_flux,p_from_u_flux,jmax,flag_p,dt,dx)
        
        # Loop over elements to update residuals and solution
        du = np.zeros_like(u)  # Initialize du to zero at each time step
        res1 = np.zeros_like(u)
        res2 = np.zeros_like(u)

        '''Time integration SSPRK3rd-2'''
        # print(Mass)
        for k in range(num_element):
            # Initialize residuals for each element
            
            # Compute residuals
            for i in range(Np):  # basis loop
                for j in range(Np):  # node loop
                    res1[k,i,0] += - u[k,j,1]* Stiff[j, i]
                    res1[k,i,1] += - (u[k,j,1]**2/u[k,j,0]+ primitive_variable[k,j,2])*Stiff[j,i]
                    res1[k,i,2] += - (u[k,j,2]+primitive_variable[k,j,2])*primitive_variable[k,j,1]*Stiff[j,i]
                res2[k,i,:] = (flux[k + 1,:] * basis_val_flux_points[i, 1] - flux[k,:] * basis_val_flux_points[i, 0])

            # Update du and u
            
            for i in range(Np):
                for j in range(Np):
                    du[k, i,:] += -dt / element_trans[k] * inverse_M[i, j] * (res1[k,j,:]+res2[k,j,:])
                u[k, i,:] = 0.25*(3.0*un[k,i,:]+u_old[k,i,:]+ du[k, i,:])
            # u1 = u.copy()
            # u_flux1 = u_flux.copy()
        # u_old = u.copy()
        # primitive_variable = np.zeros_like(u) # [rho, u, p]
        primitive_variable = compute_primitive(u)
        u_flux = np.zeros((num_element, flux_number,3))
        p_flux = np.zeros((num_element, flux_number,3))
        p_from_u_flux = np.zeros((num_element, flux_number,3))
        
        # Compute u_flux
        for k in range(num_element):
            for i in range(flux_number):
                for j in range(Np):
                    u_flux[k, i,:] += u[k, j,:] * basis_val_flux_points[j, i]
                    p_flux[k, i,:] += primitive_variable[k,j,:]*basis_val_flux_points[j, i]
        p_from_u_flux = compute_primitive(u_flux)

        '''Can add some limniters here'''
        u_flux, u, theta = PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_average,prim_v_cell_average)
        primitive_variable = compute_primitive(u)
        p_from_u_flux = compute_primitive(u_flux)
        # Compute flux coordinates
        x_flux_coord = flux_pint_coor(x, flux_points)
        
        # Compute flux values
        # flux = HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)
        flux = Rusanov(u_flux,p_flux,p_from_u_flux, jmax)
        # flux = Lax(u_flux,p_flux,p_from_u_flux,jmax,flag_p,dt,dx)
        
        # Loop over elements to update residuals and solution
        du = np.zeros_like(u)  # Initialize du to zero at each time step
        res1 = np.zeros_like(u)
        res2 = np.zeros_like(u)

        '''Time integration SSPRK3rd-2'''
        # print(Mass)
        for k in range(num_element):
            # Initialize residuals for each element
            
            # Compute residuals
            for i in range(Np):  # basis loop
                for j in range(Np):  # node loop
                    res1[k,i,0] += - u[k,j,1]* Stiff[j, i]
                    res1[k,i,1] += - (u[k,j,1]**2/u[k,j,0]+ primitive_variable[k,j,2])*Stiff[j,i]
                    res1[k,i,2] += - (u[k,j,2]+primitive_variable[k,j,2])*primitive_variable[k,j,1]*Stiff[j,i]
                res2[k,i,:] = (flux[k + 1,:] * basis_val_flux_points[i, 1] - flux[k,:] * basis_val_flux_points[i, 0])

            # Update du and u
            
            for i in range(Np):
                for j in range(Np):
                    du[k, i,:] += -2.0*dt / element_trans[k] * inverse_M[i, j] * (res1[k,j,:]+res2[k,j,:])
                u[k, i,:] =(un[k,i,:]+2.0*u_old[k,i,:]+ du[k, i,:])/3.0
            
        time += dt
        # print(time)
    

    # print(x_element.shape)
    # print(u.shape)
    cons_v_cell_average, prim_v_cell_average = compute_cell_average(u,primitive_variable,xq_weights,element_trans,dx)
    density = cons_v_cell_average[:,0]
    velocity = prim_v_cell_average[:,1]
    pressure = prim_v_cell_average[:,2]
    x2 = np.linspace(x_min,x_max,num_element)
    x_coord = x_element.flatten()
    u_coord = u[:,:,0].flatten()
    p_coord = primitive_variable[:,:,2].flatten()
    velo_coord = primitive_variable[:,:,1].flatten()

    # x_coord = x_flux_coord.flatten()
    # u_coord = p_from_u_flux[:,:,2].flatten()

    u_ini_coord = u_ini[:,:,0].flatten()
    

    plt.figure(figsize=(8, 6))
    plt.plot(x2, density, marker='o', color='b', label='Density',markersize=4)
    plt.plot(x2, velocity, marker='o', color='r', label='Velocity',markersize=4)
    plt.plot(x2, pressure, marker='o', color='g', label='Pressure',markersize=4)
    # plt.plot(x2, theta, marker='o', color='y', label='Theta',markersize=4)
    # plt.plot(x_coord, u_coord, marker='o', color='b', label='Density',markersize=4)
    # plt.plot(x_coord, velo_coord, marker='o', color='r', label='Velocity',markersize=4)
    # plt.plot(x_coord, p_coord, marker='o', color='g', label='Pressure',markersize=4)
    plt.xlabel('x')
    # plt.ylabel('D')
    # plt.title('Plot of x_coord vs u_coord')
    plt.legend()
    plt.grid(True)
    plt.show()



# def initial_baisis_setup(x,jmax):

