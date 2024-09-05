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

        '''mid state calclation'''
        a_bar = 0.5*(aL+aR)
        rho_bar = 0.5*(rhoL+rhoR)
        p_star  = 0.5*(pL+pR)-0.5*(uR-uL)*rho_bar*a_bar
        u_star  = 0.5*(uL+uR)-0.5*(pR-pL)/rho_bar/a_bar
        rhoL_star = rhoL + (uL-u_star)*rho_bar/a_bar
        rhoR_star = rhoR + (u_star-uR)*rho_bar/a_bar
        aL_star = np.sqrt(gamma*p_star/rhoL_star)
        aR_star = np.sqrt(gamma*p_star/rhoR_star)

        '''gybrid wave speed estimates'''
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
        flux[i,:] = 0.5*(fL[:]+fR[:] - dt/dx[i]*(u_flux[i-1,1,:]+u_flux[i,0,:]))
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

if __name__ == "__main__":
    jmax = 101
    num_element = jmax-1
    approx_order = 0
    flux_number = 2
    time = 0.0
    Np = approx_order+1
    u = np.zeros((num_element,Np,3))
    u_flux = np.zeros((num_element,flux_number,3))
    du = np.zeros_like(u)
    a = 1.0
    CFL = 0.1
    gamma = 1.4
    flag_p = False
    x_min, x_max = 0.0,1.0
    x = np.linspace(x_min,x_max,jmax)
    dx = np.zeros(jmax-1)
    for i in range(jmax-1):
        dx[i] = x[i+1]-x[i]
    # dt = 0.02*np.min(dx)/a
    element_trans = transform_mat(x)

    xq_points, xq_weights = gauss_legendre_points(Np)

    u,x_element = init_variable_sod(u,x,xq_points)
    primitive_variable = np.zeros_like(u) # [rho, u, p]
    primitive_variable = compute_primitive(u)
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
    print("Initial done")
    # print(basis_val_flux_points)

    while (time < 0.1):
        coefs = [0.5,1.0]
        u_old = u.copy()
        dt = compute_dt(dx,primitive_variable,CFL)
        
        for coef in coefs:
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

            # Compute flux coordinates
            x_flux_coord = flux_pint_coor(x, flux_points)
            
            # Compute flux values
            flux = HLLC(u_flux,p_flux,p_from_u_flux,jmax,flag_p)
            # flux = Lax(u_flux,p_flux,p_from_u_flux,jmax,flag_p,dt,dx)
            
            # Loop over elements to update residuals and solution
            du = np.zeros_like(u)  # Initialize du to zero at each time step
            res1 = np.zeros_like(u)
            res2 = np.zeros_like(u)

            '''Time integration RK2nd'''
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
                        du[k, i,:] += -coef*dt / element_trans[k] * inverse_M[i, j] * (res1[k,j,:]+res2[k,j,:])
                    u[k, i,:] = u_old[k,i,:]+ du[k, i,:]
            
        time += dt
        print(time)
    

    # print(x_element.shape)
    # print(u.shape)
    x_coord = x_element.flatten()
    u_coord = primitive_variable[:,:,2].flatten()

    # x_coord = x_flux_coord.flatten()
    # u_coord = p_from_u_flux[:,:,2].flatten()

    u_ini_coord = u_ini[:,:,0].flatten()
    

    plt.figure(figsize=(8, 6))
    plt.plot(x_coord, u_coord, marker='o', linestyle='-', color='b', label='Latest data')
    # plt.plot(x_coord, u_ini_coord, marker='o', linestyle='-', color='r', label='initial')
    plt.xlabel('x_coord')
    plt.ylabel('u_coord')
    plt.title('Plot of x_coord vs u_coord')
    plt.legend()
    plt.grid(True)
    plt.show()



# def initial_baisis_setup(x,jmax):

