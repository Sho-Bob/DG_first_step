import numpy as np
import euler as el



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

def PP_limiter_Lobatto(u,primitive_variable,cons_v_cell_av,prim_v_cell_av,limiter_val1,limiter_val2):
    u_shape = u.shape
    u1 = u.copy()
    u2 = u.copy()
    RES_TOL = 1.e-12
    for i in range(u_shape[0]):
        rho_bar = prim_v_cell_av[i,0]
        min_rho = np.min(primitive_variable[i,:,0])
        theta = np.abs((rho_bar-RES_TOL)/(rho_bar-min_rho+RES_TOL))
        theta =  trunc(np.minimum(1.0, theta))
        # print(theta)
        limiter_val1[i] = theta
        for j in range(u_shape[1]):
            u[i,j,:] = theta*u[i,j,:] + (1.0-theta)*cons_v_cell_av[i,:]
    primitive_variable2 = el.compute_primitive(u)

    for i in range(u_shape[0]):
        p_bar = prim_v_cell_av[i,2]
        p_min = np.min(primitive_variable2[i,:,2])
        theta = np.abs((p_bar)/(p_bar-p_min+RES_TOL))
        theta =  trunc(np.minimum(1.0, theta))
        # theta = trunc(np.minimum(1.0, np.min(theta1)))
        if(theta<1.0):
            print("LIMITER",theta,i)
        limiter_val2[i] = theta
        for j in range(u_shape[1]):
            u[i,j,:] = theta*u[i,j,:] + (1.0-theta)*cons_v_cell_av[i,:]


def PP_limiter(u_flux,u,p_from_u_flux,primitive_variable,cons_v_cell_av,prim_v_cell_av,quad_type):
    flux_shape = u_flux.shape
    u_shape = u.shape
    RES_TOL = 1.e-12
    u1 = u.copy()
    u_flux1 = u_flux.copy()
    u2 = u.copy()
    u_flux2 = u_flux.copy()
    '''Fiirst,limit with the denstiy'''
    for i in range(u_shape[0]):
        rho_bar = prim_v_cell_av[i,0]
        theta1 = np.abs((rho_bar-RES_TOL)/(rho_bar-primitive_variable[i,:,0]+RES_TOL))
        theta2 = np.abs((rho_bar-RES_TOL)/(rho_bar-p_from_u_flux[i,:,0]+RES_TOL))
        theta3 = trunc(np.minimum(1.0, np.min(theta1)))
        theta4 = trunc(np.minimum(1.0, np.min(theta2)))
        theta = min(theta3,theta4)

        if(quad_type == 'Gauss'):
            for j in range(u_shape[1]):
                u1[i,j,0] = theta*u[i,j,0] + (1.0-theta)*cons_v_cell_av[i,0]
            for j in range(flux_shape[1]):
                u_flux1[i,j,0] = theta*u_flux[i,j,0] + (1.0-theta)*cons_v_cell_av[i,0]
        elif(quad_type == 'Lobatto'):
            for j in range(u_shape[1]):
                u1[i,j,0] = theta*u[i,j,0] + (1.0-theta)*cons_v_cell_av[i,0]
    if(quad_type == 'Lobatto'):
        u_flux1[:,0,0] = u1[:,0,0]
        u_flux1[:,1,0] = u1[:,-1,0]
    primitive_variable2 = el.compute_primitive(u1)
    p_from_u_flux2 = el.compute_primitive(u_flux1)

    # unfiltered = u1.copy()
    # unfiltered_f = u_flux1.copy()
    '''Second, limit with the pressure'''
    for i in range(u_shape[0]):
        p_bar = prim_v_cell_av[i,2]
        theta1 = np.abs((p_bar)/(p_bar-primitive_variable2[i,:,2]+RES_TOL))
        theta2 = np.abs((p_bar)/(p_bar-p_from_u_flux2[i,:,2]+RES_TOL))
        theta3 = trunc(np.minimum(1.0, np.min(theta1)))
        theta4 = trunc(np.minimum(1.0, np.min(theta2)))
        theta =  min(theta3,theta4)

        if(quad_type == 'Gauss'):
            for j in range(u_shape[1]):
                u2[i,j,:] = theta*u1[i,j,:] + (1.0-theta)*cons_v_cell_av[i,:]
            for j in range(flux_shape[1]):
                u_flux2[i,j,:] = theta*u_flux1[i,j,:] + (1.0-theta)*cons_v_cell_av[i,:]
        elif(quad_type == 'Lobatto'):
            for j in range(u_shape[1]):
                u2[i,j,:] = theta*u1[i,j,:] + (1.0-theta)*cons_v_cell_av[i,:]
    if(quad_type == 'Lobatto'):
        u_flux2[:,0,:] = u2[:,0,:]
        u_flux2[:,1,:] = u2[:,-1,:]

    return u_flux2, u2

def minmod(u,u_flux,cons_v_cell_av,x,x_element):
    flux_shape = u_flux.shape
    u_shape = u.shape
    RES_TOL = 1.e-12
    unfiltered = u.copy()
    unfiltered_f = u_flux.copy()
    u1 = u.copy()
    u_flux1 = u_flux.copy()
    for i in range(1,u_shape[0]-1): # element loop
        element_center = (x[i]+x[i+1])/2
        # element_center = np.mean(x_element[i,:])
        h = x[i+1] - x[i]
        diff1 = cons_v_cell_av[i,:]-u_flux[i,0,:]
        diff2 = cons_v_cell_av[i,:]-cons_v_cell_av[i-1,:]
        diff3 = cons_v_cell_av[i+1,:]-cons_v_cell_av[i,:]
        diff4 = u_flux[i,1,:] - cons_v_cell_av[i,:]
        diff5 = cons_v_cell_av[i,:] - cons_v_cell_av[i-1,:]
        diff6 = cons_v_cell_av[i+1,:] - cons_v_cell_av[i,:]
        for k in range(3):
            filtered_val1 = cons_v_cell_av[i,k] - minmod_func(diff1[k],diff2[k],diff3[k])
            filtered_val2 = cons_v_cell_av[i,k] + minmod_func(diff4[k],diff5[k],diff6[k])
            if (filtered_val1 == u_flux[i,0,k] and filtered_val2 == u_flux[i,1,k]):
                print("No filter",i)
                continue
            else:
                u_flux1[i,0,k] = filtered_val1 
                u_flux1[i,1,k] = filtered_val2
                
                for l in range(u_shape[1]):
                    grad = minmod_func(diff1[k]*2/h, 2.0*diff2[k]/h, 2.0*diff3[k]/h)
                    u1[i,l,k] = cons_v_cell_av[i,k] + grad*(x_element[i,l]-element_center)
                    u_flux1[i,0,k] = cons_v_cell_av[i,k] + grad*(x[i]-element_center)
                    u_flux1[i,1,k] = cons_v_cell_av[i,k] + grad*(x[i+1]-element_center)
                    # u1[i,l,k] = cons_v_cell_av[i,k] 
    #Dirichlet boundary condition
    u1[0,:,:] = cons_v_cell_av[0,:]
    u1[-1,:,:] = cons_v_cell_av[-1,:]
    u_flux1[0,:,:] = cons_v_cell_av[0,:]
    u_flux1[-1,:,:] = cons_v_cell_av[-1,:]

    return u_flux1, u1

def minmod_func(a,b,c):
    if(np.sign(a) == np.sign(b) and np.sign(a) == np.sign(c)):
        return np.sign(a)*min(np.abs(a),np.abs(b),np.abs(c))
    else:
        return 0.0