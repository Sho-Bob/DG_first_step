import numpy as np

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

def init_variable_contact_d(u,x,xq):
    u_shape = u.shape
    x_elem = np.zeros((u.shape[0],u.shape[1]))
    gamma = 1.4
    rhoL = 1.0
    pL = 1.0
    uL = 1.0
    rhoR= 0.125
    pR=1.0
    uR=1.0
    for i in range(u_shape[0]): # element loop
        for j in range(u_shape[1]): # np loop
            x_elem[i,j] = 0.5*(x[i+1]+x[i]) + 0.5*(x[i+1]-x[i])*xq[j]
            # u[i,j,0] = 2.0+np.sin(2.0*x_elem[i,j]*np.pi)
            # u[i,j,1] = u[i,j,0]
            # u[i,j,2] = (pL)/(gamma-1)+0.5*u[i,j,1]**2/u[i,j,0]
            if(x_elem[i,j]<0.75 and x_elem[i,j]>0.25):
                u[i,j,0] = rhoL
                u[i,j,1] = rhoL*uL
                u[i,j,2] = pL/(gamma-1)+0.5*rhoL*uL**2
            else:
                u[i,j,0] = rhoR
                u[i,j,1] = rhoR*uR
                u[i,j,2] = pR/(gamma-1)+0.5*rhoR*uR**2
    return u,x_elem