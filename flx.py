import numpy as np

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
        sound[i,0] = np.sqrt(gamma*p_from_u[i,0,2]/max(p_from_u[i,0,0],1.e-10))
        if(p_from_u[i,1,2]<0):
            print("ERROR negative pressure",i,p_from_u[i,1,2]  )
        sound[i,1] = np.sqrt(gamma*p_from_u[i,1,2]/max(p_from_u[i,1,0],1.e-10))
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

        '''hybrid wave speed estimates/ linearised wave speed estimates'''
        sL = min(uL-aL,uR-aR)
        sR = max(uR+aR,uL+aL)
        '''mid state calclation'''
        a_bar = 0.5*(aL+aR)
        rho_bar = 0.5*(rhoL+rhoR)
        p_star  = 0.5*(pL+pR)-0.5*(uR-uL)*rho_bar*a_bar
        u_star  = 0.5*(uL+uR)-0.5*(pR-pL)/rho_bar/a_bar
        sM = (pR-pL+rhoL*uL*(sL-uL)-rhoR*uR*(sR-uR))/(rhoL*(sL-uL)-rhoR*(sR-uR))
        rhoL_star = rhoL * (sL-uL)/(sL-sM)
        rhoR_star = rhoR * (sR-uR)/(sR-sM)

        '''Flux computation of L, R, L_star, and R_star'''
        fL = [rhoL*uL,rhoL*uL**2+pL, (eL+pL)*uL]
        fR = [rhoR*uR,rhoR*uR**2+pR, (eR+pR)*uR]
        uvL = [rhoL,rhoL*uL,eL]
        uvR = [rhoR,rhoR*uR,eR]
        uv_starL = [rhoL_star,rhoL_star*sM,eL+rhoL_star*(sM-uL)*(sM+pL/(rhoL*(sL-uL)))]
        uv_starR = [rhoR_star,rhoR_star*sM,eR+rhoR_star*(sM-uR)*(sM+pR/(rhoR*(sR-uR)))]
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


def Rusanov(u_flux,p_flux,p_from_u, jmax,time_step):
    '''upwind flux'''
    nf = jmax
    flux = np.zeros((jmax,3))
    gamma = 1.4
    sound = np.zeros((nf-1,2))
    # Compute flux for internal points
    for i in range(nf-1):
        if(p_from_u[i,0,2]<=0):
            print("ERROR negative pressure",i,p_from_u[i,0,2],time_step  )
        sound[i,0] = np.sqrt(gamma*p_from_u[i,0,2]/p_from_u[i,0,0])
        if(p_from_u[i,1,2]<=0):
            print("ERROR negative pressure",i,p_from_u[i,1,2],time_step  )
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
        flux[i,:] = 0.5*(fL[:]+fR[:] - 0.5*max((np.abs(u_flux[i-1,1,1]/u_flux[i-1,1,0])+aL),(np.abs(u_flux[i,0,1]/u_flux[i,0,0])+aR))*(u_flux[i,0,:]-u_flux[i-1,1,:]))
    
    # flux[0,:] = [u_flux[0,0,1],(u_flux[0,0,1]**2/u_flux[0,0,0]+p_from_u[0,0,2]),(u_flux[0,0,2]+p_from_u[0,0,2])*p_from_u[0,0,1]]
    # flux[nf-1,:] = [u_flux[nf-2,1,1],(u_flux[nf-2,1,1]**2/u_flux[nf-2,1,0]+p_from_u[nf-2,1,2]),(u_flux[nf-2,1,2]+p_from_u[nf-2,1,2])*p_from_u[nf-2,1,1]]
    #only shu-osher
    rho1, u1, p1 = 3.857143, 2.629369, 10.3333
    rhou1 = rho1*u1
    e1 = p1/(gamma-1.0)+0.5*rho1*u1**2

    rho2 = u_flux[nf-2,1,0]
    u2 = 0.0
    p2 = p_from_u[nf-2,1,2]
    e2 = p2/(gamma-1.0)+0.5*rho2*u2**2
    rhou2 = rho2*u2
    flux[0,:] = [rhou1, rhou1**2/rho1+p1, (e1+p1)*u1]
    flux[nf-1,:] = [rhou2, rhou2**2/rho2+p2, (e2+p2)*u2]
    '''Periodic boundary condition'''
    # aL = sound[nf-2,1]
    # aR = sound[0,0]
    # eL = u_flux[nf-2,1,2]
    # eR = u_flux[0  ,0,2]
    # rhoL = u_flux[nf-2,1,0]
    # rhoR = u_flux[0,0,0]
    # uL = p_from_u[nf-2,1,1]
    # uR = p_from_u[0,  0,1]
    # pL = p_from_u[nf-2,1,2]
    # pR = p_from_u[0,  0,2]
    # fL = [rhoL*uL,rhoL*uL**2+pL, (eL+pL)*uL]
    # fR = [rhoR*uR,rhoR*uR**2+pR, (eR+pR)*uR]
    # fL = np.array(fL)
    # fR = np.array(fR)
    # flux[0,:] = 0.5*(fL[:]+fR[:] - 0.5*max((np.abs(u_flux[nf-2,1,1]/u_flux[nf-2,1,0])+sound[nf-2,0]),(np.abs(u_flux[0,0,1]/u_flux[0,0,0])+sound[0,1]))*(u_flux[0,0,:]-u_flux[nf-2,1,:]))
    # flux[nf-1,:] = 0.5*(fL[:]+fR[:] - 0.5*max((np.abs(u_flux[nf-2,1,1]/u_flux[nf-2,1,0])+sound[nf-2,0]),(np.abs(u_flux[0,0,1]/u_flux[0,0,0])+sound[0,1]))*(u_flux[0,0,:]-u_flux[nf-2,1,:]))
    return flux

def Rusanov_lobatto(u,p_from_u, jmax,time_step):
    '''upwind flux'''
    nf = jmax
    flux = np.zeros((jmax,3))
    gamma = 1.4
    sound = np.zeros((nf-1,2))
    # Compute flux for internal points
    for i in range(nf-1):
        if(p_from_u[i,0,2]<=0):
            print("ERROR negative pressure",i,p_from_u[i,0,2],time_step  )
        sound[i,0] = np.sqrt(gamma*p_from_u[i,0,2]/p_from_u[i,0,0])
        if(p_from_u[i,1,2]<=0):
            print("ERROR negative pressure",i,p_from_u[i,1,2],time_step  )
        sound[i,1] = np.sqrt(gamma*p_from_u[i,1,2]/p_from_u[i,1,0])
    for i in range(1, nf-1):
        '''Extract L and R'''
        aL = sound[i-1,1]
        aR = sound[i,0]
        eL = u[i-1,-1,2]
        eR = u[i  ,0,2]
        rhoL = u[i-1,-1,0]
        rhoR = u[i  ,0,0]
        uL = p_from_u[i-1,-1,1]
        uR = p_from_u[i,  0,1]
        pL = p_from_u[i-1,-1,2]
        pR = p_from_u[i,  0,2]
        fL = [rhoL*uL,rhoL*uL**2+pL, (eL+pL)*uL]
        fR = [rhoR*uR,rhoR*uR**2+pR, (eR+pR)*uR]
        fL = np.array(fL)
        fR = np.array(fR)
        flux[i,:] = 0.5*(fL[:]+fR[:] - 0.5*max((np.abs(u[i-1,-1,1]/u[i-1,-1,0])+aL),(np.abs(u[i,0,1]/u[i,0,0])+aR))*(u[i,0,:]-u[i-1,-1,:]))
    
    flux[0,:] = [u[0,0,1],(u[0,0,1]**2/u[0,0,0]+p_from_u[0,0,2]),(u[0,0,2]+p_from_u[0,0,2])*p_from_u[0,0,1]]
    flux[nf-1,:] = [u[nf-2,-1,1],(u[nf-2,-1,1]**2/u[nf-2,-1,0]+p_from_u[nf-2,-1,2]),(u[nf-2,-1,2]+p_from_u[nf-2,-1,2])*p_from_u[nf-2,-1,1]]
    #only shu-osher
    # rho1, u1, p1 = 3.857143, 2.629369, 10.3333
    # rhou1 = rho1*u1
    # e1 = p1/(gamma-1.0)+0.5*rho1*u1**2

    # rho2 = u_flux[nf-2,1,0]
    # u2 = 0.0
    # p2 = p_from_u[nf-2,1,2]
    # e2 = p2/(gamma-1.0)+0.5*rho2*u2**2
    # rhou2 = rho2*u2
    # flux[0,:] = [rhou1, rhou1**2/rho1+p1, (e1+p1)*u1]
    # flux[nf-1,:] = [rhou2, rhou2**2/rho2+p2, (e2+p2)*u2]
    '''Periodic boundary condition'''
    # aL = sound[nf-2,1]
    # aR = sound[0,0]
    # eL = u[nf-2,-1,2]
    # eR = u[0  ,0,2]
    # rhoL = u[nf-2,-1,0]
    # rhoR = u[0,0,0]
    # uL = p_from_u[nf-2,1,1]
    # uR = p_from_u[0,  0,1]
    # pL = p_from_u[nf-2,1,2]
    # pR = p_from_u[0,  0,2]
    # fL = [rhoL*uL,rhoL*uL**2+pL, (eL+pL)*uL]
    # fR = [rhoR*uR,rhoR*uR**2+pR, (eR+pR)*uR]
    # fL = np.array(fL)
    # fR = np.array(fR)
    # flux[0,:] = 0.5*(fL[:]+fR[:] - 0.5*max((np.abs(u[nf-2,-1,1]/u[nf-2,-1,0])+aL),(np.abs(u[0,0,1]/u[0,0,0])+aR))*(u[0,0,:]-u[nf-2,-1,:]))
    # flux[nf-1,:] = 0.5*(fL[:]+fR[:] - 0.5*max((np.abs(u[nf-2,-1,1]/u[nf-2,-1,0])+aL),(np.abs(u[0,0,1]/u[0,0,0])+aR))*(u[0,0,:]-u[nf-2,-1,:]))
    return flux