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

        '''mid state calclation'''
        a_bar = 0.5*(aL+aR)
        rho_bar = 0.5*(rhoL+rhoR)
        p_star  = 0.5*(pL+pR)-0.5*(uR-uL)*rho_bar*a_bar
        u_star  = 0.5*(uL+uR)-0.5*(pR-pL)/rho_bar/a_bar
        rhoL_star = rhoL + (uL-u_star)*rho_bar/a_bar
        rhoR_star = rhoR + (u_star-uR)*rho_bar/a_bar
        aL_star = np.sqrt(gamma*p_star/max(rhoL_star,1.e-10))
        aR_star = np.sqrt(gamma*p_star/max(rhoR_star,1.e-10))

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
        if(p_from_u[i,0,0]<=0):
            print("ERROR negative density",i,p_from_u[i,0,0]  )
        elif(p_from_u[i,0,2]<=0):
            print("ERROR negative pressure",i,p_from_u[i,0,2]  )
        sound[i,0] = np.sqrt(gamma*p_from_u[i,0,2]/p_from_u[i,0,0])
        if(sound[i,0] is np.nan):
            print("ERROR sound speed",i,p_from_u[i,0,2],p_from_u[i,0,0]  )
        if(p_from_u[i,1,0]<=0):
            print("ERROR negative density",i,p_from_u[i,1,0]  )
        elif(p_from_u[i,1,2]<=0):
            print("ERROR negative pressure",i,p_from_u[i,1,2]  )
        sound[i,1] = np.sqrt(gamma*p_from_u[i,1,2]/p_from_u[i,1,0])
        if(sound[i,1] is np.nan):
            print("ERROR sound speed",i,p_from_u[i,1,2],p_from_u[i,1,0]  )
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
        u_vec_L = [rhoL, rhoL*uL, eL]
        u_vec_R = [rhoR, rhoR*uR, eR]
        fL = np.array(fL)
        fR = np.array(fR)
        u_L = np.array(u_vec_L)
        u_R = np.array(u_vec_R)
        flux[i,:] = 0.5*(fL[:]+fR[:]) - 0.5*max(aL+np.abs(uL),aR+np.abs(uR))*(u_R[:]-u_L[:])
    
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