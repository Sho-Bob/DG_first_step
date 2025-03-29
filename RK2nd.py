import numpy as np

"""
    u_old is the old conservative variable vector
    u is the new conservative variable vector
    Stiff is the stiffness matrix
    flux is the flux
    basis_val_flux_points is the basis value at the flux points
    dt is the time step
    inverse_M is the inverse of the mass matrix
    element_trans is the element transformation matrix
    num_element is the number of elements
    Np is the number of points
"""
"""Original version: for clarity"""
def RK2nd_1st1(u,un,primitive_variable,Stiff,flux,basis_val_flux_points,dt,inverse_M,element_trans,num_element,Np):
    du = np.zeros_like(u)
    res1 = np.zeros_like(u)
    res2 = np.zeros_like(u)
    for k in range(num_element):
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
    return u

def RK2nd_1st(u, un, primitive_variable, Stiff, flux, basis_val_flux_points,
              dt, inverse_M, element_trans, num_element, Np):
    """
    Fully vectorized version of the DG update. 
    Each array is assumed to have the shapes consistent with:
        u, un, primitive_variable: (num_element, Np, 3)
        Stiff, inverse_M: (Np, Np)
        flux: (num_element+1, 3)
        basis_val_flux_points: (Np, 2)
        element_trans: (num_element,)
    """

    # 1) Precompute the flux-terms for the volume integral
    #    flux_term_0 = rhou
    #    flux_term_1 = (rhou^2 / rho) + p
    #    flux_term_2 = (rho*E + p) * u
    flux_terms = np.empty_like(u)  # shape (num_element, Np, 3)
    flux_terms[:, :, 0] = u[:, :, 1]
    flux_terms[:, :, 1] = (u[:, :, 1]**2 / u[:, :, 0]) + primitive_variable[:, :, 2]
    flux_terms[:, :, 2] = (u[:, :, 2] + primitive_variable[:, :, 2]) * primitive_variable[:, :, 1]

    # 2) Volume integral: res1 = - sum_j( flux_terms[...,j,:] * Stiff[j,i] )
    #    Using einstein summation: 'kjv, ji -> kiv'
    #    k = element index, j = summation over Np, i = new node index, v = variable
    res1 = -np.einsum('kjv,ji->kiv', flux_terms, Stiff)  # shape (num_element, Np, 3)

    # 3) Surface flux: res2[k, i, :] = flux[k+1,:]*phi_right(i) - flux[k,:]*phi_left(i)
    #    Broadcast over i:
    #      flux[1:, None, :] has shape (num_element, 1, 3)
    #      basis_val_flux_points[None, :, 1, None] has shape (1, Np, 1)
    #    giving final shape (num_element, Np, 3)
    res2 = (
        flux[1:, None, :] * basis_val_flux_points[None, :, 1, None]
      - flux[:-1, None, :] * basis_val_flux_points[None, :, 0, None]
    )  # shape (num_element, Np, 3)

    # 4) Sum volume + surface residual
    total_res = res1 + res2  # shape (num_element, Np, 3)

    # 5) Multiply by inverse mass matrix, then by dt/element_trans
    #    du[k, i, v] = - (dt / element_trans[k]) * sum_j( inverse_M[i, j] * total_res[k, j, v] )
    #    'ij, kjv -> kiv'
    du = np.einsum('ij,kjv->kiv', inverse_M, total_res)

    # Multiply by -dt / element_trans (broadcast in k dimension only)
    dt_factor = -dt / element_trans  # shape (num_element,)
    dt_factor = dt_factor[:, None, None]  # shape (num_element, 1, 1)
    du = dt_factor * du  # shape (num_element, Np, 3)

    # 6) Update solution
    u = un + du  # shape (num_element, Np, 3)

    return u

""" Original version: for clarity
for k in range(num_element):            
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
                u[k, i,:] = 0.5*(un[k,i,:]+u_old[k,i,:]+ du[k, i,:])
"""


def RK2nd_2nd_step(
    u, u_old, un, primitive_variable,
    Stiff, flux, basis_val_flux_points,
    dt, inverse_M, element_trans
):
    """
    Vectorized RK2 second step.

    Parameters
    ----------
    u : ndarray, shape (num_element, Np, 3)
        Current solution guess (will be updated).
    u_old : ndarray, shape (num_element, Np, 3)
        Saved solution from previous stage (or from previous iteration).
    un : ndarray, shape (num_element, Np, 3)
        Intermediate solution from RK first step, etc.
    primitive_variable : ndarray, shape (num_element, Np, 3)
        E.g. [rho, velocity, pressure] for each node.
    Stiff : ndarray, shape (Np, Np) or (num_element, Np, Np)
        Stiffness matrix. If shape (Np, Np), the same for all elements.
    flux : ndarray, shape (num_element+1, 3)
        Flux at each element boundary.
    basis_val_flux_points : ndarray, shape (Np, 2)
        The shape function values at left (col 0) and right (col 1).
    dt : float
        Time step.
    inverse_M : ndarray, shape (Np, Np) or (num_element, Np, Np)
        Inverse mass matrix.
    element_trans : ndarray, shape (num_element,)
        Element transformation factors, e.g. geometric scaling.

    Returns
    -------
    u : ndarray, shape (num_element, Np, 3)
        Updated solution after the second RK step.
    """
    num_element, Np, _ = u.shape
    
    # 1) Build the volume flux terms for each node
    #    flux_term_0 = rhou
    #    flux_term_1 = (rhou^2 / rho) + p
    #    flux_term_2 = (rho*E + p) * u
    flux_terms = np.empty_like(u)  # shape (num_element, Np, 3)
    flux_terms[:, :, 0] = u[:, :, 1]
    flux_terms[:, :, 1] = (u[:, :, 1]**2 / u[:, :, 0]) + primitive_variable[:, :, 2]
    flux_terms[:, :, 2] = (u[:, :, 2] + primitive_variable[:, :, 2]) * primitive_variable[:, :, 1]
    
    # 2) Volume integral: res1 = - sum_j( flux_terms[k,j,v] * Stiff[j,i] )
    #    Using Einstein summation:
    #    If Stiff is shape (Np, Np), do 'kjv,ji->kiv'
    #    If Stiff is shape (num_element, Np, Np), do 'kjv,kji->kiv'
    
    if Stiff.ndim == 2:
        # Same stiffness matrix for every element
        res1 = -np.einsum('kjv,ji->kiv', flux_terms, Stiff)  # (num_element, Np, 3)
    else:
        # Per-element stiffness matrix
        res1 = -np.einsum('kjv,kji->kiv', flux_terms, Stiff)  # (num_element, Np, 3)

    # 3) Surface flux: res2[k,i,:] = flux[k+1,:]*phi_right(i) - flux[k,:]*phi_left(i)
    res2 = (
        flux[1:, None, :] * basis_val_flux_points[None, :, 1, None]
      - flux[:-1, None, :] * basis_val_flux_points[None, :, 0, None]
    )  # shape (num_element, Np, 3)
    
    # Sum them up
    total_res = res1 + res2  # (num_element, Np, 3)
    
    # 4) Multiply by inverse mass matrix and factor dt/element_trans:
    #    du[k,i,v] = - (dt / element_trans[k]) * sum_j( inverse_M[i,j] * total_res[k,j,v] )
    #    or in Einstein notation:
    #      If inverse_M is shape (Np, Np) -> 'ij, kjv->kiv'
    #      If inverse_M is shape (num_element, Np, Np) -> 'kij,kjv->kiv'
    
    if inverse_M.ndim == 2:
        du = np.einsum('ij,kjv->kiv', inverse_M, total_res)
    else:
        du = np.einsum('kij,kjv->kiv', inverse_M, total_res)
    
    dt_factor = -dt / element_trans  # shape (num_element,)
    dt_factor = dt_factor[:, None, None]  # (num_element, 1, 1)
    du = dt_factor * du  # shape (num_element, Np, 3)
    
    # 5) Final solution update
    #    u[k,i,:] = 0.5*(un[k,i,:] + u_old[k,i,:] + du[k,i,:])
    #    => u = 0.5*(un + u_old) + 0.5*du
    # or you could do: u = 0.5*(un + u_old) + du * 0.5
    
    u = 0.5*(un + u_old) + 0.5*du
    
    return u