import numpy as np

"""
        u_flux is the conservative variable vector at the flux points
        p_flux is the primitive variable vector at the flux points
        p_from_u_flux is the primitive variable vector at the flux points from u_flux
"""
def compute_flux_point_values(u, primitive_variable, basis_val_flux_points, num_element, flux_number, Np):
    # Use numpy's optimized matrix operations instead of explicit loops
    # Reshape arrays for matrix multiplication
    u_reshaped = u.reshape(num_element, Np, 3)
    p_reshaped = primitive_variable.reshape(num_element, Np, 3)
    
    # Perform matrix multiplication for all elements at once
    u_flux = np.einsum('ijk,jl->ilk', u_reshaped, basis_val_flux_points)
    p_flux = np.einsum('ijk,jl->ilk', p_reshaped, basis_val_flux_points)
    
    return u_flux, p_flux

def compute_lobatto_point_values(u, primitive_variable, basis_val_lobatto_points, num_element, flux_number, Np):
    u_reshaped = u.reshape(num_element, Np, 3)
    p_reshaped = primitive_variable.reshape(num_element, Np, 3)
    
    u_lobatto = np.einsum('ijk,jl->ilk', u_reshaped, basis_val_lobatto_points)
    p_lobatto = np.einsum('ijk,jl->ilk', p_reshaped, basis_val_lobatto_points)
    
    return u_lobatto, p_lobatto