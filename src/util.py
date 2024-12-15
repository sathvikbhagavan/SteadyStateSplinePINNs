import torch

def dynamic_viscosity(T, mu_ref=1.716e-5, T_ref=273.15, S=110.4):
    """
    Calculate dynamic viscosity using Sutherland's law with robust error handling.
    
    Args:
    T: Temperature field (tensor)
    mu_ref: Reference viscosity (default: 1.716e-5 Pa.s)
    T_ref: Reference temperature (default: 273.15 K)
    S: Sutherland constant (default: 110.4 K)
    
    Returns:
    Tensor of dynamic viscosity values
    """
    try:
        # Ensure T is a floating point tensor
        #T = T.float()
        
        # Clamp temperature to prevent extreme values
        #T_clamped = torch.clamp(T, min=100, max=2000)
        
        # Compute viscosity with safe computation
        mu = mu_ref * ((torch.abs(T) / T_ref) ** 1.5) * ((T_ref + S) / (torch.abs(T) + S))
        
        # Replace any remaining NaNs or infs with reference viscosity
       # mu = torch.nan_to_num(mu, nan=mu_ref, posinf=mu_ref, neginf=mu_ref)
        
        return mu

    except Exception as e:
        print(f"Viscosity calculation error: {e}")
        print(f"Problematic temperature tensor: {T}")
        return torch.full_like(T, mu_ref, dtype=T.dtype)