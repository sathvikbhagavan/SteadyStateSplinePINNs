import trimesh
import numpy as np
import torch


# Hermite Spline Kernels for 2nd Order
def h0(diff):
    abs_diff = torch.abs(diff)
    return (1-abs_diff)**2*(1+2*abs_diff)

def h1(diff):
    abs_diff = torch.abs(diff)
    return diff*(1-abs_diff)**2

# Derivatives for the kernels
def dh0(diff): 
    abs_diff = torch.abs(diff)
    return 6*diff*(abs_diff-1)

def dh1(diff):
    abs_diff = torch.abs(diff)
    return 3*abs_diff**2 - 4*abs_diff + 1

# Double Derivatives for the kernels
def d2h0(diff): 
    abs_diff = torch.abs(diff)
    return 12*abs_diff-6

def d2h1(diff):
    abs_diff = torch.abs(diff)
    return 6*diff - 4*abs_diff

hs = [h0, h1]
dhs = [dh0, dh1]
d2hs = [d2h0, d2h1]

def get_gradient_function_from_product_rule(i,j,k):
    def dh_dx(x,y,z):
        return dhs[i](x) * hs[j](y) * hs[k](z)
    def dh_dy(x,y,z):
        return hs[i](x) * dhs[j](y) * hs[k](z)
    def dh_dz(x,y,z):
        return hs[i](x) * hs[j](y) * dhs[k](z)
    
    def grad_h(x,y,z):
        return torch.vstack([dh_dx(x,y,z),dh_dy(x,y,z),dh_dz(x,y,z)])
    
    return grad_h

def get_2nd_gradient_function_from_product_rule(i,j,k):
    def d2h_dx2(x,y,z):
        return d2hs[i](x) * hs[j](y) * hs[k](z)
    def d2h_dy2(x,y,z):
        return hs[i](x) * d2hs[j](y) * hs[k](z)
    def d2h_dz2(x,y,z):
        return hs[i](x) * hs[j](y) * d2hs[k](z)
    
    def second_grad_h(x,y,z):
        return torch.vstack([d2h_dx2(x,y,z),d2h_dy2(x,y,z),d2h_dz2(x,y,z)])
    
    return second_grad_h

def binary_array(n):
    if n < 0 or n > 7:
        raise ValueError("Number must be between 0 and 7 inclusive.")
    # Convert the number to a binary string, remove the '0b' prefix, and pad it to 3 bits
    binary_str = format(n, '03b')
    binary_array = [int(bit) for bit in binary_str]
    return binary_array[0], binary_array[1], binary_array[2]

def h(i,j,k,x,y,z):
    return hs[i](x) * hs[j](y) * hs[k](z)

def dh(i,j,k,x,y,z):
    return get_gradient_function_from_product_rule(i,j,k)(x,y,z)

def d2h(i,j,k,x,y,z):
    return get_2nd_gradient_function_from_product_rule(i,j,k)(x,y,z)
