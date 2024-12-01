import trimesh
import numpy as np
import torch


def binary_array(n):
    if n < 0 or n > 7:
        raise ValueError("Number must be between 0 and 7 inclusive.")
    # Convert the number to a binary string, remove the '0b' prefix, and pad it to 3 bits
    binary_str = format(n, '03b')
    binary_array = [int(bit) for bit in binary_str]
    return binary_array[0], binary_array[1], binary_array[2]

# Hermite Spline Kernels for 2nd Order
def h0(diff, der = 0):
    abs_diff = torch.abs(diff)
    match der:
        case 0:
            return (1-abs_diff)**2*(1+2*abs_diff)
        case 1:
            return 6*diff*(abs_diff-1)
        case 2:
            return 12*abs_diff-6

def h1(diff, der = 0):
    abs_diff = torch.abs(diff)
    match der:
        case 0:
            return diff*(1-abs_diff)**2
        case 1:
            return 3*abs_diff**2 - 4*abs_diff + 1
        case 2:
            return 6*diff - 4*abs_diff

hermite_kernel_1d = [h0, h1]

def hermite_kernel_3d(i,j,k,x,y,z,der_x=0,der_y=0,der_z=0):
    return hermite_kernel_1d[i](x, der_x) * hermite_kernel_1d[j](y, der_y) * hermite_kernel_1d[k](z, der_z)