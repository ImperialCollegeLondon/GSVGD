import torch
import numpy as np
from matplotlib import pyplot as plt
from src.Sliced_KSD_Clean.Util import *
def SE_kernel(sample1,sample2,**kwargs):
    # RBF kernel for projected samples (used for SKSD, S-SVGD)
    '''
    Compute the square exponential kernel
    :param sample1: x
    :param sample2: y
    :param kwargs: kernel hyper-parameter: bandwidth
    :return:
    '''
    if 'bandwidth_array' in kwargs['kernel_hyper']:
        bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or * x g
        if len(sample1.shape)==4:
            if len(bandwidth.shape)==1:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1),dim=-1)# g x 1 x 1 x 1
            else:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1)# * x g x 1 x 1

        else:
            bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1) # g x 1 x 1
        K = torch.exp(-(sample1 - sample2) ** 2 / (bandwidth_exp ** 2+1e-9)) # g x sam1 x sam2 or g x 1 x sam1 x sam2 or * x g x sam1 x sam2
    else:
        bandwidth=kwargs['kernel_hyper']['bandwidth']
        K=torch.exp(-(sample1 - sample2) ** 2 / (bandwidth ** 2+1e-9))
    return K
def SE_kernel_multi(sample1,sample2,**kwargs):
     # RBF kernel for high dimensional samples (used for KSD,SVGD)
    '''
    Compute the multidim square exponential kernel
    :param sample1: x : N x N x dim
    :param sample2: y : N x N x dim
    :param kwargs: kernel hyper-parameter:bandwidth
    :return:
    '''
    bandwidth = kwargs['kernel_hyper']['bandwidth']
    if len(sample1.shape)==4: # * x N x d, determine the data shape
        bandwidth=bandwidth.unsqueeze(-1).unsqueeze(-1)

    sample_diff=sample1-sample2 # N x N x dim

    norm_sample=torch.norm(sample_diff,dim=-1)**2 # N x N or * x N x N

    K=torch.exp(-norm_sample/(bandwidth**2+1e-9))
    return K



def repulsive_SE_kernel_multi(sample1,sample2,**kwargs):
    # gradient of RBF kernel, used for SVGD
    bandwidth=kwargs['kernel_hyper']['bandwidth'] # *
    if len(sample1.shape)==3:
        bandwidth=bandwidth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    K=kwargs['K'] # * x N x N x 1

    sample1=torch.unsqueeze(sample1,dim=-2) # * x N x 1 x D
    sample2=torch.unsqueeze(sample2,dim=-3) # * x 1 x N x D

    r_K=K*(-1./(bandwidth**2+1e-9)*2*(sample1-sample2)) # * x N x N x D
    return r_K


def d_SE_kernel(sample1,sample2,**kwargs):
    # Gradient of RBF kernel on projected samples (Used for SKSD,S-SVGD)
    'The gradient of RBF kernel'
    K=kwargs['K'] # * x g x sam1 x sam2
    if 'bandwidth_array' in kwargs['kernel_hyper']:
        bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or r x g or * x g
        if len(sample1.shape)==4:
            if len(bandwidth.shape)==1:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1),dim=-1)# g x 1 x 1 x 1
            else:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1)# r x g x 1 x 1 or * x g x 1 x 1

        else:
            bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1) # g x 1 x 1
        d_K=K*(-1/(bandwidth_exp**2+1e-9)*2*(sample1-sample2)) # * x g x sam1 x sam2
    else:
        bandwidth = kwargs['kernel_hyper']['bandwidth']
        d_K=K*(-1/(bandwidth**2+1e-9)*2*(sample1-sample2))
    return d_K
def dd_SE_kernel(sample1,sample2,**kwargs):
    K=kwargs['K'] # * x g x sam1 x sam2
    if 'bandwidth_array' in kwargs['kernel_hyper']:
        bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or r x g or * x g
        if len(sample1.shape)==4:
            if len(bandwidth.shape)==1:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1),dim=-1)# g x 1 x 1 x 1
            else:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1)# r x g x 1 x 1 or * x g x 1 x 1

        else:
            bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1) # g x 1 x 1
        dd_K=K*(2/(bandwidth_exp**2+1e-9)-4/(bandwidth_exp**4+1e-9)*(sample1-sample2)**2)
    else:
        bandwidth = kwargs['kernel_hyper']['bandwidth']
        dd_K=K*(2/(bandwidth**2)-4/(bandwidth**4)*(sample1-sample2)**2)
    return dd_K # g x N x N or * x g x sam1 x sam2

def repulsive_SE_kernel(sample1,sample2,**kwargs):
    # Repulsive force computation for S-SVGD
    # sample 1 is g x 1 x sam1 x 1 or g x sam1 x 1 or * x g x sam1 x 1
    # sample 2 is g x 1 x 1 x sam2 or g x 1 x sam2 or * x g x 1 x sam2
    K=kwargs['K']
    if 'bandwidth_array' in kwargs['kernel_hyper']:
        bandwidth = kwargs['kernel_hyper']['bandwidth_array']  #  g or * x g
        if len(sample1.shape)==4:
            if len(bandwidth.shape)==1:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1),dim=-1)# g x 1 x 1 x 1
            else:
                bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1)# r x g x 1 x 1 or * x g x 1 x 1

        else:
            bandwidth_exp=torch.unsqueeze(torch.unsqueeze(bandwidth,dim=-1),dim=-1) # g x 1 x 1
        r_K=K*(-1./(bandwidth_exp**2+1e-9)*2*(sample1-sample2)) # g x 1 x sam1 x  or g x sam1 x sam2 or r x g x sam1 x sam2 or * x g x sam1 x sam2
    else:
        bandwidth = kwargs['kernel_hyper']['bandwidth']
        r_K = K*(-1. / (bandwidth ** 2+1e-9) * 2 * (sample1 - sample2))  # g x 1 x sam1 x sam2
    return r_K

