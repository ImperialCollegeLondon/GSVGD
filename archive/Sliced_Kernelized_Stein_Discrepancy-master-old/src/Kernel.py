import torch
def SE_kernel(sample1,sample2,**kwargs):
    # compute RBF kernel with 1 dimensional input
    '''
    Compute the square exponential kernel
    :param sample1: x
    :param sample2: y
    :param kwargs: kernel hyper-parameter: bandwidth
    :return:
    '''
    if 'bandwidth_array' in kwargs['kernel_hyper']:
        # bandwidth could be an array because each g_r has a unique corresponding median heuristic bandwdith.
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
def d_SE_kernel(sample1,sample2,**kwargs):
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

def SE_kernel_multi(sample1,sample2,**kwargs):
    '''
    Compute the multidim square exponential kernel
    :param sample1: x : N x N x dim
    :param sample2: y : N x N x dim
    :param kwargs: kernel hyper-parameter:bandwidth
    :return:
    '''
    bandwidth = kwargs['kernel_hyper']['bandwidth']
    if len(sample1.shape)==4: # * x N x d
        bandwidth=bandwidth.unsqueeze(-1).unsqueeze(-1)

    sample_diff=sample1-sample2 # N x N x dim

    norm_sample=torch.norm(sample_diff,dim=-1)**2 # N x N or * x N x N

    K=torch.exp(-norm_sample/(bandwidth**2+1e-9))
    return K

def trace_SE_kernel_multi(sample1,sample2,**kwargs):
    # compute trace of second-order derivative of RBF kernel
    '''
    Compute the trace of 2 order gradient of K
    :param sample1: x : N x N x dim
    :param sample2: y : N x N x dim
    :param kwargs: kernel hyper: bandwidth
    :return:
    '''
    bandwidth = kwargs['kernel_hyper']['bandwidth']
    K=kwargs['K']

    diff=sample1-sample2 # N x N x dim
    H=K*(2./(bandwidth**2+1e-9)*sample1.shape[-1]-4./(bandwidth**4+1e-9)*torch.sum(diff*diff,dim=-1)) # N x N
    return H


def repulsive_SE_kernel_multi(sample1,sample2,**kwargs):
    # Manually derive the gradient of the RBF kernel
    bandwidth=kwargs['kernel_hyper']['bandwidth'] # *
    if len(sample1.shape)==3:
        bandwidth=bandwidth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    K=kwargs['K'] # * x N x N x 1

    sample1=torch.unsqueeze(sample1,dim=-2) # * x N x 1 x D
    sample2=torch.unsqueeze(sample2,dim=-3) # * x 1 x N x D

    r_K=K*(-1./(bandwidth**2+1e-9)*2*(sample1-sample2)) # * x N x N x D
    return r_K

def repulsive_SE_kernel(sample1,sample2,**kwargs):
    # Compute repulsive force in maxSVGD (S-SVGD)

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
