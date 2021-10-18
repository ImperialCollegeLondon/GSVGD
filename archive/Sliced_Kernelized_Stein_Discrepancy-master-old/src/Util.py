import torch

def para_store(dict,**kwargs):
    # Store experiment parameters
    if 'parameter' not in dict:
        dict['parameter']={}

    for key,value in kwargs.items():
        dict['parameter']['%s'%(key)]=value

    return dict

def median_heruistic(sample1,sample2):
    with torch.no_grad():
        G=torch.sum(sample1*sample1,dim=-1)# N or * x N
        G_exp=torch.unsqueeze(G,dim=-2) # 1 x N or * x1 x N

        H=torch.sum(sample2*sample2,dim=-1)
        H_exp=torch.unsqueeze(H,dim=-1) # N x 1 or * * x N x 1
        dist=G_exp+H_exp-2*sample2.matmul(torch.transpose(sample1,-1,-2)) #N x N or  * x N x N
        if len(dist.shape)==3:
            dist=dist[torch.triu(torch.ones(dist.shape))==1].view(dist.shape[0],-1)# * x (NN)
            median_dist,_=torch.median(dist,dim=-1) # *
        else:
            dist=(dist-torch.tril(dist)).view(-1)
            median_dist=torch.median(dist[dist>0.])
    return median_dist.clone().detach()


def median_heruistic_proj(sample1,sample2):
    # samples 1 is * x g x N x 1
    # samples 2 is * x g x N x 1
    with torch.no_grad():
        G=torch.sum(sample1*sample1,dim=-1) # * x num_g x N or r x g x N
        G_exp = torch.unsqueeze(G, dim=-2)  # * x num_g x 1 x N or * x r x g x 1 x N

        H=torch.sum(sample2*sample2,dim=-1) # * x num_g x N or * x r x g x N
        H_exp=torch.unsqueeze(H, dim=-1) # * x numb_g x N x 1 or * x r x g x N x 1

        dist = G_exp + H_exp - 2*torch.matmul(sample2,torch.transpose(sample1,-1,-2)) # * x G x N x N

        if len(dist.shape)==4:
            dist=dist[torch.triu(torch.ones(dist.shape))==1] .view(dist.shape[0],dist.shape[1],-1) # r x g x (NN) or * x g x (NN)

        else:
            dist=dist[torch.triu(torch.ones(dist.shape))==1] .view(dist.shape[0],-1) # g x (NN)

        median_dist,_=torch.median(dist,dim=-1) # num_g or * x g
    return median_dist.clone().detach()


def gradient_estimate_im(x,bandwidth,lam=0.5):
    num_samples=int(x.data.shape[0])
    K_e,G_K_e=rbf_kernel_matrix_eff(x,x,bandwidth)
    G_e=-(K_e+lam*torch.eye(num_samples)).inverse().matmul(G_K_e)
    return G_e
def rbf_kernel_matrix_eff(x,y,bandwidth):
    num_samples=int(x.data.shape[0])
    x=x.clone().requires_grad_()
    y=y.clone().requires_grad_()
    x_batch=torch.unsqueeze(x,dim=0).repeat(num_samples,1,1) # Nx(rep) x Nx x d
    y_batch=torch.unsqueeze(y,dim=1) # Ny x 1 x d
    # Kernel Matrix
    K=torch.exp(-0.5*torch.sum(torch.abs(y_batch-x_batch)**2,dim=2)/(bandwidth)) # N x N
    # G_K
    K_batch=torch.unsqueeze(K,dim=2)
    G_K=torch.sum(1./(bandwidth)*K_batch*(y_batch-x_batch),dim=1)
    return K,G_K