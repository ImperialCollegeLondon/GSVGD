import torch
import numpy as np
from matplotlib import pyplot as plt
from src.Util import *
from src.Network import *
import math


def compute_max_DSSD_eff(samples1,samples2,score_func,kernel,d_kernel,dd_kernel,**kwargs):
    # compute the maxSKSD-g/-rg
    'Optimize the speed of computation'

    if 'repeat_samples' in kwargs:
        # if samples1= samples2, normally set it to True
        flag_repeat_samples=kwargs['repeat_samples']
    else:
        flag_repeat_samples=False

    if 'flag_U' in kwargs:
        # whether use U-statistics
        flag_U=kwargs['flag_U']
    else:
        flag_U=False

    kernel_hyper = kwargs['kernel_hyper']
    # slice directions
    r = kwargs['r']
    g = kwargs['g']

    if 'flag_median' in kwargs:
        # use median heuristic
            flag_median = kwargs['flag_median']
            if flag_median:
                with torch.no_grad():
                    if 'bandwidth_scale' not in kwargs:
                        # scale parameter for bandwidth
                        bandwidth_scale=1
                    else:
                        bandwidth_scale=kwargs['bandwidth_scale']

                    if samples1.shape[0]>500:
                        # if too many samples, crop to 500 samples
                        idx_crop=500
                    else:
                        idx_crop=samples1.shape[0]

                    g_cp_exp=torch.unsqueeze(g,dim=1) # g x 1 x dim
                    samples1_exp=torch.unsqueeze(samples1[0:idx_crop,:],dim=0) #1 x N x dim
                    samples2_exp=torch.unsqueeze(samples2[0:idx_crop,:],dim=0) # 1 x N x dim
                    # Compute projected samples
                    if flag_repeat_samples:
                        proj_samples1 = torch.sum(samples1_exp * g_cp_exp, dim=-1, keepdim=True)  # g x N x 1
                        proj_samples2 = proj_samples1  # g x N x 1
                    else:
                        proj_samples1=torch.sum(samples1_exp*g_cp_exp,dim=-1,keepdim=True)# g x N x 1
                        proj_samples2=torch.sum(samples2_exp*g_cp_exp,dim=-1,keepdim=True)# g x N x 1
                    # median heuristic for projected samples
                    median_dist=median_heruistic_proj(proj_samples1,proj_samples2).clone() # g
                    bandwidth_array = bandwidth_scale*2 * torch.pow(0.5 * median_dist,kwargs['median_power'])
                    # put it back to kernel_hyper
                    kernel_hyper['bandwidth_array']=bandwidth_array
    else:
        flag_median = False

    # Note that there are 4 terms in maxSKSD, we compute each of them and then add them together. Compute Term1

    if 'score_samples1' in kwargs:
        # Compute score outside
        score_samples1=kwargs['score_samples1']
        score_samples2=kwargs['score_samples2']
    else:
        # if we give the score function
        score_samples1 = score_func(samples1)
        score_samples1 = torch.autograd.grad(torch.sum(score_samples1), samples1)[0]  # sample 1 x dim
        score_samples2 = score_func(samples2)
        score_samples2 = torch.autograd.grad(torch.sum(score_samples2), samples2)[0]  # sample 2 x dim

    g_exp = g.reshape((g.shape[0], 1, g.shape[-1]))  # g x 1 x D
    samples1_crop_exp=torch.unsqueeze(samples1,dim=0) # 1 x N x D
    samples2_crop_exp=torch.unsqueeze(samples2,dim=0) # 1 x N x D


    if flag_repeat_samples:
        # compute x^Tg
        proj_samples1_crop_exp = torch.sum(samples1_crop_exp * g_exp, dim=-1)
        proj_samples2_crop_exp=proj_samples1_crop_exp
    else:
        proj_samples1_crop_exp = torch.sum(samples1_crop_exp * g_exp, dim=-1)  # g x sam1
        proj_samples2_crop_exp = torch.sum(samples2_crop_exp * g_exp, dim=-1)  # g x sam2

    r_exp = torch.unsqueeze(r, dim=1)  # r x 1 x dim
    if flag_repeat_samples:
        # compute score_p^Tr
        proj_score_orig=torch.sum(r_exp * torch.unsqueeze(score_samples1, dim=0), dim=-1)
        proj_score1=proj_score_orig.unsqueeze(-1)
        proj_score2 =proj_score_orig
    else:
        proj_score1 = torch.sum(r_exp * torch.unsqueeze(score_samples1, dim=0), dim=-1, keepdim=True)  # r x sam1 x 1
        proj_score2 = torch.sum(r_exp * torch.unsqueeze(score_samples2, dim=0), dim=-1)  # r x sam2

    proj_score1_exp = proj_score1  # r x sam1 x 1
    proj_score2_exp = proj_score2.reshape((proj_score2.shape[0], 1, proj_score2.shape[-1]))  # r x 1 x sam2
    # compute kernel matrix
    K = kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper)  # g x sam1 x sam 2

    if flag_U:
        d_K = torch.diagonal(K, dim1=1, dim2=2).reshape((K.shape[0], K.shape[1], 1))  # g x sam1 x 1
        e_K = torch.eye(K.shape[-1]).reshape((1, K.shape[-2], K.shape[-1])).repeat(K.shape[0],1,1)  # g x sam1 x sam2
        d_K = d_K * e_K  # g x 1 x sam1 x sam2
        Term1 = proj_score1_exp * (K - d_K) * proj_score2_exp
    else:
        Term1 = proj_score1_exp * K * proj_score2_exp  # g x sam1 x sam2

    # Compute Term2, here we explicit derive the gradient of the kernel called d_kernel
    r_exp_exp = torch.unsqueeze(r_exp, dim=1)  # r x 1 x 1 x dim
    rg = torch.sum(r_exp_exp * torch.unsqueeze(g_exp,dim=-2), dim=-1)  # r x 1 x 1, r^Tg
    if flag_U:
        grad_2_K = -d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K-d_K)  # g x N x N

    else:
        grad_2_K = -d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K)  # g x N x N

    Term2 = rg * proj_score1_exp * grad_2_K  # g x sam1 x sam2

    # Compute Term3
    if flag_repeat_samples:
        Term3 = -rg * proj_score2_exp * grad_2_K
    else:
        if flag_U:
            grad_1_K=d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K-d_K)  # g x N x N
        else:
            grad_1_K =d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K)  # g x N x N
        Term3=rg*proj_score2_exp*grad_1_K

    # Compute Term4, we also explicit compute the higher order derivative of the kernel called dd_kernel.
    if flag_U:
        grad_21_K=dd_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K-d_K) # g x N x N
    else:
        grad_21_K=dd_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K) # g x N x N
    Term4=(rg**2)*grad_21_K # g x N x N

    divergence = Term1+Term2+Term3+Term4  # g x sam1  x sam2
    if flag_U:
        KDSSD = torch.sum(divergence) / ((samples1.shape[0] - 1) * samples2.shape[0])

    else:
        KDSSD = torch.sum(divergence) / (samples1.shape[0] * samples2.shape[0])

    return KDSSD, divergence



def compute_KSD(samples1,samples2,score_func,kernel,trace_kernel,**kwargs):
    # Compute KSD

    if 'flag_U' in kwargs:
        flag_U=kwargs['flag_U']
    else:
        flag_U=False

    if 'flag_retain' in kwargs:
        flag_retain = kwargs['flag_retain']
        flag_create = kwargs['flag_create']
    else:
        flag_retain = False
        flag_create = False


    kernel_hyper = kwargs['kernel_hyper']
    divergence_accum = 0

    samples1_crop_exp=torch.unsqueeze(samples1,dim=1).repeat(1,samples2.shape[0],1) # N x N(rep) x dim
    samples2_crop_exp=torch.unsqueeze(samples2,dim=0).repeat(samples1.shape[0],1,1) # N(rep) x N x dim

    # Compute Term1
    if 'score_sample1' in kwargs:
        # compute score outside of this function
        score_sample1 = kwargs['score_sample1']
        score_sample2=kwargs['score_sample2']
    else:

        score_sample1=score_func(samples1) # N
        score_sample1=torch.autograd.grad(torch.sum(score_sample1),samples1)[0] # N x dim
        score_sample2 = score_func(samples2)  # N
        score_sample2 = torch.autograd.grad(torch.sum(score_sample2), samples2)[0]  # N x dim

    score_sample1_exp=torch.unsqueeze(score_sample1,dim=1) # N x 1 x dim
    score_sample2_exp=torch.unsqueeze(score_sample2,dim=0) # 1 x N x dim

    K=kernel(samples1_crop_exp,samples2_crop_exp,kernel_hyper=kernel_hyper)

    if flag_U:
        Term1=(K-torch.diag(torch.diag(K)))*torch.sum(score_sample1_exp*score_sample2_exp,dim=-1) # N x N
    else:
        Term1 = (K) * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)  # N x N

    # Compute Term 2, directly use autograd for kernel gradient
    if flag_U:
        grad_K_2=torch.autograd.grad(torch.sum((K-torch.diag(torch.diag(K)))),samples2_crop_exp,retain_graph=flag_retain,create_graph=flag_create)[0] # N x N x dim
    else:
        grad_K_2 = torch.autograd.grad(torch.sum((K)), samples2_crop_exp,retain_graph=flag_retain,create_graph=flag_create)[0]  # N x N x dim
    Term2=torch.sum(score_sample1_exp*grad_K_2,dim=-1)# N x N

    # Compute Term 3
    if flag_U:
        K=kernel(samples1_crop_exp,samples2_crop_exp,kernel_hyper=kernel_hyper)
        grad_K_1=torch.autograd.grad(torch.sum((K-torch.diag(torch.diag(K)))),samples1_crop_exp,retain_graph=flag_retain,create_graph=flag_create)[0] # N x N x dim

    else:
        K=kernel(samples1_crop_exp,samples2_crop_exp,kernel_hyper=kernel_hyper)

        grad_K_1=torch.autograd.grad(torch.sum((K)),samples1_crop_exp,retain_graph=flag_retain,create_graph=flag_create)[0] # N x N x dim

    Term3=torch.sum(score_sample2_exp*grad_K_1,dim=-1) # N x N

    # Compute Term 4, manually derive the trace of high-order derivative of kernel called trace_kernel
    K=kernel(samples1_crop_exp,samples2_crop_exp,kernel_hyper=kernel_hyper)

    if flag_U:
        T_K = trace_kernel(samples1_crop_exp, samples2_crop_exp, kernel_hyper=kernel_hyper,K=K-torch.diag(torch.diag(K)))

        grad_K_12=T_K# N x N
    else:
        T_K = trace_kernel(samples1_crop_exp, samples2_crop_exp, kernel_hyper=kernel_hyper,K=K)

        grad_K_12=T_K# N x N

    Term4=grad_K_12

    KSD_comp=torch.sum(Term1+1*Term2+1*Term3+1*Term4)

    divergence_accum+=KSD_comp

    if flag_U:
        KSD=divergence_accum/((samples1.shape[0]-1)*samples2.shape[0])

    else:
        KSD=divergence_accum/(samples1.shape[0]*samples2.shape[0])

    return KSD, Term1+Term2+Term3+Term4


def compute_MMD(samples1,samples2,kernel,kernel_hyper,flag_U=True,flag_simple_U=True):
    # samples1: N x dim
    # samples2: N x dim
    n=samples1.shape[0]
    m=samples2.shape[0]

    if m!=n and flag_simple_U:
        raise ValueError('If m is not equal to n, flag_simple_U must be False')

    samples1_exp1=torch.unsqueeze(samples1,dim=1) # N x 1 x dim
    samples1_exp2=torch.unsqueeze(samples1,dim=0) # 1 x N x dim

    samples2_exp1 = torch.unsqueeze(samples2, dim=1)  # N x 1 x dim
    samples2_exp2 = torch.unsqueeze(samples2, dim=0)  # 1 x N x dim

    # Term1
    K1=kernel(samples1_exp1,samples1_exp2,kernel_hyper=kernel_hyper) # N x N
    if flag_U:
        K1=K1-torch.diag(torch.diag(K1))
    # Term3
    K3 = kernel(samples2_exp1, samples2_exp2, kernel_hyper=kernel_hyper)  # N x N
    if flag_U:
        K3 = K3 - torch.diag(torch.diag(K3))

    # Term2
    if flag_simple_U:
        K2_comp=kernel(samples1_exp1, samples2_exp2, kernel_hyper=kernel_hyper)
        K2_comp=K2_comp-torch.diag(torch.diag(K2_comp))
        K2=K2_comp+K2_comp.t()
    else:
        K2=2*kernel(samples1_exp1, samples2_exp2, kernel_hyper=kernel_hyper) # N x N


    if flag_U:
        if flag_simple_U:
            MMD=torch.sum(K1)/(n*(n-1))+torch.sum(K3)/(m*(m-1))-1./(m*(m-1))*torch.sum(K2)

        else:
            MMD=torch.sum(K1)/(n*(n-1))+torch.sum(K3)/(m*(m-1))-1./(m*n)*torch.sum(K2)
    else:
        MMD=torch.sum(K1+K3-K2)/(m*n)



    return MMD, K1+K3-K2



def compute_optimal_g_MCMC(burnin,dim,score_func,kernel,batch_size,g,r,d_kernel,dd_kernel,**kwargs):
    q_x_ds=kwargs['q_x_ds'] # data source
    ##### Has not implemented optimization with the fixed samples

    if 'flag_optimal_r' in kwargs:
        flag_optimal_r=kwargs['flag_optimal_r']
    else:
        flag_optimal_r=False


    g = g.clone().requires_grad_()
    r = r.clone().requires_grad_()

    if 'flag_U' in kwargs:
        flag_U=kwargs['flag_U']
    else:
        flag_U=False


    Adam = torch.optim.Adam([g],eps=0.002,betas=(0.9,0.99))
    Adam_r = torch.optim.Adam([r], eps=0.002, betas=(0.9, 0.99))


    if 'update_interval' in kwargs:
        update_interval=kwargs['update_interval']
    else:
        update_interval=1



    for idx in range(burnin):

        ###### MCMC update data
        if idx==0:
            samples1, H = kwargs['fix_sample']
        else:
            samples1_cp = samples1.clone().detach()
            samples1, H = q_x_ds.sample(None, return_latent=True, X=samples1_cp, H=H, burnin=None)
            samples1 = samples1.clone().detach().requires_grad_()

        if (idx+1)%update_interval==0:

            # Subsampling data to batch size
            batch_perm= torch.randperm(samples1.shape[0])
            samples1_batch = samples1.index_select(0, batch_perm)
            samples1_batch=samples1_batch[0:batch_size,:]

            samples1_batch = samples1_batch.clone().detach().requires_grad_()
            samples2_batch = samples1_batch.clone().detach().requires_grad_()
            # update g,r
            Adam.zero_grad()
            if flag_optimal_r:
                Adam_r.zero_grad()

            # normalize to unit vector
            g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)

            r_n=r / (torch.norm(r, 2, dim=-1, keepdim=True) + 1e-10)

            diver,_=compute_max_DSSD_eff(samples1_batch,samples2_batch,score_func,kernel,d_kernel,dd_kernel,
                            r=r_n,g=g_n,kernel_hyper=kwargs['kernel_hyper'],flag_median=True,flag_U=flag_U,median_power=kwargs['median_power'])


            (-diver).backward()
            Adam.step()
            if flag_optimal_r:
                Adam_r.step()

    return g,r

def compute_SD(samples1,test_func,score_func,m,lam=0.1,**kwargs):
    # Compute Stein Discrepancy using neural net
    test_out=test_func.forward(samples1) # N x D
    samples1_dup=samples1.unsqueeze(-2).repeat(1,m,1) # N x M X D
    test_out_dup=test_func.forward(samples1_dup)# N x  M X D

    if 'score' in kwargs:
        score=kwargs['score']
    else:
        score=score_func(samples1)
        score=torch.autograd.grad(score.sum(),samples1)[0] # N x D
    Term1=torch.sum(score*test_out,dim=-1) # N

    # compute term 2 with Hutchson's trick
    eps=torch.randn(1,m,samples1.shape[-1]) # 1 x M x D
    f_eps=torch.sum(test_out_dup*eps,dim=-1) # N x M
    f_eps_grad=torch.autograd.grad(f_eps.sum(),samples1_dup,create_graph=True,retain_graph=True)[0] # N x M x D
    eps_f_eps=torch.sum(eps*f_eps_grad,dim=-1).mean(-1) # N

    Term2=eps_f_eps

    divergence=torch.mean(Term1+Term2) # 1

    # regularize
    reg=torch.sum(test_out*test_out+1e-6,dim=-1).mean()
    divergence_reg=divergence-lam*reg
    return divergence,divergence_reg


############## Only for Amortized SVGD MNIST

def compute_max_DSSD_eff_Tensor(samples1,samples2,score_func,kernel,d_kernel,dd_kernel,**kwargs):
    # Tensor version for computing maxSKSD


    if 'flag_U' in kwargs:
        flag_U=kwargs['flag_U']
    else:
        flag_U=False

    if flag_U:
        samples2=samples2
    kernel_hyper = kwargs['kernel_hyper']
    r = kwargs['r'] # r x dim

    g = kwargs['g'] # g x dim

    if 'flag_median' in kwargs:
            flag_median = kwargs['flag_median']
            if flag_median:
                with torch.no_grad():
                    if 'bandwidth_scale' not in kwargs:
                        bandwidth_scale=1
                    else:
                        bandwidth_scale=kwargs['bandwidth_scale']
                    if samples1.shape[0]>500:
                        idx_crop=500
                    else:
                        idx_crop=samples1.shape[0]
                    g_cp_exp=torch.unsqueeze(g,dim=1).unsqueeze(0) # 1 x g x 1 x dim
                    samples1_exp=torch.unsqueeze(samples1[0:idx_crop,:],dim=-3) #* x 1 x N x dim
                    samples2_exp=torch.unsqueeze(samples2[0:idx_crop,:],dim=-3) #* x  1 x N x dim
                    proj_samples1=torch.sum(samples1_exp*g_cp_exp,dim=-1,keepdim=True)# * x g x N x 1
                    proj_samples2=torch.sum(samples2_exp*g_cp_exp,dim=-1,keepdim=True)# * x g x N x 1
                    median_dist=median_heruistic_proj(proj_samples1,proj_samples2).clone() # * x g
                    #bandwidth_array=2*torch.sqrt(0.5*median_dist)
                    bandwidth_array = bandwidth_scale*2 * torch.pow(0.5 * median_dist,kwargs['median_power'])

                    kernel_hyper['bandwidth_array']=bandwidth_array
    else:
        flag_median = False

    # Compute Term1
    # Compute score

    if 'score_samples1' in kwargs:
        score_samples1=kwargs['score_samples1'] # * x N x D
        score_samples2=kwargs['score_samples2']
    else:
        score_samples1 = score_func(samples1)
        score_samples1 = torch.autograd.grad(torch.sum(score_samples1), samples1)[0]  # * x sample 1 x dim
        score_samples2 = score_func(samples2)
        score_samples2 = torch.autograd.grad(torch.sum(score_samples2), samples2)[0]  # * x sample 2 x dim

    g_exp = g.reshape((1,g.shape[0], 1, g.shape[-1]))  # * x g x 1 x D
    samples1_crop_exp=torch.unsqueeze(samples1,dim=-3) # * x 1 x N x D
    samples2_crop_exp=torch.unsqueeze(samples2,dim=-3) # * x 1 x N x D
    proj_samples1_crop_exp = torch.sum(samples1_crop_exp * g_exp, dim=-1)  # * x g x sam1
    proj_samples2_crop_exp = torch.sum(samples2_crop_exp * g_exp, dim=-1)  # * x g x sam2

    r_exp = torch.unsqueeze(r, dim=1).unsqueeze(0)  # 1 x r x 1 x dim
    proj_score1 = torch.sum(r_exp * torch.unsqueeze(score_samples1, dim=-3), dim=-1, keepdim=True)  # * x r x sam1 x 1
    proj_score2 = torch.sum(r_exp * torch.unsqueeze(score_samples2, dim=-3), dim=-1)  # * x r x sam2

    proj_score1_exp = proj_score1  # * x r x sam1 x 1
    proj_score2_exp = proj_score2.reshape((*proj_score2.shape[0:-1], 1, proj_score2.shape[-1]))  # * x r x 1 x sam2

    K = kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper)  # * x g x sam1 x sam 2

    if flag_U:
        d_K = torch.diagonal(K, dim1=-2, dim2=-1).reshape((*K.shape[0:-2], K.shape[-1], 1))  # * x g x sam1 x 1
        e_K = torch.eye(K.shape[-1]).reshape((1,1, K.shape[-2], K.shape[-1])).repeat(K.shape[0],K.shape[1],1,1)  # * x g x sam1 x sam2
        d_K = d_K * e_K  # * x g  x sam1 x sam2
        Term1 = proj_score1_exp * (K - d_K) * proj_score2_exp # * x g x sam1 x sam2
    else:
        K=K
        Term1 = proj_score1_exp * K * proj_score2_exp  # * x g x sam1 x sam2

    # Compute Term2
    r_exp_exp = torch.unsqueeze(r_exp, dim=-3)  # * x r x 1 x 1 x dim
    rg = torch.sum(r_exp_exp * torch.unsqueeze(g_exp,dim=-2), dim=-1)  # * x r x 1 x 1
    if flag_U:
        grad_2_K = -d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K-d_K)  # * x  g x N x N

    else:
        grad_2_K = -d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K)  # * x  g x N x N

    Term2 = rg * proj_score1_exp * grad_2_K  # * x g x sam1 x sam2

    # Compute Term3
    if flag_U:
        grad_1_K=d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K-d_K)  # g x N x N
    else:
        grad_1_K =d_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K)  # g x N x N
    Term3=rg*proj_score2_exp*grad_1_K # * x g x sam1 x sam2
    # Compute Term4

    if flag_U:
        grad_21_K=dd_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K-d_K) # * x g x N x N
    else:
        grad_21_K=dd_kernel(torch.unsqueeze(proj_samples1_crop_exp,dim=-1), torch.unsqueeze(proj_samples2_crop_exp,dim=-2), kernel_hyper=kernel_hyper,K=K) # * x g x N x N
    Term4=(rg**2)*grad_21_K # * x g x N x N


    divergence = 1 * Term1 + 1 * (1 * Term2 + 1 * Term3 + 1 * Term4)  # * x g x sam1  x sam2
    if flag_U:

        KDSSD = torch.sum(divergence,dim=-1).sum(-1).sum(-1) / ((samples1.shape[0] - 1) * samples2.shape[0]) # *

    else:

        KDSSD = torch.sum(divergence,dim=-1).sum(-1).sum(-1) / (samples1.shape[0] * samples2.shape[0]) # *
    # if torch.isnan(KDSSD).sum() > 0:
    #     A = 1
    return KDSSD, divergence


def max_DSSVGD_Tensor(samples,score_func,kernel,repulsive_kernel,r,g,**kwargs):
    # maxSVGD
    # g is g x dim
    # r is r x dim
    if g.shape[0] != r.shape[0]:
        raise ValueError('The number of g must match the number of r')
    elif r.shape[0] != r.shape[1]:
        raise ValueError('The number of r must match the dimensions dim')

    kernel_hyper = kwargs['kernel_hyper']
    sample1 = samples # * x N x D
    sample2 = sample1.clone().detach().requires_grad_() # * x N x D


    if 'repulsive_coef' in kwargs:
        r_coef=kwargs['repulsive_coef']
    else:
        r_coef=1

    if 'flag_repulsive_output' in kwargs:
        # whether output repulsive force
        flag_repulsive_output=kwargs['flag_repulsive_output']
    else:
        flag_repulsive_output=False

    if 'flag_median' in kwargs:
            flag_median = kwargs['flag_median']
            if flag_median:
                with torch.no_grad():
                    g_cp_exp=torch.unsqueeze(g,dim=-2).unsqueeze(0) # 1 x g x 1 x dim
                    samples1_exp=torch.unsqueeze(sample1,dim=-3) #* x 1 x N x dim
                    samples2_exp=torch.unsqueeze(sample2,dim=-3) # * x 1 x N x dim
                    proj_samples1=torch.sum(samples1_exp*g_cp_exp,dim=-1,keepdim=True)# * x g x N x 1
                    proj_samples2=torch.sum(samples2_exp*g_cp_exp,dim=-1,keepdim=True)# * x g x N x 1
                    median_dist=median_heruistic_proj(proj_samples1,proj_samples2).clone() # * x g
                    bandwidth_array = kwargs['bandwidth_scale']*2 * torch.pow(0.5 * median_dist,kwargs['median_power'])

                    kernel_hyper['bandwidth_array']=bandwidth_array # * x g
    else:
        flag_median = False

    if 'score' in kwargs:
        score = kwargs['score'] # * x N x D
    else:
        score = score_func(sample1)  # * x sam1
        score = torch.autograd.grad(torch.sum(score), sample1)[0]  # * x sam1 x D

    with torch.no_grad():
        r_exp = torch.unsqueeze(r, dim=-2) .unsqueeze(0) # 1 x r x 1 x d
        proj_score = torch.sum(r_exp * score.unsqueeze(-3), dim=-1)  # 1 x r x sam 1

        sample1_exp = sample1.reshape((*sample1.shape[0:-2],1, sample1.shape[-2], sample1.shape[-1]))  #  * x 1 x sam1 x d
        sample2_exp = samples.reshape((*sample2.shape[0:-2],1, sample2.shape[-2], sample2.shape[-1]))  #  * x 1 x sam2 x d

        g_exp = g.reshape((1,g.shape[0], 1, g.shape[1]))  # 1 x g x 1 x d
        proj_samples1 = torch.unsqueeze(torch.sum(g_exp * sample1_exp, dim=-1), dim=-1)  # * x g x sam1 x 1
        proj_samples2 = torch.unsqueeze(torch.sum(g_exp * sample2_exp, dim=-1), dim=-2)  # * x g x 1 x sam2
        K = kernel(proj_samples1, proj_samples2, kernel_hyper=kernel_hyper)  # * x g x sam1 x sam2
        proj_score_exp = proj_score.unsqueeze(-1)  # 1 x r x sam1 x 1

        Term1 = proj_score_exp * K  # * x r x sam1 x sam2

        rg = torch.sum(r * g, dim=-1).reshape(1,r.shape[0], 1, 1)  # 1 x r x 1 x 1
        r_K = repulsive_kernel(proj_samples1, proj_samples2, kernel_hyper=kernel_hyper, K=K)  # * x g x sam 1 x sam2
        Term2 = rg * r_K  # * x g x sam1 x sam2

        force = torch.mean(Term1 + r_coef * Term2, dim=-2)  # * x r(g) x sam 2
        force = torch.transpose(force,-1,-2)  # * x sam2 x r

    if flag_repulsive_output:
        return force,torch.mean(Term2,dim=-2).t() # * x sam2 x dim
    else:
        return force


def SVGD_Tensor(samples,score_func,kernel,repulsive_kernel,**kwargs):
    kernel_hyper = kwargs['kernel_hyper'] # *  e.g. 100
    if 'repulsive_coef' in  kwargs:
        r_coef=kwargs['repulsive_coef']
    else:
        r_coef=1

    if 'flag_repulsive_output' in kwargs:
        flag_repulsive_output = kwargs['flag_repulsive_output']
    else:
        flag_repulsive_output = False

    samples_cp = samples.clone().detach().requires_grad_()  # * x N x D
    if 'score' in kwargs:
        score = kwargs['score'] # * x sam1 x D
    else:
        score = score_func(samples)  # sam1
        score = torch.autograd.grad(torch.sum(score), samples)[0]  # * x sam1 x D

    with torch.no_grad():
        score = score.reshape((*score.shape[0:-1], 1, score.shape[-1]))  # * x N x 1 x D
        K = kernel(torch.unsqueeze(samples, dim=-2), torch.unsqueeze(samples_cp, dim=-3) # * x N x 1 x D or * x 1 x N x D
                   , kernel_hyper=kernel_hyper)  # * x N x N

        Term1 = torch.unsqueeze(K, dim=-1) * score  # * x N x N x D

        Term2 = repulsive_kernel(samples, samples_cp, kernel_hyper=kernel_hyper,
                                 K=torch.unsqueeze(K, dim=-1))  # * x N x N xD

        force = torch.mean(Term1 + r_coef * Term2, dim=-3)  # * x N x D
    if flag_repulsive_output:
        return force, torch.mean(Term2, dim=0)
    else:
        return force

