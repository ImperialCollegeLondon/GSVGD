
import torch
import autograd.numpy as np
from src.Sliced_KSD_Clean.Util import *
import math


def SVGD(samples, score_func, kernel, repulsive_kernel, **kwargs):
    kernel_hyper = kwargs['kernel_hyper']

    # Control the coefficient of the repulsive force
    if 'repulsive_coef' in kwargs:
        r_coef = kwargs['repulsive_coef']
    else:
        r_coef = 1

    # if output the repulsive force
    if 'flag_repulsive_output' in kwargs:
        flag_repulsive_output = kwargs['flag_repulsive_output']
    else:
        flag_repulsive_output = False

    samples_cp = samples.clone().detach().requires_grad_()  # N x D

    if 'score' in kwargs:
        score = kwargs['score']
    else:
        # Typically, the score function is the joint/likelihood function, so if scores are needed, we need to differentiate it again
        score = score_func(samples)
        score = torch.autograd.grad(torch.sum(score), samples)[0]  # sam1 x D
    with torch.no_grad():
        score = score.reshape((score.shape[0], 1, score.shape[1]))  # N x 1 x D

        K = kernel(torch.unsqueeze(samples, dim=1), torch.unsqueeze(samples_cp, dim=0)  # Kernel matrix computation
                   , kernel_hyper=kernel_hyper)  # N x N

        Term1 = torch.unsqueeze(K, dim=-1)*score  # N x N x D

        Term2 = repulsive_kernel(samples, samples_cp, kernel_hyper=kernel_hyper, K=torch.unsqueeze(
            K, dim=-1))  # N x N xD # Repulsive force computation

        force = torch.mean(Term1+r_coef*Term2, dim=0)  # N x D
    if flag_repulsive_output:
        return force, torch.mean(Term2, dim=0)
    else:
        return force


def compute_max_DSSD_eff(samples1, samples2, score_func, kernel, d_kernel, dd_kernel, **kwargs):
    # Compute the SKSD
    'Optimize the speed of computation'
    if 'repeat_samples' in kwargs:
        # If the samples samples1=samples2
        flag_repeat_samples = kwargs['repeat_samples']
    else:
        flag_repeat_samples = False
    # If enable U-statistics or V-statistics
    if 'flag_U' in kwargs:
        flag_U = kwargs['flag_U']
    else:
        flag_U = False

    kernel_hyper = kwargs['kernel_hyper']
    r = kwargs['r']

    g = kwargs['g']

    if 'flag_median' in kwargs:
        flag_median = kwargs['flag_median']
        if flag_median:
            with torch.no_grad():
                # Control the scale for bandwidth
                if 'bandwidth_scale' not in kwargs:
                    bandwidth_scale = 1
                else:
                    bandwidth_scale = kwargs['bandwidth_scale']
                if samples1.shape[0] > 500:
                    # Crop the data
                    idx_crop = 500
                else:
                    idx_crop = samples1.shape[0]
                g_cp_exp = torch.unsqueeze(g, dim=1)  # g x 1 x dim
                samples1_exp = torch.unsqueeze(
                    samples1[0:idx_crop, :], dim=0)  # 1 x N x dim
                samples2_exp = torch.unsqueeze(
                    samples2[0:idx_crop, :], dim=0)  # 1 x N x dim
                if flag_repeat_samples:
                    proj_samples1 = torch.sum(
                        samples1_exp * g_cp_exp, dim=-1, keepdim=True)  # g x N x 1
                    proj_samples2 = proj_samples1  # g x N x 1
                else:
                    proj_samples1 = torch.sum(
                        samples1_exp*g_cp_exp, dim=-1, keepdim=True)  # g x N x 1
                    proj_samples2 = torch.sum(
                        samples2_exp*g_cp_exp, dim=-1, keepdim=True)  # g x N x 1

                median_dist = median_heruistic_proj(
                    proj_samples1, proj_samples2).clone()  # g
                bandwidth_array = torch.sqrt(median_dist / np.log(proj_samples1.shape[1])) # magic bandwidth
                # print("med", median_dist)
                # bandwidth_array = bandwidth_scale*2 * \
                #     torch.pow(0.5 * median_dist, kwargs['median_power'])

                kernel_hyper['bandwidth_array'] = bandwidth_array
    # Compute Term1
    # Compute score

    if 'score_samples1' in kwargs:
        score_samples1 = kwargs['score_samples1']
        score_samples2 = kwargs['score_samples2']
    else:
        score_samples1 = score_func(samples1)
        score_samples1 = torch.autograd.grad(torch.sum(score_samples1), samples1)[
            0]  # sample 1 x dim
        score_samples2 = score_func(samples2)
        score_samples2 = torch.autograd.grad(torch.sum(score_samples2), samples2)[
            0]  # sample 2 x dim

    g_exp = g.reshape((g.shape[0], 1, g.shape[-1]))  # g x 1 x D
    samples1_crop_exp = torch.unsqueeze(samples1, dim=0)  # 1 x N x D
    samples2_crop_exp = torch.unsqueeze(samples2, dim=0)  # 1 x N x D

    # Projected samples
    if flag_repeat_samples:

        proj_samples1_crop_exp = torch.sum(samples1_crop_exp * g_exp, dim=-1)
        proj_samples2_crop_exp = proj_samples1_crop_exp
    else:
        proj_samples1_crop_exp = torch.sum(
            samples1_crop_exp * g_exp, dim=-1)  # g x sam1
        proj_samples2_crop_exp = torch.sum(
            samples2_crop_exp * g_exp, dim=-1)  # g x sam2

    r_exp = torch.unsqueeze(r, dim=1)  # r x 1 x dim
    # Projected score function (for S-SVGD, as the r is identity matrix, so is just the normal score function)
    if flag_repeat_samples:
        proj_score_orig = torch.sum(
            r_exp * torch.unsqueeze(score_samples1, dim=0), dim=-1)
        proj_score1 = proj_score_orig.unsqueeze(-1)
        proj_score2 = proj_score_orig
    else:
        proj_score1 = torch.sum(
            r_exp * torch.unsqueeze(score_samples1, dim=0), dim=-1, keepdim=True)  # r x sam1 x 1
        proj_score2 = torch.sum(
            r_exp * torch.unsqueeze(score_samples2, dim=0), dim=-1)  # r x sam2

    proj_score1_exp = proj_score1  # r x sam1 x 1
    proj_score2_exp = proj_score2.reshape(
        (proj_score2.shape[0], 1, proj_score2.shape[-1]))  # r x 1 x sam2
    # Kernel matrix computation
    K = kernel(torch.unsqueeze(proj_samples1_crop_exp, dim=-1), torch.unsqueeze(
        proj_samples2_crop_exp, dim=-2), kernel_hyper=kernel_hyper)  # g x sam1 x sam 2

    #?
    # return torch.sum(K) / (samples1.shape[0] * samples2.shape[0]), torch.sum(K) 

    if flag_U:  # U-statistics
        d_K = torch.diagonal(K, dim1=1, dim2=2).reshape(
            (K.shape[0], K.shape[1], 1))  # g x sam1 x 1

        e_K = torch.eye(K.shape[-1]).to(K.device).reshape((1, K.shape[-2],
                                              K.shape[-1])).repeat(K.shape[0], 1, 1)  # g x sam1 x sam2
        d_K = d_K * e_K  # g x 1 x sam1 x sam2
        Term1 = proj_score1_exp * (K - d_K) * proj_score2_exp
    else:
        Term1 = proj_score1_exp * K * proj_score2_exp  # g x sam1 x sam2

    # Compute Term2
    r_exp_exp = torch.unsqueeze(r_exp, dim=1)  # r x 1 x 1 x dim
    rg = torch.sum(r_exp_exp * torch.unsqueeze(g_exp, dim=-2),
                   dim=-1)  # r x 1 x 1
    if flag_U:
        grad_2_K = -d_kernel(torch.unsqueeze(proj_samples1_crop_exp, dim=-1), torch.unsqueeze(
            proj_samples2_crop_exp, dim=-2), kernel_hyper=kernel_hyper, K=K-d_K)  # g x N x N

    else:
        grad_2_K = -d_kernel(torch.unsqueeze(proj_samples1_crop_exp, dim=-1), torch.unsqueeze(
            proj_samples2_crop_exp, dim=-2), kernel_hyper=kernel_hyper, K=K)  # g x N x N


    Term2 = rg * proj_score1_exp * grad_2_K  # g x sam1 x sam2

    # Compute Term3
    if flag_repeat_samples:
        Term3 = -rg * proj_score2_exp * grad_2_K
    else:
        if flag_U:
            grad_1_K = d_kernel(torch.unsqueeze(proj_samples1_crop_exp, dim=-1), torch.unsqueeze(
                proj_samples2_crop_exp, dim=-2), kernel_hyper=kernel_hyper, K=K-d_K)  # g x N x N
        else:
            grad_1_K = d_kernel(torch.unsqueeze(proj_samples1_crop_exp, dim=-1), torch.unsqueeze(
                proj_samples2_crop_exp, dim=-2), kernel_hyper=kernel_hyper, K=K)  # g x N x N
        Term3 = rg*proj_score2_exp*grad_1_K
    # Compute Term4
    if flag_U:
        grad_21_K = dd_kernel(torch.unsqueeze(proj_samples1_crop_exp, dim=-1), torch.unsqueeze(
            proj_samples2_crop_exp, dim=-2), kernel_hyper=kernel_hyper, K=K-d_K)  # g x N x N
    else:
        grad_21_K = dd_kernel(torch.unsqueeze(proj_samples1_crop_exp, dim=-1), torch.unsqueeze(
            proj_samples2_crop_exp, dim=-2), kernel_hyper=kernel_hyper, K=K)  # g x N x N

    Term4 = (rg**2)*grad_21_K  # g x N x N

    divergence = 1 * Term1 + 1 * \
        (1 * Term2 + 1 * Term3 + 1 * Term4)  # g x sam1  x sam2
    if flag_U:
        KDSSD = torch.sum(divergence) / \
            ((samples1.shape[0] - 1) * samples2.shape[0])

    else:
        # V-Statistic
        KDSSD = torch.sum(divergence) / (samples1.shape[0] * samples2.shape[0])
    # if torch.isnan(KDSSD).sum() > 0:
    #     A = 1
    
    return KDSSD, divergence
    # return 0, 1


def max_DSSVGD(samples, score_func, kernel, repulsive_kernel, r, g, **kwargs):
    # Compute the S-SVGD update force
    # g is g x dim
    # r is r x dim
    if g.shape[0] != r.shape[0]:
        raise ValueError('The number of g must match the number of r')
    elif r.shape[0] != r.shape[1]:
        raise ValueError('The number of r must match the dimensions dim')

    kernel_hyper = kwargs['kernel_hyper']
    sample1 = samples
    sample2 = sample1.clone().detach().requires_grad_()

    # Control the repulsive force coefficient
    if 'repulsive_coef' in kwargs:
        r_coef = kwargs['repulsive_coef']
    else:
        r_coef = 1
    # whether output the repulsive force
    if 'flag_repulsive_output' in kwargs:
        flag_repulsive_output = kwargs['flag_repulsive_output']
    else:
        flag_repulsive_output = False
    # Compute the bandwidth
    if 'flag_median' in kwargs:
        flag_median = kwargs['flag_median']
        if flag_median:
            with torch.no_grad():
                g_cp_exp = torch.unsqueeze(g, dim=1)  # g x 1 x dim
                samples1_exp = torch.unsqueeze(sample1, dim=0)  # 1 x N x dim
                samples2_exp = torch.unsqueeze(sample2, dim=0)  # 1 x N x dim
                proj_samples1 = torch.sum(samples1_exp*g_cp_exp, dim=-1, keepdim=True)  # g x N x 1
                proj_samples2 = proj_samples1
                # proj_samples2=torch.sum(samples2_exp*g_cp_exp,dim=-1,keepdim=True)# g x N x 1
                median_dist = median_heruistic_proj(proj_samples1, proj_samples2).clone()  # g
                bandwidth_array = torch.sqrt(median_dist / np.log(proj_samples1.shape[1])) # magic bandwidth
                # bandwidth_array = 2*torch.sqrt(0.5*median_dist)
                # bandwidth_array = kwargs['bandwidth_scale']*2 * torch.pow(0.5 * median_dist, kwargs['median_power'])
                kernel_hyper['bandwidth_array'] = bandwidth_array

    if 'score' in kwargs:
        score = kwargs['score']
    else:
        score = score_func(sample1)  # sam1
        score = torch.autograd.grad(torch.sum(score), sample1)[0]  # sam1 x D

    with torch.no_grad():
        r_exp = torch.unsqueeze(r, dim=1)  # r x 1 x d
        proj_score = torch.sum(r_exp * score, dim=-1)  # r x sam 1

        sample1_exp = sample1.reshape(
            (1, sample1.shape[0], sample1.shape[1]))  # 1 x sam1 x d
        # sample2_exp = samples.reshape((1, sample2.shape[0], sample2.shape[1]))  #  1 x sam2 x d

        g_exp = g.reshape((g.shape[0], 1, g.shape[1]))  # g x 1 x d
        proj_orig_samples = torch.sum(g_exp * sample1_exp, dim=-1)
        proj_samples1 = torch.unsqueeze(
            proj_orig_samples, dim=-1)  # g x sam1 x 1
        proj_samples2 = torch.unsqueeze(proj_orig_samples, dim=-2)
        # proj_samples2 = torch.unsqueeze(torch.sum(g_exp * sample2_exp, dim=-1), dim=-2)  # g x 1 x sam2
        K = kernel(proj_samples1, proj_samples2,
                   kernel_hyper=kernel_hyper)  # g x sam1 x sam2
        proj_score_exp = proj_score.reshape(
            (proj_score.shape[0], proj_score.shape[1], 1))  # r x sam1 x 1

        Term1 = proj_score_exp * K  # r x sam1 x sam2, the drift force

        rg = torch.sum(r*g, dim=-1).reshape(r.shape[0], 1, 1)  # r x 1 x 1
        r_K = repulsive_kernel(proj_samples1, proj_samples2, kernel_hyper=kernel_hyper, K=K)  # g x sam 1 x sam2
        Term2 = rg * r_K  # g x sam1 x sam2 # repulsive force

        force = torch.mean(Term1+r_coef*Term2, dim=1)  # r(g) x sam 2
        force = force.t()  # sam2 x r

    if flag_repulsive_output:
        return force, torch.mean(Term2, dim=1).t()  # sam2 x dim
    else:
        return force
