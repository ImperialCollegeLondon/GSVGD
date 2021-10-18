import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

def UCI_evaluation(test_loader,Net,W,UCI_state_dict,**kwargs):
    #raise NotImplementedError
    if 'flag_train' in kwargs:
        flag_train=kwargs['flag_train']
    else:
        flag_train=False

    mean_Y_train,std_Y_train=UCI_state_dict['mean_Y_train'],UCI_state_dict['std_Y_train'] # 1 and 1
    loggamma=W[:,-2].unsqueeze(-1)-2*np.log(std_Y_train) # np x1

    if 'gamma_heuristic' in kwargs:
        flag_gamma_heuristic=kwargs['gamma_heuristic']
    else:
        flag_gamma_heuristic=False


    if flag_gamma_heuristic: # From original SVGD paper, but not used in our experiemnt
        dev_size=np.min([UCI_state_dict['X_test'].shape[0],500])
        X=torch.from_numpy(UCI_state_dict['X_test'][0:dev_size,:]).float().cuda()
        Y=torch.from_numpy(UCI_state_dict['Y_test'][0:dev_size]).float().cuda().unsqueeze(0)
        Y_pred = torch.squeeze(Net.forward(X, W)) * std_Y_train + mean_Y_train # np x N
        loggamma=-torch.log(torch.mean((Y_pred-Y)**2,dim=-1,keepdim=True))# np x 1


    if flag_train:
        RMSE_mat = torch.ones(W.shape[0], UCI_state_dict['X_train'].shape[0])
        ll_mat = torch.ones(W.shape[0], UCI_state_dict['X_train'].shape[0])
    else:
        RMSE_mat=torch.ones(W.shape[0],UCI_state_dict['X_test'].shape[0])
        ll_mat=torch.ones(W.shape[0],UCI_state_dict['X_test'].shape[0])
    idx_start=0


    with torch.no_grad():

        for idx,data in enumerate(test_loader):
            X,Y=data[0],data[1]
            Y_pred=torch.squeeze(Net.forward(X,W))*std_Y_train+mean_Y_train # np x N
            if flag_train:
                Y=Y*std_Y_train+mean_Y_train

            Y=Y.unsqueeze(0) # 1 x N
            diff=(Y_pred-Y)**2 # np x N
            # RMSE
            mean_diff=torch.mean(Y_pred-Y,dim=0)**2 # N

            RMSE_mat[:,idx_start:idx_start+diff.shape[-1]]=mean_diff



            # NLL
            ll=-0.5 * 1 * (np.log(2 * np.pi) - loggamma) - (torch.exp(loggamma) / 2) \
                       * diff # np x N
            #ll=torch.sqrt(torch.exp(loggamma))/np.sqrt(2*np.pi)*torch.exp(-1*(diff)/2*torch.exp(loggamma))
            ll_mat[:,idx_start:idx_start+diff.shape[-1]]=ll
            idx_start+=diff.shape[-1]


    RMSE=torch.sqrt(torch.mean(RMSE_mat))

    mean_ll=torch.logsumexp(ll_mat,dim=0)-np.log(W.shape[0]) # N
    mean_ll=torch.mean(mean_ll)

    #mean_ll=torch.mean(torch.log(torch.mean(ll_mat,dim=0)))
    return RMSE,mean_ll


def median_heruistic(sample1,sample2):
    with torch.no_grad():
        G=torch.sum(sample1*sample1,dim=-1)# N or * x N
        G_exp=torch.unsqueeze(G,dim=-2) # 1 x N or * x1 x N

        H=torch.sum(sample2*sample2,dim=-1)
        H_exp=torch.unsqueeze(H,dim=-1) # N x 1 or * * x N x 1
        dist=G_exp+H_exp-2*sample2.matmul(torch.transpose(sample1,-1,-2)) #N x N or  * x N x N
        if len(dist.shape)==3:
            dist=dist[torch.triu(torch.ones(dist.shape, device=sample1.device))==1].view(dist.shape[0],-1)# * x (NN)
            median_dist,_=torch.median(dist,dim=-1) # *
        else:
            dist=(dist-torch.tril(dist)).view(-1)
            median_dist=torch.median(dist[dist>0.])
    return median_dist.clone().detach()

def SVGD_AdaGrad_update(samples,force,eps,state_dict):
    M=state_dict['M']
    t=state_dict['t']
    beta1=state_dict['beta1']
    if t==1:
        M=M+force*force
    else:
        M=beta1*M+(1-beta1)*(force*force)
    adj_grad=force/(torch.sqrt(M)+1e-10)
    samples.data=samples.data+eps*adj_grad
    t=t+1
    state_dict['M']=M
    state_dict['t']=t
    return samples,state_dict

def dist_stat(sample1,sample2):
    # Summarize the sample distance, this is to check how spread-out the samples are
    with torch.no_grad():
        G=torch.sum(sample1*sample1,dim=-1)# N
        G_exp=torch.unsqueeze(G,dim=0) # 1 x N

        H=torch.sum(sample2*sample2,dim=-1)
        H_exp=torch.unsqueeze(H,dim=-1) # N x 1
        dist=G_exp+H_exp-2*sample2.matmul(sample1.t()) # N x N
        max_dist=torch.max(dist)
        dist_tril = (dist - torch.tril(dist)).view(-1)
        sum_dist=torch.sum(dist_tril)
    return max_dist,sum_dist

def median_heruistic_proj(sample1,sample2):
    # Compute the median heuristic for projected samples
    # samples 1 is * x g x N x 1
    # samples 2 is * x g x N x 1
    with torch.no_grad():
        G=torch.sum(sample1*sample1,dim=-1) # * x num_g x N or r x g x N
        G_exp = torch.unsqueeze(G, dim=-2)  # * x num_g x 1 x N or * x r x g x 1 x N

        H=torch.sum(sample2*sample2,dim=-1) # * x num_g x N or * x r x g x N
        H_exp=torch.unsqueeze(H, dim=-1) # * x numb_g x N x 1 or * x r x g x N x 1

        dist = G_exp + H_exp - 2*torch.matmul(sample2,torch.transpose(sample1,-1,-2)) # * x G x N x N

        if len(dist.shape)==4:
            dist=dist[torch.triu(torch.ones(dist.shape, device=sample1.device))==1].view(dist.shape[0],dist.shape[1],-1) # r x g x (NN) or * x g x (NN)

        else:
            ind_array = torch.triu(torch.ones(dist.shape, device=sample1.device))
            dist=dist[torch.triu(ind_array)==1].view(dist.shape[0],-1) # g x (NN)

        dist = (dist - torch.tril(dist)).view(dist.shape[0],-1) # num_g x (NxN)
        #median_dist=torch.median(dist[dist>0].view(dist.shape[0],-1),dim=-1) # num_g
        median_dist,_=torch.median(dist,dim=-1) # num_g or * x g
    return median_dist.clone().detach()


def get_distance(sample1,sample2):
    # Compute the distance between samples
    return ((sample1-sample2)**2).sum(-1).sqrt().mean()