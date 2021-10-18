import torch
import numpy as np

class multivariate_Laplace(object):
    def __init__(self,loc,scale):
        self.comp=torch.distributions.laplace.Laplace(loc, scale)
        self.dim=loc.shape[0]
    def rsample(self,size):
        return self.comp.rsample(size)
    def log_prob(self,X):
        log_prob=torch.sum(self.comp.log_prob(X),dim=-1)
        return log_prob
    def sample(self,size):
        return self.rsample(torch.Size([size]))

class multivariate_t(object):
    def __init__(self,loc,df,scale=1):
        self.scale=scale
        self.loc=loc
        self.df=df

        # Create univariate student t distribution
        self.comp_dist=torch.distributions.studentT.StudentT(self.df,loc=self.loc,scale=self.scale)
    def rsample(self,size):
        # size should be N x D
        return self.comp_dist.rsample(size)
    def log_prob(self,X):
        log_prob=torch.sum(self.comp_dist.log_prob(X),dim=-1)
        return log_prob
    def sample(self,size):
        return self.rsample(torch.Size([size]))

class GaussBernRBM(object):
    # This is used for computation
    def __init__(self,B,b,c):
        # B is D x Dh
        # b is D
        # c is Dh
        self.B=B
        self.b=b
        self.c=c

        dh=c.shape[0]
        dx=b.shape[0]
        assert B.shape[0]==dx
        assert B.shape[1]==dh
        assert dx >0
        assert dh>0
    def log_prob(self,X):
        B=self.B
        b=self.b
        c=self.c
        b=b.unsqueeze(-1)
        XBC=0.5*torch.matmul(X,B) +c.unsqueeze(0) # N x Dh
        XBC=torch.clamp(XBC,min=-80,max=80)
        exp_XBC=torch.exp(XBC)
        n_exp_XBC=torch.exp(-XBC)

        max_XBC=XBC
        n_XBC=-XBC
        max_XBC[max_XBC<n_XBC]=n_XBC[max_XBC<n_XBC]

        unden=torch.matmul(X,b)-0.5*torch.sum(X**2,dim=-1,keepdim=True)+torch.sum(torch.log(exp_XBC+n_exp_XBC),dim=-1,keepdim=True)# N
        unden_large=torch.matmul(X,b)-0.5*torch.sum(X**2,dim=-1,keepdim=True)+torch.sum(max_XBC,dim=-1,keepdim=True)# N

        unden[torch.isinf(unden)]=unden_large[torch.isinf(unden)]
        assert unden.shape[0]==X.shape[0]
        assert unden.shape[-1]==1

        return unden
    def grad_log_prob(self,X):
        B = self.B
        b = self.b
        c = self.c
        b = b.unsqueeze(0) # 1 x dx
        XBC=0.5*torch.matmul(X,B)+c.unsqueeze(0) # N x dh
        sig_XBC=torch.exp(2*XBC)
        Phi=(sig_XBC-1.)/(sig_XBC+1)
        grad_log=b-X+torch.matmul(Phi,0.5*B.t())
        return grad_log

    def dim(self):
        return self.b.shape[0]


class DSGaussBernRBM_GPU(object):
    # This is used for Data sampling
    def __init__(self,B,b,c,burnin=2000):
        # c is Dh b is Dx Note B,b,c should be Tensor now
        self.B=B
        self.b=b
        self.c=c
        self.burnin=burnin
        assert self.burnin>=0
    def sigmoid(self,x):
        return 1./(1+torch.exp(-x))
    def _blocked_gibbs_next(self,X,H):
        '''
        Sample from mutual conditional distributions
        :param X: N x Dx
        :param H: N x Dh
        :return:
        '''
        with torch.no_grad():
            dh=H.shape[1]

            n,dx=X.shape
            B=self.B
            b=self.b
            # Draw H
            XB2C=torch.matmul(X,self.B)+2.0*self.c # N x dh
            # Ph: n x dh matrix
            Ph=self.sigmoid(XB2C)

            # H: n x dh
            H=((torch.rand(n,dh)<=Ph).float())*2.-1. # only contains +1 or -1
            assert (torch.abs(H)-1>=1e-6).sum()<1
            # Draw X
            # mean: n x dx
            mean=torch.matmul(H,B.t())/2.+b
            X=torch.randn(n,dx)+mean
        return X,H
    def sample(self,n,return_latent=False,**kwargs):
        B=self.B
        b=self.b
        c=self.c
        dh=c.shape[0]
        dx=b.shape[0]
        # initialize the state of Markov Chain
        if 'burn_samples' in kwargs:
            flag_burn_samples=kwargs['burn_samples']
            n_burn_samples=kwargs['burn_in_tr_sample']
            burn_interval=10
        else:
            flag_burn_samples=False
        if 'burnin' in kwargs:
            burnin=kwargs['burnin']
        else:
            burnin=self.burnin

        if 'X' in kwargs:
            X=kwargs['X']
            H=kwargs['H']
        else:
            X=torch.randn(n,dx)
            H=np.random.randint(1,2,(n,dh))*2-1

        if flag_burn_samples:
            n_burn_sample_each=int(n_burn_samples/(burnin/burn_interval))


        # burnin
        if burnin is not None:
            for t in range(burnin):
                X,H=self._blocked_gibbs_next(X,H)
                if flag_burn_samples: # collecting pseudo samples
                    if t%burn_interval==0:
                        X_burn_comp=X[torch.randperm(X.shape[0])[0:n_burn_sample_each],:]
                        if t==0:
                            X_burn=X_burn_comp
                        else:
                            X_burn=torch.cat((X_burn,X_burn_comp),dim=0)

        # sampling
        X,H=self._blocked_gibbs_next(X,H)
        if flag_burn_samples:
            if return_latent:
                return X,H,X_burn
            else:
                return X,X_burn
        else:
            if return_latent:
                return X,H
            else:
                return X
    def dim(self):
        return self.B.shape[0]



############# For ICA
def ICA_generate_random_W_matrix(dim,scale=None,flag_from_inverse=False,dtype=None):
    flag_ok=False
    if scale is None:
        scale=1./np.sqrt(dim)
    while flag_ok==False:
        W_candidate=scale*np.random.randn(dim,dim)
        if flag_from_inverse:
            W_candidate=np.linalg.inv(W_candidate)
        cond_W=np.linalg.cond(W_candidate)
        if cond_W<dim:
            # make sure the W condition number is smaller than dim
            flag_ok=True
    return torch.from_numpy(W_candidate).float().type(dtype)

def ICA_generate_data(W,num,base_dist):
    z=base_dist.rsample(torch.Size([num])) # num x D
    z_exp=z.unsqueeze(-1) # num x D x 1
    W_exp=W.unsqueeze(0) # 1 x D x D
    x=torch.matmul(W_exp,z_exp).squeeze() # num x D
    return x
def ICA_log_likelihood(x,base_dist,W=None,W_inv=None,flag_no_grad=True):
    if W_inv is not None:
        if flag_no_grad:
            x=x.clone().detach()
            W_inv=W_inv.clone().detach()
        x_exp=x.unsqueeze(-1) # num x D x 1
        W_inv_exp=W_inv.unsqueeze(0) # num x D x D
        z=torch.matmul(W_inv_exp,x_exp).squeeze() # num x D

        log_likelihood=base_dist.log_prob(z)+1.*torch.slogdet(W_inv)[-1] # num

    elif W is not None:
        if flag_no_grad:
            W=W.clone().detach()
            x=x.clone().detach()
        W_inv=torch.inverse(W)

        x_exp = x.unsqueeze(-1)  # num x D x 1
        W_inv_exp = W_inv.unsqueeze(0)  # num x D x D
        z = torch.matmul(W_inv_exp, x_exp).squeeze()  # num x D

        #det_W=torch.abs(torch.det(W))
        log_likelihood = base_dist.log_prob(z) - 1.*torch.slogdet(W)[-1]  # num
    else:
        raise NotImplementedError('Either W or W_inv should be not None')

    return torch.mean(log_likelihood)


def ICA_un_log_likelihood(x,base_dist,W_inv):
    x_exp = x.unsqueeze(-1)  # num x D x 1
    W_inv_exp = W_inv.unsqueeze(0)  # num x D x D
    z = torch.matmul(W_inv_exp, x_exp).squeeze()  # num x D
    log_likelihood = base_dist.log_prob(z)  # num
    return log_likelihood
