import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import collections
import math
from tqdm import tqdm


def Swish(x):
    return x*F.sigmoid(x)
class fc_Encoder_Decoder(nn.Module):
    '''
    This class implements the base network structure for fully connected encoder or decoder.
    '''
    def __init__(self,input_dim,output_dim,hidden_layer_num=2,hidden_unit=[100,50],activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False,flag_addnoise=False):
        super(fc_Encoder_Decoder,self).__init__()
        self.drop_out_rate=drop_out_rate
        self.flag_drop_out=flag_drop_out
        self.output_dim=output_dim
        self.input_dim=input_dim
        self.hidden_layer_num=hidden_layer_num
        self.hidden_unit=hidden_unit
        self.flag_only_output_layer=flag_only_output_layer
        self.flag_linear=False
        self.flag_add_noise=flag_addnoise
        self.enable_output_act=False
        self.drop_out=nn.Dropout(self.drop_out_rate)
        # activation functions
        self.activations = activations
        if activations=='ReLU':
            self.act=F.relu
        elif activations=='Sigmoid':
            self.act=F.sigmoid
        elif activations=='Tanh':
            self.act=F.tanh
        elif activations=='Elu':
            self.act=F.elu
        elif activations=='Selu':
            self.act=F.selu
        elif activations=='Softplus':
            self.act=F.softplus
        elif activations=='Swish':
            self.act=Swish
        elif activations==None:
            self.flag_linear=True
        elif activations=='LReLU':
            self.act=lambda x:F.leaky_relu(x,0.2,False)

        else:
            raise NotImplementedError

        if activations_output=='ReLU':
            self.enable_output_act=True
            self.act_out=F.relu
        elif activations_output=='Sigmoid':
            self.enable_output_act = True
            self.act_out=F.sigmoid
        elif activations_output=='Tanh':
            self.enable_output_act = True
            self.act_out=F.tanh
        elif activations_output=='Elu':
            self.enable_output_act = True
            self.act_out=F.elu
        elif activations_output=='Selu':
            self.enable_output_act = True
            self.act_out=F.selu
        elif activations_output=='Softplus':
            self.enable_output_act = True
            self.act_out=F.softplus
        elif activations_output=='Swish':
            self.enable_output_act = True
            self.act_out = Swish

        # whether to use multi NN or single layer NN
        if self.flag_only_output_layer==False:
            assert len(self.hidden_unit)==hidden_layer_num,'Hidden layer unit length %s inconsistent with layer number %s'%(len(self.hidden_unit),self.hidden_layer_num)

            # build hidden layers
            self.hidden=nn.ModuleList()

            for layer_ind in range(self.hidden_layer_num):
                if layer_ind==0:
                    self.hidden.append(nn.Linear(self.input_dim,self.hidden_unit[layer_ind]))
                else:
                    self.hidden.append(nn.Linear(self.hidden_unit[layer_ind-1],self.hidden_unit[layer_ind]))


            # output layer
            self.out=nn.Linear(self.hidden_unit[-1],self.output_dim)


            # Xavier initializer
            # for layers in self.hidden:
            #     torch.nn.init.xavier_uniform(layers.weight)
        else:
            self.out=nn.Linear(self.input_dim,self.output_dim)

        #torch.nn.init.xavier_uniform(self.out.weight)
    def _assign_weight(self,W_dict):
        if self.flag_only_output_layer==False:
            for layer_ind in range(self.hidden_layer_num):
                layer_weight=W_dict['weight_layer_%s'%(layer_ind)]
                layer_bias=W_dict['bias_layer_%s'%(layer_ind)]

                self.hidden[layer_ind].weight=torch.nn.Parameter(layer_weight.data)
                self.hidden[layer_ind].bias=torch.nn.Parameter(layer_bias.data)
            out_weight=W_dict['weight_out']
            out_bias=W_dict['bias_out']
            self.out.weight=torch.nn.Parameter(out_weight.data)
            self.out.bias=torch.nn.Parameter(out_bias.data)
        else:
            out_weight = W_dict['weight_out']
            out_bias = W_dict['bias_out']
            self.out.weight = torch.nn.Parameter(out_weight.data)
            self.out.bias = torch.nn.Parameter(out_bias.data)

    def _get_W_dict(self):
        W_dict=collections.OrderedDict()
        if self.flag_only_output_layer == False:
            for layer_ind in range(self.hidden_layer_num):
                W_dict['weight_layer_%s'%(layer_ind)]=self.hidden[layer_ind].weight.clone().detach()
                W_dict['bias_layer_%s'%(layer_ind)]=self.hidden[layer_ind].bias.clone().detach()
            W_dict['weight_out']=self.out.weight.clone().detach()
            W_dict['bias_out']=self.out.bias.clone().detach()
        else:
            W_dict['weight_out'] = self.out.weight.clone().detach()
            W_dict['bias_out'] = self.out.bias.clone().detach()
        return W_dict
    def _get_grad_W_dict(self):
        G_dict=collections.OrderedDict()
        if self.flag_only_output_layer == False:
            for layer_ind in range(self.hidden_layer_num):
                if self.hidden[layer_ind].weight.grad is not None:
                    G_dict['weight_layer_%s'%(layer_ind)]=-self.hidden[layer_ind].weight.grad.clone().detach()
                    G_dict['bias_layer_%s'%(layer_ind)]=-self.hidden[layer_ind].bias.grad.clone().detach()
            G_dict['weight_out']=-self.out.weight.grad.clone().detach()
            G_dict['bias_out']=-self.out.bias.grad.clone().detach()
        else:
            G_dict['weight_out'] = -self.out.weight.grad.clone().detach()
            G_dict['bias_out'] = -self.out.bias.grad.clone().detach()
        return G_dict
    def _flatten_stat(self):
        if self.flag_only_output_layer==False:
            for idx,layer in enumerate(self.hidden):
                W_weight,b_weight=layer.weight.view(1,-1),layer.bias.view(1,-1) # 1 x dim
                weight_comp=torch.cat((W_weight,b_weight),dim=1) # 1 x dim
                if idx==0:
                    weight_flat=weight_comp
                else:
                    weight_flat=torch.cat((weight_flat,weight_comp),dim=1)

            # Output layer (need to account for the mask)
            Out_weight,Out_b_weight=self.out.weight,self.out.bias # N_in x N_out or N_out

            return weight_flat,Out_weight,Out_b_weight
        else:
            Out_weight, Out_b_weight = self.out.weight, self.out.bias  # N_in x N_out or N_out
            return [],Out_weight, Out_b_weight


    def forward(self,x):
        '''
        The forward pass
        :param x: Input Tensor
        :type x: Tensor
        :return: output from the network
        :rtype: Tensor
        '''
        if self.flag_only_output_layer==False:
            for layer in self.hidden:
                if self.flag_linear:
                    x=layer(x)
                else:
                    x=self.act(layer(x))
                if self.flag_drop_out:
                    x=self.drop_out(x)
                if self.flag_add_noise:
                    x=x+torch.randn(x.shape)
            if self.enable_output_act==True:
                output=self.act_out(self.out(x))
            else:
                output=self.out(x)


        else:
            if self.enable_output_act==True:
                output=self.act_out(self.out(x))
            else:
                output=self.out(x)

        return output


################### For Amortized SVGD MNIST

class VAE(nn.Module):
    def __init__(self,encoder_setting,decoder_setting):
        super(VAE, self).__init__()
        self.encoder_setting = encoder_setting
        self.decoder_setting = decoder_setting
        self.dim = encoder_setting['dim']
        self.latent_dim = encoder_setting['latent_dim']
        self.noise_dim = encoder_setting['noise_dim']
        self.flag_dropout = encoder_setting['flag_dropout']

        self.encoder = fc_Encoder_Decoder(input_dim=self.dim, output_dim=2*self.latent_dim,
                                          hidden_layer_num=encoder_setting['hidden_layer_num'],
                                          hidden_unit=encoder_setting['hidden_unit'],
                                          activations=encoder_setting['activations'],
                                          activations_output=encoder_setting['activations_output'],
                                          flag_only_output_layer=encoder_setting['flag_only_output_layer'],
                                          flag_drop_out=encoder_setting['flag_dropout'],
                                          drop_out_rate=encoder_setting['drop_out_rate'],
                                          flag_addnoise=encoder_setting['flag_addnoise'])

        self.decoder = fc_Encoder_Decoder(
            input_dim=self.latent_dim, output_dim=self.dim, hidden_layer_num=decoder_setting['hidden_layer_num'],
            hidden_unit=decoder_setting['hidden_unit'], activations=decoder_setting['activations'],
            activations_output=decoder_setting['activations_output'],
            flag_only_output_layer=decoder_setting['flag_only_output_layer']
        )

        self.module_list=nn.ModuleList()
        self.module_list.append(self.encoder)
        self.module_list.append(self.decoder)

    def _encoding(self,x,z_sample=5):


        encoder_out = self.module_list[0].forward(x)  # N x 2latent
        encoder_mean=encoder_out[:,0:self.latent_dim] # N x latent
        log_encoder_sigma=encoder_out[:,self.latent_dim:] # N x latent
        encoder_sigma=torch.exp(log_encoder_sigma) # N x latent
        noise=torch.randn(x.shape[0],z_sample,self.latent_dim) # N x z x latent
        encoder_out=encoder_mean.unsqueeze(-2)+encoder_sigma.unsqueeze(-2)*noise # N x z x latent
        encoder_out_reshape=encoder_out.view(-1,self.latent_dim) # (N x z )x latent
        return encoder_out,encoder_out_reshape,encoder_mean,encoder_sigma
    def _decoding(self,z):
        # z should be N x z_sample x latent
        out=self.module_list[-1].forward(z) # N x z_sample x dim
        out_reshape=out.view(-1,self.dim)
        return out,out_reshape

    def _reconstruct(self,x,z_sample=5):
        # x is N x dim
        encoder_out,_,_,_=self._encoding(x,z_sample=z_sample)

        out,_=self._decoding(encoder_out) # N x z x dim
        return out



    def loss(self,x,z_sample=5):
        encoder_out,_,encoder_mean,encoder_sigma=self._encoding(x,z_sample=z_sample) #N x z x dim
        recons,_=self._decoding(encoder_out) # N x z x dim
        recons=recons.clamp(1e-7,1-1e-7)
        x_exp=x.unsqueeze(-2).repeat(1,z_sample,1) # N x  z x dim
        log_likelihood=torch.sum(x_exp*torch.log(recons)+(1-x_exp)*torch.log(1-recons),dim=-1) # N x z
        recons_loss=log_likelihood.mean(-1).mean()

        KL_loss=self._KL(encoder_mean,encoder_sigma)

        loss=log_likelihood.mean(-1) # N
        loss=loss-KL_loss # N
        loss=loss.mean(-1)
        return loss,log_likelihood,recons_loss,encoder_out


    def _KL(self,encoder_mean,encoder_sigma):
        loss=-torch.log(encoder_sigma)+0.5*(encoder_sigma**2+encoder_mean**2)/1-0.5 # N x dim
        loss=loss.sum(-1) # N
        return loss

    def _generate(self,n_sample=100):
        encoding_out=torch.randn(n_sample,self.latent_dim)
        encoding_out=encoding_out.unsqueeze(-2) # N x 1 x dim
        decoding,_=self._decoding(encoding_out) # N x 1 x dim
        decoding=decoding.squeeze() # N x dim
        return decoding
    def log_prior(self,z):
        log_likelihood = -0.5 * self.latent_dim * math.log(2 * np.pi) - torch.sum(
            0.5 * 2 * np.log(1) + 1. / (2 * (1 ** 2)) * \
            (z - 0) ** 2, dim=-1)  # N x K
        return log_likelihood
    def log_decoder(self,x,z):
        output = torch.clamp(self._decoding(z)[0], min=1e-7, max=1 - (1e-7))  # N x K x obs
        x_exp=x.unsqueeze(-2).repeat(1,z.shape[-2],1) # N x  K x dim

        log_likelihood = torch.sum(
            x_exp * torch.log(output + 1e-7) + (1. - x_exp) * torch.log(1. - output + 1e-7),
            dim=-1)  # N x K
        return log_likelihood




class VAE_n(nn.Module):
    def __init__(self,encoder_setting,decoder_setting):
        super(VAE_n, self).__init__()
        self.encoder_setting=encoder_setting
        self.decoder_setting=decoder_setting
        self.dim=encoder_setting['dim']
        self.latent_dim=encoder_setting['latent_dim']
        self.noise_dim=encoder_setting['noise_dim']
        self.flag_dropout=encoder_setting['flag_dropout']

        self.encoder=fc_Encoder_Decoder(input_dim=self.dim+self.noise_dim,output_dim=self.latent_dim,hidden_layer_num=encoder_setting['hidden_layer_num'],
                     hidden_unit=encoder_setting['hidden_unit'],activations=encoder_setting['activations'],
                     activations_output=encoder_setting['activations_output'],flag_only_output_layer=encoder_setting['flag_only_output_layer'],flag_drop_out=encoder_setting['flag_dropout'],drop_out_rate=encoder_setting['drop_out_rate'],flag_addnoise=encoder_setting['flag_addnoise'])

        self.decoder=fc_Encoder_Decoder(
input_dim=self.latent_dim,output_dim=self.dim,hidden_layer_num=decoder_setting['hidden_layer_num'],
                     hidden_unit=decoder_setting['hidden_unit'],activations=decoder_setting['activations'],
                     activations_output=decoder_setting['activations_output'],flag_only_output_layer=decoder_setting['flag_only_output_layer']
)

        self.module_list=nn.ModuleList()
        self.module_list.append(self.encoder)
        self.module_list.append(self.decoder)

        # for GSVGD only for storing batch data
        # self.data = data

    def _encoding(self,x,z_sample=5,n=None,flag_output_n=False):
        if n is None:
            n = torch.randn(*x.shape[0:-1], z_sample, self.noise_dim)  # N x zn x zdim

        x_exp = x.unsqueeze(-2).repeat(1, z_sample, 1)  # N x zn x dim

        input_x = torch.cat((x_exp, n), dim=-1)  # N x  zn x (dim+zdim)

        encoder_out = self.module_list[0].forward(input_x)  # N x zn x latent
        encoder_out_reshape=encoder_out.view(-1,self.latent_dim)
        if flag_output_n:
            return encoder_out, encoder_out_reshape,n
        else:
            return encoder_out,encoder_out_reshape
    def _decoding(self,z):
        # z should be N x z_sample x latent
        out=self.module_list[-1].forward(z) # N x z_sample x dim
        out_reshape=out.view(-1,self.dim)
        return out,out_reshape

    def _reconstruct(self,x,z_sample=5,z=None):
        # x is N x dim
        encoder_out,_=self._encoding(x,z_sample=z_sample,n=z)

        out,_=self._decoding(encoder_out) # N x z x dim
        return out
    def _loss(self,x,z_sample=5,n=None):
        encoder_out,_=self._encoding(x,z_sample=z_sample,n=n) #N x z x dim
        recons,_=self._decoding(encoder_out) # N x z x dim
        recons=recons.clamp(1e-7,1-1e-7)
        x_exp=x.unsqueeze(-2).repeat(1,z_sample,1) # N x  z x dim
        log_likelihood=torch.sum(x_exp*torch.log(recons+1e-10)+(1-x_exp)*torch.log(1-recons+1e-10),dim=-1) # N x z x dim
        loss=log_likelihood.mean(-1)
        loss=loss.mean(-1)
        return loss,log_likelihood,encoder_out

    def loss(self,x,encoder_out):
        recons, _ = self._decoding(encoder_out)  # N x z x dim
        recons = recons.clamp(1e-7, 1 - 1e-7)
        x_exp = x.unsqueeze(-2).repeat(1, encoder_out.shape[-2], 1)  # N x  z x dim
        log_likelihood = torch.sum(x_exp * torch.log(recons+1e-10) + (1 - x_exp) * torch.log(1 - recons+1e-10),
                                   dim=-1)  # N x z
        loss = log_likelihood.mean(-1)
        loss = loss.mean(-1)
        return loss,log_likelihood


    def _generate(self,n_sample=100):
        encoding_out=torch.randn(n_sample,self.latent_dim)
        encoding_out=encoding_out.unsqueeze(-2) # N x 1 x dim
        decoding,_=self._decoding(encoding_out) # N x 1 x dim
        decoding=decoding.squeeze() # N x dim
        return decoding
    def _Gaussian_log_likelihood(self,encoding_out):
        # encoding_out is N x z x dim
        log_likelihood=(-0.5*np.log(2*np.pi)-0.5*(encoding_out)**2).sum(-1) # N x z
        return log_likelihood
    def score(self,x,encoder_out,flag_create=True):
        _,log_likelihood=self.loss(x,encoder_out) # N x z


        prior_log_likelihood=self._Gaussian_log_likelihood(encoder_out) # N x z

        joint_log_likelihood=log_likelihood+prior_log_likelihood # Nx z

        score=torch.autograd.grad(joint_log_likelihood.sum(),encoder_out,create_graph=flag_create,retain_graph=flag_create)[0]# N xz x dim
        return score,score.view(-1,score.shape[-1])

    def log_prob(self, encoder_out):
        recons, _ = self._decoding(encoder_out)  # N x z x dim
        recons = recons.clamp(1e-7, 1 - 1e-7)
        x_exp = self.data.unsqueeze(-2).repeat(1, encoder_out.shape[-2], 1)  # N x  z x dim
        log_likelihood = torch.sum(x_exp * torch.log(recons+1e-10) + (1 - x_exp) * torch.log(1 - recons+1e-10),
                                   dim=-1)  # N x z
        return log_likelihood


    def log_prior(self,z):
        log_likelihood = -0.5 * self.latent_dim * math.log(2 * np.pi) - torch.sum(
            0.5 * 2 * np.log(1) + 1. / (2 * (1 ** 2)) * \
            (z - 0) ** 2, dim=-1)  # N x K
        return log_likelihood
    def log_decoder(self,x,z):
        output = torch.clamp(self._decoding(z)[0], min=1e-7, max=1 - (1e-7))  # N x K x obs
        x_exp=x.unsqueeze(-2).repeat(1,z.shape[-2],1) # N x  K x dim

        log_likelihood = torch.sum(
            x_exp * torch.log(output + 1e-7) + (1. - x_exp) * torch.log(1. - output + 1e-7),
            dim=-1)  # N x K
        return log_likelihood



class HAIS(object):
    def __init__(self,leap_length,sample_size):
        self.leap_length,self.sample_size=leap_length,sample_size
    def annealed_density(self, z, x, log_decoder, log_prior, temp):
        '''
        compute log likelihood of annealed density
        :param z: with size N x K x latent
        :param x: with size N x obs
        :param log_decoder: ...... for decoder net
        :param log_prior: ...... for prior log likelihood
        :param temp: annealed temp
        :return: log_likelihood
        '''
        log_likelihood = log_prior(z)+temp*log_decoder(x,z)  # N x K
        return log_likelihood
    def _grad_U(self,z,x,log_prior,log_decoder,temp):
        log_likelihood=self.annealed_density(z,x,log_decoder,log_prior,temp) #N x K
        gradient=torch.autograd.grad(log_likelihood.sum(),z)[0] # N x K
        return -gradient
    def hmc_trajectory(self,eps,z,x,log_decoder,log_prior,temp):
        v = torch.randn(z.shape) # N x K x dim
        z_step = z.clone().requires_grad_()
        v_step = v.data - 0.5 * eps * self._grad_U(z_step, x, log_prior, log_decoder, temp)

        # leap frog
        for i in range(1, self.leap_length + 1):
            z_step.data = z_step.data + eps * v_step.data
            if i < self.leap_length:
                v_step.data = v_step.data - eps * self._grad_U(z_step, x, log_prior, log_decoder, temp
                                                              )

        v_step.data = v_step.data - 0.5 * eps * self._grad_U(z_step, x, log_prior, log_decoder, temp
                                                                )
        v_step = -v_step
        return z_step.detach(), v_step.detach(), v.detach()
    def accept_reject(self,x,z_pre,z_after,v_pre,v_after,log_decoder,log_prior,temp,accept_list=None):
        with torch.no_grad():
            current_H = self.annealed_density(z_after, x, log_decoder, log_prior, temp
                                              ) - 0.5 * torch.sum(v_after ** 2, dim=-1)
            pre_H = self.annealed_density(z_pre, x,  log_decoder, log_prior, temp
                                          ) - 0.5 * torch.sum(v_pre ** 2, dim=-1)
            prob = torch.exp(current_H - pre_H) # N x K
            uniform_sample = torch.rand(prob.shape)
            if accept_list is None:
                accept = torch.unsqueeze((prob > uniform_sample), dim=-1).float()  # N x K x 1
            else:
                accept = torch.unsqueeze(accept_list, dim=-1).clone()
            z = z_after * accept + z_pre * (1 - accept)
        z = z.clone().detach().requires_grad_()
        return z, torch.squeeze(accept)
    def AIS(self,loader,model,eps,log_prior,log_decoder,temp_schedule,flag_stepsize=False,limit_size=5000):
        print('Runing HAIS')
        label_list = []
        if limit_size is not None:
            limit_size=limit_size
        else:
            limit_size=int(1e9)
        tot_count=0
        for i, (x, label) in enumerate(loader):
            stepsize=eps

            if tot_count>=limit_size:
                break

            if flag_stepsize:
                t = 0
                mu = np.log(2 * eps)  # proposals are biased upwards to stay away from 0.
                target_accept = 0.65
                gamma = 0.05
                t = 10
                kappa = 0.75
                error_sum = 0
                log_averaged_step = 0

            x = x.cuda()
            x = x.view(-1, 784)
            z=torch.randn(*x.shape[0:-1],self.sample_size,model.latent_dim) #N x K x dim

            z=z.requires_grad_()

            logw = torch.zeros(z.shape[0], z.shape[1])  # Initialized weights with size N x K
            label_list.append(label)

            for j in tqdm(range(temp_schedule.shape[0] - 1)):
                # Importance weight
                t0 = temp_schedule[j]
                t1 = temp_schedule[j + 1]
                with torch.no_grad():
                    annealed_0 = self.annealed_density(z, x, log_decoder, log_prior, t0)
                    annealed_1 = self.annealed_density(z, x, log_decoder, log_prior, t1)

                    logw = logw + annealed_1 - annealed_0  # N x K



                z_after, v_after, v_pre = self.hmc_trajectory(stepsize, z, x, log_decoder, log_prior, t1
                                                              )
                z, accept = self.accept_reject(x, z, z_after, v_pre, v_after, log_decoder, log_prior,
                                               t1
                                               )  # N x K
                p_accept=accept.sum()/(accept.shape[0]*accept.shape[1])



                if flag_stepsize:
                    # Running tally of absolute error. Can be positive or negative. Want to be 0.
                    error_sum += target_accept - p_accept

                    # This is the next proposed (log) step size. Note it is biased towards mu.
                    log_step = mu - error_sum / (np.sqrt(t) * gamma)

                    # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
                    eta = t ** -kappa

                    # Smoothed average step size
                    log_averaged_step = eta * log_step + (1 - eta) * log_averaged_step

                    # This is a stateful update, so t keeps updating
                    t += 1
                    stepsize=torch.exp(log_averaged_step)

            logw = torch.logsumexp(logw, dim=-1) - math.log(logw.shape[-1])  # N

            tot_count+=x.shape[0]
            if i == 0:
                logws = logw
            else:
                logws = torch.cat((logws, logw), dim=0) # N

            print('The batch mean log probability:%s' % (logw.mean().cpu().data.numpy()))

        print('Mean Log Probability (after logsumexp):%s' % (logws.mean().cpu().data.numpy()))
        return logws,z_after.clone().detach()