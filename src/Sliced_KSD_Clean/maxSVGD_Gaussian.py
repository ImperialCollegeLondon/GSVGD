import sys
import os
import argparse

cwd=os.getcwd()
cwd_parent=os.path.abspath('..')
sys.path.append(cwd)
sys.path.append(cwd_parent+'/Divergence')
sys.path.append(cwd_parent)
import random
import torch
from Util import *
from src.Sliced_KSD_Clean.Divergence.Network import *
from src.Sliced_KSD_Clean.Divergence.Def_Divergence import *
from src.Sliced_KSD_Clean.Divergence.Kernel import *
from src.Sliced_KSD_Clean.Divergence.Dataloader import *
import matplotlib.pylab as plt
import torch.distributions as D
from src.Sliced_KSD_Clean.density_ import xshaped_gauss_experiment
path='' # Change to your own path
if not os.path.exists(path + 'results_figs'):
    os.mkdir(path + 'results_figs')

results_folder = path + 'results_figs'



# Comment this if GPU is not used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser=argparse.ArgumentParser(description='SVGD with Gaussian')
parser.add_argument('--num_samples',type=int,default=500)

parser.add_argument('--method',type=str,default='maxSVGD',metavar='Method',help='SVGD or maxSVGD')
parser.add_argument('--epoch',type=int,default=6000)
parser.add_argument('--median_power',type=float,default=0.5)
parser.add_argument('--result_interval',type=int,default=1)
parser.add_argument('--init_scale',type=float,default=5) # variance initialized samples
parser.add_argument('--lr',type=float,default=1e-2) # variance initialized samples
parser.add_argument('--dim',type=int,default=50) # variance initialized samples


args=parser.parse_args()
method=args.method
n_samples=args.num_samples

band_scale=1 # control the bandwidth scale


#dim_array=np.unique(np.around(np.linspace(2,100,num=20))) # different dimensions
dim_array=[args.dim]

init_mean=2 # gaussian mean for initialized samples
init_scale=args.init_scale # gaussian Scale for initialized samples

eps=1e-3 # Update step size

Variance_list=[]
MMD_list=[]
repulsive_list=[]
mean_list=[]

# LL tuning
LL = []
n_epoch = args.epoch
lr = args.lr


for dim in (dim_array):
    dim=int(dim)
    print('Dim:%s'%(dim))
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # n_comp = 2
    # p_mean = torch.zeros(dim).reshape(1, -1) # zero mean target
    # p_mean2 = 2 * torch.ones(dim).reshape(1, -1) # zero mean target
    # p_cov = torch.eye(dim) # unit variance target

    # torch.cat([p_mean, p_mean2], 1)
    #p_x = torch.distributions.multivariate_normal.MultivariateNormal(p_mean2, p_cov) #+ torch.distributions.multivariate_normal.MultivariateNormal(p_mean2, p_cov)
    # mix = D.Categorical(torch.rand(2, ))
    # comp = D.Independent(D.Normal(torch.cat([p_mean, p_mean2], 0), 0.5 * torch.ones(n_comp,dim)), 1)
    # p_x = torch.distributions.MixtureSameFamily(mix, comp)
 
    correlation = 0.95

    mix_means = torch.zeros((2, dim))

    p_x  = xshaped_gauss_experiment(mixture_dist=D.Categorical(torch.ones(mix_means.shape[0],)),
                                    means=mix_means,
                                    correlation=correlation)

    # import pdb;pdb.set_trace

    init_samples=init_mean+init_scale*torch.randn(n_samples,dim) # initialized samples

    # process the list to store the results
    len_comp=int(n_epoch/args.result_interval)
    Variance_comp=np.zeros(len_comp)-100000
    MMD_comp=np.zeros(len_comp)-100000

    repulsive_comp=np.zeros(len_comp)-100000
    mean_comp=np.zeros(len_comp)-100000

    counter_record=0

    if method=='SVGD':
        print('Run SVGD')
        samples = init_samples.clone().detach().requires_grad_()
        # SVGD update state
        mixSVGD_state_dict = {'M': torch.zeros(samples.shape),
                              'V': torch.zeros(samples.shape),
                              't': 1,
                              'beta1': 0.9,
                              'beta2': 0.99
                              }

        for ep in tqdm(range(n_epoch)):
            # Compute Score
            samples1=samples.clone().detach().requires_grad_()

            log_like1=p_x.log_prob(samples1)

            # score of target
            score1=torch.autograd.grad(log_like1.sum(),samples1)[0]


            # Median heuristics
            median_dist = median_heruistic(samples1, samples1.clone())
            bandwidth = 2 * 1 * torch.pow(0.5 * median_dist, args.median_power) #np.sqrt(1. / (2 * np.log(n_samples)))

            kernel_hyper_KSD = {
                'bandwidth': 0.717 * bandwidth
            }
            # Compute SVGD update force
            SVGD_force,repulsive = SVGD(samples.clone().detach(), None, SE_kernel_multi, repulsive_SE_kernel_multi,
                              kernel_hyper=kernel_hyper_KSD,
                              score=score1, repulsive_coef=1,flag_repulsive_output=True
                              )

            repulsive_max, _ = torch.max(repulsive, dim=1)

            samples, mixSVGD_state_dict = SVGD_AdaGrad_update(samples, SVGD_force, eps, mixSVGD_state_dict)
            samples = samples.clone().requires_grad_()
            # Evaluation
            if (ep)%args.result_interval==0:
                # Record results
                samples_np=samples.cpu().data.numpy() # N x dim

                samples_cov = np.var(samples_np, axis=0).mean()
                Variance_comp[counter_record] = samples_cov

                repulsive_comp[counter_record]=repulsive_max.mean().cpu().data.numpy()

                mean_comp[counter_record]=samples.mean(0).mean(-1).cpu().data.numpy()

                if (ep)%(50*args.result_interval)==0:
                    print('ep:%s var:%s rep:%s'%(ep,samples_cov,repulsive_max.mean().cpu().data.numpy()))

                counter_record+=1

    elif method=='maxSVGD':
        print('Run maxSVGD')
        samples = init_samples.clone().detach().requires_grad_()
        # initialize r,g directions
        g = torch.eye(samples.shape[-1]).requires_grad_()
        r = torch.eye(samples.shape[-1])

        Adam_g = torch.optim.Adam([g], lr=lr,betas=(0.5,0.9))
        # g update epoch
        g_update = 1
        # S-SVGD update state
        mixSVGD_state_dict = {'M': torch.zeros(samples.shape),
                              'V': torch.zeros(samples.shape),
                              't': 1,
                              'beta1': 0.9,
                              'beta2': 0.99
                              }
        # the bandwidth is computed inside functions, so here we set it to None
        kernel_hyper_maxSVGD = {
            'bandwidth': None
        }
        # Record the previous samples. This will be used for determining whether S-SVGD converged. If the change is not huge between samples, then we stop the g optimization, and
        # continue the S-SVGD update until the accumulated changes between samples are large enough. This is to avoid the overfitting of g to small number of samples.
        samples_pre_fix=samples.clone().detach()
        avg_change=0
        counter_not_opt=0
        for ep in tqdm(range(int(n_epoch))):
            # Optimize g
            # if (ep+1) %1== 0 and ep<8000:
            #     flag_opt = True
            # else:
            #     flag_opt = False

            #if (samples_pre_fix-samples).abs().max()>5*avg_change: #3e-2
            if get_distance(samples_pre_fix,samples)>np.sqrt(dim)*0.15:
                # the change between samples are large enough, so S-SVGD not coverged, we update g direction
                flag_opt=True
                samples_pre_fix=samples.clone().detach()
                counter_not_opt=0
            else:
                # accumulate the epochs that the changes between samples are small.
                counter_not_opt+=1
                # accumulate 10 epochs, we update g.
                if counter_not_opt%40==0:
                    flag_opt=True
                else:
                    flag_opt=False

            # Update g direction
            for i_g in range(1):
                Adam_g.zero_grad()
                # Normalize
                g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
                # whether update g or not
                if flag_opt:
                    samples1 = samples.clone().detach().requires_grad_()
                    # compute target score
                    log_like1 = p_x.log_prob(samples1)

                    score1 = torch.autograd.grad(log_like1.sum(), samples1)[0]
                    diver, _ = compute_max_DSSD_eff(samples1.detach(), samples1.clone().detach(), None, SE_kernel,
                                                d_kernel=d_SE_kernel,
                                                dd_kernel=dd_SE_kernel,
                                                r=r, g=g_n, kernel_hyper=kernel_hyper_maxSVGD,
                                                score_samples1=score1, score_samples2=score1.clone()
                                                , flag_median=True, flag_U=False, median_power=0.5,
                                                bandwidth_scale=band_scale
                                                )
                    (-diver).backward()
                    Adam_g.step()

                    g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
                    g_n = g_n.clone().detach()
                    # print(g_n, 'THIS IS G_N')
                    # print(r, 'THIS IS R')

                log_like1 = p_x.log_prob(samples)

                score1 = torch.autograd.grad(log_like1.sum(), samples)[0]

                maxSVGD_force, repulsive = max_DSSVGD(samples, None, SE_kernel, repulsive_SE_kernel, r=r, g=g_n,
                                                      flag_median=True, median_power=0.5,
                                                      kernel_hyper=kernel_hyper_maxSVGD, score=score1,
                                                      bandwidth_scale=band_scale,
                                                      repulsive_coef=1, flag_repulsive_output=True)

                repulsive_max, _ = torch.max(repulsive, dim=1)

                samples, mixSVGD_state_dict = SVGD_AdaGrad_update(samples, maxSVGD_force, eps, mixSVGD_state_dict)
                samples = samples + 0. * torch.randn(samples.shape)
                samples = samples.clone().requires_grad_()

                if (ep) % args.result_interval == 0:
                    # Record results
                    samples_np = samples.cpu().data.numpy()  # N x dim
                    samples_cov = samples.var(0).cpu().data.numpy().mean()  # np.var(samples_np, axis=0).mean()
                    Variance_comp[counter_record] = samples_cov
                    repulsive_comp[counter_record] = repulsive_max.mean().cpu().data.numpy()
                    mean_comp[counter_record] = samples.mean(0).mean(-1).cpu().data.numpy()

                    if (ep) % (50 * args.result_interval) == 0:
                        print('ep:%s var:%s rep:%s' % (
                        ep, samples_cov, repulsive_max.mean().cpu().data.numpy()))
                    counter_record += 1

                LL.append(p_x.log_prob(samples).mean().cpu().data.numpy())

    #import pdb;pdb.set_trace()
    true_samples = p_x.sample([args.num_samples]).cpu().data.numpy()

    plt.scatter(true_samples[:, 0],true_samples[:, 1], label = 'TRUE samples', alpha=0.3)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], label = '{} samples'.format(args.method), alpha=0.3)
    plt.legend()
    plt.savefig(results_folder + f'/results_debug_{args.method}_epoch{n_epoch}_lr{lr}_samples{n_samples}_dim{dim}.png')
    plt.close()

    print("num_epochs:", len(LL))
    plt.plot(range(n_epoch), LL)
    plt.xlabel("Epochs")
    plt.ylabel("Loglikelihood")
    plt.legend()
    plt.savefig(results_folder + f'/results_loglik_{args.method}_epoch{n_epoch}_lr{lr}_samples{n_samples}_dim{dim}.png')
    plt.close()
    import pandas as pd
    pd.DataFrame(LL, columns=["loglik"]).to_csv(results_folder + f'/results_loglik_{args.method}_epoch{n_epoch}_lr{lr}_samples{n_samples}_dim{dim}.csv')


    # Store results
    Variance_list.append(Variance_comp)
    MMD_list.append(MMD_comp)
    repulsive_list.append(repulsive_comp)
    mean_list.append(mean_comp)

    Variance_list_np = np.stack(Variance_list, axis=0)
    MMD_list_np = np.stack(MMD_list, axis=0)
    repulsive_list_np = np.stack(repulsive_list, axis=0)
    mean_list_np = np.stack(mean_list, axis=0)






