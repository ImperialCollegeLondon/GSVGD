import sys
import os
import argparse

import torch
import numpy as np
import random

from Util import *
from Divergence.Network import *
from Divergence.Def_Divergence import *
from Divergence.Kernel import *
from Divergence.Dataloader import *
from prettytable import PrettyTable


cwd=os.getcwd()
cwd_parent=os.path.abspath('..')
sys.path.append(cwd)
sys.path.append(cwd_parent+'/Divergence')
sys.path.append(cwd_parent)

path='/home/paperspace/Projects/SlicedKSD/' # Change to your path

# comment this if you don't want to use GPU
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0'
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser=argparse.ArgumentParser(description='BNN_UCI Improper Initialization')
parser.add_argument('--method',type=str,default='maxSVGD',metavar='Method',help='SVGD or maxSVGD')
parser.add_argument('--result_interval',type=int,default=4) # kin8nm:4 Naval:6 Combined:5 Wine:2 Protein:30
parser.add_argument('--init_scale',default=1e-8,type=float)# other: 0.0001 kin8nm 1e-8 Naval:1e-8 Combined:1e-8 Wine:1e-8
parser.add_argument('--epoch',type=int,default=2000) # kin8nm:200 Naval:200 Combined:200 Boston:2000 Energy:2000 Concrete:2000, Yacht:2000 Wine:500 Protein:50
parser.add_argument('--dataset',type=str,default='d0.xls')
parser.add_argument('--flag_enable_iter',type=str,default='False') # Careful! changing flag_enable_iter requires the re-tune of ep_repuslive/counter_repuslive
parser.add_argument('--ep_repulsive',type=int,default=1000)# Boston Housing: 1000ep # Careful! Tune this if flag_enable_iter is False
parser.add_argument('--counter_repulsive',type=int,default=1) #Careful! Tune this if flag_enable_iter is True

args=parser.parse_args()

method=args.method
if args.flag_enable_iter=='True':
    flag_enable_iter=True

elif args.flag_enable_iter=='False':
    flag_enable_iter=False
else:
    raise NotImplementedError

result_interval=args.result_interval

ep_repulsive=args.ep_repulsive # This is to control the repulsive epoch
counter_repulsive=args.counter_repulsive # For wine dataset, also control repulsive epoch

rand_seed_list=[1,10,20,30,40,50,60,70,80,90,100,110,140,170,190]

RMSE_list=[] #np.zeros(len(rand_seed_list))-1000
ll_list=[] #np.zeros(len(rand_seed_list))-1000
max_dist_list=[]
sum_dist_list=[]

best_RMSE_list=[]
best_RMSE_LL_list=[]
best_LL_list=[]
best_LL_RMSE_list=[]

counter_seed=0

Tab=PrettyTable()
Tab.field_names= ['Seed','RMSE', "RMSE_LL", "LL", "LL_RMSE",'max_dist_RMSE','max_dist_LL','sum_dist_RMSE','sum_dist_LL']

n_epoch = args.epoch # epochs



for seed in rand_seed_list:
    # Fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # Dataset path
    UCI_path = '/home/paperspace/Projects/SlicedKSD/Experiments/Data/UCI/%s' % (args.dataset)

    # Define UCI dataloader
    # Define dataset and DataLoader
    UCI_state_dict = UCI_preprocess(UCI_path, flag_normalize=True)

    UCI_train_set = UCI_Dataset(UCI_state_dict, flag_test=False)
    UCI_test_set = UCI_Dataset(UCI_state_dict, flag_test=True)
    UCI_train_loader = DataLoader(UCI_train_set, batch_size=100, shuffle=False)
    UCI_test_loader = DataLoader(UCI_test_set, batch_size=100, shuffle=False)

    # Compute total number of training iterations if n_epoch is used
    tot_iter = int(np.ceil(UCI_state_dict['X_train'].shape[0] / 100)) * n_epoch
    list_len = int(np.floor(tot_iter / result_interval))

    # How many interavals we use for data recording
    result_interval=args.result_interval


    # how fine the results are stored, here we use iteration number instead of epoch number
    if not flag_enable_iter:
        RMSE_comp = np.zeros(n_epoch) - 10000
        ll_comp = np.zeros(n_epoch) - 10000
        max_dist_comp = np.zeros(n_epoch) - 10000
        sum_dist_comp = np.zeros(n_epoch) - 10000
    else:
        RMSE_comp = np.zeros(list_len) - 10000
        ll_comp = np.zeros(list_len) - 10000
        max_dist_comp = np.zeros(list_len) - 10000
        sum_dist_comp = np.zeros(list_len) - 10000


    # Learning rate and training hyper-parameters
    eps = 0.0015
    Kernel_Choice = 'RBF'
    Median_Power = 0.5 # Square root of the median heuristic

    n_particles = 50 # SVGD particle number
    n_train = UCI_state_dict['X_train'].shape[0] # training data set size
    a0 = 1 # For Gamma distribution prior hyperparameter
    b0 = 0.1 # For Gamma distribution prior hyperparameter

    # BNN definition
    BNN = Bayesian_NN_eff(UCI_state_dict['X_train'].shape[-1], n_h=50)  # for year and protein it is 100
    # BNN Weight initialization
    # init weights
    weights = init_weights(n_particles, BNN, a0, b0, scale=args.init_scale) # init_scale controls how close the partiles are
    # better initialization for gamma (From original SVGD Paper)
    ridx = np.random.choice(range(UCI_state_dict['X_train'].shape[0]), \
                            np.min([UCI_state_dict['X_train'].shape[0], 1000]), replace=False
                            )
    x_hat = torch.from_numpy(UCI_state_dict['X_train'][ridx, :]).float().cuda()
    y_hat = torch.from_numpy(UCI_state_dict['Y_train'][ridx]).float().cuda().unsqueeze(0)  # 1 x N

    y_pred = torch.squeeze(BNN.forward(x_hat, weights))  # np x n
    loggamma = -torch.log(torch.mean((y_hat - y_pred) ** 2, dim=-1))  # np
    weights[:, -2] = loggamma
    # make weights to be used for auto-diff
    weights = weights.clone().requires_grad_()
    counter_record = 0
    counter = 0

    if method == 'SVGD':
        print('Run SVGD')
        # SVGD update step state
        mixSVGD_state_dict = {'M': torch.zeros(weights.shape),
                              'V': torch.zeros(weights.shape),
                              't': 1,
                              'beta1': 0.9,
                              'beta2': 0.99
                              }

        # Training script
        for ep in tqdm(range(int(n_epoch))):
            for idx, data in enumerate(UCI_train_loader):
                X, Y = data[0], data[1]
                # Compute score
                score,_ = BNN_compute_score_eff(BNN, weights, X, Y, n_train, a0, b0, flag_gamma=True)

                # Comptute SVGD
                median_dist = median_heruistic(weights, weights.clone()) # Compute the median distance
                bandwidth = 2 * np.sqrt(1. / (2 * np.log(n_particles))) * torch.pow(0.5 * median_dist, Median_Power) # Compute the bandwidth

                kernel_hyper_KSD = {
                    'bandwidth': 1. * bandwidth
                }

                SVGD_force = SVGD(weights.clone().detach(), None, SE_kernel_multi, repulsive_SE_kernel_multi,
                                  kernel_hyper=kernel_hyper_KSD,
                                  score=score, repulsive_coef=1
                                  )

                weights, mixSVGD_state_dict = SVGD_AdaGrad_update(weights, SVGD_force, eps, mixSVGD_state_dict)
                weights = weights.clone().requires_grad_()

                if flag_enable_iter: # Evaluation
                    if (counter + 1) % result_interval == 0:
                        # Evaluation
                        RMSE, mean_ll = UCI_evaluation(UCI_test_loader, BNN, weights, UCI_state_dict, gamma_heuristic=False)
                        RMSE_comp[counter_record], ll_comp[
                            counter_record] = RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy()
                        max_dist, sum_dist = dist_stat(weights, weights.clone())
                        max_dist_comp[counter_record], sum_dist_comp[
                            counter_record] = max_dist.cpu().data.numpy(), sum_dist.cpu().data.numpy()

                        counter_record += 1
                        weights = weights.clone().requires_grad_()
                        if (counter + 1) % (result_interval * 50) == 0:
                            print('seed:%s ep:%s counter:%s RMSE:%s LL:%s max_dist:%s sum_dist:%s' % (
                                seed, ep, counter,
                                RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy(),
                                max_dist.cpu().data.numpy(),
                                sum_dist.cpu().data.numpy()
                            ))

                counter += 1

            if not flag_enable_iter: # store the results accoding to epoch
                # Evaluation
                RMSE, mean_ll = UCI_evaluation(UCI_test_loader, BNN, weights, UCI_state_dict, gamma_heuristic=False)
                RMSE_comp[ep], ll_comp[ep] = RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy()
                max_dist,sum_dist=dist_stat(weights,weights.clone())
                max_dist_comp[ep],sum_dist_comp[ep]=max_dist.cpu().data.numpy(),sum_dist.cpu().data.numpy()

                counter_record+=1
                weights = weights.clone().requires_grad_()
                if (ep + 1) % 50 == 0:
                    print('seed:%s ep:%s RMSE:%s LL:%s max_dist:%s sum_dist:%s' % (
                        seed, ep,
                        RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy(),
                        max_dist.cpu().data.numpy(),
                        sum_dist.cpu().data.numpy()
                    ))

        # Final Output

        best_RMSE = RMSE_comp.min()
        best_RMSE_idx = np.argmin(RMSE_comp)
        best_RMSE_ll = ll_comp[best_RMSE_idx]

        best_LL = ll_comp.max()
        best_LL_idx = np.argmax(ll_comp)
        best_LL_RMSE = RMSE_comp[best_LL_idx]

        max_dist_RMSE = max_dist_comp[best_RMSE_idx]
        max_dist_LL = max_dist_comp[best_LL_idx]

        sum_dist_RMSE = sum_dist_comp[best_RMSE_idx]
        sum_dist_LL = sum_dist_comp[best_LL_idx]

        best_RMSE_list.append(best_RMSE)
        best_LL_list.append(best_LL)
        best_RMSE_LL_list.append(best_RMSE_ll)
        best_LL_RMSE_list.append(best_LL_RMSE)

        Tab.add_row(['seed:%s' % (seed), '%.4f' % best_RMSE, '%.4f' % best_RMSE_ll, '%.4f' % best_LL,
                     '%.4f' % best_LL_RMSE, '%.4f' % max_dist_RMSE, '%.4f' % max_dist_LL,
                     '%.4f' % sum_dist_RMSE, '%.4f' % sum_dist_LL])

        print('seed:%s RMSE:%s RMSE_LL:%s LL:%s LL_RMSE:%s dist_RMSE:%s dist_LL:%s' % (
        seed, best_RMSE, best_RMSE_ll, best_LL, best_LL_RMSE, max_dist_RMSE, max_dist_LL))

    elif method=='maxSVGD': # S-SVGD
        print('Run maxSVGD')

        # g,r initialization and setup
        g = torch.eye(weights.shape[-1]).requires_grad_()
        r = torch.eye(weights.shape[-1]) # r is fixed to be the identity matrix.
        Adam_g = torch.optim.Adam([g], lr=1.e-3, betas=(0.9, 0.99))
        g_update = 1 # Epochs for update g direction

        # S-SVGD update state
        mixSVGD_state_dict = {'M': torch.zeros(weights.shape),
                              'V': torch.zeros(weights.shape),
                              't': 1,
                              'beta1': 0.9,
                              'beta2': 0.99
                              }


        # Training Hyperparameters
        flag_opt = True # whether update g direction
        band_scale = 0.05 # The scale for bandwidth of S-SVGD

        for ep in tqdm(range(int(n_epoch))):
            if not flag_enable_iter: # Note this is to control the repulsive according to epoch
                # Control the repulsive coefficient, initially small coefficint due to HUGE repulsive force, then increasing the coefficient to 1.
                r_coef = np.clip(ep / ep_repulsive, 0.001, 1)
            for idx, data in enumerate(UCI_train_loader):

                if flag_enable_iter: # Control the repulsive force using iterations
                    # Control the repulsive coefficient, initially small coefficint due to HUGE repulsive force, then increasing the coefficient to 1.
                    r_coef = np.clip(counter / counter_repulsive, 0.001, 1)

                X, Y = data[0], data[1]
                # Compute score
                score, _ = BNN_compute_score_eff(BNN, weights, X, Y, n_train, a0, b0, flag_gamma=True)
                # The median heuristic are computed inside the function, so we use None here
                kernel_hyper_maxSVGD = {
                    'bandwidth': None
                }

                # Optimize g direction
                if flag_opt:
                    for i_g in range(g_update):
                        Adam_g.zero_grad()
                        g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
                        weights_cp = weights.clone().detach()
                        diver, _ = compute_max_DSSD_eff(weights_cp, weights.clone().detach(), None, SE_kernel,
                                                        d_kernel=d_SE_kernel,
                                                        dd_kernel=dd_SE_kernel,
                                                        r=r, g=g_n, kernel_hyper=kernel_hyper_maxSVGD,
                                                        score_samples1=score, score_samples2=score.clone()
                                                        , flag_median=True, flag_U=False, median_power=Median_Power,
                                                        bandwidth_scale=band_scale
                                                        )
                        (-diver).backward()
                        Adam_g.step()
                        # print('Diver:%s'%(diver.cpu().data.numpy()))

                g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10) # Normalize the g
                g_n = g_n.clone().detach() # decouple the g
                # Compute the S-SVGD update
                maxSVGD_force = max_DSSVGD(weights, None, SE_kernel, repulsive_SE_kernel, r=r, g=g_n,
                                           flag_median=True, median_power=Median_Power,
                                           kernel_hyper=kernel_hyper_maxSVGD, score=score, bandwidth_scale=band_scale,
                                           repulsive_coef=r_coef)

                weights, mixSVGD_state_dict = SVGD_AdaGrad_update(weights, maxSVGD_force, eps, mixSVGD_state_dict)
                weights = weights.clone().requires_grad_()
                # Evaluation
                if flag_enable_iter:
                    if (counter+1)%result_interval==0:
                        # Evaluation
                        RMSE, mean_ll = UCI_evaluation(UCI_test_loader, BNN, weights, UCI_state_dict, gamma_heuristic=False)
                        RMSE_comp[counter_record], ll_comp[
                            counter_record] = RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy()
                        max_dist, sum_dist = dist_stat(weights, weights.clone())
                        max_dist_comp[counter_record], sum_dist_comp[
                            counter_record] = max_dist.cpu().data.numpy(), sum_dist.cpu().data.numpy()

                        counter_record += 1
                        weights = weights.clone().requires_grad_()
                        if (counter+1)%(result_interval*50)==0:
                            print('seed:%s ep:%s counter:%s RMSE:%s LL:%s max_dist:%s sum_dist:%s' % (
                                seed, ep,counter,
                                RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy(),
                                max_dist.cpu().data.numpy(),
                                sum_dist.cpu().data.numpy()
                            ))
                counter += 1
            if not flag_enable_iter:
                # Evaluation
                RMSE, mean_ll = UCI_evaluation(UCI_test_loader, BNN, weights, UCI_state_dict, gamma_heuristic=False)
                RMSE_comp[ep], ll_comp[
                    ep] = RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy()
                max_dist, sum_dist = dist_stat(weights, weights.clone())
                max_dist_comp[ep], sum_dist_comp[
                    ep] = max_dist.cpu().data.numpy(), sum_dist.cpu().data.numpy()

                counter_record += 1
                weights = weights.clone().requires_grad_()
                if (ep + 1) % 50 == 0:
                    print('seed:%s ep:%s RMSE:%s LL:%s max_dist:%s sum_dist:%s' % (
                        seed, ep,
                        RMSE.cpu().data.numpy(), mean_ll.cpu().data.numpy(),
                        max_dist.cpu().data.numpy(),
                        sum_dist.cpu().data.numpy()
                    ))

        # Final output Evaluation

        best_RMSE = RMSE_comp.min()
        best_RMSE_idx = np.argmin(RMSE_comp)
        best_RMSE_ll = ll_comp[best_RMSE_idx]

        best_LL = ll_comp.max()
        best_LL_idx = np.argmax(ll_comp)
        best_LL_RMSE = RMSE_comp[best_LL_idx]

        max_dist_RMSE = max_dist_comp[best_RMSE_idx]
        max_dist_LL = max_dist_comp[best_LL_idx]
        sum_dist_RMSE = sum_dist_comp[best_RMSE_idx]
        sum_dist_LL = sum_dist_comp[best_LL_idx]

        best_RMSE_list.append(best_RMSE)
        best_LL_list.append(best_LL)
        best_RMSE_LL_list.append(best_RMSE_ll)
        best_LL_RMSE_list.append(best_LL_RMSE)

        Tab.add_row(['seed:%s' % (seed), '%.4f' % best_RMSE, '%.4f' % best_RMSE_ll, '%.4f' % best_LL,
                     '%.4f' % best_LL_RMSE, '%.4f' % max_dist_RMSE, '%.4f' % max_dist_LL,
                     '%.4f' % sum_dist_RMSE, '%.4f' % sum_dist_LL])

        print('seed:%s RMSE:%s RMSE_LL:%s LL:%s LL_RMSE:%s dist_RMSE:%s dist_LL:%s' % (
            seed, best_RMSE, best_RMSE_ll, best_LL, best_LL_RMSE, max_dist_RMSE, max_dist_LL))






























