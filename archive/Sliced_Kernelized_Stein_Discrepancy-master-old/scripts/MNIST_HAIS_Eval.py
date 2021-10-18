# Hamiltonian Annealed Importance sampling for evaluating Amortized SVGD VAE
import sys
import os
import argparse
import torch,torchvision

cwd=os.getcwd()
cwd_parent=os.path.abspath('..')
sys.path.append(cwd)
sys.path.append(cwd_parent+'/src')
sys.path.append(cwd_parent)

from src.Util import *
from src.Network import *
from src.Divergence import *
from src.Kernel import *
from src.distribution import *
from src.GOF_Test import *
from src.active_slice import *
from src.Dataloader import *
import pickle
import random
import time


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

path='...' # Change to your own path

parser=argparse.ArgumentParser(description='HAIS_Eval')

parser.add_argument('--disable_gpu',action='store_true') # set this to disable gpu
parser.add_argument('--method',type=str,default='Amortized_maxSVGD',metavar='Method',help='Amortized_maxSVGD or Amortized_SVGD or ELBO')


args=parser.parse_args()

method = args.method

dtype = torch.FloatTensor
if not args.disable_gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dtype = torch.cuda.FloatTensor

torch.set_default_tensor_type(dtype)


dim_list= None ## Change this to your latent dimensions, e.g. [16,32,48,64]. Note: this need to match the model you load.


method_list=['Amortized_maxSVGD','Amortized_SVGD','ELBO']

# PrettyTable to print results
from prettytable import PrettyTable

Tab=PrettyTable()
Tab.field_names=['Dimension','Amortized maxSVGD','Amortized SVGD','ELBO']


# Start evaluation
for dim_lat in dim_list:
    logws_list=['%s'%(dim_lat)]
    z_list=['%s'%(dim_lat)]
    for method in method_list:
        print('Dimension:%s Method:%s'%(dim_lat,method))
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        test_loader=torch.utils.data.DataLoader(stochMNIST('../data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=1000, shuffle=False)
        # Because stochMNIST is dynamically binarized MNIST. So for fair comparision, we need to fixed the MNIST data set.
        te = FixedstochMNIST(test_loader)
        test_loader = torch.utils.data.DataLoader(te,
                                                  batch_size=1000, shuffle=False)








        path_n='...' # Change to your model storage path
        model_name = '...' # Change this to the name of your model save file


        state_dict=torch.load(path_n+model_name)

        # Load model parameters
        encoder_setting=state_dict['encoder_setting']
        decoder_setting=state_dict['decoder_setting']

        flag_dropout=encoder_setting['flag_dropout']

        dim=encoder_setting['dim']
        latent=encoder_setting['latent_dim']
        noise_dim=encoder_setting['noise_dim']

        if method == 'Amortized_maxSVGD' or method == 'Amortized_SVGD':
            VAE_noise=VAE_n(encoder_setting,decoder_setting)
            VAE_noise.load_state_dict(state_dict['state_dict'])
        elif method =='ELBO':
            VAE_vanilla=VAE(encoder_setting,decoder_setting)
            VAE_vanilla.load_state_dict(state_dict['state_dict'])

        VAE_HAIS=HAIS(leap_length=10,sample_size=100)

        if flag_dropout:
            VAE_noise.eval()

        # HAIS
        eps=0.01
        temp_schedule=np.linspace(0, 1., 500)
        if method == 'Amortized_maxSVGD' or method =='Amortized_SVGD':
            # Implicit VAE
            logws,z_after=VAE_HAIS.AIS(test_loader,VAE_noise,eps,VAE_noise.log_prior,VAE_noise.log_decoder,temp_schedule,flag_stepsize=True,limit_size=5000) #20ep:89.37 60ep:88.2190 99ep:87.94/87.90/87.07(0.15) SVGD:90,90.57
            logws_list.append('%.4f'%(logws.mean().cpu().data.numpy()))
            z_list.append(z_after)
        elif method=='ELBO':
            # Vanilla VAE
            logws,z_after=VAE_HAIS.AIS(test_loader,VAE_vanilla,eps,VAE_vanilla.log_prior,VAE_vanilla.log_decoder,temp_schedule,flag_stepsize=True,limit_size=5000) #89.65
            logws_list.append('%.4f'%(logws.mean().cpu().data.numpy()))
            z_list.append(z_after)

        else:
            raise NotImplementedError
    Tab.add_row(logws_list)
    Tab_str = Tab.get_string()
    path_NLL = '...' # Change this to your own result storage path
    with open(path_NLL +'MNIST_NLL', 'w') as f:
        f.write(Tab_str)

    z_check={
        'z_list':z_list
    }
    torch.save(z_check,path_NLL+'z_list_dim%s'%(dim_lat))
