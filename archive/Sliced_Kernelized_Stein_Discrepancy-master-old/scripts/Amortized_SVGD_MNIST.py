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
from src.Dataloader import *
import pickle
import random
import time
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

path="./" # Change to your own path

os.makedirs(path,exist_ok=True)
os.makedirs(path+'model_save/',exist_ok=True)


parser=argparse.ArgumentParser(description='BNN_UCI Improper Initialization')
parser.add_argument('--method',type=str,default='Amortized_maxSVGD',metavar='Method',help='Amortized_maxSVGD or Amortized_SVGD or ELBO')
parser.add_argument('--r_coef',type=float,default=1.) # Repulsive coefficient. Large value = large repulsive force.

parser.add_argument('--SVGD_step',type=int,default=1) # Amortized SVGD/maxSVGD steps
parser.add_argument('--eps_SVGD',type=float,default=0.1) # SVGD/maxSVGD step size
parser.add_argument('--drop_rate',type=float,default=0.3) # Dropout rate for noisy encoder

parser.add_argument('--latent_dim',type=int,default=32) # # Latent dimensions
parser.add_argument('--disable_gpu',action='store_true') # set this to disable gpu

args=parser.parse_args()


dtype = torch.FloatTensor
if not args.disable_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # dtype = torch.cuda.FloatTensor

torch.set_default_tensor_type(dtype)

# Draw and save image
def save_image(x, path, nrow=10):
    x = x.view(-1, 28, 28)  # N x 28 x 28
    x = torch.unsqueeze(x, dim=1)  # N x 1 x 28 x 28
    grid_img = torchvision.utils.make_grid(x, nrow=nrow)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().data.numpy())
    plt.savefig(path, dpi=100)

# Model hyperparameter settings
dim=28*28
latent=args.latent_dim
noise_dim=args.latent_dim # noise input dimensions
eps_de=1e-4 # learning rate for decoder
eps_en=1e-4 # learning rate for encoder
batch_size=100
epoch=100 # training epoch
z_sample=10 # number of latent variables drawn for each update.
method=args.method
SVGD_step=args.SVGD_step
encoder_update=1
eps_SVGD=args.eps_SVGD
n_g_update = 1 # g update steps
r_coef = args.r_coef # repulsive coefficient



if method=='ELBO':
    # For vanilla VAE, no drop out is used in encoder
    flag_dropout=False
else:
    # otherwise, use dropout in encoder.
    flag_dropout=True

# Model architecture
encoder_setting={
    'dim':dim,'latent_dim':latent,
    'noise_dim':noise_dim,'flag_dropout':flag_dropout,
    'hidden_layer_num':2,'hidden_unit':[300,200],
    'activations':'ReLU','activations_output':None,
    'flag_only_output_layer':False,
    'drop_out_rate':args.drop_rate,'flag_addnoise':False
}

decoder_setting={
    'flag_dropout':False,
    'hidden_layer_num':2,'hidden_unit':[200,300],
    'activations':'ReLU','activations_output':'Sigmoid',
    'flag_only_output_layer':False,
    'drop_out_rate':None,'flag_addnoise':False
}

# Define VAE model
if method == 'Amortized_maxSVGD' or method == 'Amortized_SVGD':
    VAE_noise=VAE_n(encoder_setting,decoder_setting)
    Adam_VAE_n_encoder = torch.optim.Adam(list(VAE_noise.module_list[0].parameters()), lr=eps_en, betas=(0.9, 0.99))
    Adam_VAE_n_decoder = torch.optim.Adam(list(VAE_noise.module_list[1].parameters()), lr=eps_de, betas=(0.9, 0.99))

elif method == 'ELBO':
    VAE_vanilla = VAE(encoder_setting, decoder_setting)
    Adam_VAE = torch.optim.Adam(list(VAE_vanilla.parameters()), lr=eps_de, betas=(0.9, 0.99))

# Define Dataset
train_loader=torch.utils.data.DataLoader(stochMNIST('../data', train=True, download=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(stochMNIST('../data', train=False, download=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

# Start training
if method=='Amortized_maxSVGD':
    print('Run:%s'%(method))
    # Identity initialization for r,g
    g = torch.eye(latent).requires_grad_()
    r = torch.eye(latent)
    # For maxSVGD, r is fixed and not updated.
    Adam_g = torch.optim.Adam([g], lr=0.001, betas=(0.9, 0.99))



    kernel_hyper_maxSVGD = {
        'bandwidth': None
    }
    counter_iter = 0 # count number of training iterations

    g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
    r_n = r / (torch.norm(r, 2, dim=-1, keepdim=True) + 1e-10)
    time_SSVGD = 0
    # Start training

    for ep in tqdm(range(epoch)):
        for idx, (data, label) in enumerate((train_loader)):
            VAE_noise.train()
            data = data.type(dtype)
            data = data.view(-1, dim)

            # Update encoder
            start = time.time()
            for k in range(encoder_update):
                ## generate latent samples
                # batch x num_latent x latent_dim
                latent_samples, _, noise = VAE_noise._encoding(data, z_sample=z_sample,
                                                               flag_output_n=True)  # N x z x latent
                # Run maxSVGD to get updated samples
                latent_samples_cp = latent_samples.clone().detach().requires_grad_()
                for i in range(SVGD_step):
                    for j in range(n_g_update):
                        # find slice direction g
                        Adam_g.zero_grad()
                        # batch x num_latent x latent_dim
                        score, score_reshape = VAE_noise.score(data, latent_samples_cp,
                                                               flag_create=False)  # N x z x dim and (N x z) x dim
                        # batch x num_latent x latent_dim
                        samples1 = latent_samples_cp.clone().detach()
                        samples2 = latent_samples_cp.clone().detach()
                        g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
                        maxSSD, _ = compute_max_DSSD_eff_Tensor(samples1, samples2, None,
                                                                SE_kernel, d_SE_kernel, dd_SE_kernel,
                                                                flag_U=False, kernel_hyper=kernel_hyper_maxSVGD,
                                                                r=r_n, g=g_n, score_samples1=score,
                                                                score_samples2=score.clone(), flag_median=True,
                                                                median_power=0.5,
                                                                bandwidth_scale=0.35
                                                                )

                        (-maxSSD).mean().backward()
                        Adam_g.step()

                    maxSVGD_force = max_DSSVGD_Tensor(latent_samples_cp, None, SE_kernel, repulsive_SE_kernel, r=r,
                                                      g=g_n,
                                                      flag_median=True, median_power=0.5,
                                                      kernel_hyper=kernel_hyper_maxSVGD, score=score,
                                                      bandwidth_scale=0.35,
                                                      repulsive_coef=r_coef)  # * x sam x dim


                    # Update samples
                    latent_samples_cp.data = latent_samples_cp.data + eps_SVGD * maxSVGD_force
                    latent_samples_cp = latent_samples_cp.clone().detach().requires_grad_()  # N x z x latent

                # time usage
                duration = time.time() - start
                time_SSVGD += duration

                # Update encoder by MSE
                Adam_VAE_n_encoder.zero_grad()

                loss_MSE = (((latent_samples - latent_samples_cp) ** 2).sum(-1)).mean()
                loss_MSE.backward()
                Adam_VAE_n_encoder.step()

            ### Update decoder
            Adam_VAE_n_decoder.zero_grad()
            latent_samples = latent_samples_cp.clone().detach()
            loss, _ = VAE_noise.loss(data, latent_samples)
            (-loss).backward()
            Adam_VAE_n_decoder.step()
            if counter_iter % 50 == 0:
                print('Amortized S-SVGD time per iter:%s' % (time_SSVGD / (counter_iter + 1)))
                print('Counter:%s Reconstruct loss:%s maxSKSD:%s MSE:%s mean_var1:%s mean_var2:%s' % (
                    counter_iter, loss.cpu().data.numpy(), maxSSD.mean().cpu().data.numpy(),
                    loss_MSE.cpu().data.numpy(), latent_samples.var(-2).mean().cpu().data.numpy(),
                    latent_samples.view(-1, latent).var(0).mean().cpu().data.numpy()))

            counter_iter += 1

    # final image geneartion
    VAE_noise.eval()
    with torch.no_grad():
        # Generation
        sample = VAE_noise._generate(n_sample=100)  # N X dim
        save_image(sample, path + '%s_lat%s_ep%s.png' % (method, latent, ep), nrow=10
                   )

        # Reconstruct
        recons = VAE_noise._reconstruct(data, z_sample=1, z=None)
        recons = recons.squeeze()
        save_image(data, path + '%s_lat%s_ep%s_orig.png' % (method, latent, ep), nrow=10
                   )
        save_image(recons, path + '%s_lat%s_ep%s_recon.png' % (method, latent, ep), nrow=10
                   )
        # Save model
        check_point = {
            'state_dict': VAE_noise.state_dict(),
            'Adam_encoder': Adam_VAE_n_encoder.state_dict(),
            'Adam_decoder': Adam_VAE_n_decoder.state_dict(),
            'loss': loss,
            'g': g,
            'Adam_g': Adam_g.state_dict(), 'encoder_setting': encoder_setting, 'decoder_setting': decoder_setting,

            'SVGD_step': SVGD_step,
            'encoder_update': encoder_update,
            'eps_SVGD': eps_SVGD, 'r_coef': r_coef
        }
        torch.save(check_point,
                   path + '/model_save/' + '%s_lat%s_CP_ep%s' % (method, latent, ep))


# SVGD
elif method=='Amortized_SVGD':
    print('Run:%s' % (method))

    counter_iter = 0
    time_SVGD=0
    for ep in tqdm(range(epoch)):
        for idx, (data, label) in enumerate((train_loader)):
            VAE_noise.train()
            data = data.type(dtype)
            data = data.view(-1, dim)
            # Update Encoder
            start = time.time()
            for k in range(encoder_update):
                ## generate latent samples
                latent_samples, _ = VAE_noise._encoding(data, z_sample=z_sample)  # N x z x latent

                ## update directions and SVGD Updates
                latent_samples_cp = latent_samples.clone().detach().requires_grad_()
                for i in range(SVGD_step):
                    # Comptute SVGD
                    median_dist = median_heruistic(latent_samples_cp, latent_samples_cp.clone())
                    bandwidth = 2 * torch.pow(0.5 * median_dist,0.5)

                    kernel_hyper_KSD = {
                        'bandwidth': 1.*0.35 * bandwidth
                    }

                    # SVGD updates
                    score, score_reshape = VAE_noise.score(data, latent_samples_cp,
                                                           flag_create=False)  # N x z x dim and (N x z) x dim
                    SVGD_force = SVGD_Tensor(latent_samples_cp.clone().detach(), None, SE_kernel_multi,
                                             repulsive_SE_kernel_multi,
                                             kernel_hyper=kernel_hyper_KSD,
                                             score=score, repulsive_coef=r_coef
                                             )

                    latent_samples_cp.data = latent_samples_cp.data + eps_SVGD * SVGD_force
                    latent_samples_cp = latent_samples_cp.clone().detach().requires_grad_()  # N x z x latent

                # Time usage
                duration = time.time() - start
                time_SVGD += duration

                # Update encoder by MSE
                Adam_VAE_n_encoder.zero_grad()
                loss_MSE = (((latent_samples - latent_samples_cp) ** 2).sum(-1)).mean()
                loss_MSE.backward()
                Adam_VAE_n_encoder.step()

            # Decoder Update
            Adam_VAE_n_decoder.zero_grad()
            latent_samples, _ = VAE_noise._encoding(data, z_sample=z_sample)  # N x z x dim and (N x z) x dim
            latent_samples = latent_samples.clone().detach()

            loss, _ = VAE_noise.loss(data, latent_samples)
            (-loss).backward()
            Adam_VAE_n_decoder.step()
            if counter_iter % 50 == 0:
                print('SVGD time per iter:%s' % (time_SVGD / (counter_iter+1)))
                print('Counter:%s Reconstruct loss:%s MSE:%s mean_var1:%s mean_var2:%s' % (
                    counter_iter, loss.cpu().data.numpy(),  loss_MSE.cpu().data.numpy(),latent_samples.var(-2).mean().cpu().data.numpy(),latent_samples.view(-1,latent).var(0).mean().cpu().data.numpy()))

            counter_iter += 1

    # Evaluation
    VAE_noise.eval()
    with torch.no_grad():
        sample = VAE_noise._generate(n_sample=100)  # N X dim
        save_image(sample, path + '%s_lat%s_ep%s_Setting%s.png' % (method, latent, ep, args.Setting), nrow=10
                   )

        # reconstruct
        recons = VAE_noise._reconstruct(data, z_sample=1, z=None)
        recons = recons.squeeze()
        save_image(data, path + '%s_lat%s_ep%s_orig.png' % (method, latent, ep), nrow=10
                   )
        save_image(recons, path + '%s_lat%s_ep%s_recon.png' % (method, latent, ep), nrow=10
                   )

        check_point = {
            'state_dict': VAE_noise.state_dict(),
            'Adam_encoder': Adam_VAE_n_encoder.state_dict(),
            'Adam_decoder': Adam_VAE_n_decoder.state_dict(),
            'loss': loss, 'encoder_setting': encoder_setting, 'decoder_setting': decoder_setting,
            'SVGD_step': SVGD_step,
            'encoder_update': encoder_update,
            'eps_SVGD': eps_SVGD, 'r_coef': r_coef
        }
        torch.save(check_point,
                   path + '/model_save/' + '%s_lat%s_CP_ep%s' % (method, latent, ep))



elif method=='ELBO':
    z_sample=1 # only need 1 latent sample for ELBO
    print('Run:%s' % (method))
    counter_iter = 0
    for ep in range(epoch):
        for idx, (data, label) in enumerate((train_loader)):
            #VAE_noise.train()
            data = data.type(dtype)
            data = data.view(-1, dim)
            Adam_VAE.zero_grad()

            loss,log_likelihood,recons_loss,latent_samples=VAE_vanilla.loss(data,z_sample=z_sample)

            (-loss).backward()
            Adam_VAE.step()

            if counter_iter % 50 == 0:
                print('Counter:%s Reconstruct loss:%s mean_var2:%s' % (
                    counter_iter, recons_loss.cpu().data.numpy(),
                    latent_samples.view(-1, latent).var(0).mean().cpu().data.numpy()))

            counter_iter+=1

    VAE_vanilla.eval()
    with torch.no_grad():
        sample = VAE_vanilla._generate(n_sample=100)  # N X dim
        save_image(sample, path + '%s_lat%s_ep%s_ReLU_Setting.png' % (method, latent, ep), nrow=10
                   )

        # reconstruct
        recons = VAE_vanilla._reconstruct(data, z_sample=1)
        recons = recons.squeeze()
        save_image(data, path + '%s_lat%s_ep%s_ReLU_orig.png' % (method, latent, ep), nrow=10
                   )
        save_image(recons, path + '%s_lat%s_ep%s_ReLU_recon.png' % (method, latent, ep), nrow=10
                   )

        check_point = {
            'state_dict': VAE_vanilla.state_dict(),
            'Adam': Adam_VAE.state_dict(),
            'loss': loss,
            'recons': recons_loss, 'encoder_setting': encoder_setting, 'decoder_setting': decoder_setting
        }
        torch.save(check_point, path + '/model_save/' + '%s_lat%s_CP_ep%s_ReLU' % (method,latent, ep))











