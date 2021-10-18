import sys
import os
import argparse

cwd=os.getcwd()
cwd_parent=os.path.abspath('.')
sys.path.append(cwd)
sys.path.append(cwd_parent+'/Divergence')
sys.path.append(cwd_parent)
import random
import torch
from tqdm import tqdm, trange
from src.Sliced_KSD_Clean.Util import *
from src.Sliced_KSD_Clean.Divergence.Network import *
from src.Sliced_KSD_Clean.Divergence.Def_Divergence import *
from src.Sliced_KSD_Clean.Divergence.Kernel import *
from src.Sliced_KSD_Clean.Divergence.Dataloader import *
import torch.distributions as D
from tqdm import trange

class MaxSVGD:

  def __init__(self, target, result_interval=1, device="cpu"):
    self.target = target
    self.device = device
    self.result_interval = result_interval

  def fit(
    self, 
    samples, 
    n_epoch, 
    lr, 
    eps, 
    g_nupdates=1, 
    metric=None, 
    save_every=100, 
    threshold=0,
    X_test=None,
    y_test=None
  ):

    g = torch.eye(samples.shape[-1], device=self.device).requires_grad_(True)
    # g = torch.nn.init.orthogonal_(
    #   torch.empty(samples.shape[-1], samples.shape[-1], device=self.device)
    # ).requires_grad_(True)

    self.metrics = [-1000] * (n_epoch//save_every)
    self.particles = [-1000] * (1 + n_epoch//save_every)
    self.particles[0] = samples.clone().detach().cpu()
    self.g = [-1000] * (1 + n_epoch//save_every)
    self.g[0] = g.clone().detach()
    self.pam = [0] * (n_epoch//save_every)
    self.pamrf = [0] * (n_epoch//save_every)
    self.phi_list = [0] * (n_epoch//save_every)
    self.repulsion_list = [0] * (n_epoch//save_every)

    band_scale=1 # control the bandwidth scale
    counter_record = 0

    r = torch.eye(samples.shape[-1], device=self.device)

    Adam_g = torch.optim.Adam([g], lr=lr,betas=(0.5,0.9))
    # g update epoch
    g_update = 1
    # S-SVGD update state
    mixSVGD_state_dict = {'M': torch.zeros(samples.shape, device=self.device),
                        'V': torch.zeros(samples.shape, device=self.device),
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

      # if get_distance(samples_pre_fix,samples)>np.sqrt(samples.shape[1])*0.15:
      #     # the change between samples are large enough, so S-SVGD not coverged, we update g direction
      #     flag_opt=True
      #     samples_pre_fix=samples.clone().detach()
      #     counter_not_opt=0
      # else:
      #     # accumulate the epochs that the changes between samples are small.
      #     counter_not_opt+=1
      #     # accumulate 40 epochs, we update g.
      #     if counter_not_opt%40==0:
      #         flag_opt=True
      #     else:
      #         flag_opt=False
      flag_opt = True   # remove hack

      # Update g direction
      for i_g in range(g_nupdates):
        Adam_g.zero_grad()
        # Normalize
        g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True).to(self.device) + 1e-10)
        # whether update g or not
        if flag_opt:
            samples1 = samples.clone().detach().requires_grad_()
            # compute target score
            log_like1 = self.target.log_prob(samples1)
            score1 = torch.autograd.grad(log_like1.sum(), samples1)[0]

            diver, divergence = compute_max_DSSD_eff(samples1.detach(), samples1.clone().detach(), None, SE_kernel,
                                        d_kernel=d_SE_kernel,
                                        dd_kernel=dd_SE_kernel,
                                        r=r, g=g_n, kernel_hyper=kernel_hyper_maxSVGD,
                                        score_samples1=score1, score_samples2=score1.clone()
                                        , flag_median=True, flag_U=False, median_power=0.5,
                                        bandwidth_scale=band_scale
                                        )

            (-diver).backward()
            Adam_g.step()
            g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True).to(self.device) + 1e-10)
            g_n = g_n.clone().detach()

        log_like1 = self.target.log_prob(samples)

        score1 = torch.autograd.grad(log_like1.sum(), samples)[0]

        maxSVGD_force, repulsive = max_DSSVGD(samples, None, SE_kernel, repulsive_SE_kernel, r=r, g=g_n,
                                              flag_median=True, median_power=0.5,
                                              kernel_hyper=kernel_hyper_maxSVGD, score=score1,
                                              bandwidth_scale=band_scale,
                                              repulsive_coef=1, flag_repulsive_output=True)

        repulsive_max, _ = torch.max(repulsive, dim=1)

        # particle-averaged magnitude (batch x num_particles)
        # pam = torch.linalg.norm(maxSVGD_force.detach(), dim=1).mean().item()
        pam = torch.max(maxSVGD_force.abs().detach(), dim=1)[0].mean().item()
        pamrf = torch.max(repulsive.abs().detach(), dim=1)[0].mean().item()
        # update particles
        samples, mixSVGD_state_dict = SVGD_AdaGrad_update(samples, maxSVGD_force, eps, mixSVGD_state_dict)
        samples = samples.clone().requires_grad_()

        # if (ep) % self.result_interval == 0:
        #     Record results
        #     samples_np = samples.cpu().data.numpy()  # N x dim
        #     samples_cov = samples.var(0).cpu().data.numpy().mean()  # np.var(samples_np, axis=0).mean()
        #     Variance_comp[counter_record] = samples_cov
        #     repulsive_comp[counter_record] = repulsive_max.mean().cpu().data.numpy()
        #     mean_comp[counter_record] = samples.mean(0).mean(-1).cpu().data.numpy()

        #     if (ep) % (50 * self.result_interval) == 0:
        #         print('ep:%s var:%s rep:%s' % (
        #         ep, samples_cov, repulsive_max.mean().cpu().data.numpy()))
        #     counter_record += 1
          
      if (ep+1)%save_every==0:
          # self.metrics[ep//save_every] = metric(samples.detach())
          self.particles[1 + ep//save_every] = samples.clone().detach().cpu()
          self.g[1 + ep//save_every] = g_n.clone().detach().cpu()
          self.pam[ep//save_every] = pam
          self.pamrf[ep//save_every] = pamrf
          self.phi_list[ep//save_every] = maxSVGD_force
          self.repulsion_list[ep//save_every] = repulsive

      # evaluate test accuracy if appropriate
      if X_test is not None and (ep % 500) == 0:
          _, _, acc = self.target.evaluation(samples.clone().detach(), X_test, y_test)
          print(f"Epoch {ep} batch {ep} accuracy:", acc)

      # early stop
      if pam < threshold:
        print(f"GSVGD converged in {ep+1} epochs as PAM {pam} is less than {threshold}")
        break  
    return samples, self.metrics





class MaxSVGDLR(MaxSVGD):
  def fit(
    self, 
    samples, 
    n_epoch, 
    lr, 
    eps, 
    g_nupdates=1, 
    metric=None, 
    save_every=100, 
    threshold=0,
    train_loader=None,
    test_data=None,
    valid_data=None
  ):

    self.metrics = [-1000] * (n_epoch//save_every)
    self.particles = [samples.clone().detach()]
    self.pam = [0] * (n_epoch//save_every)
    self.test_accuracy = []
    self.valid_accuracy = []
    
    X_valid, y_valid = valid_data
    X_test, y_test = test_data

    band_scale=1 # control the bandwidth scale
    counter_record = 0

    g = torch.eye(samples.shape[-1], device=self.device).requires_grad_(True)
    # g = torch.nn.init.orthogonal_(
    #   torch.empty(samples.shape[-1], samples.shape[-1], device=self.device)
    # ).requires_grad_(True)
    r = torch.eye(samples.shape[-1], device=self.device)

    Adam_g = torch.optim.Adam([g], lr=lr,betas=(0.5,0.9))
    # g update epoch
    g_update = 1
    # S-SVGD update state
    mixSVGD_state_dict = {'M': torch.zeros(samples.shape, device=self.device),
                        'V': torch.zeros(samples.shape, device=self.device),
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

    pbar = trange(int(n_epoch))
    # for ep in tqdm(range(int(n_epoch))):
    for ep in pbar:
      for j, (X_batch, y_batch) in enumerate(train_loader):
  
        flag_opt = True   # remove hack

        # Update g direction
        for i_g in range(g_nupdates):
          Adam_g.zero_grad()
          # Normalize
          g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True).to(self.device) + 1e-10)
          # whether update g or not
          if flag_opt:
              samples1 = samples.clone().detach().requires_grad_()
              # compute target score
              log_like1 = self.target.log_prob(samples1, X_batch, y_batch)
              score1 = torch.autograd.grad(log_like1.sum(), samples1)[0]
              #! this is using a lot of CPU!
              diver, divergence = compute_max_DSSD_eff(samples1.detach(), samples1.clone().detach(), None, SE_kernel,
                                          d_kernel=d_SE_kernel,
                                          dd_kernel=dd_SE_kernel,
                                          r=r, g=g_n, kernel_hyper=kernel_hyper_maxSVGD,
                                          score_samples1=score1, score_samples2=score1.clone()
                                          , flag_median=True, flag_U=False, median_power=0.5,
                                          bandwidth_scale=band_scale
                                          )

              (-diver).backward()
              Adam_g.step()
              g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True).to(self.device) + 1e-10)
              g_n = g_n.clone().detach()

          log_like1 = self.target.log_prob(samples, X_batch, y_batch)

          score1 = torch.autograd.grad(log_like1.sum(), samples)[0]

          maxSVGD_force, repulsive = max_DSSVGD(samples, None, SE_kernel, repulsive_SE_kernel, r=r, g=g_n,
                                                flag_median=True, median_power=0.5,
                                                kernel_hyper=kernel_hyper_maxSVGD, score=score1,
                                                bandwidth_scale=band_scale,
                                                repulsive_coef=1, flag_repulsive_output=True)

          repulsive_max, _ = torch.max(repulsive, dim=1)

          # particle-averaged magnitude (batch x num_particles)
          # pam = torch.linalg.norm(maxSVGD_force.detach(), dim=1).mean().item()
          pam = torch.max(maxSVGD_force.detach().abs(), dim=1)[0].mean().item()
          # update particles
          samples, mixSVGD_state_dict = SVGD_AdaGrad_update(samples, maxSVGD_force, eps, mixSVGD_state_dict)
          samples = samples.clone().requires_grad_()

        # evaluate test accuracy if appropriate
        train_steps = ep * len(train_loader) + j
        if X_test is not None and (train_steps % save_every == 0):
            self.particles.append((ep, samples.clone().detach()))
            _, _, test_acc, test_ll = self.target.evaluation(samples.clone().detach(), X_test, y_test)
            _, _, valid_acc, valid_ll = self.target.evaluation(samples.clone().detach(), X_valid, y_valid)
            self.test_accuracy.append((train_steps, test_acc, test_ll))
            self.valid_accuracy.append((train_steps, valid_acc, valid_ll))

            if j % 100 == 0:
              pbar.set_description(f"Epoch {ep} batch {j} accuracy: {valid_acc} ll: {valid_ll}")

      if metric and (ep+1)%save_every==0:
          # self.metrics[ep//save_every] = metric(samples.detach())
          self.particles[1 + ep//save_every] = samples.clone().detach().cpu()
          self.pam[ep//save_every] = pam
      
      # early stop
      if pam < threshold:
        print(f"GSVGD converged in {ep+1} epochs as PAM {pam} is less than {threshold}")
        break  
    
    return samples, self.metrics
