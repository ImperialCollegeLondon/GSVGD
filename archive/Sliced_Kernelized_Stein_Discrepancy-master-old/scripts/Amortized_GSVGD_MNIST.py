import sys
import os
import argparse
import torch, torchvision

cwd = os.getcwd()
cwd_parent = os.path.abspath("..")
sys.path.append(cwd)
sys.path.append(cwd_parent + "/src")
sys.path.append(cwd_parent)
sys.path.append("../../")

import torch.optim as optim
from src.gsvgd import BatchGSVGD
from src.kernel_batch import BatchRBF
from src.manifold import Grassmann

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

path = "./"  # Change to your own path

os.makedirs(path, exist_ok=True)
os.makedirs(path + "model_save/", exist_ok=True)


parser = argparse.ArgumentParser(description="BNN_UCI Improper Initialization")
parser.add_argument(
    "--r_coef", type=float, default=1.0
)  # Repulsive coefficient. Large value = large repulsive force.

parser.add_argument("--SVGD_step", type=int, default=1)  # Amortized SVGD/maxSVGD steps
parser.add_argument(
    "--drop_rate", type=float, default=0.3
)  # Dropout rate for noisy encoder

parser.add_argument("--latent_dim", type=int, default=32)  # # Latent dimensions
parser.add_argument("--disable_gpu", action="store_true")  # set this to disable gpu

args = parser.parse_args()


dtype = torch.FloatTensor
if not args.disable_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
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
dim = 28 * 28
latent = args.latent_dim
noise_dim = args.latent_dim  # noise input dimensions
eps_de = 1e-4  # learning rate for decoder
eps_en = 1e-4  # learning rate for encoder
batch_size = 100
epoch = 100  # training epoch
z_sample = 10  # number of latent variables drawn for each update.
SVGD_step = args.SVGD_step
encoder_update = 1
method = "Amortized_GSVGD"

flag_dropout = True

# Model architecture
encoder_setting = {
    "dim": dim,
    "latent_dim": latent,
    "noise_dim": noise_dim,
    "flag_dropout": flag_dropout,
    "hidden_layer_num": 2,
    "hidden_unit": [300, 200],
    "activations": "ReLU",
    "activations_output": None,
    "flag_only_output_layer": False,
    "drop_out_rate": args.drop_rate,
    "flag_addnoise": False,
}

decoder_setting = {
    "flag_dropout": False,
    "hidden_layer_num": 2,
    "hidden_unit": [200, 300],
    "activations": "ReLU",
    "activations_output": "Sigmoid",
    "flag_only_output_layer": False,
    "drop_out_rate": None,
    "flag_addnoise": False,
}

# Define VAE model
VAE_noise = VAE_n(encoder_setting, decoder_setting)
Adam_VAE_n_encoder = torch.optim.Adam(
    list(VAE_noise.module_list[0].parameters()), lr=eps_en, betas=(0.9, 0.99)
)
Adam_VAE_n_decoder = torch.optim.Adam(
    list(VAE_noise.module_list[1].parameters()), lr=eps_de, betas=(0.9, 0.99)
)

# Define Dataset
train_loader = torch.utils.data.DataLoader(
    stochMNIST("../data", train=True, download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    stochMNIST("../data", train=False, download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)

# SVGD
print("Run:%s" % (method))

counter_iter = 0
time_SVGD = 0


eff_dims = 1
manifold = Grassmann(latent, eff_dims)
A = torch.eye(latent)[:, :eff_dims][None].repeat(batch_size, 1, 1).requires_grad_(True)
k = BatchRBF()
gsvgd = BatchGSVGD(
    target=None,
    kernel=k,
    manifold=manifold,
    optimizer=None,
)
for ep in tqdm(range(epoch)):
    for idx, (data, label) in enumerate((train_loader)):
        VAE_noise.train()
        data = data.type(dtype)
        data = data.view(-1, dim)
        # Update Encoder
        start = time.time()
        for k in range(encoder_update):
            ## generate latent samples
            latent_samples, _ = VAE_noise._encoding(
                data, z_sample=z_sample
            )  # N x z x latent

            ## update directions and SVGD Updates
            latent_samples_cp = latent_samples.clone().detach().requires_grad_(True)
            VAE_noise.data = data.detach().requires_grad_(False)
            gsvgd.target = VAE_noise
            gsvgd.optim = optim.Adam([latent_samples_cp], lr=1e-3)
            A, _ = gsvgd.fit(latent_samples_cp, A, epochs=SVGD_step, projection_epochs=1, verbose=False)
            # VAE_noise.log_prob(latent_samples_cp)  # N x z x dim and (N x z) x dim
            # SVGD_force = SVGD_Tensor(latent_samples_cp.clone().detach(), None, SE_kernel_multi,
            #                             repulsive_SE_kernel_multi,
            #                             kernel_hyper=kernel_hyper_KSD,
            #                             score=score, repulsive_coef=r_coef
            #                             )
            # Update samples
            # latent_samples_cp.data = latent_samples_cp.data + eps_SVGD * maxSVGD_force
            latent_samples_cp = (
                latent_samples_cp.clone().detach().requires_grad_(True)
            )  # N x z x latent

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
        latent_samples, _ = VAE_noise._encoding(
            data, z_sample=z_sample
        )  # N x z x dim and (N x z) x dim
        latent_samples = latent_samples.clone().detach()

        loss, _ = VAE_noise.loss(data, latent_samples)
        (-loss).backward()
        Adam_VAE_n_decoder.step()
        if counter_iter % 50 == 0:
            print("SVGD time per iter:%s" % (time_SVGD / (counter_iter + 1)))
            print(
                "Counter:%s Reconstruct loss:%s MSE:%s mean_var1:%s mean_var2:%s"
                % (
                    counter_iter,
                    loss.cpu().data.numpy(),
                    loss_MSE.cpu().data.numpy(),
                    latent_samples.var(-2).mean().cpu().data.numpy(),
                    latent_samples.view(-1, latent).var(0).mean().cpu().data.numpy(),
                )
            )

        counter_iter += 1
# final image geneartion
VAE_noise.eval()
with torch.no_grad():
    # Generation
    sample = VAE_noise._generate(n_sample=100)  # N X dim
    save_image(sample, path + "%s_lat%s_ep%s.png" % (method, latent, ep), nrow=10)

    # Reconstruct
    recons = VAE_noise._reconstruct(data, z_sample=1, z=None)
    recons = recons.squeeze()
    save_image(data, path + "%s_lat%s_ep%s_orig.png" % (method, latent, ep), nrow=10)
    save_image(recons, path + "%s_lat%s_ep%s_recon.png" % (method, latent, ep), nrow=10)
    # Save model
    check_point = {
        "state_dict": VAE_noise.state_dict(),
        "Adam_encoder": Adam_VAE_n_encoder.state_dict(),
        "Adam_decoder": Adam_VAE_n_decoder.state_dict(),
        "loss": loss,
        "A": A,
        "encoder_setting": encoder_setting,
        "decoder_setting": decoder_setting,
        "SVGD_step": SVGD_step,
        "encoder_update": encoder_update,
    }
    torch.save(
        check_point, path + "/model_save/" + "%s_lat%s_CP_ep%s" % (method, latent, ep)
    )
