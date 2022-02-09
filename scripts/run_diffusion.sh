## conditioned diffucion
epochs=1000
nparticles=200
delta=0.1
lr_svgd=0.1
lr_ssvgd=0.1
lr_gsvgd=0.1

## run SVGD, S-SVGD and GSVGD
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  do
    taskset -c 16-17 python experiments/diffusion.py --epochs=$epochs --save_every=25 --gpu=0 --dim=100 --nparticles=$nparticles --lr=$lr_svgd \
    --seed=$seed &
    taskset -c 18-29 python experiments/diffusion.py --epochs=$epochs --save_every=25 --gpu=0 --dim=100 --nparticles=$nparticles --lr=$lr_ssvgd --method=s-svgd --lr_g=0.1 \
    --seed=$seed
    for effdim in 1 2 3 4 5 6 7 8 9 10 15 20 25 30
      do
        taskset -c 20-21 python experiments/diffusion.py --epochs=$epochs --save_every=25 --gpu=1 --dim=100 --nparticles=$nparticles --lr=$lr_gsvgd --method=gsvgd --delta=$delta \
        --effdim=$effdim --seed=$seed
        echo lr=$lr
      done
  done

## HMC
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
  do 
    taskset -c 7-17 python experiments/diffusion.py --epochs=$epochs --gpu=-1 --dim=100 --nparticles=$nparticles --seed=$seed --method=hmc
  done

## plot results
python plots/plot_diffusion.py --exp=diffusion --epochs=$epochs --nparticles=$nparticles --lr_svgd=$lr_svgd  --lr_ssvgd=$lr_ssvgd\
 --lr_gsvgd=$lr_gsvgd --delta=$delta --dim=100 --lr_g=0.1
python plots/plot_diffusion_dims.py --exp=diffusion --epochs=$epochs --nparticles=$nparticles  --lr_svgd=$lr_svgd  --lr_ssvgd=$lr_ssvgd\
 --lr_gsvgd=$lr_gsvgd --delta=$delta --dim=100 --lr_g=0.1