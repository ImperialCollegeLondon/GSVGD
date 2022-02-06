## run GSVGD with different lr, delta and effdim
epochs=200 # 1000
delta=0.1
lr_svgd=00.01
lr_ssvgd=0.01
lr_gsvgd=0.1

## run SVGD, S-SVGD and GSVGD
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
for seed in 0 1 2
  do
    ## 500 particles
    # taskset -c 7-8 python experiments/diffusion.py --epochs=$epochs --gpu=4 --dim=100 --nparticles=500 --lr=$lr --save_every=200 --seed=$seed &
    # taskset -c 9-10 python experiments/diffusion.py --epochs=$epochs --gpu=4 --dim=100 --nparticles=500 --lr=$lr --method=s-svgd --lr_g=0.1 --save_every=200 --seed=$seed
    # for effdim in 400 # 1 2 3 4 5 6 7 8 9 10 15 20 25 30
    #   do
    #     taskset -c 1-4 python experiments/diffusion.py --epochs=$epochs --gpu=2 --dim=100 --nparticles=500 --lr=$lr --method=gsvgd --delta=$delta --effdim=$effdim --save_every=200 --seed=$seed
    #     wait
    #     echo lr=$lr
    #   done

    # 200 particles
    taskset -c 16-17 python experiments/diffusion.py --epochs=$epochs --save_every=25 --gpu=3 --dim=100 --nparticles=200 --lr=$lr_svgd \
    --seed=$seed &
    taskset -c 18-29 python experiments/diffusion.py --epochs=$epochs --save_every=25 --gpu=3 --dim=100 --nparticles=200 --lr=$lr_ssvgd --method=s-svgd --lr_g=0.1 \
    --seed=$seed
    for effdim in 1 30 50 70 100
      do
        taskset -c 20-21 python experiments/diffusion.py --epochs=$epochs --save_every=25 --gpu=3 --dim=100 --nparticles=200 --lr=$lr_gsvgd --method=gsvgd --delta=$delta \
        --effdim=$effdim --seed=$seed
        echo lr=$lr
      done
  done

## HMC
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do 
#     # taskset -c 7-17 python experiments/diffusion.py --epochs=$epochs --gpu=5 --dim=100 --nparticles=500 --seed=$seed --method=hmc
#     cp res_1010_final/diffusion/rbf_epoch1000_lr0.001_delta0.1_n500_dim100/seed$seed/particles_hmc.p res/diffusion/rbf_epoch1000_lr0.1_delta0.1_n200_dim100/seed$seed/
#   done

## plot results
# lr=0.1
# python plots/plot_diffusion.py --exp=diffusion --epochs=$epochs --nparticles=200 --lr=$lr --delta=$delta --dim=100 --lr_g=0.1
# python plots/plot_diffusion.py --exp=diffusion --epochs=$epochs --nparticles=200 --lr=$lr --delta=$delta --dim=100 --lr_g=0.1
# python plots/plot_diffusion_dims.py --exp=diffusion --epochs=$epochs --nparticles=200 --lr=$lr --delta=$delta --dim=100 --lr_g=0.1

