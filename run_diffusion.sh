# # run SVGD with different lr
# for lr in 0.1 0.01 0.001
#   do
#   taskset -c 6-10 python experiments/diffusion.py --epochs=1000 --gpu=3 --nparticles=500 --lr=$lr --save_every=100 
#   done

# # run S-SVGD for different lr and lr_g
# for lr in 0.1 0.01 0.001
#     do
#     for lr_g in 1 0.1 0.01 0.001
#         do
#         taskset -c 9-10 python experiments/diffusion.py --epochs=1000 --gpu=4 --dim=100 --nparticles=500 --lr=$lr --method=s-svgd --lr_g=$lr_g --seed=0 --suffix=_ssvgd
#         done
#     done

# # run GSVGD with different lr, delta and effdim
epochs=1000
delta=0.1
lr_svgd=0.01
lr_ssvgd=0.01
lr_gsvgd=0.1

## methods
# for seed in 5 6 7 8 # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do
#     ## 500 particles
#     # taskset -c 7-8 python experiments/diffusion.py --epochs=$epochs --gpu=4 --dim=100 --nparticles=500 --lr=$lr --save_every=200 --seed=$seed &
#     # taskset -c 9-10 python experiments/diffusion.py --epochs=$epochs --gpu=4 --dim=100 --nparticles=500 --lr=$lr --method=s-svgd --lr_g=0.1 --save_every=200 --seed=$seed
#     # for effdim in 400 # 1 2 3 4 5 6 7 8 9 10 15 20 25 30
#     #   do
#     #     taskset -c 1-4 python experiments/diffusion.py --epochs=$epochs --gpu=2 --dim=100 --nparticles=500 --lr=$lr --method=gsvgd --delta=$delta --effdim=$effdim --save_every=200 --seed=$seed
#     #     wait
#     #     echo lr=$lr
#     #   done

#     # 200 particles
#     taskset -c 17-18 python experiments/diffusion.py --epochs=$epochs --gpu=2 --dim=100 --nparticles=200 --lr=$lr_svgd \
#     --seed=$seed &
#     taskset -c 19-20 python experiments/diffusion.py --epochs=$epochs --gpu=2 --dim=100 --nparticles=200 --lr=$lr_ssvgd --method=s-svgd --lr_g=0.1 \
#     --seed=$seed
#     for effdim in 1 2 5 10 20 30 40 50 60 70 80 90 100
#       do
#         taskset -c 21-24 python experiments/diffusion.py --epochs=$epochs --gpu=3 --dim=100 --nparticles=200 --lr=$lr_gsvgd --method=gsvgd --delta=$delta \
#         --effdim=$effdim --seed=$seed
#         echo lr=$lr
#       done
#   done

# # HMC
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do 
#     # taskset -c 7-17 python experiments/diffusion.py --epochs=$epochs --gpu=5 --dim=100 --nparticles=500 --seed=$seed --method=hmc
#     cp res_1010_final/diffusion/rbf_epoch1000_lr0.001_delta0.1_n500_dim100/seed$seed/particles_hmc.p res/diffusion/rbf_epoch1000_lr0.1_delta0.1_n200_dim100/seed$seed/
#   done

## copy results with 0.01 for S-SVGD
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do 
#     from1=res_1010_final/diffusion_n200_full/rbf_epoch1000_lr0.1_delta0.1_n200_dim100/seed$seed
#     from2=res/diffusion/rbf_epoch1000_lr0.01_delta0.1_n200_dim100/seed$seed
#     to1=res/diffusion0.01/rbf_epoch1000_lr0.1_delta0.1_n200_dim100/seed$seed
#     to2=res/diffusion0.01/rbf_epoch1000_lr0.01_delta0.1_n200_dim100/seed$seed

#     mkdir $to1
#     mkdir $to2
#     cp $from1/target_dist.p $to1
#     cp $from1/particles_hmc.p $to1
#     cp $from1/particles_svgd.p $to1
#     cp $from2/particles_s-svgd* $to2
#     cp $from1/particles_gsvgd* $to1
#   done

## plot results
lr=0.1
# python plots/plot_diffusion.py --exp=diffusion_n200_full --epochs=$epochs --nparticles=200 --lr=$lr --delta=$delta --dim=100 --lr_g=0.1 --root=res_1010_final
python plots/plot_diffusion.py --exp=diffusion0.01 --epochs=$epochs --nparticles=200 --lr=$lr --delta=$delta --dim=100 --lr_g=0.1
# python plots/plot_diffusion_dims.py --exp=diffusion0.01 --epochs=$epochs --nparticles=200 --lr=$lr --delta=$delta --dim=100 --lr_g=0.1

# # plot results for different hyperparams
# for lr in 0.1 0.01 0.001
#   do
#     for lr_g in 1 0.1 0.01 0.001
#       do
#       # taskset -c 6-10 python plots/plot_diffusion.py --exp=diffusion --epochs=1000 --nparticles=500 --lr=$lr --delta=0.01 --dim=100 --lr_g=0.1
#       python plots/plot_diffusion.py --root=res_1010_final --exp=diffusion_ssvgd --epochs=1000 --nparticles=500 --lr=$lr --delta=0.1 --lr_g=$lr_g
#       done
#   done

# ## copy full100 results
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do 
#     cp res/diffusion_n200_full2/rbf_epoch1000_lr0.1_delta0.1_n200_dim100/seed$seed/particles_gsvgd* res/diffusion_n200_full100/rbf_epoch1000_lr0.1_delta0.1_n200_dim100/seed$seed/
#   done
