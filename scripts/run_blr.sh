## covertype
epochs=10000
batch=300
nparticles=200
data=covertype_sub

for seed in 0 1 2 # 0 1 2 3 4 5 6 7 8 9
  do
    taskset -c 21-22 python experiments/blr.py --epochs=$epochs --gpu=4 --batch=$batch --nparticles=$nparticles --lr=0.1 --seed=$seed \
    --suffix=$suffix --data=$data &
    taskset -c 23-24 python experiments/blr.py --epochs=$epochs --gpu=5 --batch=$batch --nparticles=$nparticles --method=s-svgd --lr=0.1 --lr_g=0.1 \
    --seed=$seed --suffix=$suffix --data=$data &

    taskset -c 25-26 python experiments/blr.py --epochs=$epochs --gpu=6 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=55 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
    wait

    taskset -c 21-22 python experiments/blr.py --epochs=$epochs --gpu=4 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=1 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 23-24 python experiments/blr.py --epochs=$epochs --gpu=5 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=10 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 25-26 python experiments/blr.py --epochs=$epochs --gpu=6 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=20 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 27-28 python experiments/blr.py --epochs=$epochs --gpu=7 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=40 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
    wait 

    # taskset -c 5-10 python experiments/blr.py --epochs=$epochs --gpu=4 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=1 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    # taskset -c 16-20 python experiments/blr.py --epochs=$epochs --gpu=5 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=2 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    # taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=6 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=5 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    # taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=6 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=10 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
    # wait 

    # taskset -c 5-10 python experiments/blr.py --epochs=$epochs --gpu=4 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=20 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    # taskset -c 16-20 python experiments/blr.py --epochs=$epochs --gpu=7 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=30 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    # taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=5 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=40 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    # taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=6 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=50 \
    # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data

    # ## keep commented!
    # # taskset -c 5-10 python experiments/blr.py --epochs=$epochs --gpu=5 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=1 \
    # # --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    # # for effdim in 2 5 10 20 30 40 50 55 # 1 2 5 10 20 30 40 50 55
    # #   do
    # #     taskset -c 5-10 python experiments/blr.py --epochs=$epochs --gpu=7 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=$effdim \
    # #     --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
    # #   done
  done

# for seed in 0 1 2 3 4 # 5 6 7 8 9
#   do
#     # cp res/blr_hmcecs_thin_covertype/rbf_epoch10_lr0.1_delta0.1_n200/seed$seed/particles_hmc.p res/blr$suffix/rbf_epoch$epochs\_lr0.1_delta0.1_n200/seed$seed/
#     # cp res/blr_covertype_sub/rbf_epoch10000_lr0.1_delta0.1_n200/seed$seed/particles_hmc.p res/blr$suffix/rbf_epoch$epochs\_lr0.1_delta0.1_n200/seed$seed/
#     # cp res/blr_hmcecs_thin_covertype_sub_10000/rbf_epoch10_lr0.1_delta0.1_n200/seed$seed/particles_hmc.p res/blr$suffix/rbf_epoch$epochs\_lr0.1_delta0.1_n$nparticles/seed$seed/ #TODO delete
#     cp res/blr_covertype_sub/rbf_epoch10000_lr0.1_delta0.1_n200/seed$seed/particles_hmc.p res/blr$suffix/rbf_epoch$epochs\_lr0.1_delta0.1_n$nparticles/seed$seed/ #TODO delete
#   done

## run the following to generate plots
# CUDA_VISIBLE_DEVICES=3,6,2,0,1,5,4,7 python plots/plot_blr_marginals.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
# CUDA_VISIBLE_DEVICES=3,6,2,0,1,5,4,7 python plots/plot_blr_dist.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
# CUDA_VISIBLE_DEVICES=3,6,2,0,1,5,4,7 python plots/plot_blr_cov.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
# CUDA_VISIBLE_DEVICES=3,6,2,0,1,5,4,7 python plots/plot_blr.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1

