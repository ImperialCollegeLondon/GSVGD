## covertype
# epochs=10000
epochs=10
batch=300
nparticles=200
data=covertype_sub
# data=covertype
subsample_size=1000 # 0 for HMC
suffix_hmcecs=_hmcecs_$data
suffix_hmc=_hmc_$data

# run reference HMC
# for seed in 0 1 2 3 4 # 1 2 3 4 # 0 1 2 3 4 5 6 7 8 9
#   do
#     # HMCECS
#     CUDA_VISIBLE_DEVICES=7 taskset -c 6-10 python experiments/blr.py --gpu=-1 --nparticles=$nparticles --epochs=$epochs --lr=0.1 --delta=0.1 \
#     --method=hmc --suffix=_hmcecs_thin_$data\_10000 --data=$data --seed=$seed --subsample_size=$subsample_size
#   done

for seed in 0 1 2 3 4 # 5 6 7 8 9
  do
    # HMC
    CUDA_VISIBLE_DEVICES=1 taskset -c 6-10 python experiments/blr.py --gpu=-1 --nparticles=$nparticles --epochs=$epochs --lr=0.1 --delta=0.1 \
    --method=hmc --suffix=$suffix_hmc --data=$data --seed=$seed --subsample_size=0 
  done

# for seed in 1 2 3 4 5 6 7 8 9
#   do
#     # HMC with thinning
#     CUDA_VISIBLE_DEVICES=2,3 taskset -c 6-10 python experiments/blr.py --gpu=-1 --nparticles=$nparticles --epochs=$epochs --lr=0.1 --delta=0.1 \
#     --method=hmc --suffix=_hmc_thin_$data --data=$data --seed=$seed --subsample_size=0 
#   done

# for seed in 0 1 2 3 4 # 5 6 7 8 9
#   do
#     cp res/blr_hmcecs_thin_covertype/rbf_epoch10_lr0.1_delta0.1_n200/seed$seed/particles_hmc.p res/blr_batch2000_covertype/rbf_epoch50_lr0.1_delta0.1_n200/seed$seed/
#   done