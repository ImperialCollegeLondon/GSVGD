## covertype
epochs=10000
batch=300
nparticles=200
data=covertype_sub

for seed in 0 1 2 3 4 5 6 7 8 9
  do
    taskset -c 21-22 python experiments/blr.py --epochs=$epochs --gpu=0 --batch=$batch --nparticles=$nparticles --lr=0.1 --seed=$seed \
    --suffix=$suffix --data=$data &
    taskset -c 23-24 python experiments/blr.py --epochs=$epochs --gpu=0 --batch=$batch --nparticles=$nparticles --method=s-svgd --lr=0.1 --lr_g=0.1 \
    --seed=$seed --suffix=$suffix --data=$data &

    taskset -c 25-26 python experiments/blr.py --epochs=$epochs --gpu=1 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=55 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
    wait

    taskset -c 5-10 python experiments/blr.py --epochs=$epochs --gpu=0 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=1 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 16-20 python experiments/blr.py --epochs=$epochs --gpu=1 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=2 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=2 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=5 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=10 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
    wait 

    taskset -c 5-10 python experiments/blr.py --epochs=$epochs --gpu=0 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=20 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 16-20 python experiments/blr.py --epochs=$epochs --gpu=1 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=30 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=2 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=40 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data &
    taskset -c 21-25 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=50 \
    --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data

  done

## reference HMC
subsample_size=1000 # 0 for HMC; >0 for HMCECS
for seed in 0 1 2 3 4 5 6 7 8 9
  do
    # HMC
    CUDA_VISIBLE_DEVICES=1 taskset -c 6-10 python experiments/blr.py --gpu=-1 --nparticles=$nparticles --epochs=$epochs --lr=0.1 --delta=0.1 \
    --method=hmc --suffix=$suffix --data=$data --seed=$seed --subsample_size=0 
  done

## generate plots
python plots/plot_blr_dist.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
python plots/plot_blr_cov.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1

