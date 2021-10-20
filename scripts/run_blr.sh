## covertype
epochs=10000
batch=300
nparticles=200
data=covertype_sub
suffix=_$data
for seed in 0 1 2 3 4 5 6 7 8 9
  do
    taskset -c 11-15 python experiments/blr.py --epochs=$epochs --gpu=2 --batch=$batch --nparticles=$nparticles --lr=0.1 --seed=$seed \
    --suffix=$suffix --data=$data &
    taskset -c 1-5 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --method=s-svgd --lr=0.01 --lr_g=0.01 \
    --seed=$seed --suffix=$suffix --data=$data &
    for effdim in 1 2 5 10 20 30 40 50
      do
        taskset -c 11-15 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=$effdim \
        --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
      done
  done

## run the following to generate plots
# python plots/plot_blr_marginals.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
# python plots/plot_blr_dist.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
# python plots/plot_blr_cov.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1

