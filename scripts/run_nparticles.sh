## run ablation study on number of particles
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
      for dim in 10 20 30 40 50 60 70 80 90 100
        do
          for nparticles in 50 100 500 800
            do
              # GSVGD with no more than 20 projectors
              taskset -c 1-5 python experiments/gaussian.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=$nparticles --gpu=0 --dim=$dim --seed=$seed \
              --method=all --save_every=50 --lr_g=0.1 --delta=0.1 --effdim=-1 --suffix=_nparticles

              # GSVGD-Full
              taskset -c 1-5 python experiments/gaussian.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=$nparticles --gpu=0 --dim=$dim --seed=$seed \
              --method=all --save_every=50 --lr_g=0.1 --delta=0.1 --effdim=-1 --suffix=_nparticles_full --m=-1
              wait
              echo Finished dim=$dim seed=$seed
            done
        done
    done

## generate plot with different num of particles
taskset -c 11-16 python plots/plot_nparticles.py --exp=gaussian_nparticles --epochs=2000 --lr=0.1 --delta=0.1

## time complexity
taskset -c 11-16 python plots/plot_time.py --exp=gaussian_nparticles --epochs=2000 --lr=0.1 --delta=0.1