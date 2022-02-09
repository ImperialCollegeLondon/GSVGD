## ablation study on number of projectors M
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
      # SVGD
      taskset -c 11-15 python experiments/gaussian_ablation.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=0 --dim=50 --seed=$seed \
      --suffix=_ablation --method=SVGD
      echo Finished seed=$seed 

      # S-SVGD
      taskset -c 11-15 python experiments/gaussian_ablation.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=0 --dim=50 --seed=$seed \
      --suffix=_ablation --method=S-SVGD
      echo Finished seed=$seed

      # GSVGD1
      for M in 1 2 5 10 20 30 40 50
        do
          taskset -c 11-15 python experiments/gaussian_ablation.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=1 --dim=50 --seed=$seed \
          --suffix=_ablation --method=GSVGD --effdim=1 --m=$M
          echo Finished seed=$seed M=$M
        done

      # GSVGD2
      for M in 1 2 5 10 15 20 25
        do
          taskset -c 16-20 python experiments/gaussian_ablation.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=2 --dim=50 --seed=$seed \
          --suffix=_ablation --method=GSVGD --effdim=2 --m=$M
          echo Finished seed=$seed M=$M
        done

      # GSVGD5
      for M in 1 2 4 5 6 8 10
        do
          taskset -c 21-25 python experiments/gaussian_ablation.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=3 --dim=50 --seed=$seed \
          --suffix=_ablation --method=GSVGD --effdim=5 --m=$M
          echo Finished seed=$seed M=$M
        done

## ablation study on number of projectors M
taskset -c 11-16 python plots/plot_ablation.py --exp=gaussian_ablation --epochs=2000 --lr=0.1 --delta=0.1