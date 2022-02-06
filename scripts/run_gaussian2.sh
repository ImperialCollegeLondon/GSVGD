for seed in 0 1 2 # 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
      for M in 1 2 5 10 15 20 25
        do
          taskset -c 16-20 python experiments/gaussian_ablation.py --epoch=800 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=2 --dim=50 --seed=$seed \
          --suffix=_ablation --method=GSVGD --effdim=2 --m=$M
          echo Finished seed=$seed M=$M
        done
    done