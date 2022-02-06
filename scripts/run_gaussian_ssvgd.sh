for seed in 0 1 2 # 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
        taskset -c 11-15 python experiments/gaussian_ablation.py --epoch=800 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=0 --dim=50 --seed=$seed \
        --suffix=_ablation --method=S-SVGD
        echo Finished seed=$seed
    done