for seed in 19 # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
    for dim in 30 # 10 20 30 40 50 60 70 80 90 100
        do
            taskset -c 11-15 python experiments/gaussian.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=4 --dim=$dim --seed=$seed \
            --save_every=50
            echo Finished dim=$dim seed=$seed
        done
    done

