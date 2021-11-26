for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
    for dim in 10 20 30 40 50 60 70 80 90 100
        do
            taskset -c 1-5 python experiments/multimodal.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=2 --dim=$dim --seed=$seed \
            --save_every=50
            wait
            echo Finished dim=$dim seed=$seed
        done
    done
