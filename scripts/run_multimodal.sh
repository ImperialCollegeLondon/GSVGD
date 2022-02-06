# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
for seed in 0 1 2 #3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
      for dim in 5 30 50 70 100
        do
          taskset -c 6-10 python experiments/multimodal.py --epoch=800 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=1 --dim=$dim --seed=$seed \
          --method=all --save_every=50 --lr_g=0.1 --delta=0.1 --effdim=-1
          wait
          echo Finished dim=$dim seed=$seed
        done
    done