for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
      for dim in 10 20 30 40 50 60 70 80 90 100
        do
          taskset -c 6-10 python experiments/multimodal.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=1 --dim=$dim --seed=$seed \
          --method=all --save_every=50 --lr_g=0.1 --delta=0.1 --effdim=-1
          echo Finished dim=$dim seed=$seed
        done
    done

## generate plots
taskset -c 11-16 python plots/plot_final_particles.py --exp=multimodal --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1
taskset -c 11-16 python plots/plot_metric_vs_epochs.py --exp=multimodal --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1
taskset -c 11-16 python plots/plot_seeds.py --exp=multimodal --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1 --metric=energy