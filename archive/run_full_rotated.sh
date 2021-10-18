for seed in 0 1 2 3 4
    do
    for dim in 10 20 30 50 70 90 100
        do
            taskset -c 11-15 python experiments/full_multimodal_rotated.py --epoch=1000 --lr=0.01 --delta=0.01 --nparticles=500 --gpu=5 --dim=$dim --seed=$seed --nmix=1
            wait
            echo Finished dim=$dim seed=$seed
        done
    done

# for lr in 0.1 0.01 0.001 0.0001 0.00001
#     do
#     for delta in 0.1 0.01 0.001
#         do
#             # taskset -c 11-15 python experiments/full_multimodal.py --epoch=2000 --lr=$lr --delta=$delta --nparticles=500 --gpu=1 --dim=50 --seed=0 --nmix=3
#             taskset -c 11-15 python experiments/full_multimodal_shifted.py --epoch=2000 --lr=$lr --delta=$delta --nparticles=500 --gpu=3 --dim=50 --seed=0 --nmix=5
#             # taskset -c 10-12 python plots/plot_metric_vs_epochs.py --exp=full_multimodal_seq --nparticles=500 --epochs=2000 --lr=$lr --delta=$delta --metric=energy --ylab=Energy
#         done
#     echo Finished lr=$lr delta=$delta
#     done
