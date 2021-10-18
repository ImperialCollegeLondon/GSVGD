# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
#     do
#     for dim in 10 20 30 40 50 60 70 80 90 100
#         do
#             taskset -c 6-10 python experiments/xshaped.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=3 --dim=$dim --seed=$seed \
#             --save_every=50
#             echo Finished dim=$dim seed=$seed
#         done
#     done

## nparticles
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
#     do
#     for dim in 10 20 30 40 60 70 80 90 100
#         do
#             taskset -c 11-15 python experiments/xshaped.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=50 --gpu=0 --dim=$dim --seed=$seed --suffix=_nparticles
#             taskset -c 11-15 python experiments/xshaped.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=100 --gpu=1 --dim=$dim --seed=$seed --suffix=_nparticles
#             taskset -c 11-15 python experiments/xshaped.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=500 --gpu=2 --dim=$dim --seed=$seed --suffix=_nparticles
#             taskset -c 11-15 python experiments/xshaped.py --epoch=2000 --lr=0.1 --delta=0.1 --nparticles=800 --gpu=3 --dim=$dim --seed=$seed --suffix=_nparticles
#             echo Finished nparticles=$nparticles seed=$seed
#         done
#     done

## plot
# for exp in xshaped_batch_full gaussian_full multimodal_batch_nmix3_full multimodal_batch_nmix4_full
# for exp in multimodal_batch_nmix4_full gaussian_full # xshaped_batch_full multimodal_batch_nmix4_full
#     do
#         # taskset -c 11-16 python plots/plot_final_particles.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1
#         taskset -c 11-16 python plots/plot_metric_vs_epochs.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1 --root=res_1010_final
#         # taskset -c 11-16 python plots/plots/plot_seeds.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1 --root=res_1010_final
#     done

# plot nparticles
for exp in gaussian_nparticles
    do
        taskset -c 11-16 python plots/plot_nparticles.py --exp=$exp --epochs=2000 --lr=0.1 --delta=0.1 --root=res_1010_final
        taskset -c 11-16 python plots/plot_time.py --exp=$exp --epochs=2000 --lr=0.1 --delta=0.1 --root=res_1010_final
    done