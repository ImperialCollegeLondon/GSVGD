## uncomment the following to generate plots
# for exp in gaussian multimodal xshaped
#     do
#         # taskset -c 11-16 python plots/plot_final_particles.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1
#         taskset -c 11-16 python plots/plot_metric_vs_epochs.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1
#         # taskset -c 11-16 python plots/plot_seeds.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1 --root=res_1010_final
#     done

## generate plot with different num of particles
for exp in gaussian_nparticles
    do
        taskset -c 11-16 python plots/plot_time.py --exp=$exp --epochs=2000 --lr=0.1 --delta=0.1 --root=res_1010_final
    done