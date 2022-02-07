## generate plots for the 3 synthetic examples and ablation studies

## multivariate gaussian, multi-modal, x-shaped
for exp in gaussian multimodal xshaped
    do
        taskset -c 11-16 python plots/plot_final_particles.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1
        taskset -c 11-16 python plots/plot_metric_vs_epochs.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1
        taskset -c 11-16 python plots/plot_seeds.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1 --metric=var
        taskset -c 11-16 python plots/plot_seeds.py --exp=$exp --nparticles=500 --epochs=2000 --lr=0.1 --delta=0.1 --metric=energy
    done

## ablation study on number of projectors M
taskset -c 11-16 python plots/plot_ablation.py --exp=gaussian_ablation --epochs=2000 --lr=0.1 --delta=0.1

## generate plot with different num of particles
taskset -c 11-16 python plots/plot_nparticles.py --exp=gaussian_nparticles --epochs=2000 --lr=0.1 --delta=0.1

## time complexity
taskset -c 11-16 python plots/plot_time.py --exp=gaussian_nparticles --epochs=2000 --lr=0.1 --delta=0.1
