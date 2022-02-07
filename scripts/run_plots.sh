## generate plots for the ablation studies

## ablation study on number of projectors M
taskset -c 11-16 python plots/plot_ablation.py --exp=gaussian_ablation --epochs=2000 --lr=0.1 --delta=0.1

## generate plot with different num of particles
taskset -c 11-16 python plots/plot_nparticles.py --exp=gaussian_nparticles --epochs=2000 --lr=0.1 --delta=0.1

## time complexity
taskset -c 11-16 python plots/plot_time.py --exp=gaussian_nparticles --epochs=2000 --lr=0.1 --delta=0.1
