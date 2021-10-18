# for dataset in boston_housing energy concrete kin8nm naval protein wine yacht
#     do
#     for seed in 1 2 3 4 5 6 7 8 9 10
#         do
#             for method in MaxSVGD
#             do
#                 python experiments/bnn.py --seed=$seed --method=$method --dataset=$dataset --lr=0.001 --nparticles=50 --gpu=4
#             done
#         done
#         wait
#         echo Finished dataset=$dataset seed=$seed
#     done

# for dataset in boston_housing concrete energy # kin8nm naval protein wine yacht
#     do
#         for seed in 1 2 3 4 5 6 7 8 9 10
#             do
#                 python experiments/bnn.py --seed=$seed --method=S-SVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=6 --suffix=_tune2
#                 # python experiments/bnn.py --seed=$seed --method=S-SVGD --dataset=concrete --lr=1e-3 --nparticles=50 --gpu=4
#                 # python experiments/bnn.py --seed=$seed --method=S-SVGD --dataset=energy --lr=1e-3 --nparticles=50 --gpu=4
#                 # python experiments/bnn.py --seed=$seed --method=S-SVGD --dataset=debug --lr=1e-3 --nparticles=50 --gpu=4
#             done
#     done

for dataset in yacht # boston_housing concrete energy combined kin8nm naval protein wine yacht
  do
    python experiments/bnn.py --seed=1 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=0 --suffix=_tune &
    # python experiments/bnn.py --seed=2 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=3 --suffix=_tune &
    python experiments/bnn.py --seed=3 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=2 --suffix=_tune &
    python experiments/bnn.py --seed=4 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=6 --suffix=_tune &
    python experiments/bnn.py --seed=5 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=7 --suffix=_tune
    wait
    python experiments/bnn.py --seed=6 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=0 --suffix=_tune &
    # python experiments/bnn.py --seed=7 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=3 --suffix=_tune &
    python experiments/bnn.py --seed=8 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=2 --suffix=_tune &
    python experiments/bnn.py --seed=9 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=6 --suffix=_tune &
    python experiments/bnn.py --seed=10 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=7 --suffix=_tune
    wait
    python experiments/bnn.py --seed=2 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=0 --suffix=_tune &
    python experiments/bnn.py --seed=7 --method=S-SVGD --dataset=$dataset --lr=1e-2 --nparticles=50 --gpu=2 --suffix=_tune
  done
