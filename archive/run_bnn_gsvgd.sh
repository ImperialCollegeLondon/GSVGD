# for lr in 0.1 0.01 0.001
#     do
#         for delta in 0.1 0.01 0.001
#         do
#             taskset -c 6-10 python experiments/bnn_sequential.py --nparticles=20 --dataset=energy --m=20 --M=2 --method=GSVGD --lr=$lr --delta=$delta --gpu=0
#         done
#     done
#     wait
#     echo Finished dataset=energy lr=$lr delta=$delta
# done

# for dataset in boston_housing # concrete energy # kin8nm naval protein wine yacht
#     do
#         for seed in 1 2 3 4 5 6 7 8 9 10
#             do
#                 # python experiments/bnn.py --seed=$seed --method=GSVGD --dataset=concrete --lr=1e-3 --nparticles=50 --gpu=1 --m=10 --M=5 --delta=1e-5 --suffix=_tune2 &
#                 # python experiments/bnn.py --seed=$seed --method=GSVGD --dataset=energy --lr=1e-3 --nparticles=50 --gpu=2 --m=10 --M=5 --delta=1e-5 --suffix=_tune2 &
#                 python experiments/bnn.py --seed=$seed --method=GSVGD --dataset=kin8nm --lr=1e-3 --nparticles=50 --gpu=0 --m=10 --M=5 --delta=1e-5 --suffix=_tune2 &
#                 python experiments/bnn.py --seed=$seed --method=GSVGD --dataset=naval --lr=1e-3 --nparticles=50 --gpu=1 --m=10 --M=5 --delta=1e-5 --suffix=_tune2 &
#                 python experiments/bnn.py --seed=$seed --method=GSVGD --dataset=protein --lr=1e-3 --nparticles=50 --gpu=2 --m=10 --M=5 --delta=1e-5 --suffix=_tune2
#                 wait
#             done
#     done

# wait
# for dataset in boston_housing # concrete energy # kin8nm naval protein wine yacht
#     do
#         for seed in 1 2 3 4 5 6 7 8 9 10
#             do
#                 python experiments/bnn.py --seed=$seed --method=GSVGD --dataset=wine --lr=1e-3 --nparticles=50 --gpu=0 --m=10 --M=5 --delta=1e-5 --suffix=_tune2 &
#                 python experiments/bnn.py --seed=$seed --method=GSVGD --dataset=yacht --lr=1e-3 --nparticles=50 --gpu=1 --m=10 --M=5 --delta=1e-5 --suffix=_tune2
#                 wait
#             done
#     done

for dataset in yacht # boston_housing concrete energy kin8nm naval combined protein wine yacht
    do
        python experiments/bnn.py --seed=1 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=0 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        # python experiments/bnn.py --seed=2 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=1 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        python experiments/bnn.py --seed=3 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=2 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        python experiments/bnn.py --seed=4 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=6 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        python experiments/bnn.py --seed=5 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=7 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam 
        wait 
        python experiments/bnn.py --seed=6 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=0 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        # python experiments/bnn.py --seed=7 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=1 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        python experiments/bnn.py --seed=8 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=2 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        python experiments/bnn.py --seed=9 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=6 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        python experiments/bnn.py --seed=10 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=7 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam
        wait
        python experiments/bnn.py --seed=2 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=1 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam &
        python experiments/bnn.py --seed=7 --method=GSVGD --dataset=$dataset --lr=1e-3 --nparticles=50 --gpu=1 --m=10 --M=5 --delta=1e-5 --suffix=_tune_adam
    done
