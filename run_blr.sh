## MNIST
# epochs=20
# batch=300
# nparticles=20
# d1=3
# d2=5
# suffix=_$d1$d2
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do
#     taskset -c 6-10 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --lr=0.1 --seed=$seed \
#     --suffix=$suffix --d1=$d1 --d2=$d2
#     # taskset -c 1-5 python experiments/blr.py --epochs=$epochs --gpu=4 --batch=$batch --nparticles=$nparticles --method=s-svgd --lr=0.1 --lr_g=0.1 \
#     # --seed=$seed --suffix=$suffix --d1=$d1 --d2=$d2 &
#     for effdim in 1 2 5 10 20 30 40 50 60 70 80 90 100
#       do
#         taskset -c 11-15 python experiments/blr.py --epochs=$epochs --gpu=5 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=$effdim \
#         --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --d1=$d1 --d2=$d2
#       done
#   done

# # covertype
epochs=10000
batch=300
nparticles=200
data=covertype_sub
suffix=_$data
# for seed in 0 1 2 3 4 5 6 7 8 9 # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do
#     # taskset -c 11-15 python experiments/blr.py --epochs=$epochs --gpu=2 --batch=$batch --nparticles=$nparticles --lr=0.1 --seed=$seed \
#     # --suffix=$suffix --data=$data &
#     taskset -c 1-5 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --method=s-svgd --lr=0.01 --lr_g=0.01 \
#     --seed=$seed --suffix=$suffix --data=$data
#     # for effdim in 1 2 5 10 20 30 40 50
#     #   do
#     #     taskset -c 11-15 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=$effdim \
#     #     --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
#     #   done
#   done

# python plots/plot_blr_marginals.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
# python plots/plot_blr_dist.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1
python plots/plot_blr_cov.py --exp=blr$suffix --epochs=$epochs --nparticles=$nparticles --lr=0.1 --delta=0.1 --lr_g=0.1


# # sonar
# epochs=2000
# batch=300
# nparticles=50
# data=sonar
# suffix=_$data
# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#   do
#     # taskset -c 6-10 python experiments/blr.py --epochs=$epochs --gpu=3 --batch=$batch --nparticles=$nparticles --lr=0.1 --seed=$seed \
#     # --suffix=$suffix --data=$data
#     taskset -c 1-5 python experiments/blr.py --epochs=$epochs --gpu=4 --batch=$batch --nparticles=$nparticles --method=s-svgd --lr=0.001 --lr_g=0.01 \
#     --seed=$seed --suffix=$suffix --data=$data
#     for effdim in 1 2 5 10 20 30 40 50 60 70 80 90 100
#       do
#         taskset -c 11-15 python experiments/blr.py --epochs=$epochs --gpu=5 --batch=$batch --nparticles=$nparticles --method=gsvgd --effdim=$effdim \
#         --lr=0.1 --delta=0.1 --seed=$seed --suffix=$suffix --data=$data
#       done
#   done

# taskset -c 6-10 python experiments/blr.py --epochs=2000 --gpu=3 --batch=300 --nparticles=20 --lr=1e-3 --suffix=_sonar --data=sonar
# taskset -c 1-5 python experiments/blr.py --epochs=2000 --gpu=3 --batch=300 --nparticles=20 --lr=1e-3 --lr_g=0.1 --method=s-svgd --suffix=_sonar --data=sonar
# taskset -c 11-15 python experiments/blr.py --epochs=2000 --gpu=3 --batch=300 --nparticles=20 --method=gsvgd --effdim=1 \
# --lr=0.1 --delta=0.1 --suffix=_sonar --data=sonar
# lr=1e-3, delta=0.01

# ploting
# python plots/plot_blr.py --exp=blr$suffix --nparticles=20 --epochs=20 --lr=0.1 --delta=0.1

# # tuning
# for lr in 0.1 #0.01 0.001
#   do
#   taskset -c 6-10 python experiments/blr.py --epochs=4 --gpu=0 --batch=200 --nparticles=100 --lr=$lr --save_every=10
#   done

# for lr in 0.1 #0.01 0.001
#   do
#   for lr_g in 0.1 #0.1 0.01 0.001
#     do
#     taskset -c 1-5 python experiments/blr.py --epochs=4 --gpu=0 --batch=200 --nparticles=100 --lr=$lr --lr_g=$lr_g --method=s-svgd --save_every=10 &
#     done
#   wait
#   echo lr=$lr
#   done

# for effdim in 15 10 5 2 1
#   do
#   for lr in 0.1 # 0.1 0.01 0.001 1e-4
#     do
#     for delta in 0.01 # 0.1 0.01 0.001 1e-4
#       do
#       taskset -c 30-39 python experiments/blr.py --epochs=2 --gpu=1 --batch=200 --nparticles=100 --lr=$lr --delta=$delta \
#       --method=gsvgd --effdim=$effdim --save_every=10 &
#       done
#     wait
#     done
#   echo lr=$lr
#   done


# # tuning
# for data in german image ringnorm twonorm waveform breast_cancer flare_solar
# do
#   # for lr in 0.1 0.01 0.001
#   #   do
#   #   taskset -c 6-10 python experiments/blr.py --epochs=20 --gpu=0 --batch=200 --nparticles=100 --lr=$lr \
#   #   --suffix=_$data --save_every=1 --data=$data
#   #   done

#   # for lr in 0.1 0.01 0.001
#   #   do
#   #   for lr_g in 0.1 0.01 0.001
#   #     do
#   #     taskset -c 1-5 python experiments/blr.py --epochs=20 --gpu=0 --batch=200 --nparticles=100 --lr=$lr --lr_g=$lr_g \
#   #     --suffix=_$data --method=s-svgd --save_every=1 --data=$data &
#   #     done
#   #   wait
#   #   echo lr=$lr
#   #   done

#   for effdim in 15 10 5 2 1
#     do
#     for lr in 0.1 0.01 0.001
#       do
#       for delta in 0.1 0.01 0.001
#         do
#         taskset -c 30-39 python experiments/blr.py --epochs=20 --gpu=1 --batch=200 --nparticles=100 --lr=$lr --delta=$delta \
#         --suffix=_$data --method=gsvgd --effdim=$effdim --save_every=1 --data=$data &
#         done
#       wait
#       done
#     echo lr=$lr
#     done
#   echo data=$data
# done

# for data in german image ringnorm twonorm waveform breast_cancer flare_solar
# do
#   python tune_blr.py --exp=blr$data --nparticles=100 --epochs=20 --method=all
# done


# taskset -c 1-5 python experiments/blr.py --epochs=2 --gpu=0 --batch=200 --nparticles=100 --lr=0.01 --lr_g=0.01 --suffix=_batch200 --method=s-svgd

# # plot tuning
# python tune_blr.py --exp=blr --nparticles=100 --epochs=2 --method=all
# python plots/plot_blr.py --exp=blr --nparticles=20 --epochs=150 --lr=0.1 --delta=0.1
# python plots/plot_blr_nparticles.py --exp=blr_nparticles --epochs=4 --lr=0.01 --delta=0.01 --lr_g=0.01


# # nparticles
# for nparticles in 5 10 20 50 100 150 200
#   do
#   taskset -c 1-2 python experiments/blr.py --epochs=3 --gpu=0 --batch=200 --nparticles=$nparticles --lr=0.01 --suffix=_nparticles &
#   taskset -c 3-4 python experiments/blr.py --epochs=3 --gpu=0 --batch=200 --nparticles=$nparticles --lr=0.01 --lr_g=0.01 --suffix=_nparticles --method=s-svgd &

#   taskset -c 11-12 python experiments/blr.py --epochs=3 --gpu=0 --batch=200 --nparticles=$nparticles --lr=0.01 --delta=0.01 --suffix=_nparticles --method=gsvgd --effdim=15 &
#   taskset -c 9-10 python experiments/blr.py --epochs=3 --gpu=0 --batch=200 --nparticles=$nparticles --lr=0.01 --delta=0.01 --suffix=_nparticles --method=gsvgd --effdim=10 &
#   # taskset -c 7-8 python experiments/blr.py --epochs=3 --gpu=1 --batch=200 --nparticles=$nparticles --lr=0.001 --delta=0.01 --suffix=_nparticles --method=gsvgd --effdim=5 &
#   # taskset -c 13-14 python experiments/blr.py --epochs=3 --gpu=1 --batch=200 --nparticles=$nparticles --lr=0.001 --delta=0.01 --suffix=_nparticles --method=gsvgd --effdim=2
#   # taskset -c 15-16 python experiments/blr.py --epochs=3 --gpu=1 --batch=200 --nparticles=$nparticles --lr=0.01 --delta=0.001 --suffix=_nparticles --method=gsvgd --effdim=1 &
#   wait
#   done

# # repeat with the best params
# for seed in 0 1 2 3 4 5 6 7 8 9
#   do
#   taskset -c 15-16 python experiments/blr.py --epochs=3 --gpu=0 --batch=200 --nparticles=100 --lr=0.01 --suffix=_batch200 --seed=$seed &
#   taskset -c 13-14 python experiments/blr.py --epochs=3 --gpu=0 --batch=200 --nparticles=100 --lr=0.01 --lr_g=0.01 --suffix=_batch200 --method=s-svgd --seed=$seed &

#   # taskset -c 7-8 python experiments/blr.py --epochs=3 --gpu=0 --batch=200 --nparticles=100 --lr=0.01 --delta=0.01 --suffix=_batch200 --method=gsvgd --effdim=5 --seed=$seed &
#   taskset -c 9-10 python experiments/blr.py --epochs=3 --gpu=1 --batch=200 --nparticles=100 --lr=0.01 --delta=0.01 --suffix=_batch200 --method=gsvgd --effdim=10 --seed=$seed &
#   taskset -c 11-12 python experiments/blr.py --epochs=3 --gpu=1 --batch=200 --nparticles=100 --lr=0.01 --delta=0.01 --suffix=_batch200 --method=gsvgd --effdim=15 --seed=$seed
#   wait
#   echo "finished seed=$seed"
#   done

# for seed in 0 1 2 3 4 5 6 7 8 9
#   do
#   taskset -c 15-16 python experiments/blr.py --epochs=2 --gpu=0 --batch=200 --nparticles=100 --lr=0.1 --suffix=_batch200 --seed=$seed &
#   taskset -c 13-14 python experiments/blr.py --epochs=2 --gpu=0 --batch=200 --nparticles=100 --lr=0.1 --lr_g=0.01 --suffix=_batch200 --method=s-svgd --seed=$seed &

#   # taskset -c 9-10 python experiments/blr.py --epochs=2 --gpu=1 --batch=50 --nparticles=100 --lr=0.01 --delta=0.01 --suffix=_batch50 --method=gsvgd --effdim=10 --seed=$seed &
#   # taskset -c 11-12 python experiments/blr.py --epochs=2 --gpu=1 --batch=50 --nparticles=100 --lr=0.01 --delta=0.01 --suffix=_batch50 --method=gsvgd --effdim=15 --seed=$seed
#   wait
#   echo "finished seed=$seed"
#   done

