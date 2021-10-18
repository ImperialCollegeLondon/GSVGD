# # covertype
epochs=10000
batch=300
nparticles=200
data=covertype_sub
suffix=_$data

for seed in 1 2 3 4 5 # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
  do
    taskset -c 6-10 python experiments/blr.py --gpu=-1 --nparticles=$nparticles --epochs=$epochs --lr=0.1 --delta=0.1 \
    --method=hmc --suffix=$suffix --data=$data
  done
