## covertype
epochs=10000
batch=300
nparticles=200
data=covertype_sub
suffix=_$data

# run reference HMC
for seed in 0 1 2 3 4 5 6 7 8 9
  do
    taskset -c 6-10 python experiments/blr.py --gpu=-1 --nparticles=$nparticles --epochs=$epochs --lr=0.1 --delta=0.1 \
    --method=hmc --suffix=$suffix --data=$data
  done
