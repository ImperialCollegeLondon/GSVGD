for nparticles in 50 500 1000
    do
    for dim in 50 500 100
        do
        for lr in 0.1 0.01 0.001
            do 
                python maxSVGD_Gaussian.py --epoch=50000 --lr=$lr --num_samples=$nparticles --dim=$dim
            done
        done
    done