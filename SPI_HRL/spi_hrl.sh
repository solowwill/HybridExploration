#!/bin/bash



for d_size in 20 100; do
    for n_wedge in 5 20; do
        for baseline in .8; do
            python3 main.py --env 25x25_lava --d_size $d_size --n_wedge $n_wedge --baseline $baseline --method uncertainty --regret 10 --grid True --obstacle_reward -10 --movement_reward -1 --goal_reward 100 --path rewards/
            python3 main.py --env 25x25_lava --d_size $d_size --n_wedge $n_wedge --baseline $baseline --method rmax_dataset --regret 10 --grid True --obstacle_reward -10 --movement_reward -1 --goal_reward 100 --path rewards/
            python3 main.py --env 25x25_lava --d_size $d_size --n_wedge $n_wedge --baseline $baseline --method rmax --regret 10 --grid True --obstacle_reward -10 --movement_reward -1 --goal_reward 100 --path rewards/
            python3 main.py --env 25x25_lava --d_size $d_size --n_wedge $n_wedge --baseline $baseline --method uncertainty_v2 --regret 10 --grid True --obstacle_reward -10 --movement_reward -1 --goal_reward 100 --path rewards/
        done
    done
done 





