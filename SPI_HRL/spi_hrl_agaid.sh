#!/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -J spi_agaid
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -o spi_hrl_agaid.out
#SBATCH -e spi_hrl_agaid.err
#SBATCH --mem=2G

for d_size in 20 100; do
    for n_wedge in 5 20; do
        for baseline in .8; do
            python3 main.py --env_type agaid --env intvn28_act4_prec1 --d_size $d_size --n_wedge $n_wedge --baseline $baseline --method uncertainty --regret 10
            python3 main.py --env_type agaid --env intvn28_act4_prec1 --d_size $d_size --n_wedge $n_wedge --baseline $baseline --method rmax --regret 10
        done
    done
done 





